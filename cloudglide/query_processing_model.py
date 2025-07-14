# query_processing_model2.py
from cloudglide.event import Event, next_event_counter

import heapq
import random
import math
from typing import List, Tuple, Dict
from collections import deque

from cloudglide.job import Job

import logging

# Configure logging for better debugging and traceability
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def assign_memory_tier(hit_rate: float) -> str:
    """
    Assigns a memory tier ('DRAM', 'SSD', 'S3') based on hit rate, using probabilities.

    Args:
        hit_rate (float): The hit rate for caching.
        base_cores (int): Base number of CPU cores.
        cpu_cores (int): Total number of CPU cores.

    Returns:
        str: Assigned memory tier.
    """
    # Ensure hit_rate is within [0, 1]
    hit_rate = max(0, min(hit_rate, 1))

    # Define tier probabilities based on hit rate
    hit_probs = {
        "DRAM": hit_rate,
        "SSD": hit_rate / 2,
        "S3": 1 - hit_rate - (hit_rate / 2)
    }

    # Generate random number and assign memory tier
    rand_num = random.random()
    for tier, prob in hit_probs.items():
        if rand_num < prob:
            return tier
        rand_num -= prob

    return "S3"  # Default to S3 if no match


def peek_next_event_time(events):
    return events[0].time if events else float("inf")


def update_dram_nodes(
    dram_nodes: List[List[Job]],
    dram_job_counts: List[int],
    num_nodes: int
) -> Tuple[List[List[Job]], List[int]]:
    """
    Updates dram_nodes and dram_job_counts to match the number of nodes, redistributing jobs if needed.

    Args:
        dram_nodes (List[List[Job]]): Current list of DRAM nodes with their jobs.
        dram_job_counts (List[int]): Current count of jobs per DRAM node.
        num_nodes (int): Desired number of DRAM nodes.

    Returns:
        Tuple[List[List[Job]], List[int]]: Updated dram_nodes and dram_job_counts.
    """
    current_num_nodes = len(dram_nodes)

    if num_nodes > current_num_nodes:
        # Add new nodes to handle more jobs
        for _ in range(num_nodes - current_num_nodes):
            dram_nodes.append([])
            dram_job_counts.append(0)
    elif num_nodes < current_num_nodes:
        # Remove excess nodes and redistribute jobs
        for _ in range(current_num_nodes - num_nodes):
            jobs_to_reassign = dram_nodes.pop()
            dram_job_counts.pop()
            # Redistribute jobs from the removed node
            for job in jobs_to_reassign:
                new_node_index = job.job_id % num_nodes
                dram_nodes[new_node_index].append(job)
                dram_job_counts[new_node_index] += 1

    return dram_nodes, dram_job_counts

def simulate_io(
    current_second: float,
    hit_rate: float,
    num_nodes: int,
    io_jobs: deque,
    io_bandwidth: int,
    memory_bandwidth: int,
    phase: int,
    buffer_jobs: deque,
    cpu_jobs: deque,
    finished_jobs: deque,
    job_memory_tiers: Dict[int, str],
    dram_nodes: List[List[Job]],
    dram_job_counts: List[int],
    second_range: float,
    events: List[Event],
    architecture: int
):
    """
    Simulates I/O operations with cached memory tiers (DRAM, SSD, S3), allocating bandwidth per job.

    Args:
        current_second (float): Current simulation second.
        hit_rate (float): Hit rate for memory tier assignment.
        num_nodes (int): Number of DRAM nodes.
        io_jobs (deque): Queue of I/O jobs.
        io_bandwidth (int): I/O bandwidth available.
        memory_bandwidth (int): Memory bandwidth available.
        network_bandwidth (int): Network bandwidth available.
        buffer_jobs (deque): Buffer queue for jobs.
        job_memory_tiers (Dict[int, str]): Mapping of job_id to memory tier.
        dram_nodes (List[List[Job]]): List of DRAM nodes with their jobs.
        dram_job_counts (List[int]): Number of jobs per DRAM node.
        second_range (float): Simulation time step.
        - architecture < 2: cached tiers (DRAM, SSD, S3) with DRAM nodes
        - architecture == 2: elastic pool, no DRAM nodes
    """
    # only run on arrival or io_done or scale_check
    if phase not in ("arrival", "io_done", "scale_check"):
        return
    # also no‐op if no io_jobs
    if not io_jobs:
        return

    
    
    # ----------------------------
    # Case: QaaS
    # ----------------------------
    if architecture == 3:
        for job in list(io_jobs):
            elapsed = current_second - max(job.start_timestamp, current_second - second_range)
            cores = assign_cores_to_job_qaas(job)
            bw = cores * 150  # bytes per second

            if job.data_scanned_progress == 0:
                # Remove job from I/O queue and move it to the buffer once scanning is done
                io_jobs.remove(job)
                if job not in buffer_jobs and job not in cpu_jobs:
                    finished_jobs.append(job)
                continue

            # amount processable this tick
            processed = bw * elapsed
            if job.data_scanned_progress > processed:
                job.data_scanned_progress -= processed
                job.io_time += elapsed / 1000
            else:
                # compute exact time to finish remaining bytes
                rem = job.data_scanned_progress
                finish_delta = rem / (bw)
                # schedule completion event
                schedule_event(job, current_second + finish_delta, 'io_done', events)
                # account for I/O time until finish
                job.io_time += finish_delta
                job.data_scanned_progress = 0
        return
    
    # ----------------------------
    # Phase 1: Memory Tier Assignment
    # ----------------------------
    new_jobs = []
    for job in list(io_jobs):
        jid = job.job_id
        if jid not in job_memory_tiers:
            tier = assign_memory_tier(hit_rate)
            job_memory_tiers[jid] = tier
            if architecture < 2 and tier == 'DRAM':
                idx = jid % num_nodes
                job.dram_node_index = idx
                dram_nodes[idx].append(job)
                dram_job_counts[idx] += 1
        new_jobs.append((job, job_memory_tiers[jid]))

    # Pre-calc DRAM node distribution if cached
    if architecture < 2:
        dram_nodes, dram_job_counts = update_dram_nodes(
            dram_nodes, dram_job_counts, num_nodes)
        
    # ----------------------------
    # Phase 2: Bandwidth Allocation and Processing
    # ----------------------------
    # Phase 2: Bandwidth Allocation and Processing
    for job, tier in new_jobs:
        # Choose bandwidth source
        scan_bw = {'DRAM': memory_bandwidth, 'SSD': io_bandwidth, 'S3': 1000}
        bw = scan_bw.get(tier, 0)
        if bw == 0:
            continue
        # Compute number of peers
        if tier == 'DRAM' and architecture < 2:
            node_idx = job.dram_node_index
            peers = dram_job_counts[node_idx]
            eff_bw = bw
        else:
            eff_bw = bw
            peers = len([j for j in io_jobs if job_memory_tiers[j.job_id] == tier])
        per_job_bw = eff_bw / peers if peers > 0 else 0
        
        # Progress logic
        if job.data_scanned_progress == 0:
            continue
        elapsed = current_second - max(job.start_timestamp,
                                       current_second - second_range)
        # work in bytes per second, progress in bytes
        needed = job.data_scanned_progress * 1000
        avail = per_job_bw * elapsed
        if needed > avail:
            job.data_scanned_progress -= avail / 1000
            job.io_time += elapsed / 1000
            # schedule finish
            rem = job.data_scanned_progress
            finish_delta_ms = math.ceil(rem / (per_job_bw / 1000 or float('inf')))
            finish_time = current_second + finish_delta_ms
            schedule_event(job, finish_time, "io_done", events)
        else:
            job.io_time += job.data_scanned_progress / per_job_bw if per_job_bw else 0
            job.data_scanned_progress = 0
            # finalize removal
            if architecture < 2 and tier == 'DRAM':
                idx = job.dram_node_index
                dram_nodes[idx].remove(job)
                dram_job_counts[idx] -= 1
            io_jobs.remove(job)
            if job not in buffer_jobs and job not in cpu_jobs:
                finished_jobs.append(job)
                

def assign_cores_to_jobs(
    cpu_jobs: List[Job],
    shuffle,
    num_cores: int,
    current_second
) -> List[int]:
    """
    Assigns cores to CPU-bound jobs based on their CPU time requirements.

    Args:
        cpu_jobs (List[Job]): List of CPU-bound jobs.
        num_cores (int): Total available CPU cores.
        time_limit (float): Time limit for execution.

    Returns:
        List[int]: Number of cores allocated to each job.
    """
    time_limit = 10
    
    n = len(cpu_jobs)
    allocations = [0] * n

    # 1) Identify which jobs have actually started
    eligible = [
        i for i, job in enumerate(cpu_jobs)
        if (current_second - job.start_timestamp > 0)
        and (
            # either it has no shuffle left…
            job.data_shuffle == 0
            # …or, if it’s in the shuffle dict, its value is 0
            or shuffle.get(job.job_id, 0) == 0
        )
    ]
    n = len(eligible)
    if n == 0:
        return allocations  # no one gets cores

        # If fewer cores than jobs, give one core to the first num_cores jobs
    if num_cores <= n:
        alloc = [1 if i < num_cores else 0 for i in range(n)]
        return alloc

    # Everyone gets at least one core
    alloc = [1] * n
    extras = num_cores - n

    # Compute each job's 'weight' based on required CPU time
    required = [math.ceil(job.cpu_time / (time_limit * 1000)) for job in cpu_jobs]
    total_req = sum(required)
    if total_req == 0:
        # If no job requires CPU time (edge-case), spread extras evenly
        for i in range(extras):
            alloc[i % n] += 1
        return alloc

    # Compute each job's ideal extra share (as float)
    ideal = [req / total_req * extras for req in required]

    # Take the floor of each share, track remainders
    floors     = [math.floor(x) for x in ideal]
    remainders = [ideal[i] - floors[i] for i in range(n)]

    # Add the floored extras
    for i in range(n):
        alloc[i] += floors[i]

    used = sum(floors)
    left = extras - used  # how many cores remain to distribute

    # Distribute the remaining cores to jobs with largest remainder
    for i in sorted(range(n), key=lambda i: remainders[i], reverse=True)[:left]:
        alloc[i] += 1

    return alloc

def schedule_event(job: Job, timestamp: float, event_type: str, events: List[Event]):
    """
    Helper to push or reschedule job events, using flat next_* slots
    instead of a dict lookup for max speed.
    """
    # choose the correct slot on the Job
    if event_type == "io_done":
        slot_name = "next_io_done"
    elif event_type == "shuffle_done":
        slot_name = "next_shuffle_done"
    else:  # "cpu_done"
        slot_name = "next_cpu_done"

    if job.scheduled:
        # simply overwrite that slot
        setattr(job, slot_name, timestamp)
        heapq.heappush(events, Event(timestamp, next_event_counter(), job, event_type))
    else:
        # only schedule the very next event
        nxt = peek_next_event_time(events)
        if timestamp <= nxt:
            job.scheduled = True
            # initialize your slot as well
            setattr(job, slot_name, timestamp)
            heapq.heappush(events, Event(timestamp, next_event_counter(), job, event_type))


def simulate_cpu(
    current_second: float,
    cpu_jobs: List[Job],
    phase: int,
    cpu_cores: int,
    cpu_cores_per_node: int,
    network_bandwidth: int,
    finished_jobs: List[Job],
    shuffle_jobs: List[Job],
    io_jobs: deque,
    waiting_jobs: deque,
    shuffle: Dict[int, int],
    memory: List[float],
    second_range: float,
    events: List[Event],
    architecture,
    parallelizable_portion: float = 0.9,
):
    """
    Simulates CPU operations, allocating cores to jobs and handling shuffles.

    Args:
        current_second (float): Current simulation second.
        cpu_jobs (List[Job]): List of CPU-bound jobs.
        cpu_cores (int): Total available CPU cores.
        cpu_cores_per_node (int): CPU cores per node.
        network_bandwidth (int): Network bandwidth available.
        finished_jobs (List[Job]): List of finished jobs.
        shuffle_jobs (List[Job]): List of jobs currently in shuffle.
        io_jobs (deque): Queue of I/O jobs.
        shuffle (Dict[int, int]): Mapping of job_id to shuffle status.
        memory (List[float]): Current memory usage.
        second_range (float): Simulation time step.
        parallelizable_portion (float): Portion of the job that is parallelizable.

    Returns:
        None
    """
    
    if phase not in ("arrival","cpu_done","shuffle_done","scale_check"):
        return 0

    # and no-op if there’s nothing to do
    if not cpu_jobs:
        return 0

    if architecture >=3:
        return simulate_cpu_qaas(current_second, cpu_jobs, network_bandwidth,
                finished_jobs, shuffle_jobs, waiting_jobs, io_jobs, {}, memory, second_range, events)
    
    # Determine per-node core threshold
    per_node = cpu_cores_per_node
    
    # Assign cores to jobs
    num_jobs = len((cpu_jobs))
    if num_jobs == 0:
        return 0
    
    core_allocation = assign_cores_to_jobs(cpu_jobs, shuffle, cpu_cores, current_second)
    
    # Update shuffle_jobs list based on threshold
    for job, cores_assigned in zip(cpu_jobs, core_allocation):
        if cores_assigned > per_node and job.data_shuffle > 0:
            if job not in shuffle_jobs:
                shuffle_jobs.append(job)
        elif job in shuffle_jobs and job.data_shuffle <= 0:
            shuffle_jobs.remove(job)

    shuffle_count = sum(
        1 for job, cores_assigned in zip(cpu_jobs, core_allocation)
        if cores_assigned > per_node and job.data_shuffle > 0
    )

    # Process each job
    to_remove = []
    for job, cores_assigned in zip(cpu_jobs, core_allocation):
        elapsed_time = (current_second - max(job.start_timestamp, current_second - second_range))
        # Default speedup
        if cores_assigned == 0:
            job.processing_time += elapsed_time / 1000
            continue

        # Amdahl's Law if parallelizable
        if cores_assigned > per_node:
            shuffle[job.job_id] = 1
            speedup_factor = 1 / ((1 - parallelizable_portion)
                                  + (parallelizable_portion / (cores_assigned / per_node)))
            # Node involvement fraction
            if architecture < 2:
                nodes_involved = math.ceil(cores_assigned / cpu_cores_per_node) / (cpu_cores / cpu_cores_per_node)
            else:
                nodes_involved = 1  # not used in arch2 timing calc
        else:
            shuffle[job.job_id] = 0
            speedup_factor = 1
            nodes_involved = 1
        
        # Shuffle phase
        if shuffle[job.job_id] == 1 and job.data_shuffle > 0:
            
            if architecture == 2:
                per_job_bw = 50 * cores_assigned
            else:
                per_job_bw = network_bandwidth / shuffle_count if shuffle_count > 0 else 0
            # compute shuffle progress
            total_needed = job.data_shuffle * 1000
            available = per_job_bw * elapsed_time
            if total_needed > available:
                job.data_shuffle -= available / 1000
                job.shuffle_time += elapsed_time / 1000
                finish_delta = math.ceil(job.data_shuffle / (per_job_bw / 1000 or float('inf')))
                finish_time = current_second + finish_delta
                schedule_event(job, finish_time, "shuffle_done", events)
            else:
                job.shuffle_time += job.data_shuffle / per_job_bw if per_job_bw else 0
                job.data_shuffle = 0
        
        # CPU phase if shuffle done - IF YOU WANT TO SPLIT THEM LOGICALLY UNCOMMENT THIS - BUT THEN FIX THE ALLOCATION ISSUE
        if job.data_shuffle == 0 or shuffle[job.job_id] == 0:
            work = min(cores_assigned, per_node) * speedup_factor * elapsed_time
            
            denom = (min(cores_assigned, per_node)
                     * (nodes_involved if architecture < 2 else 1)
                     * speedup_factor * 1000)

            if job.cpu_time_progress > work:
                job.cpu_time_progress -= work
                job.processing_time += elapsed_time / 1000
                # Estimate finish
                finish_delta = math.ceil(job.cpu_time_progress / (min(cores_assigned, per_node) * nodes_involved * speedup_factor))
                finish_time = current_second + finish_delta
                schedule_event(job, finish_time, "cpu_done", events)
            else:
                # finalize
                time_used = job.cpu_time_progress / denom if denom else 0
                job.processing_time += time_used
                job.cpu_time_progress = 0
                to_remove.append(job)
                job_finalization(job, memory, cpu_jobs, shuffle_jobs,
                                 finished_jobs, io_jobs, waiting_jobs, current_second)
    for job in to_remove:
        cpu_jobs.remove(job)
    return sum(core_allocation) if architecture == 2 else 0

def job_finalization(
    job: Job,
    memory: List[float],
    cpu_jobs: List[Job],
    shuffle_jobs: List[Job],
    finished_jobs: List[Job],
    io_jobs: List[Job],
    waiting_jobs: List[Job],
    current_second: float,
    delta: float = 0.3,
    p: float = 4.0
):
    """
    Finalizes a job after completion, removes it from active queues, adjusts memory and shuffle status,
    and computes all five execution-time estimators.

    Args:
        job (Job): The job to finalize.
        memory (List[float]): Current memory usage.
        cpu_jobs (List[Job]): List of CPU-bound jobs.
        shuffle_jobs (List[Job]): List of jobs currently in shuffle.
        finished_jobs (List[Job]): List of finished jobs.
        io_jobs (List[Job]): List of I/O-bound jobs.
        waiting_jobs (List[Job]): List of waiting jobs.
        current_second (float): Current simulation second.
        δ (float): Fixed offset to account for parsing/coordination overhead.
        p (float): Exponent for the power-mean estimator.

    Returns:
        None
    """
    # 1. bookkeeping
    memory[0] -= job.data_scanned / 4
    job.end_timestamp = current_second

    # 2. extract per-phase times
    T_io      = job.io_time
    T_cpu     = job.processing_time
    T_shuffle = job.shuffle_time

    # 3. basic sum and cpu-only
    T_sum = T_io + T_cpu + T_shuffle
    T_cpu_only = T_cpu

    # 4. max + offset
    T_max_offset = max(T_io, T_cpu, T_shuffle) + delta

    # 5. power-mean
    T_pm = (T_io**p + T_cpu**p + T_shuffle**p)**(1.0/p)

    # 6. multi-wave: sum of per-wave maxes
    #    assumes job.io_phases, job.cpu_phases, job.shuffle_phases are lists of equal length
    # k = 3
    # waves = []
    T_mw = (
            max(0.8 * T_cpu, 0.2 * T_shuffle) +
            T_io +
            max(0.1 * T_cpu, 0.6 * T_shuffle) +
            max(0.1 * T_cpu, 0.2 * T_shuffle)
        )

    job.query_exec_time = T_max_offset
    job.query_exec_time_queueing = job.query_exec_time + \
        max(job.queueing_delay, job.buffer_delay)

    # 8. remove from active lists, mark finished
    if job in shuffle_jobs:
        shuffle_jobs.remove(job)
    if job not in waiting_jobs and job not in io_jobs:
        finished_jobs.append(job)


def assign_cores_to_job_qaas(job: Job, time_limit: float = 2) -> int:
    """
    Determines the number of cores required for a QAas job based on its CPU time.

    Args:
        job (Job): The job to assign cores to.
        time_limit (float, optional): Time limit for execution in seconds. Defaults to 2.

    Returns:
        int: Number of cores required.
    """
    # Determine the target execution time based on the given ranges
    if job.cpu_time > 6000000:
        target_execution_time = 18 * 1000  # 18 seconds in milliseconds
    elif job.cpu_time > 4000000:
        # Linearly interpolate the target execution time between 6 and 18 seconds
        target_execution_time = (
            6 + (12 * (job.cpu_time - 4000000) / (6000000 - 4000000))) * 1000
    elif job.cpu_time > 2000000:
        # Linearly interpolate the target execution time between 2 and 6 seconds
        target_execution_time = (
            2 + (4 * (job.cpu_time - 2000000) / (4000000 - 2000000))) * 1000
    else:
        target_execution_time = time_limit * 1000  # Default time limit in milliseconds

    required_cores = math.ceil(job.cpu_time / target_execution_time)

    return required_cores


def assign_cores_to_jobs_qaas(cpu_jobs: List[Job], time_limit: float) -> List[int]:
    """
    Assigns cores to QAas CPU-bound jobs based on their CPU time requirements.

    Args:
        cpu_jobs (List[Job]): List of CPU-bound QAas jobs.
        time_limit (float): Time limit for execution in seconds.

    Returns:
        List[int]: Number of cores allocated to each job.
    """
    core_allocation = []

    for job in cpu_jobs:
        required_cores = assign_cores_to_job_qaas(job, time_limit)
        core_allocation.append(required_cores)

    return core_allocation


def simulate_cpu_qaas(
    current_second: float,
    cpu_jobs: List[Job],
    network_bandwidth: int,
    finished_jobs: List[Job],
    shuffle_jobs: List[Job],
    waiting_jobs: List[Job],
    io_jobs: deque,
    shuffle: Dict[int, int],
    memory: List[float],
    second_range: float,
    events
) -> int:
    """
    Simulates CPU operations for a QAas (Query as a Service) system, allocating cores to jobs and handling shuffles.

    Args:
        current_second (float): Current simulation second.
        cpu_jobs (List[Job]): List of CPU-bound jobs.
        cpu_cores (int): Total available CPU cores.
        network_bandwidth (int): Network bandwidth available.
        finished_jobs (List[Job]): List of finished jobs.
        shuffle_jobs (List[Job]): List of jobs currently in shuffle.
        io_jobs (deque): Queue of I/O jobs.
        shuffle (Dict[int, int]): Mapping of job_id to shuffle status.
        memory (List[float]): Current memory usage.
        second_range (float): Simulation time step.

    Returns:
        int: Total cores allocated.
    """
    # Assign cores to jobs
    num_jobs = len(cpu_jobs)
    if num_jobs == 0:
        return 0
    
    # 1) assign cores (using your QA-specific allocator)
    core_allocation = assign_cores_to_jobs_qaas(cpu_jobs, 4)

     # 2) update shuffle_jobs set
    for job, cores in zip(cpu_jobs, core_allocation):
        if cores > 4 and job.data_shuffle > 0 and job not in shuffle_jobs:
            shuffle_jobs.append(job)
        if job in shuffle_jobs and job.data_shuffle == 0:
            shuffle_jobs.remove(job)

    # 3) process each job
    for job, cores in zip(list(cpu_jobs), core_allocation):
        elapsed = current_second - max(job.start_timestamp,
                                       current_second - second_range)

        # no cores → just accrue queueing/processing delay
        if cores == 0:
            job.processing_time += elapsed
            continue

        # parallel fraction
        p = 0.9
        if cores > 4:
            shuffle[job.job_id] = 1
            speedup = 1 / ((1 - p) + (p / (cores / 4)))
        else:
            shuffle[job.job_id] = 0
            speedup = 1

        # SHUFFLE PHASE
        if shuffle[job.job_id] == 1 and job.data_shuffle > 0:
            bw = cores * 50  # bytes/sec per job
            transferred = bw * elapsed
            if job.data_shuffle > transferred:
                job.data_shuffle -= transferred
                job.shuffle_time += elapsed / 1000
                # estimate finish in seconds
                delta = math.ceil(job.data_shuffle / (bw / 1000 or float('inf')))
                # print("Status at: ", current_second, "for ", job.job_id, ": Needs ", delta, "with shuffle: ",job.data_shuffle, "and bw:" , bw)
                schedule_event(job, current_second + delta, "shuffle_done", events)
            else:
                job.shuffle_time += job.data_shuffle / bw if bw else 0
                job.data_shuffle = 0
                shuffle[job.job_id] = 0
                shuffle_jobs.remove(job)

        # CPU PHASE (once shuffle is done)
        if job.data_shuffle == 0 or shuffle[job.job_id] == 0:
            # how much CPU-seconds we can do this tick
            work = cores * speedup * elapsed
            if job.cpu_time_progress > work:
                job.cpu_time_progress -= work
                job.processing_time += elapsed / 1000
                # schedule cpu_done
                remaining = job.cpu_time_progress
                # 4 cores base → effective rate = 4 * speedup per second
                rate = min(cores, 4) * speedup
                delta = math.ceil(remaining / (rate or float('inf'))) 
                # print("Status at: ", current_second, "for ", job.job_id, ": Needs ", delta, "with CPU time: ",job.cpu_time_progress, "and rate:" , 4*speedup, "for assgined cores: ", cores)           
                schedule_event(job, current_second + delta, "cpu_done", events)
            else:
                job.processing_time += job.cpu_time_progress / (1000 *(min(cores, 4) * speedup or 1))
                job.cpu_time_progress = 0
                # finalize
                job_finalization(job, memory, cpu_jobs, shuffle_jobs,
                finished_jobs, io_jobs, waiting_jobs, current_second)
                
    return sum(core_allocation)
from cloudglide.config import ArchitectureType
from cloudglide.event import Event, next_event_counter

import heapq
import random
import math
from typing import List, Tuple, Dict
from collections import deque

from cloudglide.job import Job

import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def assign_memory_tier(hit_rate, architecture, n, warmup_rate):
    """
    Assigns a memory tier ('DRAM', 'SSD', 'S3') based on cache hit rate and architecture type.
    For Elastic Pool, the DRAM hit probability warms up exponentially as more queries complete.

    Args:
        hit_rate (float): Steady-state cache hit rate (P ∈ [0,1]).
        architecture (int): Architecture type (e.g., ArchitectureType.ELASTIC_POOL).
        n (int): Number of completed queries (used for warmup progression).
        warmup_rate (float): Warmup rate constant (γ), controlling how fast DRAM cache warms up.

    Returns:
        str: Assigned memory tier ('DRAM', 'SSD', or 'S3').
    """
    # Ensure hit_rate is within [0, 1]
    hit_rate = max(0.0, min(1.0, hit_rate))

    # Default probabilities (DWaaS, QAAS, etc.)
    P_DRAM = hit_rate

    # ----------------------------
    # Elastic Pool: apply warmup model
    # ----------------------------
    if architecture == ArchitectureType.ELASTIC_POOL:
        # Exponential convergence toward steady-state hit_rate
        P_DRAM = hit_rate * (1 - math.exp(-warmup_rate * n))
        P_DRAM = max(0.0, min(1.0, P_DRAM)) 

    remaining = 1 - P_DRAM
    P_SSD = remaining * 0.67
    P_S3 = remaining * 0.33

    hit_probs = {
        "DRAM": P_DRAM,
        "SSD": P_SSD,
        "S3": P_S3,
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
    architecture: int,
    config
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
    if architecture == ArchitectureType.QAAS:
        for job in list(io_jobs):
            elapsed = current_second - max(job.start_timestamp, current_second - second_range)
            cores = assign_cores_to_job_qaas(job, config.qaas_base_time_limit)
            bw = cores * config.qaas_io_per_core_bw  # bytes per second

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
    # Update DRAM node distribution BEFORE processing jobs
    if architecture in [ArchitectureType.DWAAS, ArchitectureType.DWAAS_AUTOSCALING]:
        dram_nodes, dram_job_counts = update_dram_nodes(
            dram_nodes, dram_job_counts, num_nodes)

    new_jobs = []
    for job in list(io_jobs):
        jid = job.job_id
        if jid not in job_memory_tiers:
            tier = assign_memory_tier(hit_rate, architecture, len(finished_jobs), config.cache_warmup_gamma)
            job_memory_tiers[jid] = tier
            if architecture in [ArchitectureType.DWAAS, ArchitectureType.DWAAS_AUTOSCALING] and tier == 'DRAM':
                idx = jid % num_nodes
                job.dram_node_index = idx
                dram_nodes[idx].append(job)
                dram_job_counts[idx] += 1
        new_jobs.append((job, job_memory_tiers[jid]))
        

    # ----------------------------
    # Phase 2: Bandwidth Allocation and Processing
    for job, tier in new_jobs:
        # Choose bandwidth source
        scan_bw = {'DRAM': memory_bandwidth, 'SSD': io_bandwidth, 'S3': config.s3_bandwidth}
        bw = scan_bw.get(tier, 0)
        if bw == 0:
            continue
        # Compute number of peers
        if tier == 'DRAM' and architecture in [ArchitectureType.DWAAS, ArchitectureType.DWAAS_AUTOSCALING]:
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
            if architecture in [ArchitectureType.DWAAS, ArchitectureType.DWAAS_AUTOSCALING] and tier == 'DRAM':
                idx = job.dram_node_index
                dram_nodes[idx].remove(job)
                dram_job_counts[idx] -= 1
            io_jobs.remove(job)
            if job not in buffer_jobs and job not in cpu_jobs:
                finished_jobs.append(job)
                

def assign_cores_to_jobs(cpu_jobs: List[Job], shuffle, num_cores: int, current_second, time_limit) -> List[int]:
    """
    Assigns CPU cores fairly across eligible jobs based on their remaining CPU time.
    """
    final_alloc = [0] * len(cpu_jobs)

    # Determine eligible jobs
    eligible_indices = [
        i for i, job in enumerate(cpu_jobs)
        if (current_second >= job.start_timestamp)
        and (job.data_shuffle == 0 or shuffle.get(job.job_id, 0) == 0)
    ]
    if not eligible_indices:
        return final_alloc

    eligible_jobs = [cpu_jobs[i] for i in eligible_indices]
    n = len(eligible_jobs)

    # Case: fewer cores than jobs
    if num_cores <= n:
        for i, idx in enumerate(eligible_indices[:num_cores]):
            final_alloc[idx] = 1
        return final_alloc

    # Base allocation
    alloc = [1] * n
    extras = num_cores - n

    # Weight by required CPU work
    required = [max(1, math.ceil(job.cpu_time / (time_limit * 1000))) for job in eligible_jobs]
    total_req = sum(required)

    if total_req == 0:
        for i in range(extras):
            alloc[i % n] += 1
    else:
        ideal = [req / total_req * extras for req in required]
        floors = [math.floor(x) for x in ideal]
        remainders = [ideal[i] - floors[i] for i in range(n)]

        for i in range(n):
            alloc[i] += floors[i]
        used = sum(floors)
        left = extras - used
        for i in sorted(range(n), key=lambda i: remainders[i], reverse=True)[:left]:
            alloc[i] += 1

    # Map allocations back to full job list
    for alloc_val, idx in zip(alloc, eligible_indices):
        final_alloc[idx] = alloc_val

    return final_alloc


def schedule_event(job: Job, timestamp: float, event_type: str, events: List[Event]):
    """
    Push or update a job's next event, avoiding duplicates.
    """
    slot_name = {
        "io_done": "next_io_done",
        "shuffle_done": "next_shuffle_done",
        "cpu_done": "next_cpu_done",
    }[event_type]

    prev_time = getattr(job, slot_name, None)

    # Only reschedule if the new time is significantly different
    if prev_time is None or abs(prev_time - timestamp) > 1e-6:
        setattr(job, slot_name, timestamp)

        # Remove stale duplicates for the same (job_id, event_type)
        events[:] = [
            e for e in events
            if not (e.job and e.job.job_id == job.job_id and e.etype == event_type)
        ]
        heapq.heapify(events)

        heapq.heappush(events, Event(timestamp, next_event_counter(), job, event_type))
        job.scheduled = True
        # print(f"[SCHEDULED] {event_type:12s} for job {job.job_id} at {timestamp:.3f}s")


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
    config
):
    """
    Simulates CPU operations, allocating cores to jobs and handling shuffles.

    Args:
        current_second (float): Current simulation time (s).
        cpu_jobs (List[Job]): Active CPU-bound jobs.
        phase (int): Triggering phase ('arrival', 'cpu_done', etc.).
        cpu_cores (int): Total available cores.
        cpu_cores_per_node (int): Cores per node.
        network_bandwidth (int): Network bandwidth (bytes/s).
        finished_jobs (List[Job]): Completed jobs.
        shuffle_jobs (List[Job]): Jobs currently shuffling.
        io_jobs (deque): I/O-bound jobs.
        waiting_jobs (deque): Waiting jobs.
        shuffle (Dict[int, int]): job_id → shuffle flag.
        memory (List[float]): Total memory usage.
        second_range (float): Simulation timestep (ms).
        events (List[Event]): Global event queue.
        architecture (ArchitectureType): Current architecture type.
        config: Simulation configuration parameters.

    Returns:
        int: Total cores allocated (Elastic Pool only).
    """
    
    if phase not in ("arrival","cpu_done","shuffle_done","scale_check"):
        return 0

    # and no-op if there’s nothing to do
    if not cpu_jobs:
        return 0

    if architecture in [ArchitectureType.QAAS, ArchitectureType.QAAS_CAPACITY]:
        return simulate_cpu_qaas(current_second, cpu_jobs, network_bandwidth,
                finished_jobs, shuffle_jobs, waiting_jobs, io_jobs, {}, memory, second_range, events, config)
    
    # Determine per-node core threshold
    per_node = cpu_cores_per_node
    
    # Assign cores to jobs
    num_jobs = len((cpu_jobs))
    if num_jobs == 0:
        return 0

    core_allocation = assign_cores_to_jobs(cpu_jobs, shuffle, cpu_cores, current_second, config.core_alloc_window)

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
            speedup_factor = 1 / ((1 - config.parallelizable_portion)
                                  + (config.parallelizable_portion / (cores_assigned / per_node)))
            # Node involvement fraction
            if architecture in [ArchitectureType.DWAAS, ArchitectureType.DWAAS_AUTOSCALING]:
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
                per_job_bw = config.qaas_shuffle_bw_per_core * cores_assigned
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
        if job.data_shuffle == 0 or shuffle[job.job_id] == 0:
            work = min(cores_assigned, per_node) * speedup_factor * elapsed_time
            denom = (min(cores_assigned, per_node)
                     * (nodes_involved if architecture in [ArchitectureType.DWAAS, ArchitectureType.DWAAS_AUTOSCALING] else 1)
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
                job_finalization(
                    job,
                    memory,
                    cpu_jobs,
                    shuffle_jobs,
                    finished_jobs,
                    io_jobs,
                    waiting_jobs,
                    current_second,
                    config,
                )
    
    for job in to_remove:
        cpu_jobs.remove(job)
    return sum(core_allocation) if architecture == ArchitectureType.ELASTIC_POOL else 0

def job_finalization(
    job: Job,
    memory: List[float],
    cpu_jobs: List[Job],
    shuffle_jobs: List[Job],
    finished_jobs: List[Job],
    io_jobs: List[Job],
    waiting_jobs: List[Job],
    current_second: float,
    config,
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
        materialization_fraction (float): Fraction of intermediate data retained in memory.
    Returns:
        None
    """
    # 1. bookkeeping
    memory[0] -= job.data_scanned * config.materialization_fraction
    job.end_timestamp = current_second

    # per-phase times (seconds)
    T_io      = job.io_time
    T_cpu     = job.processing_time
    # print("Final processing time for ", job.job_id, "is ", job.processing_time)
    T_shuffle = job.shuffle_time

    # ---- compute estimators (raw, without δ) ----
    T_max = max(T_io, T_cpu, T_shuffle)
    T_sum = T_io + T_cpu + T_shuffle
    T_cpu_only = T_cpu

    T_pm = (T_io**config.pm_p + T_cpu**config.pm_p + T_shuffle**config.pm_p) ** (
        1.0 / config.pm_p
    )

    T_mw = (
        max(0.8 * T_cpu, 0.2 * T_shuffle) +
        T_io +
        max(0.1 * T_cpu, 0.6 * T_shuffle) +
        max(0.1 * T_cpu, 0.2 * T_shuffle)
    )

    job.estimators = {
        "max": T_max,
        "sum": T_sum,
        "cpu_only": T_cpu_only,
        "pm": T_pm,
        "mw": T_mw,  # rename to "mix" if you don't implement true multi-wave
    }

    # ---- choose official estimator and apply δ ----
    chosen_key = (
        config.default_estimator
        if config.default_estimator in job.estimators
        else "max"
    )
    chosen = job.estimators[chosen_key] + config.delta
    job.selected_estimator = chosen_key

    # ---- queue aggregation ----
    if config.queue_agg == "max":
        queue_total = max(job.queueing_delay, job.buffer_delay)
    else:  # default "sum"
        queue_total = job.queueing_delay + job.buffer_delay
    job.queue_total = queue_total

    job.query_exec_time = chosen
    job.query_exec_time_queueing = chosen + queue_total

    # 8. remove from active lists, mark finished
    if job in shuffle_jobs:
        shuffle_jobs.remove(job)
    if job not in waiting_jobs and job not in io_jobs:
        finished_jobs.append(job)


def assign_cores_to_job_qaas(job: Job, time_limit) -> int:

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
    events,
    config
) -> int:
    """
    Simulates CPU-stage execution for the QAaS (Query-as-a-Service) architecture.

    Each active query is assigned a number of cores based on its CPU demand and the
    configured time limit. Parallel speedup follows Amdahl's Law, and shuffling
    is modeled as a bandwidth-limited transfer between nodes. When shuffle completes,
    remaining CPU work progresses proportionally to the number of assigned cores.

    Args:
        current_second (float): Current simulation timestamp (in milliseconds).
        cpu_jobs (List[Job]): Active CPU-bound jobs being processed.
        network_bandwidth (int): Total available network bandwidth (bytes per second).
        finished_jobs (List[Job]): Jobs that have fully completed.
        shuffle_jobs (List[Job]): Jobs currently in shuffle phase.
        waiting_jobs (List[Job]): Jobs waiting for CPU resources.
        io_jobs (deque): Jobs still in or pending I/O.
        shuffle (Dict[int, int]): Mapping of job_id → shuffle activity flag (1 if in shuffle).
        memory (List[float]): List tracking current memory usage across nodes (index 0 = total MB).
        second_range (float): Simulation timestep window (milliseconds).
        events (List[Event]): Priority queue of future events (I/O, shuffle, CPU completions).
        config: Simulation configuration object containing all tunable constants:
            - qaas_base_cores
            - qaas_base_time_limit
            - qaas_shuffle_bw_per_core
            - materialization_fraction
            - parallelizable_portion

    Returns:
        int: Total number of CPU cores currently allocated across all active jobs.
    """
    if not cpu_jobs:
        return 0
    
    # 1) assign cores
    core_allocation = assign_cores_to_jobs_qaas(cpu_jobs, config.qaas_base_time_limit)

     # 2) update shuffle_jobs set
    for job, cores in zip(cpu_jobs, core_allocation):
        if cores > config.qaas_base_cores and job.data_shuffle > 0 and job not in shuffle_jobs:
            shuffle_jobs.append(job)
        if job in shuffle_jobs and job.data_shuffle == 0:
            shuffle_jobs.remove(job)

    to_remove = []
    # 3) process each job
    for job, cores in zip(list(cpu_jobs), core_allocation):
        elapsed = current_second - max(job.start_timestamp,
                                       current_second - second_range)

        # no cores → just accrue queueing/processing delay
        if cores == 0:
            job.processing_time += elapsed
            continue

        # parallel fraction
        p = config.parallelizable_portion
        if cores > config.qaas_base_cores:
            shuffle[job.job_id] = 1
            speedup = 1 / ((1 - p) + (p / (cores / config.qaas_base_cores)))
        else:
            shuffle[job.job_id] = 0
            speedup = 1

        # SHUFFLE PHASE
        if shuffle[job.job_id] == 1 and job.data_shuffle > 0:
            bw = cores * config.qaas_shuffle_bw_per_core
            transferred = bw * elapsed
            if job.data_shuffle > transferred:
                job.data_shuffle -= transferred
                job.shuffle_time += elapsed / 1000
                # estimate finish in seconds
                delta = math.ceil(job.data_shuffle / (bw / 1000 or float('inf')))
                schedule_event(job, current_second + delta, "shuffle_done", events)
            else:
                job.shuffle_time += job.data_shuffle / bw if bw else 0
                job.data_shuffle = 0
                shuffle[job.job_id] = 0
                shuffle_jobs.remove(job)

        # CPU Phase (once shuffle is done)
        if job.data_shuffle == 0 or shuffle[job.job_id] == 0:
            # how much CPU-seconds we can do this tick
            work = cores * speedup * elapsed
            if job.cpu_time_progress > work:
                job.cpu_time_progress -= work
                job.processing_time += elapsed / 1000
                # schedule cpu_done
                remaining = job.cpu_time_progress
                rate = min(cores, config.qaas_base_cores) * speedup
                delta = math.ceil(remaining / (rate or float('inf'))) 
                schedule_event(job, current_second + delta, "cpu_done", events)
            else:
                job.processing_time += job.cpu_time_progress / (1000 *(min(cores, config.qaas_base_cores) * speedup or 1))
                job.cpu_time_progress = 0 
                to_remove.append(job)
    
                job_finalization(
                    job,
                    memory,
                    cpu_jobs,
                    shuffle_jobs,
                    finished_jobs,
                    io_jobs,
                    waiting_jobs,
                    current_second,
                    config,
                )
    for job in to_remove:
        cpu_jobs.remove(job)
                
    return sum(core_allocation)
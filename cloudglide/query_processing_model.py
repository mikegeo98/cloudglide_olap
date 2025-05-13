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


def simulate_io_cached(
    current_second: float,
    hit_rate: float,
    num_nodes: int,
    io_jobs: deque,
    io_bandwidth: int,
    memory_bandwidth: int,
    network_bandwidth: int,
    buffer_jobs: deque,
    cpu_jobs: deque,
    finished_jobs: deque,
    job_memory_tiers: Dict[int, str],
    dram_nodes: List[List[Job]],
    dram_job_counts: List[int],
    second_range: float,
    events: List[Event]
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
    """
    # Define bandwidth for each memory tier -
    scan_bandwidth = {"DRAM": memory_bandwidth,
                      "SSD": io_bandwidth, "S3": 1000}

    # Update DRAM nodes for scaling
    dram_nodes, dram_job_counts = update_dram_nodes(
        dram_nodes, dram_job_counts, num_nodes)

    # ----------------------------
    # Phase 1: Memory Tier Assignment
    # ----------------------------
    new_jobs = []  # List to store jobs and their memory tiers for Phase 2

    for job in list(io_jobs):  # Create a static snapshot for safe iteration
        job_key = job.job_id

        # Assign memory tier if not already assigned
        if job_key not in job_memory_tiers:
            # Assuming parameters as per original code
            assigned_tier = assign_memory_tier(hit_rate)
            job_memory_tiers[job_key] = assigned_tier

            if assigned_tier == "DRAM":
                node_index = job_key % num_nodes  # Simple distribution based on job_id
                dram_nodes[node_index].append(job)
                dram_job_counts[node_index] += 1

        # Collect job and its memory tier for Phase 2
        memory_tier = job_memory_tiers[job_key]
        new_jobs.append((job, memory_tier))

    # ----------------------------
    # Phase 2: Bandwidth Allocation and Processing
    # ----------------------------
    for job, memory_tier in new_jobs:
        job_key = job.job_id

        # Retrieve bandwidth for the memory tier
        bandwidth = scan_bandwidth.get(memory_tier, 0)
        if bandwidth == 0:
            continue  # Skip processing if bandwidth is undefined

        # Calculate effective bandwidth based on memory tier and skew
        if memory_tier == "DRAM":
            skew = 0  # Adjust skew as needed
            effective_bandwidth = bandwidth * (1 - skew)
            node_index = job_key % num_nodes  # Consistent with assignment
            num_jobs = dram_job_counts[node_index]
        elif memory_tier == "SSD":
            skew = 0  # Adjust skew as needed
            effective_bandwidth = bandwidth * (1 - skew)
            # Correctly reference each job's job_id
            num_jobs = len(
                [j for j in io_jobs if job_memory_tiers.get(j.job_id) == memory_tier])
        else:
            effective_bandwidth = bandwidth
            num_jobs = len(
                [j for j in io_jobs if job_memory_tiers.get(j.job_id) == memory_tier])

        # Allocate bandwidth per job
        if num_jobs > 0:
            per_job_bandwidth = effective_bandwidth / num_jobs
        else:
            per_job_bandwidth = 0

        # Process job's data scanned progress
        if job.data_scanned_progress != 0:
            if per_job_bandwidth == 0:
                continue  # Decide how to handle, e.g., skip or assign default bandwidth
            elapsed_time = (current_second - max(job.start_timestamp, current_second - second_range))
            if 1000 * job.data_scanned_progress > per_job_bandwidth * elapsed_time:
                job.data_scanned_progress -= per_job_bandwidth * elapsed_time/1000
                
                # print("IO Appended job", job.job_id, "now has", job.data_scanned_progress)

                job.io_time += elapsed_time / 1000

                remaining_bytes = job.data_scanned_progress  # bytes
                # if per_job_bandwidth is in bytes per second, first convert to bytes/ms
                bandwidth_per_ms = per_job_bandwidth / 1000.0
                # compute how many ms are needed, always rounding up
                finish_delta_ms = math.ceil(remaining_bytes / bandwidth_per_ms)
                finish_time_ms = current_second + finish_delta_ms
                # print("IO Appended job", job.job_id, "expected to finish at", finish_time_ms, "bytes: ", remaining_bytes, "bandwidth", bandwidth_per_ms)
                if job.scheduled:
                    # Already in the heap—always push an updated estimate
                    job.next_time = finish_time_ms
                    heapq.heappush(events, Event(
                        finish_time_ms, next_event_counter(), job, "io_done"))
                else:
                    # No completion event yet—only schedule if it beats whatever is next
                    next_evt_time = peek_next_event_time(events)
                    # print("io", new_finish, next_evt_time)
                    if finish_time_ms <= next_evt_time:
                        job.scheduled = True
                        job.next_time = finish_time_ms
                        heapq.heappush(events, Event(
                            finish_time_ms, next_event_counter(), job, "io_done"))

            else:
                # Avoid division by zero
                increment = job.data_scanned_progress / \
                    per_job_bandwidth if per_job_bandwidth != 0 else 0
                job.io_time += increment
                job.data_scanned_progress = 0
                # print("IO Appended job", job.job_id, "now has", job.data_scanned_progress, "and finished at ", current_second)
        else:
            # Remove job if data scan is complete
            if memory_tier == "DRAM":
                node_index = job_key % num_nodes
                if job in dram_nodes[node_index]:
                    dram_nodes[node_index].remove(job)
                    dram_job_counts[node_index] -= 1
                else:
                    logging.warning(
                        f"Job ID {job_key} not found in DRAM Node {node_index} during removal.")

            io_jobs.remove(job)
            if job not in buffer_jobs and job not in cpu_jobs:
                finished_jobs.append(job)


def simulate_io_elastic_pool(
    current_second: float,
    hit_rate: float,
    base_cores: int,
    cpu_cores: int,
    io_jobs: deque,
    io_bandwidth: int,
    memory_bandwidth: int,
    buffer_jobs: deque,
    job_memory_tiers: Dict[int, str],
    second_range: float
):
    """
    Simulates I/O operations in an elastic pool, allocating bandwidth based on memory tier and core count.

    Args:
        current_second (float): Current simulation second.
        hit_rate (float): Hit rate for memory tier assignment.
        base_cores (int): Base number of CPU cores.
        cpu_cores (int): Total number of CPU cores.
        io_jobs (deque): Queue of I/O jobs.
        io_bandwidth (int): I/O bandwidth available.
        memory_bandwidth (int): Memory bandwidth available.
        buffer_jobs (deque): Buffer queue for jobs.
        job_memory_tiers (Dict[int, str]): Mapping of job_id to memory tier.
        second_range (float): Simulation time step.
    """
    # Define bandwidth for each memory tier
    scan_bandwidth = {"DRAM": memory_bandwidth,
                      "SSD": int(650 * cpu_cores / 4), "S3": 10000}

    # ----------------------------
    # Phase 1: Memory Tier Assignment
    # ----------------------------
    new_jobs = []  # List to store jobs and their memory tiers for Phase 2

    for job in list(io_jobs):
        job_key = job.job_id

        # Assign memory tier if not already assigned
        if job_key not in job_memory_tiers:
            job_memory_tiers[job_key] = assign_memory_tier(hit_rate)

        # Collect job and its memory tier for Phase 2
        memory_tier = job_memory_tiers[job_key]
        new_jobs.append((job, memory_tier))

    # ----------------------------
    # Phase 2: Bandwidth Allocation and Processing
    # ----------------------------
    for job, memory_tier in new_jobs:
        job_key = job.job_id

        # Retrieve bandwidth for the memory tier
        bandwidth = scan_bandwidth.get(memory_tier, 0)
        if bandwidth == 0:
            continue  # Skip processing if bandwidth is undefined

        # Calculate bandwidth per job for DRAM and SSD
        if memory_tier != "S3":
            skew = 0
            effective_bandwidth = bandwidth * (1 - skew)
            num = len(
                [j for j in io_jobs if job_memory_tiers[j.job_id] == memory_tier])
            if num > 0:
                per_job_bandwidth = effective_bandwidth / num
            else:
                per_job_bandwidth = 0  # Handle appropriately
        else:
            # For S3, consider all jobs in the same tier
            effective_bandwidth = bandwidth
            num = len(
                [j for j in io_jobs if job_memory_tiers[j.job_id] == memory_tier])
            if num > 0:
                per_job_bandwidth = effective_bandwidth / num
            else:
                per_job_bandwidth = 0  # Handle appropriately

        # Update job progress based on allocated bandwidth
        if job.data_scanned_progress != 0:
            if job.data_scanned_progress > per_job_bandwidth * second_range:
                job.data_scanned_progress -= per_job_bandwidth * second_range
                job.io_time += second_range
            else:
                job.io_time += job.data_scanned_progress / per_job_bandwidth
                job.data_scanned_progress = 0
        else:
            # Move job to buffer once scanning is complete
            io_jobs.remove(job)
            buffer_jobs.append(job)


def simulate_io_qaas(
    io_jobs: deque,
    network_bandwidth: int,
    buffer_jobs: deque,
    second_range: float,
    cpu_jobs: deque,
    finished_jobs: deque
):
    """
    Simulates I/O operations for a Qaas (Query as a Service) system, allocating bandwidth based on required cores.

    Args:
        io_jobs (deque): Queue of I/O jobs.
        network_bandwidth (int): Network bandwidth available.
        buffer_jobs (deque): Buffer queue for jobs.
        second_range (float): Simulation time step.
    """
    if io_jobs:
        for job in list(io_jobs):  # Convert deque to list for safe iteration
            # Assign the required number of cores to the job and calculate the per-job bandwidth
            required_cores = assign_cores_to_job_qaas(job)
            per_job_bandwidth = required_cores * 150  # Bandwidth per job based on cores

            # If there is data left to scan
            if job.data_scanned_progress != 0:
                # Process data based on bandwidth and time range
                if job.data_scanned_progress > per_job_bandwidth * second_range:
                    job.data_scanned_progress -= per_job_bandwidth * second_range
                    job.io_time += second_range
                else:
                    job.io_time += job.data_scanned_progress / per_job_bandwidth
                    job.data_scanned_progress = 0
            else:
                # Remove job from I/O queue and move it to the buffer once scanning is done
                io_jobs.remove(job)
                if job not in buffer_jobs and job not in cpu_jobs:
                    finished_jobs.append(job)
                # buffer_jobs.append(job)


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


def assign_cores_to_jobs_autoscaling(
    cpu_jobs: List[Job],
    num_cores: int,
    current_second: float
) -> List[int]:
    """
    Assigns cores to CPU-bound jobs based on their CPU time requirements for autoscaling.

    Args:
        cpu_jobs (List[Job]): List of CPU-bound jobs.
        num_cores (int): Total available CPU cores.
        time_limit (float): Time limit for execution.

    Returns:
        List[int]: Number of cores allocated to each job.
    """
    required_cores = [math.ceil(job.cpu_time / (time_limit * 1000))
                      for job in cpu_jobs]
    total_required_cores = sum(required_cores)

    initial_cores_per_job = num_cores // len(cpu_jobs) if cpu_jobs else 0
    core_allocation = [min(1, initial_cores_per_job) for _ in cpu_jobs]

    if num_cores > len(cpu_jobs):
        num_cores_remaining = num_cores - len(cpu_jobs)

        for i, job_cores in enumerate(required_cores):
            allocated_cores = (
                job_cores / total_required_cores) * num_cores_remaining
            core_allocation[i] += math.floor(allocated_cores)

    else:
        for i in range(num_cores):
            core_allocation[i] += 1

    return core_allocation


def simulate_cpu_nodes(
    current_second: float,
    cpu_jobs: List[Job],
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
    parallelizable_portion: float = 0.9
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
    # Assign cores to jobs
    num_jobs = len(list(cpu_jobs))

    if num_jobs > 0:
        # Assign cores based on job requirements
        core_allocation = assign_cores_to_jobs(cpu_jobs, shuffle, cpu_cores, current_second)

        # Iterate through jobs and their assigned cores
        for job, cores_assigned in zip(cpu_jobs.copy(), core_allocation):
            # Check for shuffling conditions
            if cores_assigned > cpu_cores_per_node and job.data_shuffle > 0:
                if job not in shuffle_jobs:
                    shuffle_jobs.append(job)
            if job in shuffle_jobs and job.data_shuffle <= 0:
                shuffle_jobs.remove(job)

        # Process each job
        for job, cores_assigned in zip(cpu_jobs.copy(), core_allocation):
            speedup_factor = 1.0
            nodes_involved = 1.0
            elapsed_time = (current_second - max(job.start_timestamp, current_second - second_range))
            if cores_assigned != 0:
                if cores_assigned > cpu_cores_per_node:
                    # Speedup calculation using Amdahl's Law (parameterized parallel portion)
                    speedup_factor = 1 / ((1 - parallelizable_portion) + (
                        parallelizable_portion / (cores_assigned / cpu_cores_per_node)))
                    shuffle[job.job_id] = 1
                    nodes_involved = math.ceil(cores_assigned/cpu_cores_per_node) 
                    nodes_involved /= cpu_cores/cpu_cores_per_node
                else:
                    # No speedup
                    shuffle[job.job_id] = 0
                
                # If job is in shuffle phase
                if shuffle[job.job_id] == 1 and job.data_shuffle > 0 and job in shuffle_jobs:
                    # print("CPU Appended job", job.job_id, "now is shuffling")
                    # Network bandwidth allocation per job
                    num_shuffle_jobs = len(shuffle_jobs)
                    per_job_bandwidth = network_bandwidth / \
                        num_shuffle_jobs if num_shuffle_jobs > 0 else 0

                    if 1000 * job.data_shuffle > per_job_bandwidth * elapsed_time:
                        job.data_shuffle -= per_job_bandwidth * elapsed_time / 1000
                        job.shuffle_time += elapsed_time / 1000
                        
                        # print("Shuffle Appended job", job.job_id, "now has", job.data_shuffle)

                        bandwidth_per_ms = per_job_bandwidth / 1000.0
                    
                        finish_delta_ms = math.ceil(
                            job.data_shuffle / bandwidth_per_ms)
                        finish_time_ms = current_second + finish_delta_ms
                        
                        # print("Shuffle Appended job", job.job_id, "expected to finishh at", finish_time_ms,"shuffle: ", job.data_shuffle, "bandwidth", per_job_bandwidth)
                        # print(current_second, required_cpu_time, job.cpu_time_progress, cores_assigned, job.job_id)
                        if job.scheduled:
                            # Already in the heap—always push an updated estimate
                            job.next_time = finish_time_ms
                            heapq.heappush(events, Event(
                                finish_time_ms, next_event_counter(), job, "shuffle_done"))
                        else:
                            # No completion event yet—only schedule if it beats whatever is next
                            next_evt_time = peek_next_event_time(events)
                            # print(events)
                            if finish_time_ms <= next_evt_time:
                                job.scheduled = True
                                job.next_time = finish_time_ms
                                heapq.heappush(events, Event(
                                    finish_time_ms, next_event_counter(), job, "shuffle_done"))
                        
                    else:
                        time_increment = job.data_shuffle / \
                            per_job_bandwidth if per_job_bandwidth != 0 else 0
                        job.shuffle_time += time_increment
                        # print("Shuffle Appended job", job.job_id, "finished at", current_second)
                        job.data_shuffle = 0
                if job.data_shuffle == 0 or shuffle[job.job_id] == 0: # IF YOU WANT TO SPLIT THEM LOGICALLY UNCOMMENT THIS - BUT THEN FIX THE ALLOCATION ISSUE
                    # Deduct CPU seconds based on cores allocated
                    required_cpu_time = min(
                        cores_assigned, cpu_cores_per_node) * speedup_factor * elapsed_time
                    if job.cpu_time_progress > required_cpu_time:
                        job.cpu_time_progress -= required_cpu_time
                        job.processing_time += elapsed_time / 1000
                        
                        # print("CPU Appended job", job.job_id, "now has", job.cpu_time_progress)
                        remaining_ms = job.cpu_time_progress
                        finish_delta_ms = math.ceil(
                            remaining_ms / (min(cores_assigned, cpu_cores_per_node) * nodes_involved * speedup_factor))
                        finish_time_ms = current_second + finish_delta_ms
                                                
                        # print("CPU Appended job", job.job_id, "expected to finish at", finish_time_ms,"cpu: ", remaining_ms, "cores: cores_assigned", cores_assigned ,"finish at", finish_delta_ms)

                        # print(current_second, required_cpu_time, job.cpu_time_progress, cores_assigned, job.job_id)
                        if job.scheduled:
                            # Already in the heap—always push an updated estimate
                            job.next_time = finish_time_ms
                            heapq.heappush(events, Event(
                                finish_time_ms, next_event_counter(), job, "cpu_done"))
                        else:
                            # No completion event yet—only schedule if it beats whatever is next
                            next_evt_time = peek_next_event_time(events)
                            # print(events)
                            if finish_time_ms <= next_evt_time:
                                job.scheduled = True
                                job.next_time = finish_time_ms
                                heapq.heappush(events, Event(
                                    finish_time_ms, next_event_counter(), job, "cpu_done"))

                    else:
                        # Finalize job execution if completed
                        time_used = job.cpu_time_progress / (min(cores_assigned, cpu_cores_per_node) * 1000 * speedup_factor) if (
                            min(cores_assigned, cpu_cores_per_node) * 1000 * speedup_factor) > 0 else 0
                        job.processing_time += time_used
                        job.cpu_time_progress = 0
                        # print("CPU Appended job", job.job_id, "now has", job.cpu_time_progress, "and finished at ", current_second)

                        job_finalization(job, memory, cpu_jobs, shuffle_jobs,
                                            finished_jobs, io_jobs, waiting_jobs, current_second)
            else:
                # If no cores are assigned, increment CPU time but no progress
                job.processing_time += elapsed_time / 1000


def job_finalization(
    job: Job,
    memory: List[float],
    cpu_jobs: List[Job],
    shuffle_jobs: List[Job],
    finished_jobs: List[Job],
    io_jobs: List[Job],
    waiting_jobs: List[Job],
    current_second: float
):
    """
    Finalizes a job after completion, removes it from active queues, adjusts memory and shuffle status.

    Args:
        job (Job): The job to finalize.
        memory (List[float]): Current memory usage.
        cpu_jobs (List[Job]): List of CPU-bound jobs.
        shuffle_jobs (List[Job]): List of jobs currently in shuffle.
        finished_jobs (List[Job]): List of finished jobs.
        current_second (float): Current simulation second.

    Returns:
        None
    """
    memory[0] -= job.data_scanned / 4  # Adjust memory usage
    job.end_timestamp = current_second
    job.query_exec_time = max(job.io_time, job.processing_time)
    job.query_exec_time_queueing = job.query_exec_time + \
        max(job.queueing_delay, job.buffer_delay)
    cpu_jobs.remove(job)
    if job in shuffle_jobs:
        shuffle_jobs.remove(job)
    if job not in waiting_jobs and job not in io_jobs:
        finished_jobs.append(job)


def simulate_cpu_autoscaling(
    current_second: float,
    cpu_jobs: List[Job],
    base_cores: int,
    cpu_cores: int,
    network_bandwidth: int,
    finished_jobs: List[Job],
    shuffle_jobs: List[Job],
    io_jobs: deque,
    shuffle: Dict[int, int],
    memory: List[float],
    second_range: float
) -> int:
    """
    Simulates CPU operations with autoscaling, allocating cores to jobs and handling shuffles.

    Args:
        current_second (float): Current simulation second.
        cpu_jobs (List[Job]): List of CPU-bound jobs.
        base_cores (int): Base number of CPU cores.
        cpu_cores (int): Total number of CPU cores.
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
    if num_jobs > 0:
        core_allocation = assign_cores_to_jobs_autoscaling(
            cpu_jobs, cpu_cores, 1)

        for job, cores_assigned in zip(cpu_jobs, core_allocation):
            if cores_assigned > 4 and job.data_shuffle != 0:
                if job not in shuffle_jobs and job.data_shuffle != 0:
                    shuffle_jobs.append(job)
            if job in shuffle_jobs and job.data_shuffle == 0:
                shuffle_jobs.remove(job)

        # Estimate completion time for each job and update progress
        for job, cores_assigned in zip(list(cpu_jobs), core_allocation):
            speedup_factor = 1
            if cores_assigned != 0:
                if cores_assigned > 4:  # Configurable threshold
                    # Calculate the proportion of the program that is parallelizable
                    parallelizable_portion = 0.9

                    shuffle[job.job_id] = 1

                    # Calculate the speedup factor according to Amdahl's Law
                    speedup_factor = 1 / \
                        ((1 - parallelizable_portion) +
                         (parallelizable_portion / (cores_assigned / 4)))

                    # Adjust completion time using the speedup factor
                    completion_time = job.cpu_time_progress / \
                        ((4 * 1000) * speedup_factor)
                else:
                    completion_time = job.cpu_time_progress / \
                        (cores_assigned * 1000)
                    shuffle[job.job_id] = 0

                if shuffle[job.job_id] == 1 and job.data_shuffle != 0:
                    # Shuffling Logic
                    per_job_bandwidth = cores_assigned * 50
                    if job.data_shuffle > per_job_bandwidth * second_range:
                        job.data_shuffle -= per_job_bandwidth * second_range
                        job.shuffle_time += second_range
                    else:
                        job.shuffle_time += job.data_shuffle / \
                            (per_job_bandwidth * second_range)
                        job.data_shuffle = 0
                        shuffle[job.job_id] = 0
                        shuffle_jobs.remove(job)
                else:
                    # Deduct CPU seconds from CPU time
                    if (completion_time > second_range and job.cpu_time_progress > cores_assigned * 1000 * second_range * speedup_factor):
                        job.processing_time += second_range
                        job.cpu_time_progress -= cores_assigned * 1000 * speedup_factor * second_range
                    else:
                        job.processing_time += completion_time
                        cpu_jobs.remove(job)
                        memory[0] -= job.data_scanned / 4
                        job.end_timestamp = current_second * 1000
                        job.query_exec_time = job.io_time + job.processing_time + job.shuffle_time
                        job.query_exec_time_queueing = job.query_exec_time + \
                            job.queueing_delay + job.buffer_delay
                        finished_jobs.append(job)
            else:
                job.processing_time += second_range
        return sum(core_allocation)
    else:
        return 0


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
    cpu_cores: int,
    network_bandwidth: int,
    finished_jobs: List[Job],
    shuffle_jobs: List[Job],
    waiting_jobs: List[Job],
    io_jobs: deque,
    shuffle: Dict[int, int],
    memory: List[float],
    second_range: float
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
    if num_jobs > 0:
        core_allocation = assign_cores_to_jobs_qaas(cpu_jobs, 4)

        for job, cores_assigned in zip(cpu_jobs, core_allocation):
            if cores_assigned > 4 and job.data_shuffle != 0:
                if job not in shuffle_jobs and job.data_shuffle != 0:
                    shuffle_jobs.append(job)
            if job in shuffle_jobs and job.data_shuffle == 0:
                shuffle_jobs.remove(job)

        # Estimate completion time for each job and update progress
        for job, cores_assigned in zip(list(cpu_jobs), core_allocation):
            speedup_factor = 1
            if cores_assigned != 0:
                if cores_assigned > 4:  # Configurable threshold
                    # Calculate the proportion of the program that is parallelizable
                    parallelizable_portion = 0.9

                    shuffle[job.job_id] = 1

                    # Calculate the speedup factor according to Amdahl's Law
                    speedup_factor = 1 / \
                        ((1 - parallelizable_portion) +
                         (parallelizable_portion / (cores_assigned / 4)))

                    # Adjust completion time using the speedup factor
                    completion_time = job.cpu_time_progress / \
                        ((4 * 1000) * speedup_factor)
                else:
                    completion_time = job.cpu_time_progress / \
                        (cores_assigned * 1000)
                    shuffle[job.job_id] = 0

                if shuffle[job.job_id] == 1 and job.data_shuffle != 0:
                    # Shuffling Logic
                    per_job_bandwidth = cores_assigned * 50
                    if job.data_shuffle > per_job_bandwidth * second_range:
                        job.data_shuffle -= per_job_bandwidth * second_range
                        job.shuffle_time += second_range
                    else:
                        job.shuffle_time += job.data_shuffle / \
                            (per_job_bandwidth * second_range)
                        job.data_shuffle = 0
                        shuffle[job.job_id] = 0
                        shuffle_jobs.remove(job)
                else:
                    # Deduct CPU seconds from CPU time
                    if (completion_time > second_range and job.cpu_time_progress > cores_assigned * 1000 * second_range * speedup_factor):
                        job.processing_time += second_range
                        job.cpu_time_progress -= cores_assigned * 1000 * speedup_factor * second_range
                    else:
                        job.processing_time += completion_time
                        cpu_jobs.remove(job)
                        memory[0] -= job.data_scanned / 4
                        job.end_timestamp = current_second * 1000
                        job.query_exec_time = job.io_time + job.processing_time + job.shuffle_time
                        job.query_exec_time_queueing = job.query_exec_time + \
                            job.queueing_delay + job.buffer_delay
                        if job not in waiting_jobs and job not in io_jobs:
                            finished_jobs.append(job)
            else:
                job.processing_time += second_range
        return sum(core_allocation)
    else:
        return 0

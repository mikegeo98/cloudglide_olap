# scheduling.py

from collections import deque
import heapq
from typing import List
import logging

from cloudglide.job import Job


def check_duplicate_job_ids(jobs: List[Job]) -> List[Job]:
    """
    Check for duplicate job IDs in the list of jobs.

    Args:
        jobs (List[Job]): List of Job instances.

    Returns:
        List[Job]: List of duplicate Job instances.
    """
    job_ids = set()
    duplicates = []
    for job in jobs:
        job_id = job.job_id
        if job_id in job_ids:
            duplicates.append(job)
        else:
            job_ids.add(job_id)
    return duplicates


def io_scheduler(
    scheduling: int,
    architecture: int,
    current_second: float,
    second_range,
    jobs: List[Job],
    waiting_jobs: deque,
    io_jobs: deque,
    cpu_cores: int,
    phase
):
    """
    Schedule I/O jobs based on the scheduling strategy and architecture.

    Args:
        scheduling (int): Scheduling strategy identifier.
        architecture (int): Architecture type identifier.
        current_second (float): Current simulation second.
        jobs (List[Job]): List of all Job instances.
        waiting_jobs (deque): Queue of waiting jobs.
        io_jobs (deque): Queue of I/O jobs.
        cpu_cores (int): Number of CPU cores available.
        second_range (float): Simulation time step.
        num_queues (int): Number of job queues.
        job_queues (List[List[Job]]): List of job queues.
    """
    # Only do anything at all on arrival, io_done, or scale_check
    if phase not in ("arrival", "io_done", "scale_check"):
        return
            
        # Compute base capacity (cpu_cores, or unlimited under QaaS)
    if architecture < 3:
        base_capacity = cpu_cores
    else:
        base_capacity = len(waiting_jobs)

    # Give yourself one extra slot if an I/O just completed
    capacity = base_capacity + (1 if phase == "io_done" else 0)

    # If we’re already at or above capacity, just age queueing delays
    if len(io_jobs) >= capacity:
        for job in waiting_jobs:
            job.queueing_delay += second_range / 1000.0
        return

    # How many new slots to fill now?
    slots = capacity - len(io_jobs)

    # FIFO / FCFS path
    if scheduling in (0, 4):
        for _ in range(slots):
            if not waiting_jobs:
                break
            job = waiting_jobs.popleft()
            if job.start_timestamp == 0.0:
                job.start_timestamp = current_second
            job.queueing_delay = (current_second - job.start) / 1000.0
            io_jobs.append(job)
        return

    # SJF / LJF path: build a one‐time heap of size len(waiting_jobs)
    heap = []
    for idx, job in enumerate(waiting_jobs):
        key = job.data_scanned if scheduling == 1 else -job.data_scanned
        # push (key, idx, job) so that if two keys tie, idx breaks it
        heap.append((key, idx, job))
    heapq.heapify(heap)

    for _ in range(slots):
        if not heap:
            break
        _, _, job = heapq.heappop(heap)
        waiting_jobs.remove(job)    # O(n), but done only `slots` times
        if job.start_timestamp == 0.0:
            job.start_timestamp = current_second
        job.queueing_delay = (current_second - job.start) / 1000.0
        io_jobs.append(job)

def cpu_scheduler(
    scheduling: int,
    architecture: int,
    current_second: float,
    jobs: List[Job],
    buffer_jobs: deque,
    cpu_jobs: deque,
    cpu_cores: int,
    memory: List[float],
    second_range: float,
    phase
):
    """
    Schedule CPU jobs based on the scheduling strategy and architecture.

    Args:
        scheduling (int): Scheduling strategy identifier.
        architecture (int): Architecture type identifier.
        current_second (float): Current simulation second.
        buffer_jobs (deque): Queue of buffered jobs.
        cpu_jobs (deque): Queue of CPU jobs.
        cpu_cores (int): Number of CPU cores available.
        memory (List[float]): List containing current memory usage.
        second_range (float): Simulation time step.
    """
    # --- early exit ---
    if phase not in ("arrival", "cpu_done", "scale_check"):
        return

    # 1) Compute *base* free slots
    if architecture < 3:
        base_free = cpu_cores - len(cpu_jobs)
    else:
        # QaaS / elastic pool override: can pull all buffer_jobs
        base_free = len(buffer_jobs)

    # 2) Give yourself an extra slot if this was a CPU completion
    free_slots = base_free + (1 if phase == "cpu_done" else 0)

    # 3) If no slots, just age delays
    if free_slots <= 0:
        for job in buffer_jobs:
            job.buffer_delay += second_range / 1000.0
        return

    # 4) Now your normal FCFS / SJF / LJF / priority logic, e.g.:
    #    FCFS:
    if scheduling in (0, 4):
        for _ in range(free_slots):
            if not buffer_jobs: break
            job = buffer_jobs.popleft()
            if job.start_timestamp == 0.0:
                job.start_timestamp = current_second
            memory[0] += job.data_scanned / 4
            cpu_jobs.append(job)
        for job in buffer_jobs:
            job.buffer_delay += second_range / 1000.0
        return

    #    SJF / LJF:
    if scheduling in (1,2):
        sign = 1 if scheduling==1 else -1
        # Build a heap of (key, job_id, job) so ties in key fall back to job_id:
        heap = [
            (sign * job.cpu_time, job.job_id, job)
            for job in buffer_jobs
        ]
        heapq.heapify(heap)

        # Pull at most free_slots jobs off that heap
        for _ in range(free_slots):
            if not heap:
                break
            _, _, job = heapq.heappop(heap)
            buffer_jobs.remove(job)
            if job.start_timestamp == 0.0:
                job.start_timestamp = current_second
            memory[0] += job.data_scanned / 4
            cpu_jobs.append(job)

        # Age any that remain
        for job in buffer_jobs:
            job.buffer_delay += second_range / 1000.0

        return
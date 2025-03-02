# scheduling.py

from collections import deque
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
    jobs: List[Job],
    waiting_jobs: deque,
    io_jobs: deque,
    cpu_cores: int,
    second_range: float,
    num_queues: int,
    job_queues: List[List[Job]]
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
    if architecture != 3:

        if scheduling == 0 or scheduling == 4:
            # Schedule queued jobs from previous seconds
            duplicates = check_duplicate_job_ids(jobs)
            if duplicates:
                logging.warning(f"Duplicate Job IDs found: {[job.job_id for job in duplicates]}")

            for job in list(waiting_jobs):  # Iterate over a static list to allow modification
                if len(io_jobs) < cpu_cores:
                    if job.start_timestamp == 0.0:
                        job.start_timestamp = current_second * 1000
                    job.queueing_delay = current_second - (job.start / 1000)
                    io_jobs.append(job)
                    waiting_jobs.remove(job)
                else:
                    break

        if scheduling == 1:
            # Schedule queued jobs based on shortest Data_Scanned
            while waiting_jobs and len(io_jobs) < cpu_cores:
                shortest_job = min(waiting_jobs, key=lambda job: job.data_scanned)  # Find the job with least data scanned
                if shortest_job.start_timestamp == 0.0:
                    shortest_job.start_timestamp = current_second * 1000
                    shortest_job.queueing_delay = 0
                else:
                    shortest_job.queueing_delay = current_second - (shortest_job.start / 1000)
                io_jobs.append(shortest_job)
                waiting_jobs.remove(shortest_job)

        if scheduling == 2:
            # Schedule queued jobs based on longest Data_Scanned
            while waiting_jobs and len(io_jobs) < cpu_cores:
                longest_job = max(waiting_jobs, key=lambda job: job.data_scanned)  # Find the job with most data scanned
                longest_job.start_timestamp = current_second * 1000
                longest_job.queueing_delay = current_second - (longest_job.start / 1000)
                io_jobs.append(longest_job)
                waiting_jobs.remove(longest_job)
        # Uncomment and refactor scheduling == 3 if needed
        # if scheduling == 3:
        #     # Initialize max_jobs_per_queue
        #     max_jobs_per_queue = cpu_cores // num_queues
        #     # Distribute jobs to different queues based on Data_Scanned
        #     for job in list(waiting_jobs):
        #         data_scanned = job.data_scanned
        #         max_data_scanned = max(job.data_scanned for job in jobs) if jobs else 1
        #         queue_index = int((data_scanned / max_data_scanned) * (num_queues - 1))
        #         job_queues[queue_index].append(job)
        #         waiting_jobs.remove(job)

        #     # Schedule jobs from each queue
        #     for queue in job_queues:
        #         while queue and len(io_jobs) < cpu_cores and len(queue) <= max_jobs_per_queue:
        #             job = queue.pop(0)
        #             job.start_timestamp = current_second * 1000
        #             job.queueing_delay = current_second - (job.start / 1000)
        #             io_jobs.append(job)

    # Check for job arrivals during the current second - First-Come-First-Served
    for job in list(jobs):  # Iterate over a static list to allow modification
        if current_second * 1000 <= job.start < (current_second + second_range) * 1000:
            waiting_jobs.append(job)
            # Concurrency Clause for I/O = Configurable
            if len(io_jobs) < cpu_cores or architecture > 2:
                # Condition for spot interruption
                if job.start_timestamp == 0.0:
                    job.start_timestamp = job.start
                    job.queueing_delay = 0
                io_jobs.append(job)
                waiting_jobs.remove(job)
        # Stop checking if next job arrives later (assuming jobs are sorted by start time)
        if job.start >= (current_second + second_range) * 1000:
            break


def cpu_scheduler(
    scheduling: int,
    architecture: int,
    current_second: float,
    jobs: List[Job],
    buffer_jobs: deque,
    cpu_jobs: deque,
    cpu_cores: int,
    memory: List[float],
    second_range: float
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
    if scheduling == 0:
        # First-Come-First-Served (FCFS) Scheduling
        while buffer_jobs and (len(cpu_jobs) < cpu_cores or architecture > 2):
            next_job = buffer_jobs.popleft()  # Pop the first job from the buffer
            memory[0] += next_job.data_scanned / 4
            cpu_jobs.append(next_job)  # Append the job to the CPU jobs list
        for job in buffer_jobs:
            job.buffer_delay += second_range

    if scheduling == 1:
        # Shortest CPU Time First Scheduling
        while buffer_jobs and (len(cpu_jobs) < cpu_cores or architecture > 2):
            shortest_job = min(buffer_jobs, key=lambda job: job.cpu_time)  # Find the job with least CPU time
            buffer_jobs.remove(shortest_job)
            memory[0] += shortest_job.data_scanned / 4
            cpu_jobs.append(shortest_job)  # Append the shortest job to the CPU jobs list
        for job in buffer_jobs:
            job.buffer_delay += second_range

    if scheduling == 2:
        # Longest CPU Time First Scheduling
        while buffer_jobs and (len(cpu_jobs) < cpu_cores or architecture > 2):
            longest_job = max(buffer_jobs, key=lambda job: job.cpu_time)  # Find the job with most CPU time
            buffer_jobs.remove(longest_job)
            memory[0] += longest_job.data_scanned / 4
            cpu_jobs.append(longest_job)  # Append the longest job to the CPU jobs list
        for job in buffer_jobs:
            job.buffer_delay += second_range

    if scheduling == 4:
        # Priority-Based Scheduling
        priority_levels = list(range(12))  # 0 to 11
        # Update priority levels for each job in buffer_jobs
        for job in buffer_jobs:
            if job.cpu_time > 3253850:
                job.priority_level = 11
            elif job.cpu_time > 1120500:
                job.priority_level = 10
            elif job.cpu_time > 791980:
                job.priority_level = 9
            elif job.cpu_time > 472360:
                job.priority_level = 8
            elif job.cpu_time_progress > 407330:
                job.priority_level = 7
            elif job.cpu_time > 324170:
                job.priority_level = 6
            elif job.cpu_time_progress > 205680:
                job.priority_level = 5
            elif job.cpu_time > 139170:
                job.priority_level = 4
            elif job.cpu_time > 112050:
                job.priority_level = 3
            elif job.cpu_time > 61090:
                job.priority_level = 2
            elif job.cpu_time > 46310:
                job.priority_level = 1
            else:
                job.priority_level = 0

        total_cores = cpu_cores
        core_distribution = {
            0: 2,
            1: 2,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 1,
            9: 1,
            10: 2,
            11: 2
        }
        # Dictionary to hold the counts of jobs scheduled by priority level
        scheduled_counts = {level: 0 for level in priority_levels}
        for job in cpu_jobs:
            scheduled_counts[job.priority_level] += 1

        # Dictionary to hold remaining jobs per priority level
        remaining_jobs = {level: [] for level in priority_levels}

        for level in priority_levels:
            for job in list(buffer_jobs):  # Iterate over a static list to allow modification
                if job.priority_level == level:
                    if scheduled_counts[level] < core_distribution[level] and len(cpu_jobs) < total_cores:
                        cpu_jobs.append(job)
                        memory[0] += job.data_scanned / 4
                        buffer_jobs.remove(job)
                        scheduled_counts[level] += 1
                    else:
                        remaining_jobs[level].append(job)

        # Fill remaining cores with jobs from any priority level
        for level in priority_levels:
            while remaining_jobs[level] and len(cpu_jobs) < total_cores:
                job = remaining_jobs[level].pop(0)
                cpu_jobs.append(job)
                memory[0] += job.data_scanned / 4
                buffer_jobs.remove(job)
                scheduled_counts[level] += 1

        # Increment wait time for all remaining jobs in the buffer
        for job in buffer_jobs:
            job.buffer_delay += second_range

    # Check for job arrivals during the current second - First-Come-First-Served
    for job in list(jobs):  # Iterate over a static list to allow modification
        if current_second * 1000 <= job.start < (current_second + second_range) * 1000:
            buffer_jobs.append(job)
            jobs.remove(job)
            # Concurrency Clause for I/O = Configurable
            if len(cpu_jobs) < cpu_cores or architecture > 2:
                # Condition for spot interruption
                if job.start_timestamp == 0.0:
                    job.start_timestamp = job.start
                    job.queueing_delay = 0
                cpu_jobs.append(job)
                buffer_jobs.remove(job)
        # Stop checking if next job arrives later (assuming jobs are sorted by start time)
        if job.start >= (current_second + second_range) * 1000:
            break

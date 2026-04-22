"""
QaaS Slot Management Policies (Scenario 2)

This module implements slot management policies for QaaS architecture:
- Strict Priority: Short jobs can preempt long jobs
- Fixed Ratio: Reserve percentage of baseline slots for short jobs
- Baseline Slots: Define a fixed baseline pool of resources
"""

from collections import deque
from typing import List, Tuple
from cloudglide.job import Job
import logging


# Job categorization threshold (10 seconds = 10000ms CPU time)
SHORT_JOB_THRESHOLD = 10000  # ms


def categorize_job(job: Job) -> str:
    """
    Categorize a job as 'short' or 'long' based on CPU time.

    Args:
        job: The job to categorize

    Returns:
        'short' if job.cpu_time <= SHORT_JOB_THRESHOLD, else 'long'
    """
    return 'short' if job.cpu_time <= SHORT_JOB_THRESHOLD else 'long'


def partition_jobs_by_length(jobs: deque) -> Tuple[List[Job], List[Job]]:
    """
    Partition jobs into short and long categories.

    Args:
        jobs: Queue of jobs to partition

    Returns:
        Tuple of (short_jobs, long_jobs)
    """
    short_jobs = []
    long_jobs = []

    for job in jobs:
        if categorize_job(job) == 'short':
            short_jobs.append(job)
        else:
            long_jobs.append(job)

    return short_jobs, long_jobs


def strict_priority_scheduler(
    waiting_jobs: deque,
    io_jobs: deque,
    cpu_jobs: deque,
    baseline_slots: int,
    now: float,
    enable_preemption: bool = True
) -> None:
    """
    Strict Priority Scheduler: Short jobs always get priority over long jobs.
    When preemption is enabled, short jobs can preempt long jobs.

    Args:
        waiting_jobs: Queue of waiting jobs
        io_jobs: Currently running I/O jobs
        cpu_jobs: Currently running CPU jobs
        baseline_slots: Total baseline slot pool
        now: Current simulation time
        enable_preemption: If True, short jobs can preempt long jobs
    """
    # Categorize waiting jobs
    short_waiting, long_waiting = partition_jobs_by_length(waiting_jobs)

    # Categorize running jobs
    short_running_io, long_running_io = partition_jobs_by_length(io_jobs)
    short_running_cpu, long_running_cpu = partition_jobs_by_length(cpu_jobs)

    total_running = len(io_jobs) + len(cpu_jobs)
    available_slots = baseline_slots - total_running

    # If preemption is enabled and we have short jobs waiting but no slots
    if enable_preemption and short_waiting and available_slots <= 0:
        # Preempt long jobs to make room for short jobs
        long_running_all = list(long_running_io) + list(long_running_cpu)

        if long_running_all:
            # Preempt the oldest long job
            jobs_to_preempt = min(len(short_waiting), len(long_running_all))

            for i in range(jobs_to_preempt):
                # Find oldest long job
                oldest = None
                for job in long_running_all:
                    if oldest is None or job.start_timestamp < oldest.start_timestamp:
                        oldest = job

                if oldest:
                    # Preempt the job - move it back to waiting queue
                    if oldest in io_jobs:
                        io_jobs.remove(oldest)
                    if oldest in cpu_jobs:
                        cpu_jobs.remove(oldest)

                    waiting_jobs.appendleft(oldest)  # Put at front to maintain fairness
                    long_running_all.remove(oldest)
                    available_slots += 1

                    logging.info(f"[QaaS] Preempted long job {oldest.job_id} for short job at {now}ms")

    # Schedule short jobs first
    scheduled_short = 0
    for job in short_waiting:
        if available_slots <= 0:
            break
        waiting_jobs.remove(job)
        if job.start_timestamp == 0.0:
            job.start_timestamp = now
        io_jobs.append(job)
        available_slots -= 1
        scheduled_short += 1

    # Schedule long jobs with remaining slots
    scheduled_long = 0
    for job in long_waiting:
        if available_slots <= 0:
            break
        waiting_jobs.remove(job)
        if job.start_timestamp == 0.0:
            job.start_timestamp = now
        io_jobs.append(job)
        available_slots -= 1
        scheduled_long += 1

    if scheduled_short > 0 or scheduled_long > 0:
        logging.debug(f"[QaaS Strict Priority] Scheduled {scheduled_short} short, {scheduled_long} long jobs")


def fixed_ratio_scheduler(
    waiting_jobs: deque,
    io_jobs: deque,
    cpu_jobs: deque,
    baseline_slots: int,
    fixed_ratio: float,
    now: float
) -> None:
    """
    Fixed Ratio Scheduler: Reserve a percentage of baseline slots for short jobs.

    Args:
        waiting_jobs: Queue of waiting jobs
        io_jobs: Currently running I/O jobs
        cpu_jobs: Currently running CPU jobs
        baseline_slots: Total baseline slot pool
        fixed_ratio: Percentage of slots reserved for short jobs (0.0-1.0)
        now: Current simulation time
    """
    # Calculate slot allocation
    short_reserved_slots = int(baseline_slots * fixed_ratio)
    long_available_slots = baseline_slots - short_reserved_slots

    # Categorize waiting jobs
    short_waiting, long_waiting = partition_jobs_by_length(waiting_jobs)

    # Categorize running jobs
    short_running_io, long_running_io = partition_jobs_by_length(io_jobs)
    short_running_cpu, long_running_cpu = partition_jobs_by_length(cpu_jobs)

    short_running_total = len(short_running_io) + len(short_running_cpu)
    long_running_total = len(long_running_io) + len(long_running_cpu)

    # Calculate available slots for each category
    short_slots_available = short_reserved_slots - short_running_total
    long_slots_available = long_available_slots - long_running_total

    # Short jobs can use reserved slots + unused long slots
    # Long jobs can only use their allocated slots (cannot use short reserved slots)

    # Schedule short jobs first (can overflow into long slots if needed)
    scheduled_short = 0
    for job in short_waiting:
        if short_slots_available > 0:
            # Use reserved short slots
            waiting_jobs.remove(job)
            if job.start_timestamp == 0.0:
                job.start_timestamp = now
            io_jobs.append(job)
            short_slots_available -= 1
            scheduled_short += 1
        elif long_slots_available > 0:
            # Overflow into long slots if available
            waiting_jobs.remove(job)
            if job.start_timestamp == 0.0:
                job.start_timestamp = now
            io_jobs.append(job)
            long_slots_available -= 1
            scheduled_short += 1
        else:
            break

    # Schedule long jobs (cannot use short reserved slots)
    scheduled_long = 0
    for job in long_waiting:
        if long_slots_available <= 0:
            break
        waiting_jobs.remove(job)
        if job.start_timestamp == 0.0:
            job.start_timestamp = now
        io_jobs.append(job)
        long_slots_available -= 1
        scheduled_long += 1

    if scheduled_short > 0 or scheduled_long > 0:
        logging.debug(
            f"[QaaS Fixed Ratio {fixed_ratio:.0%}] "
            f"Scheduled {scheduled_short} short, {scheduled_long} long jobs "
            f"(short reserved: {short_reserved_slots}, long available: {baseline_slots - short_reserved_slots})"
        )


def qaas_slot_management_scheduler(
    waiting_jobs: deque,
    io_jobs: deque,
    cpu_jobs: deque,
    baseline_slots: int,
    now: float,
    strict_priority: bool = False,
    fixed_ratio: float = 0.0
) -> None:
    """
    Main QaaS slot management scheduler that applies configured policies.

    Args:
        waiting_jobs: Queue of waiting jobs
        io_jobs: Currently running I/O jobs
        cpu_jobs: Currently running CPU jobs
        baseline_slots: Total baseline slot pool
        now: Current simulation time
        strict_priority: If True, enable strict priority preemption
        fixed_ratio: If > 0, enable fixed ratio slot reservation (0.0-1.0)
    """
    # Validate parameters
    if fixed_ratio < 0.0 or fixed_ratio > 1.0:
        logging.warning(f"Invalid fixed_ratio {fixed_ratio}, must be in [0.0, 1.0]. Using 0.0.")
        fixed_ratio = 0.0

    if baseline_slots < 1:
        logging.warning(f"Invalid baseline_slots {baseline_slots}, must be >= 1. Using 400.")
        baseline_slots = 400

    # Apply policies
    if fixed_ratio > 0.0:
        # Fixed ratio takes precedence
        fixed_ratio_scheduler(waiting_jobs, io_jobs, cpu_jobs, baseline_slots, fixed_ratio, now)
    elif strict_priority:
        # Strict priority with preemption
        strict_priority_scheduler(waiting_jobs, io_jobs, cpu_jobs, baseline_slots, now, enable_preemption=True)
    else:
        # Default: strict priority without preemption (just prioritize short jobs)
        strict_priority_scheduler(waiting_jobs, io_jobs, cpu_jobs, baseline_slots, now, enable_preemption=False)

from collections import deque
import heapq
import logging
from enum import IntEnum
from typing import List
from cloudglide.config import ArchitectureType
from cloudglide.job import Job


# ----------------------------
# Scheduling Policy Enum
# ----------------------------
class SchedulingPolicy(IntEnum):
    FCFS = 0
    SJF = 1
    LJF = 2
    MULTI = 3    # placeholder for future multi-queue or hybrid strategy


# ----------------------------
# Generic Helpers
# ----------------------------
def pick_jobs_by_metric(queue: deque, metric: str, ascending=True, limit=None) -> List[Job]:
    """
    Selects up to 'limit' jobs from queue ordered by a given metric (ascending or descending).
    Removes the selected jobs from the original queue.
    """
    sign = 1 if ascending else -1
    heap = [(sign * getattr(j, metric), j.job_id, j) for j in queue]
    heapq.heapify(heap)
    selected = []
    for _ in range(limit or len(heap)):
        if not heap:
            break
        _, _, job = heapq.heappop(heap)
        queue.remove(job)
        selected.append(job)
    return selected


def age_jobs_delay(jobs: deque, attr: str, dt: float):
    """Increment a delay field (queueing_delay or buffer_delay) for all jobs."""
    for job in jobs:
        setattr(job, attr, getattr(job, attr) + dt)


# ----------------------------
# Individual Scheduling Strategies
# ----------------------------
def fcfs_policy(source_q: deque, target_q: deque, slots: int, now: float):
    """
    First-Come-First-Served (FIFO)
    """
    for _ in range(slots):
        if not source_q:
            break
        job = source_q.popleft()
        if job.start_timestamp == 0.0:
            job.start_timestamp = now
        target_q.append(job)


def sjf_policy(source_q: deque, target_q: deque, slots: int, now: float, metric: str = "data_scanned"):
    """
    Shortest Job First (ascending by given metric)
    """
    selected = pick_jobs_by_metric(source_q, metric=metric, ascending=True, limit=slots)
    for job in selected:
        if job.start_timestamp == 0.0:
            job.start_timestamp = now
        target_q.append(job)


def ljf_policy(source_q: deque, target_q: deque, slots: int, now: float, metric: str = "data_scanned"):
    """
    Longest Job First (descending by given metric)
    """
    selected = pick_jobs_by_metric(source_q, metric=metric, ascending=False, limit=slots)
    for job in selected:
        if job.start_timestamp == 0.0:
            job.start_timestamp = now
        target_q.append(job)


def multi_queue_policy(source_q: deque, target_q: deque, slots: int, now: float):
    """
    Placeholder for future hybrid / multi-queue policies.
    Currently same as FCFS.
    """
    fcfs_policy(source_q, target_q, slots, now)


# ----------------------------
# I/O Scheduler
# ----------------------------
def io_scheduler(
    scheduling: int,
    architecture: int,
    now: float,
    dt: float,
    jobs: List[Job],
    waiting_jobs: deque,
    io_jobs: deque,
    cpu_cores: int,
    phase: str,
):
    if phase not in ("arrival", "io_done", "scale_check"):
        return

    # Determine base capacity
    if architecture in (
        ArchitectureType.DWAAS,
        ArchitectureType.DWAAS_AUTOSCALING,
        ArchitectureType.ELASTIC_POOL,
    ):
        capacity = cpu_cores
    else:
        capacity = len(jobs)

    if phase == "io_done":
        capacity += 1  # allow refill immediately after a completion

    # If full, just age delays
    if len(io_jobs) >= capacity:
        age_jobs_delay(waiting_jobs, "queueing_delay", dt)
        return

    slots = capacity - len(io_jobs)
    policy = SchedulingPolicy(scheduling)

    if policy == SchedulingPolicy.FCFS:
        fcfs_policy(waiting_jobs, io_jobs, slots, now)
    elif policy == SchedulingPolicy.SJF:
        sjf_policy(waiting_jobs, io_jobs, slots, now, metric="data_scanned")
    elif policy == SchedulingPolicy.LJF:
        ljf_policy(waiting_jobs, io_jobs, slots, now, metric="data_scanned")
    else:
        multi_queue_policy(waiting_jobs, io_jobs, slots, now)

    logging.debug(f"[{now/1000:.1f}s] I/O Scheduler: policy={policy.name}, added={slots}")


# ----------------------------
# CPU Scheduler
# ----------------------------
def cpu_scheduler(
    scheduling: int,
    architecture: int,
    now: float,
    jobs: List[Job],
    buffer_jobs: deque,
    cpu_jobs: deque,
    cpu_cores: int,
    memory: List[float],
    dt: float,
    phase: str,
    config,
):
    if phase not in ("arrival", "cpu_done", "scale_check"):
        return

    if architecture in (
        ArchitectureType.DWAAS,
        ArchitectureType.DWAAS_AUTOSCALING,
        ArchitectureType.ELASTIC_POOL,
    ):
        base_free = cpu_cores - len(cpu_jobs)
    else:
        base_free = len(buffer_jobs)

    free_slots = base_free + (1 if phase == "cpu_done" else 0)
    if free_slots <= 0:
        age_jobs_delay(buffer_jobs, "buffer_delay", dt)
        return

    policy = SchedulingPolicy(scheduling)
    if policy == SchedulingPolicy.FCFS:
        fcfs_policy(buffer_jobs, cpu_jobs, free_slots, now)
    elif policy == SchedulingPolicy.SJF:
        sjf_policy(buffer_jobs, cpu_jobs, free_slots, now, metric="cpu_time")
    elif policy == SchedulingPolicy.LJF:
        ljf_policy(buffer_jobs, cpu_jobs, free_slots, now, metric="cpu_time")
    else:
        multi_queue_policy(buffer_jobs, cpu_jobs, free_slots, now)

    # Track memory usage
    for job in list(cpu_jobs)[-free_slots:]:
        memory[0] += job.data_scanned * config.materialization_fraction


    # Age remaining jobs
    age_jobs_delay(buffer_jobs, "buffer_delay", dt)

    logging.debug(f"[{now/1000:.1f}s] CPU Scheduler: policy={policy.name}, added={free_slots}, mem={memory[0]:.2f}")

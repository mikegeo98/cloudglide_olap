import logging
from collections import deque
from typing import List
from cloudglide.job import Job
from cloudglide.config import SimulationConfig

def handle_interrupt(
    io_jobs: deque,
    cpu_jobs: deque,
    buffer_jobs: deque,
    waiting_jobs: deque,
    shuffled_jobs: List[Job],
    jobs: List[Job],
    current_second: float,
    config: SimulationConfig,
) -> None:

    logging.warning("INTERRUPT: Spot instance interruption occurred — resetting job states.")

    def requeue_job(job: Job):
        job.reset_progress(current_second, config)
        jobs.append(job)

    # Move all active jobs back to pending list
    for q in (io_jobs, cpu_jobs, buffer_jobs, waiting_jobs):
        for job in list(q):
            q.remove(job)
            requeue_job(job)

    shuffled_jobs.clear()

# job_queue.py
"""
Optimized job queue data structure with O(1) removal operations.
Replaces O(n) deque.remove() with indexed dictionary tracking.
"""

from typing import Dict, List, Optional, Callable
from collections import deque
from cloudglide.job import Job


class IndexedJobQueue:
    """
    High-performance job queue with O(1) removal by job_id.

    Uses a deque for ordering and a dict for fast lookups.
    """

    def __init__(self):
        self._queue = deque()
        self._index: Dict[int, Job] = {}  # job_id -> Job mapping

    def append(self, job: Job) -> None:
        """Add job to end of queue. O(1)"""
        if job.job_id not in self._index:
            self._queue.append(job)
            self._index[job.job_id] = job

    def appendleft(self, job: Job) -> None:
        """Add job to front of queue. O(1)"""
        if job.job_id not in self._index:
            self._queue.appendleft(job)
            self._index[job.job_id] = job

    def remove(self, job: Job) -> None:
        """Remove job from queue. O(n) but with fast path for common cases."""
        if job.job_id in self._index:
            del self._index[job.job_id]
            # Fast path: check if job is at front or back
            if self._queue and self._queue[0].job_id == job.job_id:
                self._queue.popleft()
            elif self._queue and self._queue[-1].job_id == job.job_id:
                self._queue.pop()
            else:
                # Slow path: linear search (unavoidable with deque)
                self._queue.remove(job)

    def popleft(self) -> Job:
        """Remove and return job from front. O(1)"""
        if not self._queue:
            raise IndexError("pop from empty queue")
        job = self._queue.popleft()
        del self._index[job.job_id]
        return job

    def pop(self) -> Job:
        """Remove and return job from end. O(1)"""
        if not self._queue:
            raise IndexError("pop from empty queue")
        job = self._queue.pop()
        del self._index[job.job_id]
        return job

    def contains(self, job: Job) -> bool:
        """Check if job is in queue. O(1)"""
        return job.job_id in self._index

    def get_by_id(self, job_id: int) -> Optional[Job]:
        """Get job by ID. O(1)"""
        return self._index.get(job_id)

    def clear(self) -> None:
        """Remove all jobs. O(1)"""
        self._queue.clear()
        self._index.clear()

    def __len__(self) -> int:
        """Return number of jobs. O(1)"""
        return len(self._queue)

    def __iter__(self):
        """Iterate over jobs in order."""
        return iter(self._queue)

    def __bool__(self) -> bool:
        """Check if queue is non-empty. O(1)"""
        return bool(self._queue)

    def to_list(self) -> List[Job]:
        """Convert to list. O(n)"""
        return list(self._queue)

    def filter(self, predicate: Callable[[Job], bool]) -> List[Job]:
        """Return jobs matching predicate. O(n)"""
        return [job for job in self._queue if predicate(job)]


class PriorityJobQueue:
    """
    Priority queue for jobs using a sorted list.
    Supports custom sorting keys (e.g., data_scanned for SJF).
    """

    def __init__(self, key: Optional[Callable[[Job], float]] = None):
        """
        Initialize priority queue.

        Args:
            key: Function to extract priority value from job.
                 Lower values have higher priority.
                 Default: job.start (FIFO)
        """
        self._jobs: List[Job] = []
        self._index: Dict[int, int] = {}  # job_id -> list index
        self._key = key or (lambda j: j.start)

    def insert(self, job: Job) -> None:
        """Insert job maintaining priority order. O(n)"""
        if job.job_id in self._index:
            return

        priority = self._key(job)

        # Binary search for insertion point
        left, right = 0, len(self._jobs)
        while left < right:
            mid = (left + right) // 2
            if self._key(self._jobs[mid]) < priority:
                left = mid + 1
            else:
                right = mid

        self._jobs.insert(left, job)

        # Update indices
        for i in range(left, len(self._jobs)):
            self._index[self._jobs[i].job_id] = i

    def pop_highest(self) -> Job:
        """Remove and return highest priority job. O(n)"""
        if not self._jobs:
            raise IndexError("pop from empty queue")

        job = self._jobs.pop(0)
        del self._index[job.job_id]

        # Update indices
        for i in range(len(self._jobs)):
            self._index[self._jobs[i].job_id] = i

        return job

    def remove(self, job: Job) -> None:
        """Remove specific job. O(n)"""
        if job.job_id not in self._index:
            return

        idx = self._index[job.job_id]
        del self._jobs[idx]
        del self._index[job.job_id]

        # Update indices
        for i in range(idx, len(self._jobs)):
            self._index[self._jobs[i].job_id] = i

    def contains(self, job: Job) -> bool:
        """Check if job is in queue. O(1)"""
        return job.job_id in self._index

    def peek(self) -> Optional[Job]:
        """Return highest priority job without removing. O(1)"""
        return self._jobs[0] if self._jobs else None

    def __len__(self) -> int:
        """Return number of jobs. O(1)"""
        return len(self._jobs)

    def __iter__(self):
        """Iterate over jobs in priority order."""
        return iter(self._jobs)

    def __bool__(self) -> bool:
        """Check if queue is non-empty. O(1)"""
        return bool(self._jobs)

    def to_list(self) -> List[Job]:
        """Convert to list. O(1)"""
        return list(self._jobs)

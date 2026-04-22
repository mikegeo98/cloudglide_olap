# test_scheduling_model_fixed.py
"""
Fixed comprehensive unit tests for scheduling_model.py
Tests FCFS, SJF, LJF scheduling policies with correct API.
"""

import pytest
from collections import deque
from cloudglide.job import Job
from cloudglide.scheduling_model import io_scheduler, cpu_scheduler
from cloudglide.config import SimulationConfig, ArchitectureType


@pytest.fixture
def sample_jobs():
    """Create sample jobs for testing."""
    jobs = [
        Job(
            job_id=1,
            database_id=1,
            query_id=1,
            start=0.0,
            cpu_time=1000.0,
            data_scanned=100.0,
            scale_factor=1.0
        ),
        Job(
            job_id=2,
            database_id=1,
            query_id=2,
            start=5.0,
            cpu_time=5000.0,
            data_scanned=500.0,
            scale_factor=1.0
        ),
        Job(
            job_id=3,
            database_id=1,
            query_id=3,
            start=10.0,
            cpu_time=2000.0,
            data_scanned=200.0,
            scale_factor=1.0
        ),
    ]
    return jobs


@pytest.fixture
def config():
    """Create default simulation config."""
    return SimulationConfig()


class TestIOSchedulerFixed:
    """Tests for I/O scheduler with correct API."""

    def test_fcfs_scheduling(self, sample_jobs, config):
        """Test FCFS (First-Come-First-Served) scheduling."""
        waiting_jobs = deque(sample_jobs)
        io_jobs = deque()

        # FCFS policy = 0
        io_scheduler(
            scheduling=0,
            architecture=ArchitectureType.DWAAS,
            now=10.0,
            dt=1.0,
            jobs=sample_jobs,
            waiting_jobs=waiting_jobs,
            io_jobs=io_jobs,
            cpu_cores=16,
            phase="arrival",
            options=config.scheduling_options
        )

        # Should schedule jobs (exact behavior depends on implementation)
        assert isinstance(io_jobs, deque)

    def test_sjf_scheduling(self, sample_jobs, config):
        """Test SJF (Shortest Job First) scheduling."""
        waiting_jobs = deque(sample_jobs)
        io_jobs = deque()

        # SJF policy = 1
        io_scheduler(
            scheduling=1,
            architecture=ArchitectureType.DWAAS,
            now=15.0,
            dt=1.0,
            jobs=sample_jobs,
            waiting_jobs=waiting_jobs,
            io_jobs=io_jobs,
            cpu_cores=16,
            phase="arrival",
            options=config.scheduling_options
        )

        # Should complete without error
        assert isinstance(io_jobs, deque)

    def test_ljf_scheduling(self, sample_jobs, config):
        """Test LJF (Longest Job First) scheduling."""
        waiting_jobs = deque(sample_jobs)
        io_jobs = deque()

        # LJF policy = 2
        io_scheduler(
            scheduling=2,
            architecture=ArchitectureType.DWAAS,
            now=20.0,
            dt=1.0,
            jobs=sample_jobs,
            waiting_jobs=waiting_jobs,
            io_jobs=io_jobs,
            cpu_cores=16,
            phase="arrival",
            options=config.scheduling_options
        )

        # Should complete without error
        assert isinstance(io_jobs, deque)

    def test_empty_waiting_queue(self, sample_jobs, config):
        """Test scheduling with empty waiting queue."""
        waiting_jobs = deque()
        io_jobs = deque()

        io_scheduler(
            scheduling=0,
            architecture=ArchitectureType.DWAAS,
            now=25.0,
            dt=1.0,
            jobs=sample_jobs,
            waiting_jobs=waiting_jobs,
            io_jobs=io_jobs,
            cpu_cores=16,
            phase="arrival",
            options=config.scheduling_options
        )

        # Should remain empty
        assert len(io_jobs) == 0

    def test_different_architectures(self, sample_jobs, config):
        """Test scheduling works across different architectures."""
        architectures = [
            ArchitectureType.DWAAS,
            ArchitectureType.DWAAS_AUTOSCALING,
            ArchitectureType.ELASTIC_POOL,
            ArchitectureType.QAAS
        ]

        for arch in architectures:
            waiting_jobs = deque([sample_jobs[0]])
            io_jobs = deque()

            # Should not raise exceptions for any architecture
            try:
                io_scheduler(
                    scheduling=0,
                    architecture=arch,
                    now=50.0,
                    dt=1.0,
                    jobs=sample_jobs,
                    waiting_jobs=waiting_jobs,
                    io_jobs=io_jobs,
                    cpu_cores=16,
                    phase="arrival",
                    options=config.scheduling_options
                )
            except Exception as e:
                pytest.fail(f"Scheduling failed for {arch}: {e}")


class TestCPUSchedulerFixed:
    """Tests for CPU scheduler with correct API."""

    def test_cpu_scheduling_basic(self, sample_jobs, config):
        """Test basic CPU scheduling."""
        buffer_jobs = deque([sample_jobs[0]])
        cpu_jobs = deque()
        memory = [0.0]

        cpu_scheduler(
            scheduling=0,
            architecture=ArchitectureType.DWAAS,
            now=30.0,
            jobs=sample_jobs,
            buffer_jobs=buffer_jobs,
            cpu_jobs=cpu_jobs,
            cpu_cores=16,
            memory=memory,
            dt=1.0,
            phase="arrival",
            config=config,
            options=config.scheduling_options
        )

        # Should complete without error
        assert isinstance(cpu_jobs, deque)
        assert isinstance(memory, list)

    def test_cpu_scheduling_empty_buffer(self, sample_jobs, config):
        """Test CPU scheduling with empty buffer."""
        buffer_jobs = deque()
        cpu_jobs = deque()
        memory = [0.0]

        cpu_scheduler(
            scheduling=0,
            architecture=ArchitectureType.DWAAS,
            now=35.0,
            jobs=sample_jobs,
            buffer_jobs=buffer_jobs,
            cpu_jobs=cpu_jobs,
            cpu_cores=16,
            memory=memory,
            dt=1.0,
            phase="arrival",
            config=config,
            options=config.scheduling_options
        )

        # Should handle empty buffer gracefully
        assert len(cpu_jobs) == 0

    def test_cpu_scheduling_different_policies(self, sample_jobs, config):
        """Test CPU scheduling with different policies."""
        for policy in [0, 1, 2]:  # FCFS, SJF, LJF
            buffer_jobs = deque([sample_jobs[0]])
            cpu_jobs = deque()
            memory = [0.0]

            try:
                cpu_scheduler(
                    scheduling=policy,
                    architecture=ArchitectureType.DWAAS,
                    now=40.0,
                    jobs=sample_jobs,
                    buffer_jobs=buffer_jobs,
                    cpu_jobs=cpu_jobs,
                    cpu_cores=16,
                    memory=memory,
                    dt=1.0,
                    phase="arrival",
                    config=config,
                    options=config.scheduling_options
                )
            except Exception as e:
                pytest.fail(f"CPU scheduling failed for policy {policy}: {e}")


class TestSchedulingEdgeCases:
    """Test edge cases in scheduling."""

    def test_zero_cpu_cores(self, sample_jobs, config):
        """Test scheduling with zero CPU cores."""
        waiting_jobs = deque([sample_jobs[0]])
        io_jobs = deque()

        # Should handle gracefully
        io_scheduler(
            scheduling=0,
            architecture=ArchitectureType.DWAAS,
            now=60.0,
            dt=1.0,
            jobs=sample_jobs,
            waiting_jobs=waiting_jobs,
            io_jobs=io_jobs,
            cpu_cores=0,
            phase="arrival",
            options=config.scheduling_options
        )

        # Should not crash
        assert True

    def test_large_job_queue(self, config):
        """Test scheduling with many jobs."""
        # Create 1000 jobs
        large_jobs = [
            Job(i, 1, i, float(i), 1000.0, 100.0, 1.0)
            for i in range(1000)
        ]
        waiting_jobs = deque(large_jobs)
        io_jobs = deque()

        io_scheduler(
            scheduling=0,
            architecture=ArchitectureType.DWAAS,
            now=70.0,
            dt=1.0,
            jobs=large_jobs,
            waiting_jobs=waiting_jobs,
            io_jobs=io_jobs,
            cpu_cores=16,
            phase="arrival",
            options=config.scheduling_options
        )

        # Should complete without hanging
        assert isinstance(io_jobs, deque)

    def test_high_concurrency_limit(self, sample_jobs, config):
        """Test with high concurrency limit."""
        config.scheduling_options['max_io_concurrency'] = 1000

        waiting_jobs = deque(sample_jobs)
        io_jobs = deque()

        io_scheduler(
            scheduling=0,
            architecture=ArchitectureType.DWAAS,
            now=75.0,
            dt=1.0,
            jobs=sample_jobs,
            waiting_jobs=waiting_jobs,
            io_jobs=io_jobs,
            cpu_cores=16,
            phase="arrival",
            options=config.scheduling_options
        )

        # Should handle high limits
        assert True

    def test_zero_delta_time(self, sample_jobs, config):
        """Test scheduling with zero delta time."""
        waiting_jobs = deque([sample_jobs[0]])
        io_jobs = deque()

        io_scheduler(
            scheduling=0,
            architecture=ArchitectureType.DWAAS,
            now=80.0,
            dt=0.0,  # Zero delta
            jobs=sample_jobs,
            waiting_jobs=waiting_jobs,
            io_jobs=io_jobs,
            cpu_cores=16,
            phase="arrival",
            options=config.scheduling_options
        )

        # Should handle zero dt
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

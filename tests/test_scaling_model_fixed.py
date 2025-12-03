# test_scaling_model_fixed.py
"""
Fixed comprehensive unit tests for scaling_model.py
Tests queue-based, reactive, and predictive autoscaling with correct API.
"""

import pytest
from collections import deque
from cloudglide.scaling_model import Autoscaler
from cloudglide.job import Job
from cloudglide.config import SimulationConfig


@pytest.fixture
def config():
    """Create default simulation config with scaling options."""
    cfg = SimulationConfig()
    return cfg


@pytest.fixture
def sample_jobs():
    """Create sample jobs for scaling tests."""
    jobs = [
        Job(i, 1, i, float(i), 1000.0, 100.0, 1.0)
        for i in range(50)
    ]
    return jobs


class TestAutoscalerInitialization:
    """Test Autoscaler initialization."""

    def test_create_autoscaler(self, config):
        """Test creating an Autoscaler instance."""
        autoscaler = Autoscaler(
            cold_start=60.0,
            scaling_rules=config.scaling_options
        )

        assert autoscaler is not None

    def test_autoscaler_with_zero_cold_start(self, config):
        """Test autoscaler with zero cold start delay."""
        autoscaler = Autoscaler(
            cold_start=0.0,
            scaling_rules=config.scaling_options
        )

        assert autoscaler is not None

    def test_autoscaler_with_none_scaling_rules(self):
        """Test autoscaler with None scaling rules."""
        autoscaler = Autoscaler(
            cold_start=60.0,
            scaling_rules=None
        )

        assert autoscaler is not None


class TestQueueBasedScaling:
    """Test queue-based autoscaling policy."""

    def test_scale_out_on_queue_length(self, config, sample_jobs):
        """Test scaling out when queue exceeds threshold."""
        autoscaler = Autoscaler(60.0, config.scaling_options)

        cpu_jobs = deque()
        io_jobs = deque()
        waiting_jobs = deque(sample_jobs[:15])  # 15 jobs waiting
        buffer_jobs = deque()

        initial_nodes = 2
        cpu_cores_per_node = 8
        io_bandwidth = 1000
        cpu_cores = initial_nodes * cpu_cores_per_node
        base_n = 2
        memory = 64000
        current_second = 100.0
        second_range = 1.0
        events = []

        # Queue-based strategy = 1
        new_nodes, new_cpu_cores, new_io_bw, new_memory = (
            autoscaler.autoscaling_dw(
                strategy=1,
                cpu_jobs=cpu_jobs,
                io_jobs=io_jobs,
                waiting_jobs=waiting_jobs,
                buffer_jobs=buffer_jobs,
                nodes=initial_nodes,
                cpu_cores_per_node=cpu_cores_per_node,
                io_bandwidth=io_bandwidth,
                cpu_cores=cpu_cores,
                base_n=base_n,
                memory=memory,
                current_second=current_second,
                second_range=second_range,
                events=events
            )
        )

        # Should return valid values
        assert new_nodes >= base_n
        assert new_cpu_cores >= 0
        assert new_io_bw >= 0
        assert new_memory >= 0

    def test_scale_with_empty_queues(self, config):
        """Test scaling with empty queues."""
        autoscaler = Autoscaler(60.0, config.scaling_options)

        new_nodes, new_cpu_cores, new_io_bw, new_memory = (
            autoscaler.autoscaling_dw(
                strategy=1,
                cpu_jobs=deque(),
                io_jobs=deque(),
                waiting_jobs=deque(),
                buffer_jobs=deque(),
                nodes=2,
                cpu_cores_per_node=8,
                io_bandwidth=1000,
                cpu_cores=16,
                base_n=2,
                memory=64000,
                current_second=200.0,
                second_range=1.0,
                events=[]
            )
        )

        # Should not scale below base
        assert new_nodes >= 2

    def test_no_scale_below_base(self, config):
        """Test that scaling doesn't go below base_n."""
        autoscaler = Autoscaler(60.0, config.scaling_options)

        new_nodes, _, _, _ = autoscaler.autoscaling_dw(
            strategy=1,
            cpu_jobs=deque(),
            io_jobs=deque(),
            waiting_jobs=deque(),
            buffer_jobs=deque(),
            nodes=2,
            cpu_cores_per_node=8,
            io_bandwidth=1000,
            cpu_cores=16,
            base_n=2,
            memory=64000,
            current_second=300.0,
            second_range=1.0,
            events=[]
        )

        # Should not scale below base_n
        assert new_nodes >= 2


class TestReactiveScaling:
    """Test reactive autoscaling policy."""

    def test_reactive_scaling_basic(self, config, sample_jobs):
        """Test reactive scaling with active jobs."""
        autoscaler = Autoscaler(60.0, config.scaling_options)

        cpu_jobs = deque(sample_jobs[:10])

        new_nodes, new_cpu_cores, new_io_bw, new_memory = (
            autoscaler.autoscaling_dw(
                strategy=2,  # Reactive
                cpu_jobs=cpu_jobs,
                io_jobs=deque(),
                waiting_jobs=deque(),
                buffer_jobs=deque(),
                nodes=2,
                cpu_cores_per_node=8,
                io_bandwidth=1000,
                cpu_cores=16,
                base_n=2,
                memory=64000,
                current_second=400.0,
                second_range=1.0,
                events=[]
            )
        )

        # Should return valid values
        assert new_nodes >= 2
        assert new_cpu_cores >= 0


class TestPredictiveScaling:
    """Test predictive autoscaling policy."""

    def test_predictive_scaling_basic(self, config):
        """Test predictive scaling initialization."""
        autoscaler = Autoscaler(60.0, config.scaling_options)

        new_nodes, _, _, _ = autoscaler.autoscaling_dw(
            strategy=3,  # Predictive
            cpu_jobs=deque(),
            io_jobs=deque(),
            waiting_jobs=deque(),
            buffer_jobs=deque(),
            nodes=2,
            cpu_cores_per_node=8,
            io_bandwidth=1000,
            cpu_cores=16,
            base_n=2,
            memory=64000,
            current_second=500.0,
            second_range=1.0,
            events=[]
        )

        # Should return valid nodes
        assert new_nodes >= 2


class TestElasticPoolScaling:
    """Test autoscaling for Elastic Pool architecture."""

    def test_elastic_pool_scale_basic(self, config, sample_jobs):
        """Test VPU scaling for elastic pool."""
        autoscaler = Autoscaler(60.0, config.scaling_options)

        cpu_jobs = deque(sample_jobs[:10])
        waiting_jobs = deque(sample_jobs[10:20])

        new_vpu, new_cpu_cores = autoscaler.autoscaling_ec(
            strategy=1,  # Queue-based
            cpu_jobs=cpu_jobs,
            io_jobs=deque(),
            waiting_jobs=waiting_jobs,
            buffer_jobs=deque(),
            vpu=8,
            cpu_cores=32,
            base_cores=16,
            current_second=600.0,
            second_range=1.0,
            events=[]
        )

        # Should return valid VPU and cores
        assert new_vpu >= 0
        assert new_cpu_cores >= 0

    def test_elastic_pool_empty_queues(self, config):
        """Test VPU scaling with empty queues."""
        autoscaler = Autoscaler(60.0, config.scaling_options)

        new_vpu, new_cpu_cores = autoscaler.autoscaling_ec(
            strategy=1,
            cpu_jobs=deque(),
            io_jobs=deque(),
            waiting_jobs=deque(),
            buffer_jobs=deque(),
            vpu=8,
            cpu_cores=32,
            base_cores=16,
            current_second=700.0,
            second_range=1.0,
            events=[]
        )

        # Should return valid values
        assert new_vpu >= 0
        assert new_cpu_cores >= 0


class TestScalingEdgeCases:
    """Test edge cases in autoscaling."""

    def test_all_strategies_complete(self, config):
        """Test that all strategies complete without error."""
        autoscaler = Autoscaler(60.0, config.scaling_options)

        for strategy in [1, 2, 3]:
            try:
                autoscaler.autoscaling_dw(
                    strategy=strategy,
                    cpu_jobs=deque(),
                    io_jobs=deque(),
                    waiting_jobs=deque(),
                    buffer_jobs=deque(),
                    nodes=2,
                    cpu_cores_per_node=8,
                    io_bandwidth=1000,
                    cpu_cores=16,
                    base_n=2,
                    memory=64000,
                    current_second=800.0,
                    second_range=1.0,
                    events=[]
                )
            except Exception as e:
                pytest.fail(f"Scaling strategy {strategy} failed: {e}")

    def test_large_node_count(self, config):
        """Test scaling with many nodes."""
        autoscaler = Autoscaler(60.0, config.scaling_options)

        new_nodes, _, _, _ = autoscaler.autoscaling_dw(
            strategy=1,
            cpu_jobs=deque(),
            io_jobs=deque(),
            waiting_jobs=deque(),
            buffer_jobs=deque(),
            nodes=100,  # Many nodes
            cpu_cores_per_node=8,
            io_bandwidth=1000,
            cpu_cores=800,
            base_n=2,
            memory=64000,
            current_second=900.0,
            second_range=1.0,
            events=[]
        )

        # Should handle large node counts
        assert new_nodes >= 2

    def test_different_time_ranges(self, config):
        """Test scaling with different time deltas."""
        autoscaler = Autoscaler(60.0, config.scaling_options)

        for dt in [0.1, 1.0, 10.0]:
            new_nodes, _, _, _ = autoscaler.autoscaling_dw(
                strategy=1,
                cpu_jobs=deque(),
                io_jobs=deque(),
                waiting_jobs=deque(),
                buffer_jobs=deque(),
                nodes=2,
                cpu_cores_per_node=8,
                io_bandwidth=1000,
                cpu_cores=16,
                base_n=2,
                memory=64000,
                current_second=1000.0,
                second_range=dt,
                events=[]
            )

            # Should handle different deltas
            assert new_nodes >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

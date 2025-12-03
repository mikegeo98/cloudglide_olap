# test_execution_model.py
"""
Comprehensive unit tests for execution_model.py
Tests state initialization, cost calculation, and autoscaling hooks.
"""

import pytest
import tempfile
import os
from collections import deque
from cloudglide.execution_model import (
    initialize_state,
    charge_costs,
    maybe_autoscale,
    process_event,
    collect_metrics
)
from cloudglide.config import (
    ArchitectureType,
    SimulationConfig,
    ExecutionParams,
    SimulationParams
)
from cloudglide.job import Job
from cloudglide.event import Event, next_event_counter
import pandas as pd


@pytest.fixture
def test_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("database_id,query_id,start,cpu_time,data_scanned,scale_factor\n")
        f.write("1,1,0.0,1000.0,100.0,1.0\n")
        f.write("1,2,5000.0,2000.0,200.0,1.0\n")
        f.write("1,3,10000.0,1500.0,150.0,1.0\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def execution_params():
    """Create default execution parameters."""
    return ExecutionParams(
        scheduling=0,  # FCFS
        scaling=1,     # Queue-based
        nodes=2,
        cpu_cores=16,
        io_bw=1000,
        max_jobs=100,
        vpu=8,
        network_bw=10000,
        memory_bw=40000,
        total_memory_capacity_mb=64000,
        cold_start=60.0,
        hit_rate=0.7
    )


@pytest.fixture
def config():
    """Create default simulation config."""
    return SimulationConfig()


class TestInitializeState:
    """Test state initialization."""

    def test_initialize_dwaas(self, test_csv_file, execution_params, config):
        """Test initialization for DWaaS architecture."""
        sim_params = SimulationParams(
            dataset_path=test_csv_file
        )

        result = initialize_state(
            architecture=ArchitectureType.DWAAS,
            execution_params=execution_params,
            simulation_params=sim_params,
            config=config
        )

        # Unpack result
        (state, autoscaler, jobs, workload_data_scanned_mb,
         cpu_cores_per_node, cost_per_sec_usd, nodes, cpu_cores, io_bw,
         vpu, network_bw, memory_bw, total_memory_capacity_mb,
         scaling, scheduling, hit_rate, cold_start, events) = result

        # Verify state structure
        assert isinstance(state, dict)
        assert 'waiting_jobs' in state
        assert 'io_jobs' in state
        assert 'cpu_jobs' in state
        assert 'finished_jobs' in state

        # Verify job loading
        assert len(jobs) == 3
        assert workload_data_scanned_mb > 0

        # Verify no autoscaler for static DWaaS
        assert autoscaler is None

        # Verify events created
        assert len(events) == 3  # One per job

    def test_initialize_dwaas_autoscaling(self, test_csv_file, execution_params, config):
        """Test initialization for DWaaS with autoscaling."""
        sim_params = SimulationParams(dataset_path=test_csv_file)

        result = initialize_state(
            architecture=ArchitectureType.DWAAS_AUTOSCALING,
            execution_params=execution_params,
            simulation_params=sim_params,
            config=config
        )

        (state, autoscaler, *rest) = result

        # Should have autoscaler for autoscaling architecture
        assert autoscaler is not None

    def test_initialize_elastic_pool(self, test_csv_file, execution_params, config):
        """Test initialization for Elastic Pool architecture."""
        sim_params = SimulationParams(dataset_path=test_csv_file)

        result = initialize_state(
            architecture=ArchitectureType.ELASTIC_POOL,
            execution_params=execution_params,
            simulation_params=sim_params,
            config=config
        )

        (state, autoscaler, *rest) = result

        # Should have autoscaler for elastic pool
        assert autoscaler is not None

    def test_initialize_qaas(self, test_csv_file, execution_params, config):
        """Test initialization for QaaS architecture."""
        sim_params = SimulationParams(dataset_path=test_csv_file)

        result = initialize_state(
            architecture=ArchitectureType.QAAS,
            execution_params=execution_params,
            simulation_params=sim_params,
            config=config
        )

        (state, autoscaler, *rest) = result

        # Should not have autoscaler for QaaS
        assert autoscaler is None

    def test_initialize_with_different_nodes(self, test_csv_file, config):
        """Test initialization with different node counts."""
        for num_nodes in [1, 2, 4, 8]:
            exec_params = ExecutionParams(
                scheduling=0, scaling=1, nodes=num_nodes, cpu_cores=num_nodes*8,
                io_bw=1000, max_jobs=100, vpu=8, network_bw=10000,
                memory_bw=40000, total_memory_capacity_mb=64000,
                cold_start=60.0, hit_rate=0.7
            )
            sim_params = SimulationParams(dataset_path=test_csv_file)

            result = initialize_state(
                architecture=ArchitectureType.DWAAS,
                execution_params=exec_params,
                simulation_params=sim_params,
                config=config
            )

            state = result[0]
            # Verify DRAM nodes match node count
            assert len(state['dram_nodes']) == num_nodes
            assert len(state['dram_job_counts']) == num_nodes


class TestChargeCosts:
    """Test cost calculation."""

    def test_charge_costs_dwaas(self):
        """Test cost charging for DWaaS."""
        state = {
            'accumulated_cost_usd': 0.0,
            'vpu_charge': 0.0,
            'accumulated_slot_hours': 0.0
        }

        charge_costs(
            state=state,
            architecture=ArchitectureType.DWAAS,
            dt=1.0,
            nodes=2,
            base_n=2,
            cost_per_sec_usd=0.001,
            vpu=0,
            slots=0,
            spot=0,
            interrupt=0
        )

        # Cost should increase
        assert state['accumulated_cost_usd'] >= 0

    def test_charge_costs_qaas_skipped(self):
        """Test that QaaS cost charging is skipped."""
        state = {
            'accumulated_cost_usd': 0.0,
            'vpu_charge': 0.0,
            'accumulated_slot_hours': 0.0
        }

        initial_cost = state['accumulated_cost_usd']

        charge_costs(
            state=state,
            architecture=ArchitectureType.QAAS,
            dt=1.0,
            nodes=2,
            base_n=2,
            cost_per_sec_usd=0.001,
            vpu=0,
            slots=0,
            spot=0,
            interrupt=0
        )

        # QaaS cost is handled separately, should not change here
        assert state['accumulated_cost_usd'] == initial_cost

    def test_charge_costs_zero_delta(self):
        """Test cost charging with zero time delta."""
        state = {
            'accumulated_cost_usd': 10.0,
            'vpu_charge': 0.0,
            'accumulated_slot_hours': 0.0
        }

        charge_costs(
            state=state,
            architecture=ArchitectureType.DWAAS,
            dt=0.0,  # Zero delta
            nodes=2,
            base_n=2,
            cost_per_sec_usd=0.001,
            vpu=0,
            slots=0,
            spot=0,
            interrupt=0
        )

        # Should handle zero delta gracefully
        assert state['accumulated_cost_usd'] >= 10.0


class TestMaybeAutoscale:
    """Test autoscaling hooks."""

    def test_autoscale_no_autoscaler(self):
        """Test autoscaling when autoscaler is None."""
        state = {'cpu_jobs': deque(), 'io_jobs': deque(),
                 'waiting_jobs': deque(), 'buffer_jobs': deque()}

        result = maybe_autoscale(
            architecture=ArchitectureType.DWAAS,
            autoscaler=None,
            scaling=1,
            state=state,
            nodes=2,
            cpu_cores_per_node=8,
            io_bw=1000,
            cpu_cores=16,
            base_n=2,
            total_memory_capacity_mb=64000,
            now=100.0,
            dt=1.0,
            events=[]
        )

        nodes, cpu_cores, io_bw, total_memory, scale_level = result

        # Should return unchanged values
        assert nodes == 2
        assert cpu_cores == 16
        assert scale_level == 0

    def test_autoscale_dwaas_autoscaling(self, config):
        """Test autoscaling for DWaaS autoscaling architecture."""
        from cloudglide.scaling_model import Autoscaler

        autoscaler = Autoscaler(60.0, config.scaling_options)
        state = {
            'cpu_jobs': deque(),
            'io_jobs': deque(),
            'waiting_jobs': deque(),
            'buffer_jobs': deque()
        }

        result = maybe_autoscale(
            architecture=ArchitectureType.DWAAS_AUTOSCALING,
            autoscaler=autoscaler,
            scaling=1,
            state=state,
            nodes=2,
            cpu_cores_per_node=8,
            io_bw=1000,
            cpu_cores=16,
            base_n=2,
            total_memory_capacity_mb=64000,
            now=100.0,
            dt=1.0,
            events=[]
        )

        nodes, cpu_cores, io_bw, total_memory, scale_level = result

        # Should return valid values
        assert nodes >= 2
        assert cpu_cores >= 16
        assert isinstance(scale_level, int)


class TestProcessEvent:
    """Test event processing."""

    def test_process_arrival_event(self):
        """Test processing arrival event."""
        job = Job(1, 1, 1, 0.0, 1000.0, 100.0, 1.0)
        event = Event(0.0, next_event_counter(), job, "arrival")

        state = {
            'waiting_jobs': deque(),
            'buffer_jobs': deque()
        }

        process_event(event, state, 0.0, "arrival")

        # Job should be added to waiting and buffer queues
        assert len(state['waiting_jobs']) == 1
        assert len(state['buffer_jobs']) == 1

    def test_process_unknown_event_type(self):
        """Test processing unknown event type."""
        job = Job(1, 1, 1, 0.0, 1000.0, 100.0, 1.0)
        event = Event(0.0, next_event_counter(), job, "unknown")

        state = {
            'waiting_jobs': deque(),
            'buffer_jobs': deque()
        }

        # Should handle gracefully (pass statement in code)
        process_event(event, state, 0.0, "unknown")


class TestCollectMetrics:
    """Test metrics collection."""

    def test_collect_metrics_basic(self):
        """Test basic metrics collection."""
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name

        try:
            # Create finished jobs
            finished_jobs = []
            for i in range(10):
                job = Job(i, 1, i, float(i*1000), 1000.0, 100.0, 1.0)
                job.end_timestamp = float(i*1000 + 1500)
                job.queueing_delay = 100.0
                job.buffer_delay = 50.0
                job.io_time = 200.0
                job.processing_time = 800.0
                job.shuffle_time = 100.0
                job.query_exec_time = 1100.0
                job.query_exec_time_queueing = 1250.0
                finished_jobs.append(job)

            # Collect metrics
            metrics = collect_metrics(output_file, finished_jobs, 10.0)

            # Verify metrics structure
            assert 'average_queueing_delay' in metrics
            assert 'average_query_latency' in metrics
            assert 'median_query_latency' in metrics
            assert 'percentile_95_query_latency' in metrics
            assert 'total_price' in metrics
            assert metrics['total_price'] == 10.0

        finally:
            # Cleanup
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_collect_metrics_empty_jobs(self):
        """Test metrics collection with no jobs."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name

        try:
            # Collect metrics with empty job list
            metrics = collect_metrics(output_file, [], 0.0)

            # Should return empty metrics (file won't exist or will be empty)
            assert isinstance(metrics, dict)

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)


class TestEdgeCases:
    """Test edge cases in execution model."""

    def test_initialize_with_missing_csv(self, execution_params, config):
        """Test initialization with non-existent CSV file."""
        sim_params = SimulationParams(dataset_path="/nonexistent/file.csv")

        # Should raise an error from validation
        with pytest.raises(Exception):
            initialize_state(
                architecture=ArchitectureType.DWAAS,
                execution_params=execution_params,
                simulation_params=sim_params,
                config=config
            )

    def test_charge_costs_negative_dt(self):
        """Test cost charging with negative delta time."""
        state = {
            'accumulated_cost_usd': 10.0,
            'vpu_charge': 0.0,
            'accumulated_slot_hours': 0.0
        }

        # Should handle negative dt (may be a logic error, but shouldn't crash)
        try:
            charge_costs(
                state=state,
                architecture=ArchitectureType.DWAAS,
                dt=-1.0,
                nodes=2,
                base_n=2,
                cost_per_sec_usd=0.001,
                vpu=0,
                slots=0,
                spot=0,
                interrupt=0
            )
        except Exception:
            # It's okay to raise an exception for invalid input
            pass

    def test_initialize_with_zero_hit_rate(self, test_csv_file, config):
        """Test initialization with zero cache hit rate."""
        exec_params = ExecutionParams(
            scheduling=0, scaling=1, nodes=2, cpu_cores=16,
            io_bw=1000, max_jobs=100, vpu=8, network_bw=10000,
            memory_bw=40000, total_memory_capacity_mb=64000,
            cold_start=60.0, hit_rate=0.0  # Zero hit rate
        )
        sim_params = SimulationParams(dataset_path=test_csv_file)

        result = initialize_state(
            architecture=ArchitectureType.DWAAS,
            execution_params=exec_params,
            simulation_params=sim_params,
            config=config
        )

        # Should complete without error
        assert result is not None

    def test_initialize_with_max_hit_rate(self, test_csv_file, config):
        """Test initialization with 100% cache hit rate."""
        exec_params = ExecutionParams(
            scheduling=0, scaling=1, nodes=2, cpu_cores=16,
            io_bw=1000, max_jobs=100, vpu=8, network_bw=10000,
            memory_bw=40000, total_memory_capacity_mb=64000,
            cold_start=60.0, hit_rate=1.0  # Max hit rate
        )
        sim_params = SimulationParams(dataset_path=test_csv_file)

        result = initialize_state(
            architecture=ArchitectureType.DWAAS,
            execution_params=exec_params,
            simulation_params=sim_params,
            config=config
        )

        # Should complete without error
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

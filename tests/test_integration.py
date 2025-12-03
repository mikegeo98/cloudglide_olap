# test_integration.py
"""
Integration tests for end-to-end CloudGlide simulations.
Tests complete simulation workflows across different architectures.
"""

import pytest
import tempfile
import os
from cloudglide.execution_model import schedule_jobs
from cloudglide.config import (
    ArchitectureType,
    SimulationConfig,
    ExecutionParams,
    SimulationParams
)


@pytest.fixture
def small_test_csv():
    """Create a small CSV file for fast integration tests."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("database_id,query_id,start,cpu_time,data_scanned,scale_factor\n")
        # 5 small jobs
        for i in range(5):
            f.write(f"1,{i},{i*1000.0},500.0,50.0,1.0\n")
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def output_file():
    """Create temporary output file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestDWaaSIntegration:
    """Integration tests for DWaaS architecture."""

    def test_dwaas_simple_simulation(self, small_test_csv, output_file):
        """Test basic DWaaS simulation end-to-end."""
        exec_params = ExecutionParams(
            scheduling=0,  # FCFS
            scaling=1,
            nodes=2,
            cpu_cores=8,
            io_bw=1000,
            max_jobs=100,
            vpu=0,
            network_bw=10000,
            memory_bw=40000,
            total_memory_capacity_mb=32000,
            cold_start=60.0,
            hit_rate=0.7
        )

        sim_params = SimulationParams(dataset_path=small_test_csv)
        config = SimulationConfig()

        result = schedule_jobs(
            architecture=ArchitectureType.DWAAS,
            execution_params=exec_params,
            simulation_params=sim_params,
            output_file=output_file,
            config=config
        )

        # Verify result structure
        assert len(result) == 7
        scale_observe, mem_track, avg_latency, avg_wq_latency, total_price, median_latency, p95_latency = result

        # Verify metrics are reasonable
        assert avg_latency >= 0
        assert avg_wq_latency >= avg_latency
        assert total_price >= 0
        assert median_latency >= 0
        assert p95_latency >= median_latency

        # Verify output file was created
        assert os.path.exists(output_file)


class TestQaaSIntegration:
    """Integration tests for QaaS architecture."""

    def test_qaas_simple_simulation(self, small_test_csv, output_file):
        """Test basic QaaS simulation end-to-end."""
        exec_params = ExecutionParams(
            scheduling=0,
            scaling=1,
            nodes=0,  # QaaS doesn't use fixed nodes
            cpu_cores=16,
            io_bw=1000,
            max_jobs=100,
            vpu=0,
            network_bw=10000,
            memory_bw=40000,
            total_memory_capacity_mb=0,
            cold_start=0.0,
            hit_rate=0.0
        )

        sim_params = SimulationParams(dataset_path=small_test_csv)
        config = SimulationConfig()

        result = schedule_jobs(
            architecture=ArchitectureType.QAAS,
            execution_params=exec_params,
            simulation_params=sim_params,
            output_file=output_file,
            config=config
        )

        # Verify result
        assert len(result) == 7
        _, _, avg_latency, _, total_price, _, _ = result
        assert avg_latency >= 0
        assert total_price >= 0


class TestMultiArchitectureComparison:
    """Integration tests comparing multiple architectures."""

    def test_compare_dwaas_vs_qaas(self, small_test_csv):
        """Compare DWaaS vs QaaS on same workload."""
        config = SimulationConfig()

        results = {}

        for arch_name, arch_type, nodes in [
            ("DWaaS", ArchitectureType.DWAAS, 2),
            ("QaaS", ArchitectureType.QAAS, 0)
        ]:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                output_file = f.name

            try:
                exec_params = ExecutionParams(
                    scheduling=0, scaling=1, nodes=nodes,
                    cpu_cores=16, io_bw=1000, max_jobs=100, vpu=0,
                    network_bw=10000, memory_bw=40000,
                    total_memory_capacity_mb=32000 if nodes > 0 else 0,
                    cold_start=60.0 if nodes > 0 else 0.0,
                    hit_rate=0.7 if nodes > 0 else 0.0
                )

                sim_params = SimulationParams(dataset_path=small_test_csv)

                result = schedule_jobs(
                    architecture=arch_type,
                    execution_params=exec_params,
                    simulation_params=sim_params,
                    output_file=output_file,
                    config=config
                )

                results[arch_name] = {
                    "avg_latency": result[2],
                    "total_cost": result[4],
                    "median_latency": result[5]
                }

            finally:
                if os.path.exists(output_file):
                    os.unlink(output_file)

        # Verify both architectures completed
        assert "DWaaS" in results
        assert "QaaS" in results

        # Verify all metrics are positive
        for arch, metrics in results.items():
            assert metrics["avg_latency"] > 0, f"{arch} has invalid latency"
            assert metrics["total_cost"] >= 0, f"{arch} has negative cost"
            assert metrics["median_latency"] > 0, f"{arch} has invalid median"


class TestSchedulingPolicies:
    """Integration tests for different scheduling policies."""

    @pytest.mark.parametrize("policy", [0, 1, 2])  # FCFS, SJF, LJF
    def test_scheduling_policies(self, small_test_csv, output_file, policy):
        """Test that different scheduling policies complete successfully."""
        exec_params = ExecutionParams(
            scheduling=policy,
            scaling=1,
            nodes=2,
            cpu_cores=8,
            io_bw=1000,
            max_jobs=100,
            vpu=0,
            network_bw=10000,
            memory_bw=40000,
            total_memory_capacity_mb=32000,
            cold_start=60.0,
            hit_rate=0.7
        )

        sim_params = SimulationParams(dataset_path=small_test_csv)
        config = SimulationConfig()

        result = schedule_jobs(
            architecture=ArchitectureType.DWAAS,
            execution_params=exec_params,
            simulation_params=sim_params,
            output_file=output_file,
            config=config
        )

        # Verify completion
        assert result is not None
        assert result[2] >= 0  # avg_latency


class TestConfigValidation:
    """Integration tests for configuration validation."""

    def test_invalid_config_raises_error(self):
        """Test that invalid configs are caught."""
        with pytest.raises(ValueError) as exc_info:
            SimulationConfig(interrupt_probability=1.5)  # Invalid > 1

        assert "interrupt_probability" in str(exc_info.value)

    def test_valid_config_accepted(self):
        """Test that valid configs are accepted."""
        config = SimulationConfig(
            interrupt_probability=0.01,
            spot_discount=0.5,
            cold_start_delay=60.0
        )

        assert config.interrupt_probability == 0.01
        assert config.spot_discount == 0.5


class TestEdgeCases:
    """Integration tests for edge cases."""

    def test_single_job_simulation(self, output_file):
        """Test simulation with just one job."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("database_id,query_id,start,cpu_time,data_scanned,scale_factor\n")
            f.write("1,1,0.0,1000.0,100.0,1.0\n")
            single_job_csv = f.name

        try:
            exec_params = ExecutionParams(
                scheduling=0, scaling=1, nodes=1, cpu_cores=4,
                io_bw=1000, max_jobs=100, vpu=0, network_bw=10000,
                memory_bw=40000, total_memory_capacity_mb=16000,
                cold_start=60.0, hit_rate=0.7
            )

            config = SimulationConfig()
            sim_params = SimulationParams(dataset_path=single_job_csv)

            result = schedule_jobs(
                architecture=ArchitectureType.DWAAS,
                execution_params=exec_params,
                simulation_params=sim_params,
                output_file=output_file,
                config=config
            )

            assert result is not None
            assert result[2] >= 0  # avg_latency should be computed

        finally:
            if os.path.exists(single_job_csv):
                os.unlink(single_job_csv)

    def test_zero_hit_rate(self, small_test_csv, output_file):
        """Test simulation with zero cache hit rate (all cold)."""
        exec_params = ExecutionParams(
            scheduling=0, scaling=1, nodes=2, cpu_cores=8,
            io_bw=1000, max_jobs=100, vpu=0, network_bw=10000,
            memory_bw=40000, total_memory_capacity_mb=32000,
            cold_start=60.0, hit_rate=0.0  # All cache misses
        )

        config = SimulationConfig()

        sim_params = SimulationParams(dataset_path=small_test_csv)

        result = schedule_jobs(
            architecture=ArchitectureType.DWAAS,
            execution_params=exec_params,
            simulation_params=sim_params,
            output_file=output_file,
            config=config
        )

        assert result is not None
        # With zero hit rate, latency should be higher than with caching
        assert result[2] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

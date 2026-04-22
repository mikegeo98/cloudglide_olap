# test_validation.py
"""
Unit tests for the validation module.
Tests CSV and JSON schema validation functions.
"""

import pytest
import pandas as pd
from cloudglide.validation import (
    validate_csv_schema,
    validate_csv_data,
    validate_simulation_config,
    validate_scheduling_config,
    validate_scaling_config,
    validate_execution_params,
    CSVValidationError,
    JSONValidationError
)


class TestCSVSchemaValidation:
    """Test CSV schema validation."""

    def test_valid_csv_schema(self):
        """Test validation passes for valid CSV."""
        df = pd.DataFrame({
            'database_id': [1, 2],
            'query_id': [1, 2],
            'start': [0.0, 100.0],
            'cpu_time': [1000.0, 2000.0],
            'data_scanned': [100.0, 200.0],
            'scale_factor': [1.0, 2.0]
        })

        # Should not raise
        validate_csv_schema(df, 'test.csv')

    def test_missing_required_columns(self):
        """Test validation fails when required columns are missing."""
        df = pd.DataFrame({
            'database_id': [1, 2],
            'query_id': [1, 2],
            # Missing: start, cpu_time, data_scanned, scale_factor
        })

        with pytest.raises(CSVValidationError) as exc_info:
            validate_csv_schema(df, 'test.csv')

        assert 'missing required columns' in str(exc_info.value).lower()

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(CSVValidationError) as exc_info:
            validate_csv_schema(df, 'test.csv')

        assert 'empty' in str(exc_info.value).lower()

    def test_extra_columns_allowed(self):
        """Test validation passes with extra optional columns."""
        df = pd.DataFrame({
            'database_id': [1],
            'query_id': [1],
            'start': [0.0],
            'cpu_time': [1000.0],
            'data_scanned': [100.0],
            'scale_factor': [1.0],
            'extra_column': ['extra']  # Extra column should be OK
        })

        # Should not raise
        validate_csv_schema(df, 'test.csv')


class TestCSVDataValidation:
    """Test CSV data validation."""

    def test_valid_csv_data(self):
        """Test validation passes for valid data."""
        df = pd.DataFrame({
            'database_id': [1, 2],
            'query_id': [1, 2],
            'start': [0.0, 100.0],
            'cpu_time': [1000.0, 2000.0],
            'data_scanned': [100.0, 200.0],
            'scale_factor': [1.0, 2.0]
        })

        # Should not raise
        validate_csv_data(df, 'test.csv')

    def test_negative_start_time(self):
        """Test validation fails for negative start times."""
        df = pd.DataFrame({
            'database_id': [1],
            'query_id': [1],
            'start': [-10.0],  # Negative
            'cpu_time': [1000.0],
            'data_scanned': [100.0],
            'scale_factor': [1.0]
        })

        with pytest.raises(CSVValidationError) as exc_info:
            validate_csv_data(df, 'test.csv')

        assert 'start time' in str(exc_info.value).lower()

    def test_negative_cpu_time(self):
        """Test validation fails for negative CPU time."""
        df = pd.DataFrame({
            'database_id': [1],
            'query_id': [1],
            'start': [0.0],
            'cpu_time': [-1000.0],  # Negative
            'data_scanned': [100.0],
            'scale_factor': [1.0]
        })

        with pytest.raises(CSVValidationError) as exc_info:
            validate_csv_data(df, 'test.csv')

        assert 'cpu time' in str(exc_info.value).lower()

    def test_nan_values(self):
        """Test validation fails for NaN values."""
        df = pd.DataFrame({
            'database_id': [1, 2],
            'query_id': [1, None],  # NaN value
            'start': [0.0, 100.0],
            'cpu_time': [1000.0, 2000.0],
            'data_scanned': [100.0, 200.0],
            'scale_factor': [1.0, 2.0]
        })

        with pytest.raises(CSVValidationError) as exc_info:
            validate_csv_data(df, 'test.csv')

        assert 'nan' in str(exc_info.value).lower()

    def test_zero_scale_factor(self):
        """Test validation fails for scale_factor < 1."""
        df = pd.DataFrame({
            'database_id': [1],
            'query_id': [1],
            'start': [0.0],
            'cpu_time': [1000.0],
            'data_scanned': [100.0],
            'scale_factor': [0.5]  # Less than 1
        })

        with pytest.raises(CSVValidationError) as exc_info:
            validate_csv_data(df, 'test.csv')

        assert 'scale factor' in str(exc_info.value).lower()


class TestSimulationConfigValidation:
    """Test simulation configuration validation."""

    def test_valid_simulation_config(self):
        """Test validation passes for valid config."""
        config = {
            'interrupt_probability': 0.01,
            'spot_discount': 0.5,
            'cold_start_delay': 60.0,
            'cost_per_second_redshift': 0.0003
        }

        # Should not raise
        validate_simulation_config(config)

    def test_probability_out_of_range(self):
        """Test validation fails for probability > 1."""
        config = {
            'interrupt_probability': 1.5  # Invalid
        }

        with pytest.raises(JSONValidationError) as exc_info:
            validate_simulation_config(config)

        assert 'interrupt_probability' in str(exc_info.value)
        assert '[0, 1]' in str(exc_info.value)

    def test_negative_cold_start_delay(self):
        """Test validation fails for negative delay."""
        config = {
            'cold_start_delay': -60.0  # Negative
        }

        with pytest.raises(JSONValidationError) as exc_info:
            validate_simulation_config(config)

        assert 'cold_start_delay' in str(exc_info.value)

    def test_invalid_type(self):
        """Test validation fails for wrong type."""
        config = {
            'interrupt_probability': 'not_a_number'  # Wrong type
        }

        with pytest.raises(JSONValidationError) as exc_info:
            validate_simulation_config(config)

        assert 'numeric' in str(exc_info.value).lower()


class TestSchedulingConfigValidation:
    """Test scheduling configuration validation."""

    def test_valid_scheduling_config(self):
        """Test validation passes for valid config."""
        config = {
            'policy': 'fcfs',
            'max_io_concurrency': 64,
            'max_cpu_concurrency': 32
        }

        # Should not raise
        validate_scheduling_config(config)

    def test_invalid_policy(self):
        """Test validation fails for invalid policy."""
        config = {
            'policy': 'invalid_policy'
        }

        with pytest.raises(JSONValidationError) as exc_info:
            validate_scheduling_config(config)

        assert 'invalid scheduling policy' in str(exc_info.value).lower()

    def test_negative_concurrency(self):
        """Test validation fails for negative concurrency."""
        config = {
            'max_io_concurrency': -10
        }

        with pytest.raises(JSONValidationError) as exc_info:
            validate_scheduling_config(config)

        assert 'max_io_concurrency' in str(exc_info.value)

    def test_multi_level_without_queues(self):
        """Test multi-level policy requires queue definition."""
        config = {
            'policy': 'multi_level'
            # Missing: multi_level_queues
        }

        with pytest.raises(JSONValidationError) as exc_info:
            validate_scheduling_config(config)

        assert 'multi_level_queues' in str(exc_info.value)

    def test_integer_policy_valid(self):
        """Test integer policy values are accepted."""
        config = {
            'policy': 0  # Integer policy
        }

        # Should not raise
        validate_scheduling_config(config)


class TestScalingConfigValidation:
    """Test scaling configuration validation."""

    def test_valid_scaling_config(self):
        """Test validation passes for valid config."""
        config = {
            'policy': 'queue',
            'queue': {
                'length_thresholds': [5, 10, 15],
                'scale_steps': [4, 8, 12]
            }
        }

        # Should not raise
        validate_scaling_config(config)

    def test_invalid_scaling_policy(self):
        """Test validation fails for invalid policy."""
        config = {
            'policy': 'invalid'
        }

        with pytest.raises(JSONValidationError) as exc_info:
            validate_scaling_config(config)

        assert 'invalid scaling policy' in str(exc_info.value).lower()

    def test_non_ascending_thresholds(self):
        """Test validation fails for non-ascending thresholds."""
        config = {
            'queue': {
                'length_thresholds': [10, 5, 15]  # Not ascending
            }
        }

        with pytest.raises(JSONValidationError) as exc_info:
            validate_scaling_config(config)

        assert 'ascending order' in str(exc_info.value).lower()

    def test_negative_scale_steps(self):
        """Test validation fails for negative scale steps."""
        config = {
            'queue': {
                'scale_steps': [4, -8, 12]  # Negative value
            }
        }

        with pytest.raises(JSONValidationError) as exc_info:
            validate_scaling_config(config)

        assert 'positive' in str(exc_info.value).lower()

    def test_invalid_utilization_threshold(self):
        """Test validation fails for utilization > 1."""
        config = {
            'reactive': {
                'cpu_utilization_thresholds': [0.6, 1.5, 0.8]  # > 1
            }
        }

        with pytest.raises(JSONValidationError) as exc_info:
            validate_scaling_config(config)

        assert '[0, 1]' in str(exc_info.value)

    def test_invalid_growth_factor(self):
        """Test validation fails for negative growth factor."""
        config = {
            'predictive': {
                'growth_factor': -0.5  # Negative
            }
        }

        with pytest.raises(JSONValidationError) as exc_info:
            validate_scaling_config(config)

        assert 'growth_factor' in str(exc_info.value)


class TestExecutionParamsValidation:
    """Test execution parameters validation."""

    def test_valid_execution_params(self):
        """Test validation passes for valid params."""
        # Should not raise
        validate_execution_params(
            nodes=4,
            cpu_cores=32,
            hit_rate=0.7
        )

    def test_invalid_nodes(self):
        """Test validation fails for non-positive nodes."""
        with pytest.raises(JSONValidationError) as exc_info:
            validate_execution_params(
                nodes=0,  # Invalid
                cpu_cores=32,
                hit_rate=0.7
            )

        assert 'nodes' in str(exc_info.value)

    def test_invalid_hit_rate(self):
        """Test validation fails for hit_rate > 1."""
        with pytest.raises(JSONValidationError) as exc_info:
            validate_execution_params(
                nodes=4,
                cpu_cores=32,
                hit_rate=1.5  # Invalid
            )

        assert 'hit_rate' in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

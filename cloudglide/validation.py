# validation.py
"""
Input validation for CloudGlide OLAP simulation framework.
Provides CSV and JSON schema validation to catch errors early.
"""

import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


# Required CSV columns for workload datasets
REQUIRED_CSV_COLUMNS = [
    'database_id',
    'query_id',
    'start',
    'cpu_time',
    'data_scanned',
    'scale_factor'
]

# Optional but commonly used columns
OPTIONAL_CSV_COLUMNS = [
    'shuffled_mb',
    'memory_mb',
    'io_time',
    'network_time'
]


class CSVValidationError(Exception):
    """Raised when CSV validation fails."""
    pass


class JSONValidationError(Exception):
    """Raised when JSON schema validation fails."""
    pass


def validate_csv_schema(df: pd.DataFrame, file_path: str) -> None:
    """
    Validate that a CSV DataFrame contains all required columns.

    Args:
        df: DataFrame to validate
        file_path: Path to the CSV file (for error messages)

    Raises:
        CSVValidationError: If required columns are missing
    """
    if df.empty:
        raise CSVValidationError(
            f"CSV file '{file_path}' is empty"
        )

    missing_columns = [
        col for col in REQUIRED_CSV_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        raise CSVValidationError(
            f"CSV file '{file_path}' is missing required columns: "
            f"{', '.join(missing_columns)}. "
            f"Required columns are: {', '.join(REQUIRED_CSV_COLUMNS)}"
        )

    logger.info(f"CSV schema validation passed for '{file_path}'")


def validate_csv_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Validate that CSV data values are reasonable.

    Args:
        df: DataFrame to validate
        file_path: Path to the CSV file (for error messages)

    Raises:
        CSVValidationError: If data validation fails
    """
    errors = []

    # Check for negative values in numeric columns
    numeric_checks = {
        'start': ('start time', 0),
        'cpu_time': ('CPU time', 0),
        'data_scanned': ('data scanned', 0),
        'scale_factor': ('scale factor', 1)
    }

    for col, (name, min_val) in numeric_checks.items():
        if col in df.columns:
            if (df[col] < min_val).any():
                errors.append(
                    f"{name} contains values less than {min_val}"
                )

    # Check for NaN values in required columns
    for col in REQUIRED_CSV_COLUMNS:
        if col in df.columns and df[col].isna().any():
            nan_count = df[col].isna().sum()
            errors.append(
                f"Column '{col}' contains {nan_count} NaN values"
            )

    if errors:
        raise CSVValidationError(
            f"Data validation failed for '{file_path}':\n" +
            "\n".join(f"  - {error}" for error in errors)
        )

    logger.info(f"CSV data validation passed for '{file_path}'")


def validate_simulation_config(config: Dict[str, Any]) -> None:
    """
    Validate simulation configuration parameters.

    Args:
        config: Configuration dictionary to validate

    Raises:
        JSONValidationError: If configuration is invalid
    """
    errors = []

    # Validate probability values (should be in [0, 1])
    probability_fields = [
        'interrupt_probability',
        'spot_discount',
        'parallelizable_portion',
        'materialization_fraction'
    ]

    for field in probability_fields:
        if field in config:
            value = config[field]
            if not isinstance(value, (int, float)):
                errors.append(
                    f"'{field}' must be numeric, got {type(value)}"
                )
            elif not 0 <= value <= 1:
                errors.append(
                    f"'{field}' must be in [0, 1], got {value}"
                )

    # Validate positive values
    positive_fields = [
        'cold_start_delay',
        'cache_warmup_gamma',
        'logging_interval',
        'cost_per_second_redshift',
        'cost_per_rpu_hour',
        'cost_per_slot_hour'
    ]

    for field in positive_fields:
        if field in config:
            value = config[field]
            if not isinstance(value, (int, float)):
                errors.append(
                    f"'{field}' must be numeric, got {type(value)}"
                )
            elif value < 0:
                errors.append(
                    f"'{field}' must be non-negative, got {value}"
                )

    if errors:
        raise JSONValidationError(
            "Simulation configuration validation failed:\n" +
            "\n".join(f"  - {error}" for error in errors)
        )

    logger.info("Simulation configuration validation passed")


def validate_scheduling_config(config: Dict[str, Any]) -> None:
    """
    Validate scheduling configuration parameters.

    Args:
        config: Scheduling configuration dictionary

    Raises:
        JSONValidationError: If configuration is invalid
    """
    errors = []

    # Validate policy
    valid_policies = ['fcfs', 'sjf', 'ljf', 'multi', 'multi_level']
    if 'policy' in config:
        policy = str(config['policy']).lower().strip()
        if policy not in valid_policies and not isinstance(config['policy'], int):
            errors.append(
                f"Invalid scheduling policy '{config['policy']}'. "
                f"Valid options: {', '.join(valid_policies)}"
            )

    # Validate concurrency limits
    concurrency_fields = ['max_io_concurrency', 'max_cpu_concurrency']
    for field in concurrency_fields:
        if field in config and config[field] is not None:
            value = config[field]
            if not isinstance(value, int) or value <= 0:
                errors.append(
                    f"'{field}' must be a positive integer, got {value}"
                )

    # Validate multi-level queues if policy is multi
    if config.get('policy') in ['multi', 'multi_level', 3]:
        if 'multi_level_queues' not in config:
            errors.append(
                "Multi-level scheduling requires 'multi_level_queues' config"
            )
        elif not isinstance(config['multi_level_queues'], list):
            errors.append(
                "'multi_level_queues' must be a list"
            )

    if errors:
        raise JSONValidationError(
            "Scheduling configuration validation failed:\n" +
            "\n".join(f"  - {error}" for error in errors)
        )

    logger.info("Scheduling configuration validation passed")


def validate_scaling_config(config: Dict[str, Any]) -> None:
    """
    Validate autoscaling configuration parameters.

    Args:
        config: Scaling configuration dictionary

    Raises:
        JSONValidationError: If configuration is invalid
    """
    errors = []

    # Validate policy
    valid_policies = ['queue', 'queue_based', 'reactive', 'predictive']
    if 'policy' in config:
        policy = str(config['policy']).lower().strip()
        if policy not in valid_policies and not isinstance(config['policy'], int):
            errors.append(
                f"Invalid scaling policy '{config['policy']}'. "
                f"Valid options: {', '.join(valid_policies)}"
            )

    # Validate queue-based scaling
    if 'queue' in config:
        queue_config = config['queue']
        if 'length_thresholds' in queue_config:
            thresholds = queue_config['length_thresholds']
            if not isinstance(thresholds, list):
                errors.append(
                    "'length_thresholds' must be a list"
                )
            elif len(thresholds) > 1:
                # Check if ascending
                if not all(
                    thresholds[i] < thresholds[i+1]
                    for i in range(len(thresholds)-1)
                ):
                    errors.append(
                        "'length_thresholds' must be in ascending order"
                    )

        if 'scale_steps' in queue_config:
            steps = queue_config['scale_steps']
            if not isinstance(steps, list):
                errors.append("'scale_steps' must be a list")
            elif any(s <= 0 for s in steps):
                errors.append(
                    "'scale_steps' must contain only positive values"
                )

    # Validate reactive scaling
    if 'reactive' in config:
        reactive_config = config['reactive']
        if 'cpu_utilization_thresholds' in reactive_config:
            thresholds = reactive_config['cpu_utilization_thresholds']
            if not isinstance(thresholds, list):
                errors.append(
                    "'cpu_utilization_thresholds' must be a list"
                )
            elif any(not 0 <= t <= 1 for t in thresholds):
                errors.append(
                    "'cpu_utilization_thresholds' values must be in [0, 1]"
                )

    # Validate predictive scaling
    if 'predictive' in config:
        pred_config = config['predictive']
        if 'growth_factor' in pred_config:
            gf = pred_config['growth_factor']
            if not isinstance(gf, (int, float)) or gf <= 0:
                errors.append(
                    "'growth_factor' must be a positive number"
                )

    if errors:
        raise JSONValidationError(
            "Scaling configuration validation failed:\n" +
            "\n".join(f"  - {error}" for error in errors)
        )

    logger.info("Scaling configuration validation passed")


def validate_execution_params(
    nodes: int,
    cpu_cores: int,
    hit_rate: float
) -> None:
    """
    Validate execution parameters.

    Args:
        nodes: Number of nodes
        cpu_cores: Number of CPU cores
        hit_rate: Cache hit rate

    Raises:
        JSONValidationError: If parameters are invalid
    """
    errors = []

    if nodes <= 0:
        errors.append(f"'nodes' must be positive, got {nodes}")

    if cpu_cores <= 0:
        errors.append(f"'cpu_cores' must be positive, got {cpu_cores}")

    if not 0 <= hit_rate <= 1:
        errors.append(f"'hit_rate' must be in [0, 1], got {hit_rate}")

    if errors:
        raise JSONValidationError(
            "Execution parameters validation failed:\n" +
            "\n".join(f"  - {error}" for error in errors)
        )

    logger.info("Execution parameters validation passed")

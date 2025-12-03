# config.py

from dataclasses import dataclass, field
from typing import Any, Dict, List
from copy import deepcopy
from enum import Enum

# ==========================================================
# 1. Architecture Types
# ==========================================================


class ArchitectureType(Enum):
    DWAAS = 0                # Traditional static DWaaS (no autoscaling)
    DWAAS_AUTOSCALING = 1    # DWaaS with autoscaling
    ELASTIC_POOL = 2         # Elastic compute pool (RPU-based)
    QAAS = 3                 # Query-as-a-Service (pay-per-query)
    QAAS_CAPACITY = 4        # QaaS with reserved capacity pricing
    SERVERLESS = 5           # Serverless (slot-based or ephemeral)



# ==========================================================
# 2. Instance Configuration
# ==========================================================
@dataclass
class InstanceConfig:
    cpu_cores: int
    memory: int  # in GB
    io_bandwidth: int  # in Mbps
    network_bandwidth: int  # in Mbps
    memory_bandwidth: int  # in Mbps


@dataclass
class ExecutionParams:
    """
    Structured parameters for simulation execution.
    Replaces fragile tuple unpacking with type-safe dataclass.
    """
    scheduling: int
    scaling: int
    nodes: int
    cpu_cores: int
    io_bw: int
    max_jobs: int
    vpu: int
    network_bw: int
    memory_bw: int
    total_memory_capacity_mb: int
    cold_start: float
    hit_rate: float


@dataclass
class SimulationParams:
    """
    Parameters for simulation initialization.
    """
    dataset_path: str
    output_file: str = ""


# Define all instance types
INSTANCE_TYPES: List[InstanceConfig] = [
    InstanceConfig(cpu_cores=4, memory=32, io_bandwidth=650, network_bandwidth=1000, memory_bandwidth=40000),   # ra3.xlplus
    InstanceConfig(cpu_cores=12, memory=96, io_bandwidth=2000, network_bandwidth=10000, memory_bandwidth=40000),  # ra3.4xlarge
    InstanceConfig(cpu_cores=48, memory=384, io_bandwidth=8000, network_bandwidth=10000, memory_bandwidth=40000),  # ra3.16xlarge
    InstanceConfig(cpu_cores=4, memory=8, io_bandwidth=143, network_bandwidth=10000, memory_bandwidth=40000),     # c5d.xlarge
    InstanceConfig(cpu_cores=8, memory=16, io_bandwidth=287, network_bandwidth=10000, memory_bandwidth=40000),    # c5d.2xlarge
    InstanceConfig(cpu_cores=16, memory=32, io_bandwidth=575, network_bandwidth=10000, memory_bandwidth=40000),   # c5d.4xlarge
    InstanceConfig(cpu_cores=4, memory=32, io_bandwidth=650, network_bandwidth=1000, memory_bandwidth=2500),   # ra3.xlplus
]

# ==========================================================
# 3. Simulation Configuration
# ==========================================================
def _default_scheduling_options() -> Dict[str, Any]:
    return {
        "max_io_concurrency": None,
        "max_cpu_concurrency": None,
        "multi_level_queues": []
    }


def _default_scaling_options() -> Dict[str, Any]:
    return {
        "queue": {
            "length_thresholds": [5, 10, 15, 20],
            "scale_steps": [4, 8, 12, 16],
            "scale_in_utilization": 0.4,
            "scale_in_step": 4,
        },
        "reactive": {
            "cpu_utilization_thresholds": [0.6, 0.7, 0.8],
            "scale_steps": [8, 16, 24],
            "scale_in_utilization": 0.1,
            "scale_in_step": 8,
        },
        "predictive": {
            "growth_factor": 1.2,
            "decline_factor": 0.75,
            "history": 3,
            "observation_interval": 10000,
            "scale_step": 4,
        },
    }


@dataclass
class SimulationConfig:
    # Scaling
    scale_factor_min: int = 1
    scale_factor_max: int = 100
    shuffle_percentage_min: float = 0.1
    shuffle_percentage_max: float = 0.35

    # Interrupt parameters
    interrupt_probability: float = 0.01
    interrupt_duration: int = 6000  # seconds

    # Cost parameters
    cost_per_second_redshift: float = 0.00030166666
    cost_per_rpu_hour: float = 0.375
    cost_per_slot_hour: float = 0.04
    qaas_cost_per_tb: float = 5.0       # $5 per TB scanned
    spot_discount: float = 0.5          # 50% discount for spot instances

    # Logging / simulation timing
    logging_interval: int = 60          # seconds
    default_output_prefix: str = "cloudglide/output_simulation/simulation"
    default_sleep_time: int = 0
    default_max_duration: int = 108000  # seconds

    # Cache warm-up and cold-start
    cold_start_delay: float = 60.0      # seconds, configurable
    cache_warmup_gamma: float = 0.05     # parameter for exponential warm-up
    materialization_fraction: float = 0.25
    parallelizable_portion: float = 0.9

    use_spot_instances: bool = False
    default_estimator: str = "pm"
    pm_p: float = 4.0
    delta: float = 0.3
    queue_agg: str = "sum"

    qaas_io_per_core_bw: int = 150
    qaas_shuffle_bw_per_core: int = 50
    qaas_base_cores: int = 4
    qaas_base_time_limit: int = 2
    core_alloc_window: float = 10.0
    s3_bandwidth: int = 1000

    scheduling_options: Dict[str, Any] = field(default_factory=_default_scheduling_options)
    scaling_options: Dict[str, Any] = field(default_factory=_default_scaling_options)

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        errors = []

        # Validate probability values [0, 1]
        if not 0 <= self.interrupt_probability <= 1:
            errors.append(
                f"interrupt_probability must be in [0, 1], got {self.interrupt_probability}"
            )
        if not 0 <= self.spot_discount <= 1:
            errors.append(
                f"spot_discount must be in [0, 1], got {self.spot_discount}"
            )
        if not 0 <= self.materialization_fraction <= 1:
            errors.append(
                f"materialization_fraction must be in [0, 1], got {self.materialization_fraction}"
            )
        if not 0 <= self.parallelizable_portion <= 1:
            errors.append(
                f"parallelizable_portion must be in [0, 1], got {self.parallelizable_portion}"
            )
        if not 0 <= self.shuffle_percentage_min <= 1:
            errors.append(
                f"shuffle_percentage_min must be in [0, 1], got {self.shuffle_percentage_min}"
            )
        if not 0 <= self.shuffle_percentage_max <= 1:
            errors.append(
                f"shuffle_percentage_max must be in [0, 1], got {self.shuffle_percentage_max}"
            )

        # Validate positive values
        if self.cold_start_delay < 0:
            errors.append(
                f"cold_start_delay must be non-negative, got {self.cold_start_delay}"
            )
        if self.cache_warmup_gamma < 0:
            errors.append(
                f"cache_warmup_gamma must be non-negative, got {self.cache_warmup_gamma}"
            )
        if self.logging_interval <= 0:
            errors.append(
                f"logging_interval must be positive, got {self.logging_interval}"
            )
        if self.cost_per_second_redshift < 0:
            errors.append(
                f"cost_per_second_redshift must be non-negative, got {self.cost_per_second_redshift}"
            )
        if self.cost_per_rpu_hour < 0:
            errors.append(
                f"cost_per_rpu_hour must be non-negative, got {self.cost_per_rpu_hour}"
            )
        if self.cost_per_slot_hour < 0:
            errors.append(
                f"cost_per_slot_hour must be non-negative, got {self.cost_per_slot_hour}"
            )
        if self.qaas_cost_per_tb < 0:
            errors.append(
                f"qaas_cost_per_tb must be non-negative, got {self.qaas_cost_per_tb}"
            )

        # Validate scale factor range
        if self.scale_factor_min < 1:
            errors.append(
                f"scale_factor_min must be >= 1, got {self.scale_factor_min}"
            )
        if self.scale_factor_max < self.scale_factor_min:
            errors.append(
                f"scale_factor_max ({self.scale_factor_max}) must be >= scale_factor_min ({self.scale_factor_min})"
            )

        # Validate shuffle percentage range
        if self.shuffle_percentage_max < self.shuffle_percentage_min:
            errors.append(
                f"shuffle_percentage_max ({self.shuffle_percentage_max}) must be >= shuffle_percentage_min ({self.shuffle_percentage_min})"
            )

        # Validate estimator
        valid_estimators = ["max", "sum", "cpu_only", "pm", "mw"]
        if self.default_estimator not in valid_estimators:
            errors.append(
                f"default_estimator must be one of {valid_estimators}, got '{self.default_estimator}'"
            )

        # Validate queue aggregation
        valid_queue_agg = ["max", "sum"]
        if self.queue_agg not in valid_queue_agg:
            errors.append(
                f"queue_agg must be one of {valid_queue_agg}, got '{self.queue_agg}'"
            )

        if errors:
            raise ValueError(
                "SimulationConfig validation failed:\n" +
                "\n".join(f"  - {error}" for error in errors)
            )

    def copy(self) -> "SimulationConfig":
        clone = deepcopy(self)
        return clone

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Update config attributes (including nested scheduling/scaling dictionaries)
        using values supplied in overrides.
        """
        if not overrides:
            return

        data = dict(overrides)
        scheduling_payload = data.pop("scheduling", None) or data.pop("scheduling_options", None)
        scaling_payload = data.pop("scaling", None) or data.pop("scaling_options", None)

        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if scheduling_payload:
            self.scheduling_options = _deep_merge_dicts(self.scheduling_options, scheduling_payload)
        if scaling_payload:
            self.scaling_options = _deep_merge_dicts(self.scaling_options, scaling_payload)


def _deep_merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a new dict that merges overrides into base recursively.
    """
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result

# ==========================================================
# 4. Dataset File Mapping
# ==========================================================
DATASET_FILES = {
    1: "cloudglide/datasets/cab_10_4.csv",
    3: "cloudglide/datasets/cab_10_20.csv",
    4: "cloudglide/datasets/cab_20_4.csv",
    16: "cloudglide/datasets/qs0_40_4f.csv",
    2: "cloudglide/datasets/cabby.csv",
    5: "cloudglide/datasets/concurrency1.csv",
    6: "cloudglide/datasets/concurrency2.csv",
    17: "cloudglide/datasets/qs1_40_4f.csv",
    18: "cloudglide/datasets/qs2_40_4f.csv",
    19: "cloudglide/datasets/qs3_40_4f.csv",
    20: "cloudglide/datasets/qs4_40_4f.csv",
    21: "cloudglide/datasets/pattern_2_50f.csv",
    22: "cloudglide/datasets/tenk.csv",
    997: "cloudglide/datasets/biggo_edited.csv",
    998: "cloudglide/datasets/concurrency.csv",
    999: "cloudglide/datasets/tpch_all_runs.csv",
}


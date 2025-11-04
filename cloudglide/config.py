# config.py

from dataclasses import dataclass
from typing import List
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

    qaas_io_per_core_bw = 150
    qaas_shuffle_bw_per_core = 50
    qaas_base_cores = 4
    qaas_base_time_limit = 2
    core_alloc_window = 10.0
    s3_bandwidth = 1000

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

# ==========================================================
# 5. Estimator Defaults
# ==========================================================
DEFAULT_ESTIMATOR = "pm"   # "max", "cpu_only", "sum", "pm", "mw"
PM_P = 4.0                 # p for power-mean
DELTA = 0.3                # fixed offset for MAX+δ
QUEUE_AGG = "sum"          # how to combine queue + buffer delays


# Cache and memory parameters
FMEM = 0.7                      
DELTA_SPILLED = 0.3             

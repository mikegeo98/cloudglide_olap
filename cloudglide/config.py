# config.py

from dataclasses import dataclass
from typing import List


@dataclass
class InstanceConfig:
    cpu_cores: int
    memory: int  # in GB
    io_bandwidth: int  # in Mbps
    network_bandwidth: int  # in Mbps
    memory_bandwidth: int  # in Mbps


# Define all instance types
INSTANCE_TYPES: List[InstanceConfig] = [
    InstanceConfig(cpu_cores=4, memory=32, io_bandwidth=650, network_bandwidth=1000, memory_bandwidth=30000),   # ra3.xlplus
    InstanceConfig(cpu_cores=12, memory=96, io_bandwidth=2000, network_bandwidth=10000, memory_bandwidth=40000),  # ra3.4xlarge
    InstanceConfig(cpu_cores=48, memory=384, io_bandwidth=8000, network_bandwidth=10000, memory_bandwidth=40000),  # ra3.16xlarge
    InstanceConfig(cpu_cores=4, memory=8, io_bandwidth=143, network_bandwidth=10000, memory_bandwidth=40000),     # c5d.xlarge
    InstanceConfig(cpu_cores=8, memory=16, io_bandwidth=287, network_bandwidth=10000, memory_bandwidth=40000),    # c5d.2xlarge
    InstanceConfig(cpu_cores=16, memory=32, io_bandwidth=575, network_bandwidth=10000, memory_bandwidth=40000),   # c5d.4xlarge
]

# Scaling factors and percentages
SCALE_FACTOR_MIN = 1
SCALE_FACTOR_MAX = 100
SHUFFLE_PERCENTAGE_MIN = 0.1
SHUFFLE_PERCENTAGE_MAX = 0.35

# Interrupt parameters
INTERRUPT_PROBABILITY = 0.01
INTERRUPT_DURATION = 6000  # in seconds

# Cost parameters
COST_PER_SECOND_REDSHIFT = 0.00030166666
COST_PER_RPU_HOUR = 0.375
COST_PER_SLOT_HOUR = 0.04

# Logging and simulation parameters
LOGGING_INTERVAL = 60  # in seconds
DEFAULT_OUTPUT_PREFIX = "cloudglide/output_simulation/simulation"
DEFAULT_SLEEP_TIME = 0
DEFAULT_MAX_DURATION = 108000  # in seconds

# Dataset file paths
DATASET_FILES = {
    999: 'cloudglide/datasets/tpch_all_runs.csv'
}

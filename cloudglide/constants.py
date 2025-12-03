# constants.py
"""
Named constants for CloudGlide OLAP simulation framework.
Replaces magic numbers throughout the codebase for better maintainability.
"""

# ==========================================================
# Memory Tier Probabilities
# ==========================================================

# Memory tier distribution for cache misses
MEMORY_TIER_SSD_FRACTION = 0.67  # 67% of misses go to SSD
MEMORY_TIER_S3_FRACTION = 0.33   # 33% of misses go to S3

# ==========================================================
# Simulation Timing
# ==========================================================

# Simulation timestep in Hz (used in execution_model.py)
SIMULATION_TIMESTEP_HZ = 10  # 10 Hz = 0.1 second resolution
SIMULATION_TIMESTEP_DIVISOR = 10  # For math.ceil(now * 10) / 10

# Logging and monitoring intervals
DEFAULT_LOGGING_INTERVAL_SECONDS = 60
DEFAULT_MEMORY_TRACKING_INTERVAL_SECONDS = 60

# ==========================================================
# Cost Model Constants
# ==========================================================

# Unit conversions
SECONDS_PER_HOUR = 3600
MILLISECONDS_PER_SECOND = 1000
BYTES_PER_MB = 1000  # Note: Using 1000 not 1024 for consistency with data_scanned

# ==========================================================
# CPU and Core Allocation
# ==========================================================

# Default cores per node for Elastic Pool
ELASTIC_POOL_DEFAULT_CORES_PER_NODE = 4

# Minimum cores for job execution
MIN_CORES_FOR_JOB = 1

# ==========================================================
# QaaS (Query-as-a-Service) Constants
# ==========================================================

# QaaS CPU time thresholds for core allocation (in milliseconds)
QAAS_CPU_TIME_THRESHOLD_HIGH = 6000000  # 6000 seconds
QAAS_CPU_TIME_THRESHOLD_MED = 4000000   # 4000 seconds
QAAS_CPU_TIME_THRESHOLD_LOW = 2000000   # 2000 seconds

# QaaS target execution times (in seconds)
QAAS_TARGET_EXEC_TIME_HIGH = 18  # For jobs > 6000s CPU time
QAAS_TARGET_EXEC_TIME_MED_MAX = 6   # Upper bound for 4000-6000s
QAAS_TARGET_EXEC_TIME_LOW_MAX = 2   # Upper bound for 2000-4000s

# Interpolation ranges
QAAS_INTERP_HIGH_RANGE = 12  # 18 - 6
QAAS_INTERP_MED_RANGE = 4    # 6 - 2

# ==========================================================
# Cache Warmup Model
# ==========================================================

# Default gamma for exponential warmup: P_DRAM(n) = hit_rate * (1 - exp(-γ * n))
DEFAULT_CACHE_WARMUP_GAMMA = 0.05

# ==========================================================
# Amdahl's Law and Parallelization
# ==========================================================

# Default parallelizable portion for Amdahl's Law
DEFAULT_PARALLELIZABLE_PORTION = 0.9  # 90% of work can be parallelized

# Speedup formula: 1 / ((1 - p) + (p / cores))
# where p = parallelizable_portion

# ==========================================================
# Job Finalization Estimators
# ==========================================================

# Multi-wave estimator weights (for T_mw calculation)
MW_ESTIMATOR_CPU_WEIGHT_1 = 0.8
MW_ESTIMATOR_SHUFFLE_WEIGHT_1 = 0.2
MW_ESTIMATOR_CPU_WEIGHT_2 = 0.1
MW_ESTIMATOR_SHUFFLE_WEIGHT_2 = 0.6
MW_ESTIMATOR_CPU_WEIGHT_3 = 0.1
MW_ESTIMATOR_SHUFFLE_WEIGHT_3 = 0.2

# Default p-norm power for power-mean estimator
DEFAULT_PM_P = 4.0

# Default delta adjustment for estimators
DEFAULT_ESTIMATOR_DELTA = 0.3

# ==========================================================
# Scheduling Policies
# ==========================================================

SCHEDULING_POLICY_FCFS = 0  # First-Come-First-Served
SCHEDULING_POLICY_SJF = 1   # Shortest Job First
SCHEDULING_POLICY_LJF = 2   # Longest Job First
SCHEDULING_POLICY_MULTI = 3  # Multi-level priority queues

# ==========================================================
# Autoscaling Policies
# ==========================================================

AUTOSCALING_POLICY_NONE = 0       # No autoscaling
AUTOSCALING_POLICY_QUEUE = 1      # Queue-based autoscaling
AUTOSCALING_POLICY_REACTIVE = 2   # Reactive (utilization-based)
AUTOSCALING_POLICY_PREDICTIVE = 3  # Predictive (trend-based)

# ==========================================================
# Architecture Types (reference - defined in config.py)
# ==========================================================

# Note: ArchitectureType enum is in config.py
# DWAAS = 0
# DWAAS_AUTOSCALING = 1
# ELASTIC_POOL = 2
# QAAS = 3
# QAAS_CAPACITY = 4
# SERVERLESS = 5

# ==========================================================
# Error Tolerance and Thresholds
# ==========================================================

# Minimum time difference for event rescheduling (seconds)
EVENT_RESCHEDULE_TOLERANCE = 1e-6

# Minimum value to avoid division by zero
EPSILON = 1e-10

# ==========================================================
# Data Validation
# ==========================================================

# Minimum scale factor for queries
MIN_SCALE_FACTOR = 1

# Maximum reasonable query duration (seconds) for sanity checks
MAX_REASONABLE_QUERY_DURATION = 86400  # 24 hours

# Maximum reasonable data scanned (MB) for sanity checks
MAX_REASONABLE_DATA_SCANNED_MB = 1e9  # 1 petabyte

# ==========================================================
# Output and Reporting
# ==========================================================

# Default percentiles for latency reporting
DEFAULT_LATENCY_PERCENTILES = [50, 95, 99]

# CSV output precision (decimal places)
CSV_OUTPUT_PRECISION = 4

# ==========================================================
# Performance Optimization
# ==========================================================

# Threshold for switching between O(1) and O(n) job removal strategies
INDEXED_QUEUE_SIZE_THRESHOLD = 100

# Maximum events to process before yielding (for long simulations)
MAX_EVENTS_BATCH_SIZE = 10000

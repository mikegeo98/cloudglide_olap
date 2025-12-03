# CloudGlide OLAP Simulator - Parameters Guide

This guide documents the essential and optional parameters for each architecture type.

## Table of Contents
- [DWaaS (Static)](#dwaas-static)
- [DWaaS Autoscaling](#dwaas-autoscaling)
- [Elastic Pool](#elastic-pool)
- [QaaS](#qaas)
- [Common Parameters](#common-parameters)

---

## DWaaS (Static)

**Architecture Type:** `DWAAS` (0)

**Description:** Traditional static Data Warehouse as a Service with a fixed cluster size. No autoscaling is performed.

### Essential Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `nodes` | int | Number of compute nodes (fixed) | `4` |
| `instance` | int | Instance type index (0-6, see INSTANCE_TYPES) | `0` |
| `hit_rate` | float | Cache hit rate (0.0-1.0) | `0.7` |
| `dataset` | int | Dataset identifier | `999` |
| `architecture` | string | Must be "DWAAS" | `"DWAAS"` |

### Optional Parameters

| Parameter | Type | Description | Default | Example |
|-----------|------|-------------|---------|---------|
| `network_bandwidth` | int | Network bandwidth in Mbps | From instance config | `10000` |
| `io_bandwidth` | int | I/O bandwidth in Mbps | From instance config | `650` |
| `memory_bandwidth` | int | Memory bandwidth in Mbps | From instance config | `40000` |
| `total_memory_capacity_mb` | int | Total memory in MB | Calculated from nodes | `128000` |
| `scheduling.policy` | string | Scheduling policy ("fcfs", "sjf", "ljf", "multi_level") | `"fcfs"` | `"sjf"` |
| `scheduling.max_io_concurrency` | int | Max concurrent I/O jobs | `64` | `32` |
| `scheduling.max_cpu_concurrency` | int | Max concurrent CPU jobs | `64` | `32` |

### Not Needed for DWaaS

- `vpu` - Only for Elastic Pool
- `cold_start` - Only for autoscaling architectures
- `scaling.*` - Only for autoscaling architectures

### Example Configuration

```json
{
  "name": "dwaas_static",
  "architecture": "DWAAS",
  "dataset": 999,
  "nodes": 4,
  "instance": 0,
  "hit_rate": 0.7,
  "network_bandwidth": 10000,
  "io_bandwidth": 650,
  "memory_bandwidth": 40000,
  "total_memory_capacity_mb": 128000,
  "runs": [
    {
      "name": "dwaas_fcfs",
      "scheduling": {
        "policy": "fcfs",
        "max_io_concurrency": 32,
        "max_cpu_concurrency": 32
      }
    }
  ]
}
```

---

## DWaaS Autoscaling

**Architecture Type:** `DWAAS_AUTOSCALING` (1)

**Description:** Data Warehouse as a Service with dynamic node scaling based on workload.

### Essential Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `nodes` | int | Initial number of compute nodes | `2` |
| `instance` | int | Instance type index (0-6) | `0` |
| `hit_rate` | float | Cache hit rate (0.0-1.0) | `0.7` |
| `dataset` | int | Dataset identifier | `999` |
| `architecture` | string | Must be "DWAAS_AUTOSCALING" | `"DWAAS_AUTOSCALING"` |
| `cold_start` | float | Cold start delay in seconds when scaling | `60.0` |
| `scaling.policy` | string | Scaling policy ("queue", "reactive", "predictive") | `"queue"` |

### Optional Parameters

| Parameter | Type | Description | Default | Example |
|-----------|------|-------------|---------|---------|
| `network_bandwidth` | int | Network bandwidth in Mbps | From instance | `10000` |
| `io_bandwidth` | int | I/O bandwidth in Mbps | From instance | `650` |
| `memory_bandwidth` | int | Memory bandwidth in Mbps | From instance | `40000` |
| `total_memory_capacity_mb` | int | Total memory in MB | Calculated | `64000` |
| `scheduling.policy` | string | Scheduling policy | `"fcfs"` | `"sjf"` |
| `scheduling.max_io_concurrency` | int | Max concurrent I/O jobs | `64` | `32` |
| `scheduling.max_cpu_concurrency` | int | Max concurrent CPU jobs | `64` | `32` |
| `use_spot_instances` | bool | Use spot instances (50% discount) | `false` | `true` |

### Scaling Policy Parameters

#### Queue-Based Scaling
```json
"scaling": {
  "policy": "queue",
  "queue": {
    "length_thresholds": [5, 10, 15, 20],
    "scale_steps": [4, 8, 12, 16],
    "scale_in_utilization": 0.4,
    "scale_in_step": 4
  }
}
```

#### Reactive Scaling (CPU-based)
```json
"scaling": {
  "policy": "reactive",
  "reactive": {
    "cpu_utilization_thresholds": [0.6, 0.7, 0.8],
    "scale_steps": [8, 16, 24],
    "scale_in_utilization": 0.2,
    "scale_in_step": 8
  }
}
```

#### Predictive Scaling
```json
"scaling": {
  "policy": "predictive",
  "predictive": {
    "growth_factor": 1.2,
    "decline_factor": 0.75,
    "history": 3,
    "observation_interval": 10000,
    "scale_step": 4,
    "utilization_ceiling": 0.75
  }
}
```

### Not Needed for DWaaS Autoscaling

- `vpu` - Only for Elastic Pool

### Example Configuration

```json
{
  "name": "dwaas_autoscaling",
  "architecture": "DWAAS_AUTOSCALING",
  "dataset": 999,
  "nodes": 2,
  "instance": 0,
  "hit_rate": 0.7,
  "cold_start": 60.0,
  "runs": [
    {
      "name": "dwaas_queue_scaling",
      "scheduling": {
        "policy": "sjf",
        "max_io_concurrency": 32,
        "max_cpu_concurrency": 32
      },
      "scaling": {
        "policy": "queue",
        "queue": {
          "length_thresholds": [5, 10, 15, 20],
          "scale_steps": [4, 8, 12, 16],
          "scale_in_utilization": 0.4,
          "scale_in_step": 4
        }
      }
    }
  ]
}
```

---

## Elastic Pool

**Architecture Type:** `ELASTIC_POOL` (2)

**Description:** RPU-based elastic compute pool with VPU autoscaling.

### Essential Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `vpu` | int | Virtual Processing Units (RPUs) | `16` |
| `nodes` | int | Number of nodes for memory distribution | `1` |
| `hit_rate` | float | Cache hit rate (0.0-1.0) | `0.8` |
| `dataset` | int | Dataset identifier | `999` |
| `architecture` | string | Must be "ELASTIC_POOL" | `"ELASTIC_POOL"` |
| `cold_start` | float | Cold start delay in seconds | `120.0` |
| `scaling.policy` | string | Scaling policy | `"queue"` |

### Optional Parameters

| Parameter | Type | Description | Default | Example |
|-----------|------|-------------|---------|---------|
| `network_bandwidth` | int | Network bandwidth in Mbps | `10000` | `10000` |
| `io_bandwidth` | int | I/O bandwidth in Mbps | `1200` | `1200` |
| `memory_bandwidth` | int | Memory bandwidth in Mbps | `40000` | `40000` |
| `total_memory_capacity_mb` | int | Total memory in MB | `96000` | `96000` |
| `scheduling.policy` | string | Scheduling policy | `"fcfs"` | `"fcfs"` |
| `scheduling.max_io_concurrency` | int | Max concurrent I/O jobs | `64` | `32` |
| `scheduling.max_cpu_concurrency` | int | Max concurrent CPU jobs | `64` | `32` |

### Why `nodes` is Needed

Even though Elastic Pool is RPU-based, the `nodes` parameter is required for:
- Memory distribution tracking across physical nodes
- DRAM cache partitioning
- Bandwidth allocation per node

### Scaling Policy Parameters

Same as DWaaS Autoscaling (queue, reactive, predictive).

### Not Needed for Elastic Pool

- `instance` - VPU-based, not instance-based

### Example Configuration

```json
{
  "name": "elastic_pool",
  "architecture": "ELASTIC_POOL",
  "dataset": 999,
  "vpu": 16,
  "nodes": 1,
  "hit_rate": 0.8,
  "network_bandwidth": 10000,
  "io_bandwidth": 1200,
  "memory_bandwidth": 40000,
  "total_memory_capacity_mb": 96000,
  "cold_start": 120.0,
  "runs": [
    {
      "name": "ep_queue_scaling",
      "scheduling": {
        "policy": "fcfs",
        "max_io_concurrency": 32,
        "max_cpu_concurrency": 32
      },
      "scaling": {
        "policy": "queue",
        "queue": {
          "length_thresholds": [10, 20, 30],
          "scale_steps": [8, 16, 24],
          "scale_in_utilization": 0.3,
          "scale_in_step": 8
        }
      }
    }
  ]
}
```

---

## QaaS

**Architecture Type:** `QAAS` (3)

**Description:** Query-as-a-Service with pay-per-query pricing model. No persistent infrastructure.

### Essential Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `dataset` | int | Dataset identifier | `999` |
| `architecture` | string | Must be "QAAS" | `"QAAS"` |

### Optional Parameters

| Parameter | Type | Description | Default | Example |
|-----------|------|-------------|---------|---------|
| `scheduling.policy` | string | Scheduling policy | `"fcfs"` | `"sjf"` |
| `scheduling.max_io_concurrency` | int | Max concurrent I/O jobs | `64` | `64` |
| `scheduling.max_cpu_concurrency` | int | Max concurrent CPU jobs | `64` | `64` |

### Must Be Zero/Not Used

The following parameters MUST be set to `0` or `0.0` for QaaS (no persistent infrastructure):

| Parameter | Required Value | Reason |
|-----------|----------------|--------|
| `nodes` | `0` | No persistent nodes |
| `vpu` | `0` | Not VPU-based |
| `hit_rate` | `0.0` | No persistent cache |
| `network_bandwidth` | `10` (minimal) | Per-query allocation |
| `io_bandwidth` | `0` | Dynamic per-query |
| `memory_bandwidth` | `0` | Dynamic per-query |
| `total_memory_capacity_mb` | `0` | No persistent memory |
| `cold_start` | `0` | No cold start delay |

### Not Needed for QaaS

- `instance` - No instance-based infrastructure
- `scaling.*` - Auto-scales per query automatically
- `use_spot_instances` - Not applicable

### Cost Model

QaaS uses pay-per-query pricing:
- Cost per TB scanned: `$5.00` (configurable via `qaas_cost_per_tb`)
- Dynamic core allocation per query based on data scanned
- No infrastructure costs

### Example Configuration

```json
{
  "name": "qaas",
  "architecture": "QAAS",
  "dataset": 999,
  "nodes": 0,
  "vpu": 0,
  "hit_rate": 0.0,
  "network_bandwidth": 10,
  "io_bandwidth": 0,
  "memory_bandwidth": 0,
  "total_memory_capacity_mb": 0,
  "cold_start": 0,
  "runs": [
    {
      "name": "qaas_fcfs_ondemand",
      "scheduling": {
        "policy": "fcfs",
        "max_io_concurrency": 64,
        "max_cpu_concurrency": 64
      }
    },
    {
      "name": "qaas_sjf_ondemand",
      "scheduling": {
        "policy": "sjf",
        "max_io_concurrency": 64,
        "max_cpu_concurrency": 64
      }
    }
  ]
}
```

---

## Common Parameters

These parameters apply to all architectures.

### Simulation Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `scale_factor_min` | int | Minimum scale factor for data | `1` |
| `scale_factor_max` | int | Maximum scale factor for data | `100` |
| `shuffle_percentage_min` | float | Min shuffle percentage | `0.1` |
| `shuffle_percentage_max` | float | Max shuffle percentage | `0.35` |
| `interrupt_probability` | float | Probability of interruption | `0.01` |
| `interrupt_duration` | int | Duration of interrupt (seconds) | `6000` |
| `logging_interval` | int | Logging interval (seconds) | `60` |
| `default_max_duration` | int | Max simulation duration (seconds) | `108000` |
| `default_estimator` | string | Estimator type ("pm", "mw", "max", "sum") | `"pm"` |
| `pm_p` | float | Power mean parameter | `4.0` |
| `delta` | float | Delta parameter | `0.3` |
| `queue_agg` | string | Queue aggregation ("sum", "max") | `"sum"` |
| `materialization_fraction` | float | Materialization fraction | `0.25` |
| `parallelizable_portion` | float | Parallelizable portion | `0.9` |
| `s3_bandwidth` | int | S3 bandwidth in Mbps | `1000` |

### DWaaS-Specific Defaults

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `cost_per_second_redshift` | float | Cost per node-second | `0.00030166666` |
| `spot_discount` | float | Spot instance discount | `0.5` |
| `use_spot_instances` | bool | Use spot instances | `false` |
| `cold_start_delay` | float | Cold start delay (seconds) | `60.0` |

### Elastic Pool Defaults

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `cost_per_rpu_hour` | float | Cost per RPU-hour | `0.375` |
| `cache_warmup_gamma` | float | Cache warmup parameter | `0.05` |
| `cold_start_delay` | float | Cold start delay (seconds) | `120.0` |

### QaaS Defaults

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `cost_per_slot_hour` | float | Cost per slot-hour | `0.04` |
| `qaas_cost_per_tb` | float | Cost per TB scanned | `5.0` |
| `qaas_io_per_core_bw` | int | I/O bandwidth per core (Mbps) | `150` |
| `qaas_shuffle_bw_per_core` | int | Shuffle bandwidth per core (Mbps) | `50` |
| `qaas_base_cores` | int | Base cores per query | `4` |
| `qaas_base_time_limit` | int | Base time limit (seconds) | `2` |
| `core_alloc_window` | float | Core allocation window (seconds) | `10.0` |

### Instance Types

Seven instance types are available (index 0-6):

| Index | Type | CPU Cores | Memory (GB) | I/O BW (Mbps) | Network BW (Mbps) | Memory BW (Mbps) |
|-------|------|-----------|-------------|---------------|-------------------|------------------|
| 0 | ra3.xlplus | 4 | 32 | 650 | 1000 | 40000 |
| 1 | ra3.4xlarge | 12 | 96 | 2000 | 10000 | 40000 |
| 2 | ra3.16xlarge | 48 | 384 | 8000 | 10000 | 40000 |
| 3 | c5d.xlarge | 4 | 8 | 143 | 10000 | 40000 |
| 4 | c5d.2xlarge | 8 | 16 | 287 | 10000 | 40000 |
| 5 | c5d.4xlarge | 16 | 32 | 575 | 10000 | 40000 |
| 6 | ra3.xlplus (alt) | 4 | 32 | 650 | 1000 | 2500 |

---

## Parameter Summary by Architecture

### Quick Reference

| Parameter | DWaaS | DWaaS Auto | Elastic Pool | QaaS |
|-----------|-------|------------|--------------|------|
| `architecture` | Required | Required | Required | Required |
| `dataset` | Required | Required | Required | Required |
| `nodes` | Required | Required | Required* | 0 |
| `instance` | Required | Required | - | - |
| `vpu` | - | - | Required | 0 |
| `hit_rate` | Required | Required | Required | 0.0 |
| `cold_start` | - | Required | Required | 0 |
| `scaling.*` | - | Required | Required | - |
| `network_bandwidth` | Optional | Optional | Optional | 10 |
| `io_bandwidth` | Optional | Optional | Optional | 0 |
| `memory_bandwidth` | Optional | Optional | Optional | 0 |
| `total_memory_capacity_mb` | Optional | Optional | Optional | 0 |
| `scheduling.*` | Optional | Optional | Optional | Optional |

\* Required for memory distribution tracking

---

## Configuration Tips

1. **DWaaS Static**: Use when you want predictable costs with a fixed cluster size. No scaling overhead.

2. **DWaaS Autoscaling**: Use when workload varies. Choose scaling policy based on:
   - **Queue**: React to queue length (best for bursty workloads)
   - **Reactive**: React to CPU utilization (best for sustained load changes)
   - **Predictive**: Predict future needs (best for patterns with history)

3. **Elastic Pool**: Use for RPU-based pricing model with fast scaling. Higher cold start delay than DWaaS.

4. **QaaS**: Use for serverless, pay-per-query model. No infrastructure management. Cost based on data scanned.

5. **Scheduling Policies**:
   - **FCFS**: Fair, predictable latency
   - **SJF**: Optimize for throughput, minimize average latency
   - **LJF**: Deprioritize large jobs
   - **Multi-level**: Advanced prioritization with multiple queues

6. **Spot Instances**: Enable `use_spot_instances: true` in DWaaS Autoscaling for 50% cost reduction with interruption risk.

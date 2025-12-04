# CloudGlide OLAP Configuration Guide

## Overview

This guide explains how to configure simulations for each architecture type, what parameters are required, and what metrics are produced.

---

## Architecture-Specific Requirements

### 1. DWaaS (Traditional Static)

**Architecture:** `DWAAS`

**Description:** Traditional data warehouse with fixed cluster size. No autoscaling capabilities.

#### Required Parameters

```json
{
  "architecture": "DWAAS",
  "dataset": 999,
  "nodes": 4,                    // Number of compute nodes (fixed)
  "instance": 0,                 // Instance type ID (see INSTANCE_TYPES in config.py)
  "hit_rate": 0.7,              // Cache hit rate [0.0-1.0]
  "arrival_rate": 1.0,          // Job arrival rate multiplier
  "network_bandwidth": 10000,   // Mbps
  "io_bandwidth": 650,          // Mbps
  "memory_bandwidth": 40000,    // Mbps
  "total_memory_capacity_mb": 128000
}
```

#### Parameters NOT Needed
- `cold_start`: No autoscaling = no cold starts
- `vpu`: Not applicable to DWaaS
- `scaling` config: Ignored (no autoscaling)

#### Output Metrics
1. **avg_latency** - Average query execution time (seconds)
2. **avg_wq_latency** - Average query latency including queue wait time (seconds)
3. **median_latency** - Median query execution time (seconds)
4. **p95_latency** - 95th percentile query execution time (seconds)
5. **total_cost** - Total cluster cost (USD) = `nodes × cost_per_second_redshift × simulation_time`
6. **scale_observe** - Empty (no scaling events)
7. **mem_track** - Memory utilization samples (% of capacity, sampled every 60s)

---

### 2. DWaaS with Autoscaling

**Architecture:** `DWAAS_AUTOSCALING`

**Description:** DWaaS with ability to add/remove nodes based on workload.

#### Required Parameters

```json
{
  "architecture": "DWAAS_AUTOSCALING",
  "dataset": 999,
  "nodes": 2,                    // Base/minimum number of nodes
  "instance": 0,
  "hit_rate": 0.7,
  "arrival_rate": 1.0,
  "network_bandwidth": 10000,
  "io_bandwidth": 650,
  "memory_bandwidth": 40000,
  "total_memory_capacity_mb": 64000,
  "cold_start": 60.0            // Time to add new node (seconds)
}
```

#### Parameters NOT Needed
- `vpu`: Not applicable to DWaaS

#### Scaling Configuration Required

You must specify a `scaling.policy` and its parameters:

**Option 1: Queue-based Scaling**
```json
{
  "scaling": {
    "policy": "queue",
    "queue": {
      "length_thresholds": [5, 10, 15, 20],     // Queue lengths triggering scale-out
      "scale_steps": [4, 8, 12, 16],            // Nodes to add at each threshold
      "scale_in_utilization": 0.4,              // CPU util below which to scale in
      "scale_in_step": 4                        // Nodes to remove when scaling in
    }
  }
}
```

**Option 2: Reactive Scaling (CPU-based)**
```json
{
  "scaling": {
    "policy": "reactive",
    "reactive": {
      "cpu_utilization_thresholds": [0.6, 0.7, 0.8],  // CPU% triggering scale-out
      "scale_steps": [8, 16, 24],                      // Nodes to add at each threshold
      "scale_in_utilization": 0.2,                     // CPU% below which to scale in
      "scale_in_step": 8                               // Nodes to remove
    }
  }
}
```

**Option 3: Predictive Scaling**
```json
{
  "scaling": {
    "policy": "predictive",
    "predictive": {
      "growth_factor": 1.2,           // Multiplier when predicting growth
      "decline_factor": 0.75,         // Multiplier when predicting decline
      "history": 3,                   // Number of observation windows to consider
      "observation_interval": 10000,  // Window size in seconds
      "scale_step": 4                 // Nodes to add/remove per step
    }
  }
}
```

#### Output Metrics
1. **avg_latency** - Average query execution time (seconds)
2. **avg_wq_latency** - Average query latency including queue wait time (seconds)
3. **median_latency** - Median query execution time (seconds)
4. **p95_latency** - 95th percentile query execution time (seconds)
5. **total_cost** - Total cluster cost including autoscaling (USD)
6. **scale_observe** - List of node counts over time (sampled every 60s)
7. **mem_track** - Memory utilization samples (%)

---

### 3. Elastic Pool

**Architecture:** `ELASTIC_POOL`

**Description:** RPU-based elastic compute pool with VPU autoscaling.

#### Required Parameters

```json
{
  "architecture": "ELASTIC_POOL",
  "dataset": 999,
  "vpu": 16,                     // Virtual Processing Units (base)
  "nodes": 1,                    // Base nodes (typically 1)
  "hit_rate": 0.8,
  "arrival_rate": 1.0,
  "network_bandwidth": 10000,
  "io_bandwidth": 1200,
  "memory_bandwidth": 40000,
  "total_memory_capacity_mb": 96000,
  "cold_start": 120.0            // Time to scale VPUs (seconds)
}
```

#### Scaling Configuration Required

Elastic Pool uses VPU scaling (not node scaling):

**Queue-based VPU Scaling**
```json
{
  "scaling": {
    "policy": "queue",
    "queue": {
      "length_thresholds": [10, 20, 30],
      "scale_steps": [8, 16, 24],        // VPUs to add (not nodes)
      "scale_in_utilization": 0.3,
      "scale_in_step": 8
    }
  }
}
```

**Reactive VPU Scaling**
```json
{
  "scaling": {
    "policy": "reactive",
    "reactive": {
      "cpu_utilization_thresholds": [0.7, 0.8, 0.9],
      "scale_steps": [8, 16, 24],        // VPUs to add
      "scale_in_utilization": 0.25,
      "scale_in_step": 8
    }
  }
}
```

#### Output Metrics
1. **avg_latency** - Average query execution time (seconds)
2. **avg_wq_latency** - Average query latency including queue wait time (seconds)
3. **median_latency** - Median query execution time (seconds)
4. **p95_latency** - 95th percentile query execution time (seconds)
5. **total_cost** - RPU-based cost (USD) = `(vpu_charge / 3600) × cost_per_rpu_hour`
6. **scale_observe** - VPU counts over time (sampled every 60s)
7. **mem_track** - Memory utilization samples (%)

---

### 4. QaaS (Query-as-a-Service)

**Architecture:** `QAAS`

**Description:** Serverless query service with pay-per-query pricing. Auto-scales cores per query.

#### Required Parameters

```json
{
  "architecture": "QAAS",
  "dataset": 999,
  "nodes": 0,                    // Must be 0 (no fixed infrastructure)
  "vpu": 0,                      // Must be 0
  "hit_rate": 0.0,              // Must be 0 (no persistent cache)
  "arrival_rate": 1.0,
  "network_bandwidth": 10000,    // Used for inter-core communication
  "io_bandwidth": 0,            // Not applicable (computed dynamically)
  "memory_bandwidth": 0,        // Not applicable
  "total_memory_capacity_mb": 0 // Not applicable
}
```

#### Parameters NOT Needed
- `cold_start`: QaaS has no cold start (instant provisioning)
- `scaling` config: QaaS auto-scales per query (not configurable)
- `hit_rate`: QaaS has no persistent cache between queries

#### QaaS-Specific Configuration
The following parameters in `defaults.simulation` control QaaS behavior:
- `qaas_io_per_core_bw`: 150 Mbps - I/O bandwidth per core
- `qaas_shuffle_bw_per_core`: 50 Mbps - Shuffle bandwidth per core
- `qaas_base_cores`: 4 - Minimum cores per query
- `qaas_base_time_limit`: 2 seconds - Time limit for minimum core allocation
- `core_alloc_window`: 10.0 seconds - Window for core allocation decisions

#### Output Metrics
1. **avg_latency** - Average query execution time (seconds)
2. **avg_wq_latency** - Average query latency including queue wait time (seconds)
3. **median_latency** - Median query execution time (seconds)
4. **p95_latency** - 95th percentile query execution time (seconds)
5. **total_cost** - Pay-per-query cost (USD) = `data_scanned_tb × qaas_cost_per_tb`
6. **scale_observe** - Empty (no fixed infrastructure)
7. **mem_track** - Empty (no persistent memory)

**Note:** For `QAAS_CAPACITY` architecture, cost is based on reserved slots instead of data scanned.

---

## Scheduling Configuration

Scheduling policies control how jobs are assigned to resources.

### Scheduling Policies

**1. FCFS (First-Come-First-Served)**
```json
{
  "scheduling": {
    "policy": "fcfs",
    "max_io_concurrency": 32,    // Max concurrent I/O operations
    "max_cpu_concurrency": 32    // Max concurrent CPU operations
  }
}
```
- Jobs processed in arrival order
- Simple, no prioritization
- Default policy

**2. SJF (Shortest Job First)**
```json
{
  "scheduling": {
    "policy": "sjf",
    "max_io_concurrency": 32,
    "max_cpu_concurrency": 32
  }
}
```
- Prioritizes jobs with shortest estimated execution time
- Better average latency, but may starve long jobs
- Requires job time estimation

**3. LJF (Longest Job First)**
```json
{
  "scheduling": {
    "policy": "ljf",
    "max_io_concurrency": 32,
    "max_cpu_concurrency": 32
  }
}
```
- Prioritizes jobs with longest estimated execution time
- Can improve utilization for batch workloads
- May increase average latency

**4. Multi-Level Priority Queues**
```json
{
  "scheduling": {
    "policy": "multi",
    "max_io_concurrency": 64,
    "max_cpu_concurrency": 64,
    "multi_level_queues": [
      {"priority": 1, "max_jobs": 10},
      {"priority": 2, "max_jobs": 20},
      {"priority": 3, "max_jobs": 50}
    ]
  }
}
```
- Jobs assigned to priority queues
- Higher priority queues served first
- Useful for SLA-based workloads

### Concurrency Limits

- `max_io_concurrency`: Maximum number of jobs doing I/O simultaneously
- `max_cpu_concurrency`: Maximum number of jobs in CPU phase simultaneously
- Set to `null` for unlimited (inherits from architecture limits)
- Typical values: 16-64 depending on cluster size

---

## Scaling Configuration

Scaling policies control when and how to add/remove resources.

### When Scaling Applies

| Architecture | Scales What | Configuration Required |
|--------------|-------------|------------------------|
| DWAAS | Nothing | N/A (scaling ignored) |
| DWAAS_AUTOSCALING | Nodes | **Yes - must specify** |
| ELASTIC_POOL | VPUs | **Yes - must specify** |
| QAAS | Per-query cores | No (automatic) |
| QAAS_CAPACITY | Slot reservation | No (automatic) |
| SERVERLESS | Ephemeral resources | No (automatic) |

### Scaling Policies

**1. Queue-Based Scaling**

Triggers scaling based on queue length:

```json
{
  "scaling": {
    "policy": "queue",
    "queue": {
      "length_thresholds": [5, 10, 15, 20],   // Queue lengths triggering scale-out
      "scale_steps": [4, 8, 12, 16],          // Resources to add at each threshold
      "scale_in_utilization": 0.4,            // Utilization below which to scale in
      "scale_in_step": 4                      // Resources to remove when scaling in
    }
  }
}
```

**How it works:**
- When waiting queue length ≥ threshold[i], add scale_steps[i] resources
- When CPU utilization < scale_in_utilization, remove scale_in_step resources
- Example: Queue length 12 → add 8 resources (second threshold)

**2. Reactive Scaling**

Triggers scaling based on CPU utilization:

```json
{
  "scaling": {
    "policy": "reactive",
    "reactive": {
      "cpu_utilization_thresholds": [0.6, 0.7, 0.8],  // CPU% triggering scale-out
      "scale_steps": [8, 16, 24],                      // Resources to add
      "scale_in_utilization": 0.2,                     // CPU% below which to scale in
      "scale_in_step": 8                               // Resources to remove
    }
  }
}
```

**How it works:**
- When CPU utilization ≥ threshold[i], add scale_steps[i] resources
- When CPU utilization < scale_in_utilization, remove scale_in_step resources
- Example: CPU at 75% → add 16 resources (second threshold)

**3. Predictive Scaling**

Predicts future workload and scales proactively:

```json
{
  "scaling": {
    "policy": "predictive",
    "predictive": {
      "growth_factor": 1.2,           // Multiply capacity by this when growing
      "decline_factor": 0.75,         // Multiply capacity by this when declining
      "history": 3,                   // Number of windows to analyze
      "observation_interval": 10000,  // Window size in seconds
      "scale_step": 4                 // Resources to add/remove per step
    }
  }
}
```

**How it works:**
- Analyzes last `history` windows of `observation_interval` seconds each
- Predicts if workload will grow or decline
- If growing: add `scale_step` resources
- If declining: remove `scale_step` resources

---

## Complete Simulation Configuration

All configurable parameters in `defaults.simulation`:

### Core Parameters
```json
{
  "scale_factor_min": 1,              // Min data scale factor
  "scale_factor_max": 100,            // Max data scale factor
  "shuffle_percentage_min": 0.1,      // Min fraction of data shuffled
  "shuffle_percentage_max": 0.35,     // Max fraction of data shuffled
  "materialization_fraction": 0.25,   // Fraction of intermediate results materialized
  "parallelizable_portion": 0.9,      // Fraction of work that can be parallelized
  "default_max_duration": 108000,     // Max simulation time (seconds)
  "logging_interval": 60              // Metric collection interval (seconds)
}
```

### Cost Parameters
```json
{
  "cost_per_second_redshift": 0.00030166666,  // DWaaS cost per node-second (USD)
  "cost_per_rpu_hour": 0.375,                 // Elastic Pool RPU cost (USD/hour)
  "cost_per_slot_hour": 0.04,                 // QaaS reserved slot cost (USD/hour)
  "qaas_cost_per_tb": 5.0,                    // QaaS on-demand cost (USD/TB scanned)
  "spot_discount": 0.5,                       // Spot instance discount (50%)
  "use_spot_instances": false                 // Whether to use spot instances
}
```

### Cache & Cold Start
```json
{
  "cold_start_delay": 60.0,          // Time to provision new resources (seconds)
  "cache_warmup_gamma": 0.05,        // Cache warmup rate parameter
  "s3_bandwidth": 1000               // S3 I/O bandwidth (Mbps)
}
```

### Interruption (Spot Instances)
```json
{
  "interrupt_probability": 0.01,     // Probability of spot interruption per interval
  "interrupt_duration": 6000         // Recovery time after interruption (seconds)
}
```

### Job Estimation
```json
{
  "default_estimator": "pm",         // Estimator type: "max", "sum", "cpu_only", "pm", "mw"
  "pm_p": 4.0,                       // PM estimator parameter
  "delta": 0.3,                      // Estimation delta parameter
  "queue_agg": "sum"                 // Queue aggregation: "max" or "sum"
}
```

### QaaS-Specific
```json
{
  "qaas_io_per_core_bw": 150,        // I/O bandwidth per core (Mbps)
  "qaas_shuffle_bw_per_core": 50,    // Shuffle bandwidth per core (Mbps)
  "qaas_base_cores": 4,              // Minimum cores per query
  "qaas_base_time_limit": 2,         // Time limit for base core allocation (seconds)
  "core_alloc_window": 10.0          // Core allocation decision window (seconds)
}
```

---

## Parameter Dependency Matrix

| Parameter | DWAAS | DWAAS_AUTO | ELASTIC_POOL | QAAS |
|-----------|-------|------------|--------------|------|
| **nodes** | ✅ Required | ✅ Base nodes | ⚠️ Usually 1 | ❌ Must be 0 |
| **vpu** | ❌ N/A | ❌ N/A | ✅ Required | ❌ Must be 0 |
| **hit_rate** | ✅ Required | ✅ Required | ✅ Required | ❌ Must be 0 |
| **cold_start** | ❌ N/A | ✅ Required | ✅ Required | ❌ N/A |
| **total_memory_capacity_mb** | ✅ Required | ✅ Required | ✅ Required | ❌ Must be 0 |
| **io_bandwidth** | ✅ Required | ✅ Required | ✅ Required | ❌ Must be 0 |
| **memory_bandwidth** | ✅ Required | ✅ Required | ✅ Required | ❌ Must be 0 |
| **scaling config** | ❌ Ignored | ✅ Required | ✅ Required | ❌ Auto |

Legend:
- ✅ Required - Must be specified
- ⚠️ Conditional - Depends on scenario
- ❌ Not applicable - Should be 0 or omitted
- N/A - Has no effect

---

## Example: Running Multiple Scenarios

```json
{
  "defaults": { /* ... */ },
  "scenarios": [
    {
      "name": "compare_architectures",
      "dataset": 999,
      "arrival_rate": 1.0,
      "runs": [
        {
          "name": "static_dwaas",
          "architecture": "DWAAS",
          "nodes": 4,
          "instance": 0,
          "hit_rate": 0.7
        },
        {
          "name": "autoscaling_dwaas",
          "architecture": "DWAAS_AUTOSCALING",
          "nodes": 2,
          "instance": 0,
          "hit_rate": 0.7,
          "cold_start": 60.0,
          "scaling": {
            "policy": "queue"
          }
        },
        {
          "name": "elastic_pool",
          "architecture": "ELASTIC_POOL",
          "vpu": 16,
          "nodes": 1,
          "hit_rate": 0.8,
          "cold_start": 120.0,
          "scaling": {
            "policy": "reactive"
          }
        },
        {
          "name": "qaas",
          "architecture": "QAAS",
          "nodes": 0,
          "vpu": 0,
          "hit_rate": 0.0
        }
      ]
    }
  ]
}
```

---

## Validation

The system validates configurations automatically:

1. **At Config Load** - `SimulationConfig.__post_init__()` validates:
   - Probabilities in [0, 1]
   - Positive costs
   - Valid estimator choices
   - Valid queue aggregation

2. **At Runtime** - Additional checks for:
   - Required parameters per architecture
   - Scaling policy completeness
   - Scheduling policy validity

**Common Errors:**
- ❌ `QAAS with nodes > 0` → QaaS must have nodes=0
- ❌ `DWAAS with cold_start` → Static DWaaS doesn't use cold_start
- ❌ `DWAAS_AUTOSCALING without scaling config` → Autoscaling requires policy
- ❌ `hit_rate > 1.0` → Must be in [0, 1]

---

## Web Interface Recommendations

For building a web interface, ensure every parameter in `defaults.simulation` has:

1. **Input Field** - Text, number, or select
2. **Validation** - Client-side validation matching `__post_init__()`
3. **Tooltips** - Explain what each parameter does
4. **Presets** - Quick-select common configurations
5. **Architecture Wizard** - Guide users through architecture-specific requirements

**Suggested UI Flow:**
1. Select Architecture → Show only relevant parameters
2. Configure Resources → nodes/vpu based on architecture
3. Configure Scaling → Only if DWAAS_AUTOSCALING or ELASTIC_POOL
4. Configure Scheduling → Policy + concurrency limits
5. Review & Run → Show complete JSON before execution

---

## Quick Reference

### Minimal Working Examples

**DWaaS:**
```json
{"architecture": "DWAAS", "dataset": 999, "nodes": 4, "instance": 0, "hit_rate": 0.7}
```

**DWaaS Autoscaling:**
```json
{"architecture": "DWAAS_AUTOSCALING", "dataset": 999, "nodes": 2, "instance": 0,
 "hit_rate": 0.7, "cold_start": 60.0, "scaling": {"policy": "queue"}}
```

**Elastic Pool:**
```json
{"architecture": "ELASTIC_POOL", "dataset": 999, "vpu": 16, "nodes": 1,
 "hit_rate": 0.8, "cold_start": 120.0, "scaling": {"policy": "reactive"}}
```

**QaaS:**
```json
{"architecture": "QAAS", "dataset": 999, "nodes": 0, "vpu": 0, "hit_rate": 0.0}
```

---

**For complete examples, see:** `cloudglide/simulations/architecture_examples.json`

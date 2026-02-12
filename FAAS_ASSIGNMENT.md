# FaaS Architecture Assignment

## Overview

A new **FaaS (Function-as-a-Service)** architecture type has been added to CloudGlide as `ArchitectureType.FAAS` (value `6`). All infrastructure wiring is complete, your task is to implement **three functions** that define how FaaS simulates I/O, CPU, and cost.

## What You Need to Implement

### 1. `simulate_io_faas()` in `cloudglide/query_processing_model.py`

Simulates the I/O phase for FaaS. Each function invocation fetches data from object storage (S3) with no local cache.

Key points:
- Bandwidth per function: `config.s3_bandwidth / num_concurrent_functions`
- No DRAM/SSD tiers — all data comes from S3
- Enforce `config.faas_concurrency_limit` (max concurrent functions)
- Progress each job's `data_scanned_progress` per tick and schedule events

### 2. `simulate_cpu_faas()` in `cloudglide/query_processing_model.py`

Simulates the CPU phase for FaaS. Each function gets memory-proportional vCPUs.

Key points:
- vCPUs per function: `config.faas_memory_mb / 1769.0` (AWS Lambda proportionality)
- No shuffle phase, functions are independent
- Track GB-seconds for cost: `(config.faas_memory_mb / 1024) * duration_seconds`, accumulate in `faas_gb_seconds[0]`
- Enforce `config.faas_max_duration` (timeout in seconds)
- Return `0` (no slot reporting)

### 3. `faas_cost()` in `cloudglide/cost_model.py`

Calculates total FaaS cost using the hybrid pricing model.

Key points:
- Per-invocation fee: `num_invocations * cost_per_invocation`
- Duration fee: `total_gb_seconds * cost_per_gb_second`
- Return the sum of both

## Configuration Parameters

All FaaS parameters are defined in `cloudglide/config.py` under `SimulationConfig`:

| Parameter | Default | Description |
|---|---|---|
| `faas_cost_per_invocation` | 0.0000002 | Cost per function invocation ($0.20/1M) |
| `faas_cost_per_gb_second` | 0.0000166667 | Cost per GB-second of compute |
| `faas_memory_mb` | 1024 | Memory allocated per function (MB) |
| `faas_max_duration` | 900 | Max function duration (seconds) |
| `faas_concurrency_limit` | 1000 | Max concurrent executions |
| `s3_bandwidth` | 1000 | S3 bandwidth (MB/s) |

## How to Test

Run the FaaS scenario:

```bash
python main.py faas_basic cloudglide/simulations/use_cases.json --output_prefix output/faas_test
```

Before implementation, this will raise `NotImplementedError`. Once implemented, it should produce output CSVs with query latencies and a total cost reflecting the hybrid pricing model.

Run the test suite to check for regressions:

```bash
pytest tests/ -x -q
```

## Architecture Wiring (Already Done)

For reference, FaaS is wired through the following files — you should **not** need to modify these:

- **`cloudglide/config.py`** — `FAAS = 6` enum, config fields, validation
- **`main.py`** — FaaS parameter resolution, defaults override
- **`cloudglide/simulation_runner.py`** — FaaS `ExecutionParams` (serverless defaults)
- **`cloudglide/execution_model.py`** — Cost exclusion, `faas_gb_seconds` state tracking, final pricing dispatch, `faas_gb_seconds` kwarg passed to `simulate_cpu`
- **`cloudglide/scheduling_model.py`** — No changes needed (FaaS uses existing serverless scheduling paths)
- **`cloudglide/simulations/use_cases.json`** — `faas_basic` scenario

## Useful Reference

Look at the existing QaaS implementation for patterns to follow:
- `simulate_io` QaaS branch at line ~157 in `query_processing_model.py`
- `simulate_cpu_qaas()` at line ~530 in `query_processing_model.py`
- `qaas_total_cost()` in `cost_model.py`

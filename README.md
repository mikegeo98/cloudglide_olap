# CloudGlide: A Multi-Architecture Simulation Framework

**CloudGlide** is a Python-based simulation framework for analyzing and benchmarking different data-processing architectures and configurations. It supports:

- **Multiple Architectures** (e.g., classic data warehouse, autoscaling DW, QaaS, etc.)  
- **Scheduling Strategies** (e.g., FCFS, Shortest Job, Priority)  
- **Autoscaling Approaches** (e.g., reactive, queue-based, predictive)  
- **Cost Models** (spot vs. on-demand, capacity-based pricing, data-scanned pricing)  
- **Benchmarking** (CSV/JSON-based benchmarking for single queries or entire test scenarios)

The framework is designed for research experiments, allowing you to define test scenarios via JSON files, generate performance metrics (latency, cost, percentiles), and compare simulation results against ground-truth or expected outcomes.

---

## Table of Contents

1. [Project Structure](#project-structure)  
2. [Installation](#installation)  
3. [Configuration and Input Files](#configuration-and-input-files)  
4. [Running Simulations](#running-simulations)  
5. [Benchmarking Mode](#benchmarking-mode)
6. [Output Files](#output-files)  
7. [Extending CloudGlide](#extending-cloudglide)  

---

## 1. Project Structure
```
├── cloudglide
│   ├── config.py
│   ├── cost_model.py
│   ├── datasets
│   │   ├── ...
│   │	├── ...
│   ├── execution_model.py
│   ├── job.py
│   ├── output_simulation
│   │   ├── ...
│   │	├── ...
│   ├── query_processing_model.py
│   ├── README.md
│   ├── scaling_model.py
│   ├── scheduling_model.py
│   ├── simulation_runner.py
│   ├── simulations
│   │   ├── ...
│   │	├── ...
│   ├── use_cases.py
│   └── visual_model.py
└── main.py
```

---

## 2. Installation

1. **Clone** the repository:
   ```bash
   git clone https://github.com/mikegeo98/cloudglide_olap.git
   cd cloudglide_olap
    ```

2. Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```

## 3. Configuration and Input Files

**`cloudglide/config.py`** contains global constants and parameters:

- **INSTANCE_TYPES**: Preset instance configurations (CPU, memory, bandwidth).
- **DATASET_FILES**: Maps dataset indices to CSV file paths.
- **Cost constants** such as `COST_PER_RPU_HOUR` or `COST_PER_SLOT_HOUR`.
- **Default simulation parameters** (e.g., `DEFAULT_MAX_DURATION`).

**Dataset CSV files** (e.g., `tpch_all_runs.csv`) are stored in the `cloudglide/datasets` directory. Each CSV contains query-related fields such as:
- **job_id**
- **start time**
- **CPU time**
- **data scanned**
- and more...

**JSON simulation specs** (e.g., `tpch.json`) live in `cloudglide/simulations/` and follow a flattened, web-friendly schema:

- `defaults.simulation`: override any `SimulationConfig` field (costs, logging interval, estimator, QAAS bandwidths, spot behaviour, etc.).
- `defaults.scheduling`: specify the default scheduler policy plus concurrency caps or multi-level queue definitions.
- `defaults.scaling`: declare the default autoscaling policy (`queue`, `reactive`, or `predictive`) along with queue length thresholds, utilization limits, and predictive tuning knobs.
- `scenarios[]`: list of named scenarios. Each scenario can override config defaults, provide base parameters (architecture, dataset, nodes, etc.), and declare multiple `runs`. Every run can tweak scheduling/scaling policy, set per-run overrides (e.g., enable spot instances), and set any simulation parameter (arrival rate, hit rate, bandwidths, cold-start timers, …).

Example:

```json
{
  "defaults": {
    "simulation": {
      "interrupt_probability": 0.01,
      "spot_discount": 0.5,
      "cost_per_second_redshift": 0.00030166666,
      "logging_interval": 60,
      "default_estimator": "pm",
      "materialization_fraction": 0.25,
      "parallelizable_portion": 0.9,
      "cold_start_delay": 60,
      "cache_warmup_gamma": 0.05,
      "qaas_io_per_core_bw": 150,
      "qaas_shuffle_bw_per_core": 50,
      "qaas_base_cores": 4,
      "qaas_base_time_limit": 2,
      "core_alloc_window": 10,
      "s3_bandwidth": 1000
    },
    "scheduling": {
      "policy": "fcfs",
      "max_io_concurrency": 64,
      "max_cpu_concurrency": 64
    },
    "scaling": {
      "policy": "queue",
      "queue": {
        "length_thresholds": [5, 10, 15, 20],
        "scale_steps": [4, 8, 12, 16],
        "scale_in_utilization": 0.4
      },
      "reactive": {
        "cpu_utilization_thresholds": [0.6, 0.7, 0.8],
        "scale_steps": [8, 16, 24],
        "scale_in_utilization": 0.2
      },
      "predictive": {
        "growth_factor": 1.2,
        "decline_factor": 0.75,
        "history": 3,
        "observation_interval": 10000
      }
    }
  },
  "scenarios": [
    {
      "name": "tpch_all",
      "architecture": "QAAS",
      "dataset": 999,
      "nodes": 2,
      "arrival_rate": 0.1,
      "hit_rate": 0.7,
      "runs": [
        {
          "name": "tpch_fcfs_base",
          "scheduling": {
            "policy": "fcfs",
            "max_io_concurrency": 32,
            "max_cpu_concurrency": 16
          },
          "scaling": {
            "policy": "queue",
            "queue": {
              "length_thresholds": [4, 8, 12],
              "scale_steps": [2, 4, 8]
            }
          }
        },
        {
          "name": "tpch_multi_queue",
          "scheduling": {
            "policy": "multi_level",
            "multi_level_queues": [
              { "name": "latency", "criteria": "queueing_delay", "order": "desc", "max_concurrency": 4 },
              { "name": "short", "criteria": "data_scanned", "order": "asc", "max_concurrency": 8 }
            ]
          },
          "scaling": {
            "policy": "predictive",
            "predictive": {
              "growth_factor": 1.15,
              "decline_factor": 0.8,
              "history": 4,
              "observation_interval": 8000
            }
          },
          "config_overrides": {
            "use_spot_instances": true
          }
        }
      ]
    }
  ]
}
```

## 4. Running Simulations

The primary entry point is **`main.py`**. Below is the standard usage, focusing on the core parameters:

```css
python main.py <test_case_keyword> <json_file_path> [--benchmark] [--benchmark_file BENCHMARK_FILE] [--output_prefix PREFIX]
```

- **`test_case_keyword`**: The key that appears in your JSON scenario file.  
- **`json_file_path`**: Path to your scenario JSON (e.g., `cloudglide/simulations/tpch.json`).  

**Optional arguments**:

- `--benchmark`: Enable benchmarking mode (see Benchmarking section).
- `--benchmark_file`: Path to a JSON file containing expected metrics for each scenario. Defaults to `benchmark_data.json`.
- `--output_prefix`: Prefix for generated CSV output. Defaults to `cloudglide/output_simulation/simulation`.

### Example Command (Standard Run)

```bash
python main.py tpch_all cloudglide/simulations/tpch.json
```

## 5. Benchmarking Mode

When `--benchmark` is enabled, **`main.py`** compares the simulation results against expected metrics (e.g., expected execution time, median, cost) defined in `benchmark_data.json` (or a file you provide via `--benchmark_file`).

- After completion, a JSON report (e.g., `tpch_experiment_run_benchmark_report.json`) is generated.
- Benchmark results are also printed to the terminal with color codes to highlight pass/fail within a specified tolerance.

**Sample `benchmark_data.json`:**

```json
{
  "scenario1": {
    "architecture": 0,
    "scheduling": 1,
    "nodes": 1,
    "vpu": 0,
    "scaling": 1,
    "cold_starts": false,
    "hit_rate": 0.9,
    "instance": 0,
    "arrival_rate": 10.0,
    "network_bandwidth": 10000,
    "io_bandwidth": 650,
    "memory_bandwidth": 40000,
    "dataset": 999,
    "expected_execution_time": 120.0,
    "expected_median": 90.0,
    "expected_95th": 150.0,
    "expected_cost": 10.0
  }
  // ... additional scenarios
}
```

## 6. Output Files

After a run, CloudGlide writes simulation results in CSV format under the specified `--output_prefix` (or the default `cloudglide/output_simulation/simulation`).

- Each parameter combination in your JSON scenario creates a new CSV file named `<prefix>_1.csv`, `<prefix>_2.csv`, etc.
- Columns in the CSV include `query_duration`, `query_duration_with_queue`, `queueing_delay`, `cpu`, `io`, etc.
- Logs are written to **`simulation.log`** in the project root (or wherever configured in `main.py`).

## 7. Extending CloudGlide

- **Adding new architectures**: Update `configure_execution_params()` in `simulation_runner.py` and `execution_model.py` to handle a new `architecture` index.
- **Custom scheduling strategies**: Implement in `scheduling_model.py`.
- **Custom scaling policies**: Insert logic in `scaling_model.py` (see `Autoscaler` class).
- **Custom cost models**: Modify `cost_model.py` or reference your own cost function in `execution_model.py`.

## 8. Motivated Use Cases

**`use_cases.py`** is a helper script that demonstrates how to run predefined simulation scenarios and analyze their results. Each scenario corresponds to a typical research or operational question about scheduling, scaling, caching, or cost modeling in data-processing systems. After running the simulation, **`use_cases.py`** automatically processes the output CSV files and generates relevant plots or statistics. This allows you to quickly reproduce experiments and gather insights without manually invoking `main.py` multiple times.

Below is a quick overview of each supported use case:

- **`scheduling()`**: Compares different scheduling strategies (e.g., FCFS vs. priority-based) under various node configurations.  
- **`scaling_options()`**: Evaluates fixed vs. autoscaling node configurations, highlighting how different strategies handle changing workloads.  
- **`caching()`**: Assesses the effect of caching or partial reuse on overall query runtime and resource utilization.  
- **`scaling_algorithms()`**: Investigates multiple autoscaling policies (e.g., reactive or queue-based) for performance and cost trade-offs.  
- **`spot()`**: Explores the use of spot instances, analyzing potential savings versus performance variability.  
- **`workload_patterns()`**: Simulates different workload distributions (e.g., bursty vs. steady) across multiple architectures (DWaaS, EP, QaaS).  
- **`cold_starts()`**: Examines how cold starts in on-demand or serverless environments affect query latencies.  
- **`tpch()`**: Demonstrates a TPC-H run for various scale factors and cluster sizes, commonly used for benchmarking relational query processing. Results plotted and compared against experimental data.
- **`concurrency()`**: Demonstrates how isolated or concurrent execution affects query latency on fixed hardware.

By running `python use_cases.py <example_name>`, you can reproduce these scenarios, generate CSV outputs under `cloudglide/output_simulation/`, and view any generated plots or metrics.

---

### Questions or Support

For any clarifications, contact: [geom@in.tum.de](mailto:geom@in.tum.de)

**Happy Simulating!**

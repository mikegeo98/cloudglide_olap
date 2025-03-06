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

**JSON simulation specs** (e.g., `tpch.json`) reside in the `cloudglide/simulations` directory. These files define parameter sets for the simulation runner. Below is an example:

```json
{
  "test_case_keyword": {
    "architecture_values": [0],
    "scheduling_values": [1],
    "nodes_values": [1],
    "vpu_values": [0],
    "scaling_values": [1],
    "cold_starts_values": [false],
    "hit_rate_values": [0.9],
    "instance_values": [0],
    "arrival_rate_values": [10.0],
    "network_bandwidth_values": [10000],
    "io_bandwidth_values": [650],
    "memory_bandwidth_values": [40000],
    "dataset_values": [999]
  }
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
- Columns in the CSV include `query_duration`, `query_duration_with_queue`, `Queueing Delay`, `CPU`, `I/O`, etc.
- Logs are written to **`simulation.log`** in the project root (or wherever configured in `main.py`).

## 7. Extending CloudGlide

- **Adding new architectures**: Update `configure_execution_params()` in `simulation_runner.py` and `execution_model.py` to handle a new `architecture` index.
- **Custom scheduling strategies**: Implement in `scheduling_model.py`.
- **Custom scaling policies**: Insert logic in `scaling_model.py` (see `Autoscaler` class).
- **Custom cost models**: Modify `cost_model.py` or reference your own cost function in `execution_model.py`.

---

### Questions or Support

For any clarifications, contact: [geom@in.tum.de](mailto:geom@in.tum.de)

**Happy Simulating!**

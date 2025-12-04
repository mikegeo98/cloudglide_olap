# CloudGlide OLAP Simulator - Running Simulations Guide

## Quick Start

```bash
python main.py <scenario_name> <json_config_file>
```

Example:
```bash
python main.py dwaas_static cloudglide/simulations/architecture_examples.json
```

---

## Understanding the Configuration Structure

The JSON configuration has two main sections:

1. **defaults**: Global settings applied to all scenarios
2. **scenarios**: Individual simulation scenarios, each with multiple "runs"

```json
{
  "defaults": { ... },
  "scenarios": [
    {
      "name": "scenario1",
      "architecture": "DWAAS",
      ...
      "runs": [
        { "name": "run1", ... },
        { "name": "run2", ... }
      ]
    },
    {
      "name": "scenario2",
      ...
    }
  ]
}
```

---

## Running Multiple Scenarios

### Option 1: Run Scenarios Sequentially (Bash Loop)

To run multiple scenarios from the same JSON file:

```bash
# Run all DWaaS scenarios
python main.py dwaas_static cloudglide/simulations/architecture_examples.json
python main.py dwaas_autoscaling cloudglide/simulations/architecture_examples.json

# Run DWaaS and QaaS together
python main.py dwaas_static cloudglide/simulations/architecture_examples.json
python main.py qaas cloudglide/simulations/architecture_examples.json
```

### Option 2: Bash Script for Multiple Scenarios

Create a script `run_multiple.sh`:

```bash
#!/bin/bash

CONFIG_FILE="cloudglide/simulations/architecture_examples.json"

# List of scenarios to run
SCENARIOS=(
  "dwaas_static"
  "dwaas_autoscaling"
  "elastic_pool"
  "qaas"
)

for scenario in "${SCENARIOS[@]}"; do
  echo "Running scenario: $scenario"
  python main.py "$scenario" "$CONFIG_FILE"
  if [ $? -ne 0 ]; then
    echo "Error running $scenario"
    exit 1
  fi
done

echo "All scenarios completed!"
```

Run it:
```bash
chmod +x run_multiple.sh
./run_multiple.sh
```

---

## Multiple Runs within a Scenario

Each scenario can have multiple **runs** that vary parameters. A single `python main.py` command will execute ALL runs within that scenario.

### Example: Varying Hit Rate in DWaaS

Create a custom JSON file `dwaas_hit_rate_study.json`:

```json
{
  "defaults": {
    "simulation": {
      "scale_factor_min": 1,
      "scale_factor_max": 100,
      "logging_interval": 60,
      "default_max_duration": 108000
    },
    "dwaas": {
      "cost_per_second_redshift": 0.00030166666,
      "cold_start_delay": 60.0
    },
    "scheduling": {
      "policy": "sjf",
      "max_io_concurrency": 32,
      "max_cpu_concurrency": 32
    }
  },
  "scenarios": [
    {
      "name": "dwaas_hit_rate_experiment",
      "description": "Study impact of cache hit rate on DWaaS performance",
      "architecture": "DWAAS",
      "dataset": 999,
      "nodes": 4,
      "instance": 0,
      "runs": [
        {
          "name": "hit_rate_0",
          "hit_rate": 0.0
        },
        {
          "name": "hit_rate_25",
          "hit_rate": 0.25
        },
        {
          "name": "hit_rate_50",
          "hit_rate": 0.5
        },
        {
          "name": "hit_rate_75",
          "hit_rate": 0.75
        },
        {
          "name": "hit_rate_100",
          "hit_rate": 1.0
        }
      ]
    }
  ]
}
```

Run it:
```bash
python main.py dwaas_hit_rate_experiment dwaas_hit_rate_study.json
```

This will execute 5 simulations (one for each hit rate value) in a single command.

---

## Comparing Multiple Architectures

### Option 1: Per-Run Architecture (Recommended - Single Command)

You can specify different architectures within the **runs** of a single scenario!

```json
{
  "scenarios": [
    {
      "name": "multi_architecture_comparison",
      "description": "Compare all architectures in one scenario",
      "dataset": 999,
      "runs": [
        {
          "name": "dwaas_4nodes",
          "architecture": "DWAAS",
          "nodes": 4,
          "instance": 0,
          "hit_rate": 0.7
        },
        {
          "name": "dwaas_autoscaling_2nodes",
          "architecture": "DWAAS_AUTOSCALING",
          "nodes": 2,
          "instance": 0,
          "hit_rate": 0.7,
          "cold_start": 60.0,
          "scaling": {
            "policy": "queue",
            "queue": {
              "length_thresholds": [5, 10, 15, 20],
              "scale_steps": [4, 8, 12, 16]
            }
          }
        },
        {
          "name": "elastic_pool_16vpu",
          "architecture": "ELASTIC_POOL",
          "vpu": 16,
          "nodes": 1,
          "hit_rate": 0.8,
          "cold_start": 120.0
        },
        {
          "name": "qaas_ondemand",
          "architecture": "QAAS"
        }
      ]
    }
  ]
}
```

Run ALL architectures in **one command**:
```bash
python main.py multi_architecture_comparison architecture_comparison.json
```

This executes 4 simulations (one for each architecture) sequentially.

### Option 2: Separate Scenarios (Multiple Commands)

Create `architecture_comparison.json`:

```json
{
  "defaults": {
    "simulation": { ... },
    "dwaas": { ... },
    "ep": { ... },
    "qaas": { ... }
  },
  "scenarios": [
    {
      "name": "dwaas_baseline",
      "architecture": "DWAAS",
      "dataset": 999,
      "nodes": 4,
      "instance": 0,
      "hit_rate": 0.7,
      "runs": [
        { "name": "dwaas_run" }
      ]
    },
    {
      "name": "ep_baseline",
      "architecture": "ELASTIC_POOL",
      "dataset": 999,
      "vpu": 16,
      "nodes": 1,
      "hit_rate": 0.8,
      "cold_start": 120.0,
      "runs": [
        { "name": "ep_run" }
      ]
    },
    {
      "name": "qaas_baseline",
      "architecture": "QAAS",
      "dataset": 999,
      "runs": [
        { "name": "qaas_run" }
      ]
    }
  ]
}
```

Run each scenario:
```bash
python main.py dwaas_baseline architecture_comparison.json
python main.py ep_baseline architecture_comparison.json
python main.py qaas_baseline architecture_comparison.json
```

Or use a loop:
```bash
for arch in dwaas_baseline ep_baseline qaas_baseline; do
  python main.py "$arch" architecture_comparison.json
done
```

---

## Example: Varying Node Count

```json
{
  "scenarios": [
    {
      "name": "dwaas_node_scaling",
      "architecture": "DWAAS",
      "dataset": 999,
      "instance": 0,
      "hit_rate": 0.7,
      "runs": [
        {
          "name": "nodes_2",
          "nodes": 2
        },
        {
          "name": "nodes_4",
          "nodes": 4
        },
        {
          "name": "nodes_8",
          "nodes": 8
        },
        {
          "name": "nodes_16",
          "nodes": 16
        }
      ]
    }
  ]
}
```

Run:
```bash
python main.py dwaas_node_scaling my_config.json
```

---

## Example: Varying Scheduling Policies

```json
{
  "scenarios": [
    {
      "name": "scheduling_comparison",
      "architecture": "DWAAS",
      "dataset": 999,
      "nodes": 4,
      "instance": 0,
      "hit_rate": 0.7,
      "runs": [
        {
          "name": "fcfs",
          "scheduling": {
            "policy": "fcfs",
            "max_io_concurrency": 32,
            "max_cpu_concurrency": 32
          }
        },
        {
          "name": "sjf",
          "scheduling": {
            "policy": "sjf",
            "max_io_concurrency": 32,
            "max_cpu_concurrency": 32
          }
        },
        {
          "name": "ljf",
          "scheduling": {
            "policy": "ljf",
            "max_io_concurrency": 32,
            "max_cpu_concurrency": 32
          }
        }
      ]
    }
  ]
}
```

---

## Example: Varying Scaling Policies (Autoscaling)

```json
{
  "scenarios": [
    {
      "name": "scaling_comparison",
      "architecture": "DWAAS_AUTOSCALING",
      "dataset": 999,
      "nodes": 2,
      "instance": 0,
      "hit_rate": 0.7,
      "cold_start": 60.0,
      "runs": [
        {
          "name": "queue_scaling",
          "scaling": {
            "policy": "queue",
            "queue": {
              "length_thresholds": [5, 10, 15, 20],
              "scale_steps": [4, 8, 12, 16],
              "scale_in_utilization": 0.4,
              "scale_in_step": 4
            }
          }
        },
        {
          "name": "reactive_scaling",
          "scaling": {
            "policy": "reactive",
            "reactive": {
              "cpu_utilization_thresholds": [0.6, 0.7, 0.8],
              "scale_steps": [8, 16, 24],
              "scale_in_utilization": 0.2,
              "scale_in_step": 8
            }
          }
        },
        {
          "name": "predictive_scaling",
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
        }
      ]
    }
  ]
}
```

---

## Example: Varying Simulation Defaults (pm_p, delta, etc.)

You can vary **any** simulation parameter using `config_overrides`:

```json
{
  "scenarios": [
    {
      "name": "pm_p_study",
      "architecture": "DWAAS",
      "dataset": 999,
      "nodes": 4,
      "instance": 0,
      "hit_rate": 0.7,
      "runs": [
        {
          "name": "pm_p_1",
          "config_overrides": {
            "pm_p": 1.0
          }
        },
        {
          "name": "pm_p_2",
          "config_overrides": {
            "pm_p": 2.0
          }
        },
        {
          "name": "pm_p_4",
          "config_overrides": {
            "pm_p": 4.0
          }
        },
        {
          "name": "pm_p_8",
          "config_overrides": {
            "pm_p": 8.0
          }
        }
      ]
    }
  ]
}
```

This works for **any** parameter in the simulation defaults:
- `pm_p` - Power mean parameter
- `delta` - Delta parameter
- `materialization_fraction` - Materialization fraction
- `parallelizable_portion` - Parallelizable portion
- `s3_bandwidth` - S3 bandwidth
- `logging_interval` - Logging interval
- `interrupt_probability` - Interruption probability
- etc.

---

## Multi-Dimensional Parameter Studies

Combine multiple varying parameters within runs:

```json
{
  "scenarios": [
    {
      "name": "comprehensive_study",
      "architecture": "DWAAS",
      "dataset": 999,
      "instance": 0,
      "runs": [
        {
          "name": "2nodes_hit50",
          "nodes": 2,
          "hit_rate": 0.5
        },
        {
          "name": "2nodes_hit75",
          "nodes": 2,
          "hit_rate": 0.75
        },
        {
          "name": "4nodes_hit50",
          "nodes": 4,
          "hit_rate": 0.5
        },
        {
          "name": "4nodes_hit75",
          "nodes": 4,
          "hit_rate": 0.75
        },
        {
          "name": "8nodes_hit50",
          "nodes": 8,
          "hit_rate": 0.5
        },
        {
          "name": "8nodes_hit75",
          "nodes": 8,
          "hit_rate": 0.75
        }
      ]
    }
  ]
}
```

---

## Output Files

Each run generates outputs in `cloudglide/output_simulation/`:

- `simulation_1.csv` - First run results
- `simulation_2.csv` - Second run results
- `simulation_N.csv` - Nth run results

Outputs are numbered sequentially within each scenario execution.

---

## Tips

1. **Use descriptive run names**: Makes it easier to identify results later
2. **Start small**: Test with 1-2 runs before scaling to many
3. **Check defaults**: Parameters not specified in runs inherit from scenario, then defaults
4. **One scenario = one command**: All runs within a scenario execute together
5. **Multiple scenarios = multiple commands**: Run separately or script them
6. **Output prefix**: Use `--output_prefix` to organize results:
   ```bash
   python main.py dwaas_static config.json --output_prefix experiment1_
   ```

---

## Common Patterns

### Pattern 1: Same architecture, vary one parameter
```json
"runs": [
  { "name": "param_value1", "parameter": value1 },
  { "name": "param_value2", "parameter": value2 },
  { "name": "param_value3", "parameter": value3 }
]
```

### Pattern 2: Same config, different scheduling
```json
"runs": [
  { "name": "with_fcfs", "scheduling": { "policy": "fcfs" } },
  { "name": "with_sjf", "scheduling": { "policy": "sjf" } }
]
```

### Pattern 3: Compare architectures
Create separate scenarios for each architecture, run them individually with bash loop.

### Pattern 4: Sensitivity analysis
Create many runs varying parameter values systematically, execute in one command.

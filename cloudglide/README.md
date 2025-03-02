# CloudGlide: A Simulation Framework for Cloud-Based OLAP Systems

CloudGlide is a flexible simulation tool for exploring trade-offs between cost, performance, and scalability across different OLAP architectures in cloud environments. It allows users to configure scheduling, scaling, and caching policies, while also providing a cost analysis. CloudGlide models Data Warehouse-as-a-Service (DWaaS), Query-as-a-Service (QaaS), Elastic Pool (EP), and autoscaling architectures, offering users the ability to experiment with system configurations and workload patterns to simulate real-world cloud operations.

## Prerequisites

Before you start using CloudGlide, ensure the following are installed on your system:

- **Python 3.x** (Python 3.8 or higher is recommended)
- **Pandas** for data manipulation and analysis
- **Matplotlib** for data visualization
- **Subprocess** for handling external command execution

Install the required Python packages by running:

```bash
pip install -r requirements.txt
```

## Input Data

CloudGlide takes a JSON file (`input/use_cases.json`) as input to specify simulation configurations such as the number of nodes, scheduling policies, arrival rates, and more. Each use case is defined by a series of parameters relevant to the system architecture being modeled. See below for an example configuration:

```json
{
    "scheduling_4_nodes": {
        "architecture_values": [0],
        "scheduling_values": [0,1,2,4],
        "nodes_values": [3],
        "vpu_values": [1],
        "scaling_values": [1],
        "cold_starts_values": [180],
        "hit_rate_values": [0.6],
        "instance_values": [0],
        "arrival_rate_values": [0.1],
        "network_bandwidth_values": [10],
        "io_bandwidth_values": [650],
        "memory_bandwidth_values": [40000],
        "dataset_values": [614]
    }
}
```

## Usage Instructions

### Running CloudGlide Simulations

To run a simulation for a specific use case, use the `use_cases.py` script. Below are instructions for each example:

#### Scheduling:
Run scheduling tests for 4 and 2 nodes:
```bash
python use_cases.py scheduling
```


#### Queueing:
Simulate the effects of queueing on performance:

```bash
python use_cases.py queueing
```

#### Scaling:

Test scaling strategies across various node configurations:

```bash
python use_cases.py scaling_options
```

Compare different scaling algorithms:

```bash
python use_cases.py scaling_algorithms
```

#### Scaling:
Model the impact of caching policies on performance:

```bash
python use_cases.py caching
```

#### Spot Instances:
Simulate the use of spot instances in the cloud:

```bash
python use_cases.py spot
```

#### Workload Patterns:
Experiment with varying workload patterns:

```bash
python use_cases.py workload_patterns
```

#### Granular Autoscaling:
Simulate fine-grained autoscaling behavior:

```bash
python use_cases.py granular_autoscaling
```

Each simulation generates output files in the output_simulation/ directory, which are subsequently processed and visualized.

#### Output Visualization
CloudGlide provides functions for visualizing the results of each simulation. Visualizations will be generated based on the configuration defined in use_cases.json. The generated plots will be saved in the output_visual/ directory.

Example for visualizing scheduling results:

```bash
python use_cases.py scheduling
```
After running, the processed CSV files will be plotted and saved as PNG files in the output_visual/ directory.

#### Customizing Use Cases
You can easily modify the provided use_cases.json file to suit your specific needs by altering parameters like the number of nodes, arrival rates, or scheduling policies. This allows users to test the system under a wide variety of scenarios.

For example, to adjust the number of nodes for a particular test case, modify the nodes_values field in the JSON file:

#### Extending CloudGlide

Developers can extend CloudGlide to model additional architectures, workload patterns, or pricing models by modifying the existing functions in use_cases_visual.py or by adding new configurations to the use_cases.json file.

For example, to introduce a new architecture type, you can define its parameters within the JSON and implement its specific behaviors in client_test.py.
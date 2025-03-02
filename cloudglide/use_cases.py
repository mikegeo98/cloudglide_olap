import subprocess
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from client_test import main  # Importing the function directly
from use_cases_visual import process_and_plot_scheduling,process_and_plot_queueing, process_and_plot_scaling_options, process_and_plot_scaling_algorithms

# Path to the use_cases.json file
use_cases_file = "input/use_cases.json"

# Function to run the client_test script and count lines in output
def run_example(example_name):
    # Call client_test with the given output prefix
    output_prefix = example_name
    print(f"Running client_test for {output_prefix}...")
    
    # Assuming client_test is a script, using subprocess to call it
    try:
        result = main(example_name, use_cases_file, example_name)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running client_test for {output_prefix}: {e}")
        return None

# Example 1: Scheduling
def scheduling():
    result = run_example("scheduling_4_nodes")
    output_files = ['output_simulation/scheduling_4_nodes_1.csv','output_simulation/scheduling_4_nodes_2.csv','output_simulation/scheduling_4_nodes_3.csv','output_simulation/scheduling_4_nodes_4.csv']
    process_and_plot_scheduling(output_files)

    result = run_example("scheduling_2_nodes")
    output_files = ['output_simulation/scheduling_2_nodes_1.csv','output_simulation/scheduling_2_nodes_2.csv','output_simulation/scheduling_2_nodes_3.csv','output_simulation/scheduling_2_nodes_4.csv']
    process_and_plot_scheduling(output_files)

# Example 2: Queueing
def queueing():
    result = run_example("queueing_effect")
    output_files = ['output_simulation/queueing_effect_1.csv','output_simulation/queueing_effect_1.csv','output_simulation/queueing_effect_4.csv','output_simulation/queueing_effect_4.csv','output_simulation/queueing_effect_2.csv','output_simulation/queueing_effect_2.csv']
    process_and_plot_queueing(output_files)

# Example 3: Scaling Options
def scaling_options():
    result = run_example("scaling_options")
    
    output_files = ['output_simulation/scaling_options_1.csv','output_simulation/scaling_options_2.csv','output_simulation/scaling_options_3.csv','output_simulation/scaling_options_4.csv','output_simulation/scaling_options_5.csv','output_simulation/scaling_options_6.csv']
    process_and_plot_scaling_options(output_files, result)

# Example 4: Caching
def caching():
    output_file = run_example("caching", "caching")
    
    # Plotting logic specific to the caching example
    if output_file:
        data = pd.read_csv(output_file)
        plt.plot(data['Column1'], data['Column2'], label='Caching', color='blue')
        plt.title("Plot for Caching")
        plt.xlabel('X-axis Label')
        plt.ylabel('Y-axis Label')
        plt.legend()
        plt.savefig(f"{output_file[:-4]}_plot.png")
        print(f"Plot saved as {output_file[:-4]}_plot.png.")
        plt.close()

# Example 5: Scaling Algorithms
def scaling_algorithms():
    
    result = run_example("scaling_algorithms")
    output_files = ['output_simulation/scaling_algorithms_1.csv','output_simulation/scaling_algorithms_2.csv','output_simulation/scaling_algorithms_3.csv','output_simulation/scaling_algorithms_4.csv']
    process_and_plot_scaling_algorithms(result)

# Example 6: Spot
def spot():
    output_file = run_example("spot", "spot")
    
    # Plotting logic specific to the spot example
    if output_file:
        data = pd.read_csv(output_file)
        plt.plot(data['Column1'], data['Column2'], label='Spot')
        plt.title("Plot for Spot Instances")
        plt.xlabel('X-axis Label')
        plt.ylabel('Y-axis Label')
        plt.legend()
        plt.savefig(f"{output_file[:-4]}_plot.png")
        print(f"Plot saved as {output_file[:-4]}_plot.png.")
        plt.close()

# Example 7: Workload Patterns
def workload_patterns():
    output_file = run_example("workload_patterns", "workload_patterns")
    
    # Plotting logic specific to the workload patterns example
    if output_file:
        data = pd.read_csv(output_file)
        plt.plot(data['Column1'], data['Column2'], label='Workload Patterns')
        plt.title("Plot for Workload Patterns")
        plt.xlabel('X-axis Label')
        plt.ylabel('Y-axis Label')
        plt.legend()
        plt.savefig(f"{output_file[:-4]}_plot.png")
        print(f"Plot saved as {output_file[:-4]}_plot.png.")
        plt.close()

# Example 8: Granular Autoscaling
def granular_autoscaling():
    output_file = run_example("granular_autoscaling", "granular_autoscaling")
    
    # Plotting logic specific to the granular autoscaling example
    if output_file:
        data = pd.read_csv(output_file)
        plt.plot(data['Column1'], data['Column2'], label='Granular Autoscaling')
        plt.title("Plot for Granular Autoscaling")
        plt.xlabel('X-axis Label')
        plt.ylabel('Y-axis Label')
        plt.legend()
        plt.savefig(f"{output_file[:-4]}_plot.png")
        print(f"Plot saved as {output_file[:-4]}_plot.png.")
        plt.close()

# Main function to call specific examples by passing the example name as an argument
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python use_cases.py <example_name>")
        sys.exit(1)

    example_name = sys.argv[1]

    # Run the specified example
    if example_name == "scheduling":
        scheduling()
    elif example_name == "queueing":
        queueing()
    elif example_name == "scaling_options":
        scaling_options()
    elif example_name == "caching":
        caching()
    elif example_name == "scaling_algorithms":
        scaling_algorithms()
    elif example_name == "spot":
        spot()
    elif example_name == "workload_patterns":
        workload_patterns()
    elif example_name == "granular_autoscaling":
        granular_autoscaling()
    else:
        print(f"Example '{example_name}' does not exist.")

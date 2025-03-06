import subprocess
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from use_cases_visual import process_and_plot_scheduling,process_and_plot_queueing, process_and_plot_scaling_options, process_and_plot_scaling_algorithms, process_and_plot_caching


def run_example(test_case_keyword, json_file_path, output_prefix=None,
                benchmark=False, benchmark_file=None):
    """
    Runs main.py with the provided arguments via subprocess and returns
    a list of output CSV file names if needed.

    :param test_case_keyword: The test case key in your scenario JSON.
    :param json_file_path: Path to the JSON scenario file.
    :param output_prefix: Prefix for the generated CSV output files.
    :param benchmark: Boolean flag to enable or disable benchmarking mode.
    :param benchmark_file: Path to benchmark JSON if needed.
    :return: List of generated output file names or None.
    """

    # Build the command to call main.py
    cmd = [
        sys.executable,        # e.g., /usr/bin/python3
        os.path.join(os.getcwd(), 'main.py'),  # Full path to main.py
        test_case_keyword,
        json_file_path
    ]

    # Optionally add --output_prefix
    if output_prefix:
        cmd += ["--output_prefix", output_prefix]

    # Optionally add benchmarking flags
    if benchmark:
        cmd.append("--benchmark")
        if benchmark_file:
            cmd += ["--benchmark_file", benchmark_file]

    print(f"\n[INFO] Running: {' '.join(cmd)}")

    try:
        # Run the command and wait for it to finish
        completed_process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[INFO] Simulation completed successfully.\n")
        # You can parse standard output if needed:
        # print(completed_process.stdout)
        # Return the prefix so we know how to find CSV files
        return output_prefix
    except subprocess.CalledProcessError as e:
        print("[ERROR] Simulation failed.")
        print("Return code:", e.returncode)
        print("Output:", e.output)
        return None

##############################################################################
# Example 1: Scheduling
##############################################################################

def scheduling():
    """
    Reproduce the scheduling use case:
    1) scheduling_4_nodes
    2) scheduling_2_nodes
    Then process the output CSV files for plotting or analysis.
    """
    # Example 1A: scheduling_4_nodes
    prefix = "cloudglide/output_simulation/scheduling_4_nodes"
    run_example("scheduling_4_nodes",
                "cloudglide/simulations/use_cases.json",
                output_prefix=prefix)
    # Suppose you know which CSVs get produced:
    output_files = [
        f'{prefix}_1.csv',
        f'{prefix}_2.csv',
        f'{prefix}_3.csv',
        f'{prefix}_4.csv'
    ]
    process_and_plot_scheduling(output_files, 0)  # call your old plotting function

    # Example 1B: scheduling_2_nodes
    prefix = "cloudglide/output_simulation/scheduling_2_nodes"
    run_example("scheduling_2_nodes",
                "cloudglide/simulations/use_cases.json",
                output_prefix=prefix)
    output_files = [
        f'{prefix}_1.csv',
        f'{prefix}_2.csv',
        f'{prefix}_3.csv',
        f'{prefix}_4.csv'
    ]
    process_and_plot_scheduling(output_files, 1)

##############################################################################
# Example 2: Queueing
##############################################################################

def queueing():
    """
    Reproduce the queueing_effect use case and plot/visualize results.
    """
    prefix = "cloudglide/output_simulation/queueing_effect"
    run_example("queueing_effect",
                "cloudglide/simulations/use_cases.json",
                output_prefix=prefix)
    output_files = [
        f'{prefix}_1.csv',
        # Adjust as needed for however many combos get produced
    ]
    process_and_plot_queueing(output_files)

##############################################################################
# Example 3: Scaling Options
##############################################################################

def scaling_options():
    """
    Reproduce the scaling_options use case and analyze/plot the outcome.
    """
    prefix = "cloudglide/output_simulation/scaling_options"
    run_example("scaling_options",
                "cloudglide/simulations/use_cases.json",
                output_prefix=prefix)
    output_files = [
        f'{prefix}_1.csv',
        f'{prefix}_2.csv',
        f'{prefix}_3.csv',
        f'{prefix}_4.csv',
        f'{prefix}_5.csv',
        f'{prefix}_6.csv'
    ]
    process_and_plot_scaling_options(output_files)

##############################################################################
# Example 4: Caching
##############################################################################

def caching():
    """
    Run caching scenario and do custom plotting logic.
    """
    prefix = "cloudglide/output_simulation/caching"
    run_example("caching",
                "cloudglide/simulations/use_cases.json",
                output_prefix=prefix)

    # If there's a single result file, you might do:
    output_files = [
        f'{prefix}_1.csv',
        f'{prefix}_2.csv',
        f'{prefix}_3.csv',
        f'{prefix}_4.csv',
        f'{prefix}_5.csv'
    ]
    
    process_and_plot_caching(output_files)

##############################################################################
# Example 5: Scaling Algorithms
##############################################################################

def scaling_algorithms():
    """
    Run scaling_algorithms scenario and call specialized plotting function.
    """
    prefix = "cloudglide/output_simulation/scaling_algorithms"
    run_example("scaling_algorithms",
                "cloudglide/simulations/use_cases.json",
                output_prefix=prefix)
    output_files = [
        f'{prefix}_1.csv',
        f'{prefix}_2.csv',
        f'{prefix}_3.csv',
        f'{prefix}_4.csv'
    ]
    process_and_plot_scaling_algorithms(output_files)

##############################################################################
# Example 6: Spot
##############################################################################

def spot():
    """
    Run spot scenario and do custom plotting logic.
    """
    prefix = "cloudglide/output_simulation/spot"
    run_example("spot",
                "cloudglide/simulations/use_cases.json",
                output_prefix=prefix)

    output_file = f'{prefix}_1.csv'
    if os.path.exists(output_file):
        data = pd.read_csv(output_file)
        plt.plot(data['Column1'], data['Column2'], label='Spot')
        plt.title("Plot for Spot Instances")
        plt.xlabel('X-axis Label')
        plt.ylabel('Y-axis Label')
        plt.legend()
        plt.savefig(f"{prefix}_plot.png")
        plt.close()
        print(f"[INFO] Plot saved as {prefix}_plot.png.")

##############################################################################
# Example 7: Workload Patterns
##############################################################################

def workload_patterns():
    """
    Run workload_patterns scenario and do custom plotting logic.
    """
    prefix = "cloudglide/output_simulation/workload_patterns"
    run_example("workload_patterns",
                "cloudglide/simulations/use_cases.json",
                output_prefix=prefix)

    output_file = f'{prefix}_1.csv'
    if os.path.exists(output_file):
        data = pd.read_csv(output_file)
        plt.plot(data['Column1'], data['Column2'], label='Workload Patterns')
        plt.title("Plot for Workload Patterns")
        plt.xlabel('X-axis Label')
        plt.ylabel('Y-axis Label')
        plt.legend()
        plt.savefig(f"{prefix}_plot.png")
        plt.close()
        print(f"[INFO] Plot saved as {prefix}_plot.png.")

##############################################################################
# Example 8: Granular Autoscaling
##############################################################################

def granular_autoscaling():
    """
    Run granular_autoscaling scenario and do custom plotting logic.
    """
    prefix = "cloudglide/output_simulation/granular_autoscaling"
    run_example("granular_autoscaling",
                "cloudglide/simulations/use_cases.json",
                output_prefix=prefix)

    output_file = f'{prefix}_1.csv'
    if os.path.exists(output_file):
        data = pd.read_csv(output_file)
        plt.plot(data['Column1'], data['Column2'], label='Granular Autoscaling')
        plt.title("Plot for Granular Autoscaling")
        plt.xlabel('X-axis Label')
        plt.ylabel('Y-axis Label')
        plt.legend()
        plt.savefig(f"{prefix}_plot.png")
        plt.close()
        print(f"[INFO] Plot saved as {prefix}_plot.png.")


##############################################################################
# Main CLI: Choose which example to run
##############################################################################

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python use_cases.py <example_name>")
        print("Available examples: scheduling, queueing, scaling_options, caching,")
        print("scaling_algorithms, spot, workload_patterns, granular_autoscaling")
        sys.exit(1)

    example_name = sys.argv[1].lower()

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
        print(f"Example '{example_name}' not recognized.")
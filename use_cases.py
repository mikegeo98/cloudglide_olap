import subprocess
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from use_cases_visual import process_and_plot_scheduling,process_and_plot_queueing, process_and_plot_scaling_options, process_and_plot_scaling_algorithms, process_and_plot_caching, tpch_results, process_and_plot_workload_pattern_1, process_and_plot_workload_pattern_2, process_and_plot_workload_pattern_3, process_and_plot_workload_pattern_4, process_and_plot_workload_pattern_5


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
# Evaluation 1: TPC-H
##############################################################################

def tpch():
    """
    Reproduce the scheduling use case:
    1) scheduling_4_nodes
    2) scheduling_2_nodes
    Then process the output CSV files for plotting or analysis.
    """
    # Example 1A: scheduling_4_nodes
    prefix = "cloudglide/output_simulation/tpch"
    run_example("tpch_all",
                "cloudglide/simulations/tpch.json",
                output_prefix=prefix)
    # Suppose you know which CSVs get produced:
    output_files = [
        f'{prefix}_1.csv',
        f'{prefix}_2.csv',
        f'{prefix}_3.csv',
        f'{prefix}_4.csv'
    ]
    tpch_results()  # call your old plotting function

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
        f'{prefix}_4.csv'
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
    Dynamically handle different workload patterns producing different numbers
    of CSV files for each config.

    We'll define a dictionary that maps (pattern_prefix, config) -> number_of_files
    so you can easily control how many CSVs each scenario yields.
    """

    # 1) Define your patterns and the function that will plot them
    patterns = [
        ("p1", process_and_plot_workload_pattern_1),
        ("p2", process_and_plot_workload_pattern_2),
        ("p3", process_and_plot_workload_pattern_3),
        ("p4", process_and_plot_workload_pattern_4),
        ("p5", process_and_plot_workload_pattern_5),
    ]

    # 2) A dictionary controlling how many CSVs each (pattern, config) produces
    #    For example, p1_dwaas: 8 CSVs, p1_ep: 3 CSVs, etc.
    #    The defaults below are just examples – change them to match your real outputs!
    num_csv_map = {
        ("p1", "dwaas"): 8,
        ("p1", "ep"): 3,
        ("p1", "qaas"): 1,

        ("p2", "dwaas"): 8,
        ("p2", "ep"): 6,   # e.g. p2 has 6 CSV for EP
        ("p2", "qaas"): 1,

        ("p3", "dwaas"): 8,
        ("p3", "ep"): 6,   # e.g. p3 also has 6 CSV for EP
        ("p3", "qaas"): 1,

        ("p4", "dwaas"): 8,
        ("p4", "ep"): 3,
        ("p4", "qaas"): 1,

        ("p5", "dwaas"): 8,
        ("p5", "ep"): 4,   # e.g. p5 has 4 CSV for EP
        ("p5", "qaas"): 1,
    }

    # 3) The prefix used for all output paths
    base_prefix = "cloudglide/output_simulation/workload_pattern_"

    # 4) The different config suffixes we want to run
    configs = ["dwaas", "ep", "qaas"]

    for pattern_prefix, plot_function in patterns:
        # For each config, run the example
        for config in configs:
            # Compose the final prefix, e.g. "cloudglide/output_simulation/workload_pattern_p1_dwaas"
            out_prefix = f"{base_prefix}{pattern_prefix}_{config}"

            # Run the simulation, e.g. "p1_dwaas"
            scenario_key = f"{pattern_prefix}_{config}"

            # Actually run the scenario
            run_example(scenario_key, "cloudglide/simulations/use_cases.json", output_prefix=out_prefix)

        # Now build the combined list of output CSVs for each config
        output_files = []
        for config in configs:
            # How many CSV files does this pattern/config produce?
            n_csv = num_csv_map.get((pattern_prefix, config), 0)
            # If not found, it returns 0, or you can raise an error if it's mandatory

            # For each file index, create the path, e.g. "prefix_1.csv"
            out_prefix = f"{base_prefix}{pattern_prefix}_{config}"
            for i in range(1, n_csv + 1):
                csv_path = f"{out_prefix}_{i}.csv"
                output_files.append(csv_path)

        # At this point, output_files is a single combined list for all 3 configs
        # If you prefer separate lists, handle them differently
        # Finally, call your plotting function
        plot_function(output_files)

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
    elif example_name == "tpch":
        tpch()
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
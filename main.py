# main.py

import sys
import json
import logging
from typing import Tuple, Dict
from cloudglide.simulation_runner import run_simulation
from cloudglide.config import DEFAULT_OUTPUT_PREFIX


def load_test_cases(json_file_path: str) -> dict:
    """
    Load test cases from a JSON configuration file.
    """
    try:
        with open(json_file_path, 'r') as file:
            test_cases = json.load(file)
        return test_cases
    except FileNotFoundError:
        logging.error(f"Configuration file '{json_file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from '{json_file_path}'. Please check the file format.")
        sys.exit(1)


def load_benchmark_data(benchmark_file_path: str) -> Dict:
    """
    Load benchmark data from a JSON file.
    """
    try:
        with open(benchmark_file_path, 'r') as file:
            benchmark_data = json.load(file)
        return benchmark_data
    except FileNotFoundError:
        logging.error(f"Benchmark file '{benchmark_file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from '{benchmark_file_path}'. Please check the file format.")
        sys.exit(1)


def parse_arguments() -> Tuple[str, str, str, bool, str]:
    """
    Parse command-line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="CloudGlide Simulation Runner with Benchmarking.")
    parser.add_argument('test_case_keyword', type=str, help='Keyword of the test case to run.')
    parser.add_argument('json_file_path', type=str, help='Path to the JSON configuration file.')
    parser.add_argument('--benchmark', action='store_true', help='Enable benchmarking mode.')
    parser.add_argument('--benchmark_file', type=str, default='benchmark_data.json', help='Path to the benchmark data JSON file.')
    parser.add_argument('--output_prefix', type=str, default=DEFAULT_OUTPUT_PREFIX, help='Prefix for output files.')

    args = parser.parse_args()

    return args.test_case_keyword, args.json_file_path, args.output_prefix, args.benchmark, args.benchmark_file


def compare_results(simulation_time: float, expected_time: float, tolerance: float = 0.05) -> bool:
    """
    Compare simulation execution time with expected execution time within a tolerance.
    
    Returns True if within tolerance, else False.
    """
    lower_bound = expected_time * (1 - tolerance)
    upper_bound = expected_time * (1 + tolerance)
    return lower_bound <= simulation_time <= upper_bound


def generate_benchmark_report(comparisons: Dict[str, Dict], report_file: str = "benchmark_report.json"):
    """
    Generate a JSON report summarizing the benchmarking results.
    """
    try:
        with open(report_file, 'w') as file:
            json.dump(comparisons, file, indent=4)
        logging.info(f"Benchmark report generated at '{report_file}'.")
    except Exception as e:
        logging.error(f"Failed to write benchmark report: {e}")


def main():
    # Configure logging
    logging.basicConfig(
        filename='simulation.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

    # Parse command-line arguments
    test_case_keyword, json_file_path, output_prefix, benchmark_mode, benchmark_file_path= parse_arguments()

    # Load test cases
    test_cases = load_test_cases(json_file_path)

    # Validate test case keyword
    if test_case_keyword not in test_cases:
        logging.error(f"Test case keyword '{test_case_keyword}' not found in '{json_file_path}'.")
        sys.exit(1)

    test_case = test_cases[test_case_keyword]

    # Extract values from the test case
    architecture_values = test_case.get("architecture_values", [0])
    scheduling_values = test_case.get("scheduling_values", [1])
    nodes_values = test_case.get("nodes_values", [1])
    vpu_values = test_case.get("vpu_values", [0])
    scaling_values = test_case.get("scaling_values", [1])
    cold_starts_values = test_case.get("cold_starts_values", [False])
    hit_rate_values = test_case.get("hit_rate_values", [0.9])
    instance_values = test_case.get("instance_values", [0])
    arrival_rate_values = test_case.get("arrival_rate_values", [10.0])
    network_bandwidth_values = test_case.get("network_bandwidth_values", [10000])
    io_bandwidth_values = test_case.get("io_bandwidth_values", [650])
    memory_bandwidth_values = test_case.get("memory_bandwidth_values", [40000])
    dataset_values = test_case.get("dataset_values", [1])

    # Run simulations
    output_files, scaling_sequences, memory_sequences, results = run_simulation(
        architecture=architecture_values,
        scheduling=scheduling_values,
        nodes=nodes_values,
        vpu=vpu_values,
        scaling=scaling_values,
        cold_starts=cold_starts_values,
        hit_rate=hit_rate_values,
        instance=instance_values,
        arrival_rate=arrival_rate_values,
        network_bandwidth=network_bandwidth_values,
        io_bandwidth=io_bandwidth_values,
        memory_bandwidth=memory_bandwidth_values,
        dataset_index=dataset_values,
        output_prefix=output_prefix
    )

    # Check if benchmarking mode is enabled
    if benchmark_mode:
        # Load benchmark data
        benchmark_data = load_benchmark_data(benchmark_file_path)

        comparisons = {}

        for idx, (key, expected) in enumerate(benchmark_data.items()):
            # Match test case parameters
            if all(test_case.get(param) == expected.get(param) for param in [
                "architecture", "scheduling", "nodes", "vpu", "scaling", "cold_starts",
                "hit_rate", "instance", "arrival_rate", "network_bandwidth",
                "io_bandwidth", "memory_bandwidth", "dataset"
            ]):
                # Use the index to fetch the simulation result for simulation time
                simulation_time = results[idx][0] if idx < len(results) else None
                expected_time = expected.get("expected_execution_time")             
                # Extract additional expected metrics
                simulation_median = results[idx][3] if idx < len(results) else None
                expected_median = expected.get("expected_median")            
                simulation_95 = results[idx][4] if idx < len(results) else None
                expected_95th = expected.get("expected_95th")            
                simulation_cost = results[idx][2] if idx < len(results) else None
                expected_cost = expected.get("expected_cost")            
                # Compare each metric if available
                within_time = compare_results(simulation_time, expected_time) if simulation_time and expected_time else False
                within_median = compare_results(simulation_median, expected_median) if simulation_median and expected_median else False
                within_95th = compare_results(simulation_95, expected_95th) if simulation_95 and expected_95th else False
                within_cost = compare_results(simulation_cost, expected_cost) if simulation_cost and expected_cost else False             
                overall_within = within_time and within_median and within_95th and within_cost           
                comparisons[key] = {
                    "simulation_time": simulation_time,
                    "expected_time": expected_time,
                    "median_sim_time": simulation_median,
                    "expected_median": expected_median,
                    "perc95_sim_time": simulation_95,
                    "expected_95th": expected_95th,
                    "avg_cost": simulation_cost,
                    "expected_cost": expected_cost,
                    "within_tolerance": bool(overall_within)
                }
            else:
                comparisons[key] = {
                    "simulation_time": None,
                    "expected_time": expected.get("expected_execution_time"),
                    "median_sim_time": None,
                    "expected_median": expected.get("expected_median"),
                    "perc95_sim_time": None,
                    "expected_95th": expected.get("expected_95th"),
                    "avg_cost": None,
                    "expected_cost": expected.get("expected_cost"),
                    "within_tolerance": False,
                    "note": "Test case parameters do not match."
                }

        # Generate benchmark report
        generate_benchmark_report(comparisons, report_file=f"{output_prefix}_benchmark_report.json")

        # Print colored results to the terminal
        # ANSI escape codes: green: \033[92m, yellow: \033[93m, red: \033[91m, reset: \033[0m
        print("\nBenchmark Results:")
        for key, comp in comparisons.items():
            # For now, we'll print green if overall tolerance passes, red otherwise.
            color = "\033[92m" if comp["within_tolerance"] else "\033[91m"
            print(f"{color}{key}: Simulation time = {comp['simulation_time']} (Expected = {comp['expected_time']}), "
                f"Median = {comp['median_sim_time']} (Expected = {comp['expected_median']}), "
                f"95th Perc = {comp['perc95_sim_time']} (Expected = {comp['expected_95th']}), "
                f"Avg Cost = {comp['avg_cost']} (Expected = {comp['expected_cost']}), "
                f"Within tolerance = {comp['within_tolerance']}\033[0m")
            

        def load_simulation_results(simulation_csv_path: str) -> dict:
            """
            Load simulation results from CSV and return a dictionary keyed by (database_id, query_id).
            If 'database_id' is missing, it uses an empty string.
            """
            sim_results = {}
            try:
                with open(simulation_csv_path, 'r') as sim_file:
                    reader = csv.DictReader(sim_file)
                    for row in reader:
                        db_id = row.get("database_id", "")
                        key = (db_id, row["query_id"])
                        sim_results[key] = row
            except Exception as e:
                logging.error(f"Error reading simulation output file '{simulation_csv_path}': {e}")
                sys.exit(1)
            return sim_results

        def load_benchmark_csv(benchmark_csv_path: str) -> dict:
            """
            Load benchmark query data from CSV and return a dictionary keyed by (database_id, query_id).
            Expected columns: 'database_id', 'query_id', 'measured_time'
            """
            bench_results = {}
            try:
                with open(benchmark_csv_path, 'r') as bench_file:
                    reader = csv.DictReader(bench_file)
                    for row in reader:
                        key = (row["database_id"], row["query_id"])
                        bench_results[key] = row
            except Exception as e:
                logging.error(f"Error reading benchmark CSV file '{benchmark_csv_path}': {e}")
                sys.exit(1)
            return bench_results

        def compare_query_times(sim_time, benchmark_time, tolerance=0.05):
            try:
                sim_time = float(sim_time)
                benchmark_time = float(benchmark_time)
            except Exception:
                return False
            lower = benchmark_time * (1 - tolerance)
            upper = benchmark_time * (1 + tolerance)
            return lower <= sim_time <= upper

        def benchmark_queries(simulation_csv_path: str, benchmark_csv_path: str) -> dict:
            sim_results = load_simulation_results(simulation_csv_path)
            bench_results = load_benchmark_csv(benchmark_csv_path)
            comparisons = {}

            for key, bench_row in bench_results.items():
                bench_time = bench_row["measured_time"]
                sim_row = sim_results.get(key)
                if sim_row is not None:
                    sim_time = sim_row.get("query_duration")
                    match = compare_query_times(sim_time, bench_time)
                    comparisons[key] = {
                        "sim_time": sim_time,
                        "bench_time": bench_time,
                        "within_tolerance": bool(match)
                    }
                else:
                    comparisons[key] = {
                        "sim_time": None,
                        "bench_time": bench_time,
                        "within_tolerance": False,
                        "note": "No simulation result found for key."
                    }
            return comparisons

        # Define the file paths for the simulation output and the benchmark CSV
        simulation_csv = "simulation_output.csv"  # update with your actual simulation CSV path
        benchmark_csv = "benchmark_queries.csv"    # update with your actual benchmark CSV path

        single_query_comparisons = benchmark_queries(simulation_csv, benchmark_csv)

        # Print colored results to the terminal
        # ANSI escape codes: green: \033[92m, red: \033[91m, reset: \033[0m
        print("\nPer-Query Benchmark Results:")
        for key, comp in single_query_comparisons.items():
            color = "\033[92m" if comp["within_tolerance"] else "\033[91m"
            print(f"{color}{key}: Simulation time = {comp['sim_time']}, "
                f"Benchmark time = {comp['bench_time']}, "
                f"Within tolerance = {comp['within_tolerance']}\033[0m")

        # Optionally, you could also generate a report file for per-query benchmarks
        # generate_benchmark_report(single_query_comparisons, report_file=f"{output_prefix}_benchmark_single_query_report.json")



if __name__ == "__main__":
    main()

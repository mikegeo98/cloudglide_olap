# main.py

import sys
import json
import logging
import itertools
from typing import Tuple, Dict, List
from cloudglide.simulation_runner import run_simulation, SimulationRun
from cloudglide.config import SimulationConfig, ArchitectureType

def load_test_cases(json_file_path: str) -> Tuple[Dict, Dict[str, Dict]]:
    """
    Load test cases from a JSON configuration file. Supports the new schema
    (`defaults` + `scenarios`) as well as the legacy dictionary-of-cases format.
    """
    try:
        with open(json_file_path, 'r') as file:
            test_cases = json.load(file)

        if "scenarios" in test_cases:
            defaults = test_cases.get("defaults", {})
            scenarios_raw = test_cases.get("scenarios", [])
            scenario_map = {}
            if isinstance(scenarios_raw, dict):
                for name, payload in scenarios_raw.items():
                    payload.setdefault("name", name)
                    scenario_map[name] = payload
            else:
                for scenario in scenarios_raw:
                    if "name" not in scenario:
                        raise ValueError("Each scenario entry must include a 'name'.")
                    scenario_map[scenario["name"]] = scenario
            return defaults, scenario_map

        # Legacy format fallback: no defaults block, treat the full dict as the scenario map.
        return {}, test_cases
    except FileNotFoundError:
        logging.error(f"Configuration file '{json_file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from '{json_file_path}'. Please check the file format.")
        sys.exit(1)
    except ValueError as exc:
        logging.error(str(exc))
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


SCHEDULING_POLICIES = {
    "fcfs": 0,
    "sjf": 1,
    "ljf": 2,
    "multi": 3,
    "multi_level": 3,
}

SCALING_POLICIES = {
    "queue": 1,
    "queue_based": 1,
    "reactive": 2,
    "predictive": 3,
}


def _normalize_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def map_scheduling_policy(value, fallback: int) -> int:
    if value is None:
        return fallback
    if isinstance(value, int):
        return value
    key = _normalize_name(str(value))
    if key not in SCHEDULING_POLICIES:
        raise ValueError(f"Unknown scheduling policy '{value}'.")
    return SCHEDULING_POLICIES[key]


def map_scaling_policy(value, fallback: int) -> int:
    if value is None:
        return fallback
    if isinstance(value, int):
        return value
    key = _normalize_name(str(value))
    if key not in SCALING_POLICIES:
        raise ValueError(f"Unknown scaling policy '{value}'.")
    return SCALING_POLICIES[key]


def parse_architecture(value) -> ArchitectureType:
    if isinstance(value, ArchitectureType):
        return value
    if isinstance(value, int):
        return ArchitectureType(value)
    if isinstance(value, str):
        key = _normalize_name(value).upper()
        return ArchitectureType[key]
    raise ValueError(f"Unsupported architecture value '{value}'.")


def extract_option_payload(payload: Dict) -> Dict:
    return {k: v for k, v in (payload or {}).items() if k != "policy"}


def build_base_config(defaults: Dict) -> Tuple[SimulationConfig, int, int]:
    config = SimulationConfig()

    # Apply general simulation overrides
    simulation_overrides = defaults.get("simulation", {})
    config.apply_overrides(simulation_overrides)

    # Apply architecture-specific overrides
    dwaas_overrides = defaults.get("dwaas", {})
    if dwaas_overrides:
        config.apply_overrides(dwaas_overrides)

    ep_overrides = defaults.get("ep", {})
    if ep_overrides:
        config.apply_overrides(ep_overrides)

    qaas_overrides = defaults.get("qaas", {})
    if qaas_overrides:
        config.apply_overrides(qaas_overrides)

    # Apply scheduling defaults
    sched_defaults = defaults.get("scheduling", {}) or {}
    sched_options = extract_option_payload(sched_defaults)
    if sched_options:
        config.apply_overrides({"scheduling": sched_options})
    default_sched_policy = map_scheduling_policy(
        sched_defaults.get("policy"),
        SCHEDULING_POLICIES["fcfs"],
    )

    # Apply scaling defaults
    scaling_defaults = defaults.get("scaling", {}) or {}
    scaling_options = extract_option_payload(scaling_defaults)
    if scaling_options:
        config.apply_overrides({"scaling": scaling_options})
    default_scaling_policy = map_scaling_policy(
        scaling_defaults.get("policy"),
        SCALING_POLICIES["queue"],
    )

    return config, default_sched_policy, default_scaling_policy


def resolve_value(run_payload: Dict, scenario_payload: Dict, key: str, default=None, required: bool = False):
    if key in run_payload:
        return run_payload[key]
    if scenario_payload and key in scenario_payload:
        return scenario_payload[key]
    if required and default is None:
        raise ValueError(f"Scenario '{scenario_payload.get('name')}' missing required field '{key}'.")
    return default


def build_runs_for_scenario(
    scenario_name: str,
    scenario_payload: Dict,
    base_config: SimulationConfig,
    default_sched_policy: int,
    default_scaling_policy: int,
) -> List[SimulationRun]:
    scenario_config = base_config.copy()
    scenario_config.apply_overrides(scenario_payload.get("config_overrides", {}))

    scenario_sched_policy = default_sched_policy
    scenario_sched_payload = scenario_payload.get("scheduling")
    if scenario_sched_payload:
        scenario_sched_policy = map_scheduling_policy(
            scenario_sched_payload.get("policy"),
            default_sched_policy,
        )
        scen_sched_options = extract_option_payload(scenario_sched_payload)
        if scen_sched_options:
            scenario_config.apply_overrides({"scheduling": scen_sched_options})

    scenario_scaling_policy = default_scaling_policy
    scenario_scaling_payload = scenario_payload.get("scaling")
    if scenario_scaling_payload:
        scenario_scaling_policy = map_scaling_policy(
            scenario_scaling_payload.get("policy"),
            default_scaling_policy,
        )
        scen_scaling_options = extract_option_payload(scenario_scaling_payload)
        if scen_scaling_options:
            scenario_config.apply_overrides({"scaling": scen_scaling_options})

    run_definitions = scenario_payload.get("runs")
    legacy_value_keys = [key for key in scenario_payload.keys() if key.endswith("_values")]
    if not run_definitions and legacy_value_keys:
        param_names = [key.replace("_values", "") for key in legacy_value_keys]
        value_lists = [scenario_payload[key] for key in legacy_value_keys]
        run_definitions = [
            dict(zip(param_names, combo))
            for combo in itertools.product(*value_lists)
        ]
    if not run_definitions:
        run_definitions = [scenario_payload]

    runs: List[SimulationRun] = []
    for idx, run_payload in enumerate(run_definitions, start=1):
        run_config = scenario_config.copy()

        run_sched_payload = run_payload.get("scheduling")
        run_sched_policy = map_scheduling_policy(
            run_sched_payload.get("policy") if run_sched_payload else None,
            scenario_sched_policy,
        )
        run_sched_options = extract_option_payload(run_sched_payload or {})
        if run_sched_options:
            run_config.apply_overrides({"scheduling": run_sched_options})

        run_scaling_payload = run_payload.get("scaling")
        run_scaling_policy = map_scaling_policy(
            run_scaling_payload.get("policy") if run_scaling_payload else None,
            scenario_scaling_policy,
        )
        run_scaling_options = extract_option_payload(run_scaling_payload or {})
        if run_scaling_options:
            run_config.apply_overrides({"scaling": run_scaling_options})

        run_config.apply_overrides(run_payload.get("config_overrides", {}))
        if "use_spot_instances" in run_payload or "spot" in run_payload:
            run_config.use_spot_instances = run_payload.get(
                "use_spot_instances",
                run_payload.get("spot", run_config.use_spot_instances),
            )

        architecture_value = run_payload.get("architecture", scenario_payload.get("architecture"))
        if architecture_value is None:
            raise ValueError(f"Scenario '{scenario_name}' is missing 'architecture' in run #{idx}.")
        architecture = parse_architecture(architecture_value)

        dataset_value = run_payload.get(
            "dataset",
            run_payload.get(
                "dataset_index",
                scenario_payload.get("dataset", scenario_payload.get("dataset_index")),
            ),
        )
        if dataset_value is None:
            raise ValueError(f"Scenario '{scenario_name}' is missing 'dataset' for run #{idx}.")

        run_name = run_payload.get("name", f"{scenario_name}_{idx}")

        runs.append(
            SimulationRun(
                name=run_name,
                architecture=architecture,
                scheduling_policy=run_sched_policy,
                nodes=resolve_value(run_payload, scenario_payload, "nodes", 1),
                vpu=resolve_value(run_payload, scenario_payload, "vpu", 0),
                scaling_policy=run_scaling_policy,
                cold_start=resolve_value(run_payload, scenario_payload, "cold_start", 0),
                hit_rate=resolve_value(run_payload, scenario_payload, "hit_rate", 0.9),
                instance=resolve_value(run_payload, scenario_payload, "instance", 0),
                network_bandwidth=resolve_value(run_payload, scenario_payload, "network_bandwidth", 10000),
                io_bandwidth=resolve_value(run_payload, scenario_payload, "io_bandwidth", 650),
                memory_bandwidth=resolve_value(run_payload, scenario_payload, "memory_bandwidth", 40000),
                dataset_index=dataset_value,
                config=run_config,
            )
        )

    return runs

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
    parser.add_argument('--output_prefix', type=str, default='cloudglide/output_simulation/simulation', help='Prefix for output files.')

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

    # Load test cases using the new flattened schema
    defaults, scenarios = load_test_cases(json_file_path)

    # Validate test case keyword
    if test_case_keyword not in scenarios:
        logging.error(f"Test case keyword '{test_case_keyword}' not found in '{json_file_path}'.")
        sys.exit(1)

    scenario = scenarios[test_case_keyword]
    base_config, default_sched_policy, default_scaling_policy = build_base_config(defaults)
    try:
        runs = build_runs_for_scenario(
            test_case_keyword,
            scenario,
            base_config,
            default_sched_policy,
            default_scaling_policy,
        )
    except ValueError as exc:
        logging.error(str(exc))
        sys.exit(1)

    run_metadata = [
        {
            "architecture": run.architecture.value,
            "scheduling": run.scheduling_policy,
            "nodes": run.nodes,
            "vpu": run.vpu,
            "scaling": run.scaling_policy,
            "cold_starts": run.cold_start,
            "hit_rate": run.hit_rate,
            "instance": run.instance,
            "network_bandwidth": run.network_bandwidth,
            "io_bandwidth": run.io_bandwidth,
            "memory_bandwidth": run.memory_bandwidth,
            "dataset": run.dataset_index,
        }
        for run in runs
    ]

    # Run simulations
    output_files, scaling_sequences, memory_sequences, results = run_simulation(
        runs=runs,
        output_prefix=output_prefix
    )

    # Check if benchmarking mode is enabled
    if benchmark_mode:
        # Load benchmark data
        benchmark_data = load_benchmark_data(benchmark_file_path)

        comparisons = {}
        comparison_params = [
            "architecture", "scheduling", "nodes", "vpu", "scaling", "cold_starts",
            "hit_rate", "instance", "network_bandwidth",
            "io_bandwidth", "memory_bandwidth", "dataset"
        ]

        for key, expected in benchmark_data.items():
            match_idx = next(
                (
                    i for i, meta in enumerate(run_metadata)
                    if all(meta.get(param) == expected.get(param) for param in comparison_params)
                ),
                None,
            )
            if match_idx is not None:
                simulation_time = results[match_idx][0]
                expected_time = expected.get("expected_execution_time")             
                # Extract additional expected metrics
                simulation_median = results[match_idx][3]
                expected_median = expected.get("expected_median")            
                simulation_95 = results[match_idx][4]
                expected_95th = expected.get("expected_95th")            
                simulation_cost = results[match_idx][2]
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
                    "note": "Test case parameters do not match any simulated run."
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

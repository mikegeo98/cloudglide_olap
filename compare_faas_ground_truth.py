"""
Compare CloudGlide FaaS simulation results against Lambda ground truth.
Usage: python compare_faas_ground_truth.py
"""

import csv
import sys

# Ground truth: (query_number, memory_mb) -> runtime_ms
GROUND_TRUTH = {
    (1, 4096): 292000,
    (1, 8192): 222000,
    (3, 4096): 24000,
    (3, 8192): 22000,
    (5, 2048): 118000,
    (5, 4096): 33000,
    (5, 8192): 26000,
    (6, 2048): 10000,
    (6, 4096): 7000,
    (6, 8192): 4500,
    (12, 4096): 18000,
    (12, 8192): 16000,
}

# Simulation output files mapped to memory config
SIM_FILES = {
    2048: "cloudglide/output_simulation/simulation_1.csv",
    4096: "cloudglide/output_simulation/simulation_2.csv",
    8192: "cloudglide/output_simulation/simulation_3.csv",
}


def load_sim_results(filepath):
    """Load simulation results, return dict of query_id -> query_duration_ms."""
    results = {}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["query_id"])
            # query_duration is in seconds, convert to ms
            duration = float(row["query_duration"]) * 1000
            results[qid] = duration
    return results


def main():
    print(f"{'Query':<8} {'Mem MB':<10} {'Simulated':>12} {'Ground Truth':>14} {'Error':>10}")
    print("-" * 58)

    # Collect all rows first, then sort by (query_id, memory)
    rows = []

    for mem_mb, filepath in sorted(SIM_FILES.items()):
        try:
            sim = load_sim_results(filepath)
        except FileNotFoundError:
            print(f"  [missing: {filepath}]")
            continue

        for qid, sim_ms in sim.items():
            key = (qid, mem_mb)
            gt_ms = GROUND_TRUTH.get(key)
            rows.append((qid, mem_mb, sim_ms, gt_ms))

    # Sort by query_id first, then memory size
    rows.sort(key=lambda r: (r[0], r[1]))

    total_error = 0
    count = 0
    prev_qid = None

    for qid, mem_mb, sim_ms, gt_ms in rows:
        # Blank line between different queries for readability
        if prev_qid is not None and qid != prev_qid:
            print()
        prev_qid = qid

        if gt_ms is not None:
            error_pct = ((sim_ms - gt_ms) / gt_ms) * 100
            total_error += abs(error_pct)
            count += 1
            sign = "+" if error_pct >= 0 else ""
            print(f"Q{qid:<7} {mem_mb:<10} {sim_ms:>10.0f}ms {gt_ms:>12}ms {sign}{error_pct:>8.1f}%")
        else:
            print(f"Q{qid:<7} {mem_mb:<10} {sim_ms:>10.0f}ms {'N/A':>12} {'':>10}")

    if count > 0:
        print("-" * 58)
        print(f"Mean absolute error: {total_error / count:.1f}%  ({count} comparisons)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to verify estimator calculations by running single-query simulations
and comparing with the existing models_tpch.csv values.

For each query in models_tpch.csv:
1. Run a single-query simulation with DWaaS (hit_rate=1.0, ra3.xlplus)
2. Calculate estimates using all estimators (sum, max, pm, mw) with offsets (0, 0.3)
3. Compute QERROR comparing estimates to exec_wallclock_time
4. Compare with existing values in the CSV
"""

import pandas as pd
import numpy as np
import sys
import os


def calculate_qerror(predicted, actual):
    """Calculate QERROR: max(predicted/actual, actual/predicted)"""
    if actual == 0 or predicted == 0:
        return float('inf')
    ratio = predicted / actual
    return max(ratio, 1/ratio)


def estimate_execution_time(io_time, cpu_time, shuffle_time,
                            estimator='sum', offset=0.0):
    """
    Calculate execution time estimate using different estimators.

    Args:
        io_time: I/O time in seconds (O2)
        cpu_time: CPU time in seconds (N2)
        shuffle_time: Shuffle time in seconds (M2)
        estimator: 'sum', 'max', 'pm' (powermean), 'mw' (multiwave)
        offset: Offset to add to the estimate (default 0.0)

    Returns:
        Estimated execution time in seconds
    """
    # Note: In Excel formulas, O2=io, N2=cpu, M2=shuffle
    O2 = shuffle_time
    N2 = cpu_time
    M2 = io_time

    if estimator == 'sum':
        # Sum: O2 + N2 + M2
        estimate = max((O2 + N2 + M2),0.2)

    elif estimator == 'max':
        # Max: MAX(O2, N2, M2)
        estimate = max(O2, N2, M2)

    elif estimator == 'pm':
        # Power mean with p=1.5 (AK2)
        # POWER(POWER(O2,AK2)+POWER(N2,AK2)+POWER(M2,AK2), 1/AK2)
        p = 1.5
        estimate = (O2**p + N2**p + M2**p) ** (1/p)

    elif estimator == 'mw':
        # Multiwave (incorrectly labeled as holder_norm in CSV)
        # max(0.8*M2, 0.2*N2) + O2 + max(0.1*M2, 0.6*N2) + max(0.1*M2, 0.2*N2)
        term1 = max(0.8*M2, 0.2*N2)
        term2 = O2
        term3 = max(0.1*M2, 0.6*N2)
        term4 = max(0.1*M2, 0.2*N2)
        estimate = term1 + term2 + term3 + term4

    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    return estimate + offset


def main():
    """Main function to process the CSV and verify estimators."""

    # Read the input CSV
    input_csv = 'tpcds_2_resource_requirements_plus_estimates.csv'
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)

    print(f"Found {len(df)} queries to process")
    print()

    # Initialize results dictionary
    results = []
    mismatches = []

    # Process each query (rows 2-110, which is index 1-109 in DataFrame)
    for idx, row in df.iterrows():
        if 'tpcds' in input_csv:
            query_name = f"{row['query']}"
        elif 'tpch' in input_csv:
            query_name = f"{row['query']}_{row['size']}"

        # Extract pre-calculated times from CSV (already computed in the original file)
        io_time = row['io_time']
        cpu_time = row['cpu_time']
        shuffle_time = row['shuffle_time']
        actual_time = row['exec_wallclock_s']

        # Calculate estimates for all estimator types
        # Offset: AL2 = 0.36 for max/powermean/multiwave
        estimates = {
            'summed': estimate_execution_time(
                io_time, cpu_time, shuffle_time, 'sum', 0),
            'cpu_only': cpu_time + 0.2,  # Special case: N2 + 0.2
            'powermean': estimate_execution_time(
                io_time, cpu_time, shuffle_time, 'pm', 0),
            'max_offset': estimate_execution_time(
                io_time, cpu_time, shuffle_time, 'max', 0.36),
            'powermean_offset': estimate_execution_time(
                io_time, cpu_time, shuffle_time, 'pm', 0.36),
            'multiwave_offset': estimate_execution_time(
                io_time, cpu_time, shuffle_time, 'mw', 0.36),
        }

        # Calculate QERROR for each estimator
        qerrors = {}
        for est_name, est_value in estimates.items():
            qerrors[est_name] = calculate_qerror(est_value, actual_time)

        # Get existing QERROR values from CSV
        # Note: Columns now have _qerror suffix after renaming
        existing_qerrors = {
            'summed': row.get('sum_qerror', np.nan),
            'cpu_only': row.get('cpu_only_qerror', np.nan),
            'powermean': row.get('powermean_qerror', np.nan),
            'max_offset': row.get('max_offset_qerror', np.nan),
            'powermean_offset': row.get('powermean_offset_qerror', np.nan),
            'multiwave_offset': row.get('multiwave_offset_qerror', np.nan),
        }

        # Compare and track mismatches
        row_results = {
            'query': query_name,
            'actual_time': actual_time,
            'io_time': io_time,
            'cpu_time': cpu_time,
            'shuffle_time': shuffle_time,
        }

        has_mismatch = False
        for est_name in estimates.keys():
            calc_qerror = qerrors[est_name]
            existing_qerror = existing_qerrors[est_name]

            # Handle nan/inf values and convert to float if needed
            try:
                existing_qerror = float(existing_qerror)
                if pd.isna(existing_qerror) or np.isinf(existing_qerror):
                    diff = np.nan
                else:
                    diff = abs(calc_qerror - existing_qerror)
            except (ValueError, TypeError):
                diff = np.nan
                existing_qerror = np.nan

            row_results[f'{est_name}_calc'] = calc_qerror
            row_results[f'{est_name}_existing'] = existing_qerror
            row_results[f'{est_name}_diff'] = diff

            # Check for significant mismatch (> 0.01 difference)
            if not pd.isna(diff) and not np.isinf(diff) and abs(diff) > 0.01:
                has_mismatch = True

        results.append(row_results)

        if has_mismatch:
            mismatches.append(query_name)

    # Create output DataFrame
    results_df = pd.DataFrame(results)
    output_csv = 'estimator_verification.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"✓ Results saved to {output_csv}")
    print()

    # Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total queries processed: {len(results)}")
    print(f"Queries with mismatches (diff > 0.01): {len(mismatches)}")
    print()

    if mismatches:
        print("Queries with mismatches:")
        for query in mismatches[:10]:  # Show first 10
            print(f"  - {query}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
        print()

        # Show statistics for mismatches
        mismatch_rows = results_df[results_df['query'].isin(mismatches)]
        print("Mismatch statistics:")
        for est_name in ['summed', 'cpu_only', 'powermean', 'max_offset', 'powermean_offset', 'multiwave_offset']:
            diff_col = f'{est_name}_diff'
            if diff_col in mismatch_rows.columns:
                mean_diff = mismatch_rows[diff_col].mean()
                max_diff = mismatch_rows[diff_col].max()
                print(f"  {est_name:20s}: mean={mean_diff:8.4f}, max={max_diff:8.4f}")
    else:
        print("✓ All values match! No significant differences found.")

    print()
    print("=" * 80)

    return 0 if len(mismatches) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

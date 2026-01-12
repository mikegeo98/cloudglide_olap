#!/usr/bin/env python3
"""
Script to verify phase time calculations using direct formula calculations.

For each query in the CSV files:
1. Calculate phase times using bandwidth formulas
2. Use total_scan_mb, total_cpu_sec, shuffle_mb as resource requirements
3. Compare calculated phase times with pre-calculated values in the CSV
"""

import pandas as pd
import numpy as np
import sys
import os


def calculate_phase_times(scan_mb, cpu_sec, shuffle_mb,
                          cpu_cores=8, io_bw=2000, net_bw=1250):
    """
    Calculate phase times using bandwidth formulas.

    Args:
        scan_mb: Total data scanned in MB
        cpu_sec: Total CPU seconds
        shuffle_mb: Shuffle data in MB
        cpu_cores: Number of CPU cores (default: 8)
        io_bw: I/O bandwidth in MB/s (default: 2000)
        net_bw: Network bandwidth in MB/s (default: 1250)

    Returns:
        (io_time, cpu_time, shuffle_time) in seconds
    """
    # Calculate I/O time: total_scan_mb / io_bandwidth
    io_time = scan_mb / io_bw if scan_mb > 0 else 0

    # Calculate CPU time: total_cpu_sec / cpu_cores
    cpu_time = cpu_sec / cpu_cores if cpu_cores > 0 else 0

    # Calculate shuffle time: shuffle_mb / network_bandwidth
    shuffle_time = shuffle_mb / net_bw if shuffle_mb > 0 else 0

    return io_time, cpu_time, shuffle_time


def verify_csv_file(csv_path, output_path):
    """
    Verify phase times for all queries in a CSV file.

    Args:
        csv_path: Path to the CSV file to verify
        output_path: Path to save verification results

    Returns:
        Number of mismatches found
    """
    print(f"\n{'='*80}")
    print(f"Processing: {csv_path}")
    print(f"{'='*80}")

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} queries to process")

    # Results storage
    results = []
    mismatches = []

    # Process each query
    for idx, row in df.iterrows():
        # Generate query name (handle different column names)
        if 'size' in row:
            query_name = f"{row['query']}_{row['size']}"
        elif 'query_id' in row:
            query_name = row['query_id']
        else:
            query_name = f"query_{idx}"

        # Get input parameters
        total_scan_mb = row['total_scan_mb']
        total_cpu_sec = row['total_cpu_sec']
        shuffle_mb = row['shuffle_mb']

        # Get expected phase times from CSV
        expected_io = row['io_time']
        expected_cpu = row['cpu_time']
        expected_shuffle = row['shuffle_time']

        # Get configuration from CSV (or use defaults)
        # Handle both named and unnamed columns
        cpu_cores = row.get('cpu', row.get('Unnamed: 35', 8))
        io_bw = row.get('ssd', row.get('Unnamed: 33', 2000))
        net_bw = row.get('net', row.get('Unnamed: 34', 1250))

        # Calculate phase times using formulas
        calc_io, calc_cpu, calc_shuffle = calculate_phase_times(
            total_scan_mb, total_cpu_sec, shuffle_mb,
            cpu_cores=cpu_cores, io_bw=io_bw, net_bw=net_bw
        )

        # Calculate differences
        io_diff = abs(calc_io - expected_io)
        cpu_diff = abs(calc_cpu - expected_cpu)
        shuffle_diff = abs(calc_shuffle - expected_shuffle)

        # Store results
        result = {
            'query': query_name,
            'total_scan_mb': total_scan_mb,
            'total_cpu_sec': total_cpu_sec,
            'shuffle_mb': shuffle_mb,
            'expected_io': expected_io,
            'calculated_io': calc_io,
            'io_diff': io_diff,
            'expected_cpu': expected_cpu,
            'calculated_cpu': calc_cpu,
            'cpu_diff': cpu_diff,
            'expected_shuffle': expected_shuffle,
            'calculated_shuffle': calc_shuffle,
            'shuffle_diff': shuffle_diff,
        }
        results.append(result)

        # Check for significant mismatch (> 0.01 seconds)
        if io_diff > 0.01 or cpu_diff > 0.01 or shuffle_diff > 0.01:
            mismatches.append(query_name)
            if len(mismatches) <= 10:  # Show first 10
                print(f"  Mismatch: {query_name}")
                print(f"    IO: expected={expected_io:.6f}, "
                      f"calculated={calc_io:.6f}, diff={io_diff:.6f}")
                print(f"    CPU: expected={expected_cpu:.6f}, "
                      f"calculated={calc_cpu:.6f}, diff={cpu_diff:.6f}")
                print(f"    Shuffle: expected={expected_shuffle:.6f}, "
                      f"calculated={calc_shuffle:.6f}, "
                      f"diff={shuffle_diff:.6f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total queries: {len(results)}")
    print(f"Mismatches (diff > 0.01s): {len(mismatches)}")

    if len(results) > 0:
        print(f"\nPhase time differences:")
        print(f"  IO time    - mean: {results_df['io_diff'].mean():.6f}, max: {results_df['io_diff'].max():.6f}")
        print(f"  CPU time   - mean: {results_df['cpu_diff'].mean():.6f}, max: {results_df['cpu_diff'].max():.6f}")
        print(f"  Shuffle    - mean: {results_df['shuffle_diff'].mean():.6f}, max: {results_df['shuffle_diff'].max():.6f}")

    return len(mismatches)


def main():
    """Main function to verify all CSV files."""

    # List of CSV files to verify (in current directory)
    csv_files = [
        'tpcds_2_resource_requirements_plus_estimates.csv',
        'tpcds_4_resource_requirements_plus_estimates.csv',
        'tpcds_8_resource_requirements_plus_estimates.csv',
        'tpch_2_resource_requirements_plus_estimates.csv',
        'tpch_4_resource_requirements_plus_estimates.csv',
        'tpch_8_resource_requirements_plus_estimates.csv',
    ]

    print("="*80)
    print("Phase Time Verification")
    print("="*80)
    print("\nVerifying phase times using formula calculations")
    print("Configuration: cpu_cores=8, io_bw=2000 MB/s, net_bw=1250 MB/s")

    total_mismatches = 0

    for csv_file in csv_files:
        # Check if file exists
        if not os.path.exists(csv_file):
            print(f"\n⚠️  File not found: {csv_file}, skipping...")
            continue

        # Generate output filename
        base_name = os.path.basename(csv_file).replace('.csv', '')
        output_file = f'phase_verification_{base_name}.csv'

        # Verify the file
        mismatches = verify_csv_file(csv_file, output_file)
        total_mismatches += mismatches

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Total mismatches across all files: {total_mismatches}")

    if total_mismatches == 0:
        print("\n✅ All phase times match perfectly!")
        return 0
    else:
        print(f"\n⚠️  Found {total_mismatches} mismatches")
        return 1


if __name__ == "__main__":
    sys.exit(main())

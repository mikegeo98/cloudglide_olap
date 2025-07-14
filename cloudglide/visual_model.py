# visual_model.py

import csv
import io
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import logging
from cloudglide.job import Job
from dataclasses import dataclass, asdict

# Configure Logging: Only console output, INFO level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def print_info(current_second, nodes, io_jobs, waiting_jobs, cpu_jobs, jobs, buffer_jobs, memory_jobs, disk_jobs, s3_jobs):
    """
    Prints the current state of various job queues every 60 seconds.
    """
    if current_second % 60 == 0:
        logging.info(
            f"Second {current_second} - Remaining: {len(jobs)}, I/O: {len(io_jobs)}, "
            f"DRAM: {len(memory_jobs)}, SSD: {len(disk_jobs)}, S3: {len(s3_jobs)}, "
            f"Queued: {len(waiting_jobs)}, Buffered: {len(buffer_jobs)}, CPU: {len(cpu_jobs)}"
        )
        print(
            f"Second {current_second} - Remaining: {len(jobs)}, I/O: {len(io_jobs)}, "
            f"DRAM: {len(memory_jobs)}, SSD: {len(disk_jobs)}, S3: {len(s3_jobs)}, "
            f"Queued: {len(waiting_jobs)}, Buffered: {len(buffer_jobs)}, CPU: {len(cpu_jobs)}"
        )

def write_to_csv(path: str, data: List[Job], total_price: float):
    if not data:
        logging.warning("No data provided to write to CSV.")
        return

    # column header
    columns = [
        'job_id','query_id','database_id','start','start_timestamp','end_timestamp',
        'Queueing Delay','Buffer Delay','I/O','CPU','Shuffle',
        'query_duration_with_queue','query_duration','mon_cost'
    ]

    # use StringIO as an in-memory text buffer
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)

    for job in data:
        writer.writerow([
            job.job_id,
            job.query_id,
            job.database_id,
            job.start,
            job.start_timestamp,
            job.end_timestamp,
            job.queueing_delay,
            job.buffer_delay,
            job.io_time,
            job.processing_time,
            job.shuffle_time,
            job.query_exec_time_queueing,
            job.query_exec_time,
            total_price,
        ])

    # now write *once* to disk
    with open(path, 'w', newline='') as f:
        f.write(buf.getvalue())

    logging.info(f"Wrote {len(data)} rows to {path} (buffered).")

def analyze_results(filepath):
    """
    Analyzes the CSV results file to extract and print metrics.
    """
    try:
        df = pd.read_csv(filepath)

        if df.empty:
            logging.warning(f"The CSV file '{filepath}' is empty.")
            print(f"The CSV file '{filepath}' is empty.")
            return ""

        max_4th_column = (df['Queueing Delay'].max() - 360000) / 1000
        average_5th_column = df['query_duration'].mean() / 1000

        thresholds = [180000, 240000, 300000, 360000]
        count_less = [(df['Queueing Delay'] < threshold).sum() for threshold in thresholds]

        print("Filename:", filepath)
        print("Maximum of 4th column (Adjusted):", max_4th_column)
        print("Average of 5th column:", average_5th_column)
        for i, threshold in enumerate(thresholds):
            print(f"# of rows whose 4th column value is < {threshold}:", count_less[i])

        summary = f"{filepath}, {max_4th_column}, {average_5th_column}, {', '.join(map(str, count_less))}"
        logging.info(f"Analysis Summary: {summary}")
        return summary

    except FileNotFoundError:
        logging.error(f"Output file '{filepath}' not found.")
        print(f"Output file '{filepath}' not found.")
        return ""
    except KeyError as e:
        logging.error(f"Missing expected column in '{filepath}': {e}")
        print(f"Missing expected column in '{filepath}': {e}")
        return ""
    except Exception as e:
        logging.error(f"Error reading or analyzing '{filepath}': {e}")
        print(f"Error reading or analyzing '{filepath}': {e}")
        return ""

def process_and_plot_csv(file_paths, num_plots=4, save_path=None):
    """
    Processes multiple CSV files and plots the number of active queries in minute ranges.
    """
    all_range_labels = []
    all_range_counts = []

    global_range_start, global_range_end = float('inf'), float('-inf')
    step_size = 120000  # 2 minutes in milliseconds

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            actual_range_start = df['Start_Timestamp'].min()
            actual_range_end = df['End_Timestamp'].max()

            global_range_start = min(global_range_start, actual_range_start)
            global_range_end = max(global_range_end, actual_range_end)
        except Exception as e:
            logging.error(f"Error reading '{file_path}' for plotting: {e}")
            continue

    if global_range_start == float('inf') or global_range_end == float('-inf'):
        logging.error("No valid data found in the provided CSV files.")
        return

    global_range_start = int(global_range_start // step_size) * step_size

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            range_counts = [0] * ((global_range_end - global_range_start) // step_size + 1)

            for _, row in df.iterrows():
                actual_start_time, end_time = float(row['Start_Timestamp']), float(row['End_Timestamp'])
                for i, start in enumerate(range(global_range_start, global_range_end + 1, step_size)):
                    range_lower, range_upper = start, start + step_size - 1
                    if (range_lower <= actual_start_time <= range_upper) \
                       or (range_lower <= end_time <= range_upper) \
                       or (actual_start_time < range_lower and end_time > range_upper):
                        range_counts[i] += 1

            range_labels = [
                f"{start // 60000}-{(start + step_size - 1) // 60000}"
                for start in range(global_range_start, global_range_end + 1, step_size)
            ]
            all_range_labels.append(range_labels)
            all_range_counts.append(range_counts)
            logging.info(f"Processed '{file_path}' for plotting.")
        except Exception as e:
            logging.error(f"Error processing '{file_path}' for plotting: {e}")
            continue

    if not all_range_labels or not all_range_counts:
        logging.error("No data available to plot after processing CSV files.")
        return

    num_plots = len(file_paths)
    num_columns = 2
    num_rows = (num_plots + num_columns - 1) // num_columns

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axs = axs.flatten() if num_rows > 1 else [axs]

    for i, (ax, range_labels, range_counts, file_path) in enumerate(zip(axs, all_range_labels, all_range_counts, file_paths)):
        ax.bar(range_labels, range_counts, color='skyblue')
        ax.set_xlabel('Minute Range')
        ax.set_ylabel('Number of Active Queries')
        ax.set_title(f'Active Queries per Minute Range\n({file_path})')
        ax.set_xticks(range(len(range_labels)))
        ax.set_xticklabels(range_labels, rotation=45, ha='right')
        ax.set_ylim(0, max(range_counts) * 1.1 if range_counts else 10)
        ax.grid(True, linestyle='--', alpha=0.5)

    for j in range(num_plots, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path)
            logging.info(f"Saved plot to '{save_path}'.")
        except Exception as e:
            logging.error(f"Failed to save plot to '{save_path}': {e}")
    else:
        plt.show()

def process_and_plot_scaling(servers_per_minute_list, num_plots=4, save_path=None):
    """
    Plots the number of servers over time for multiple scaling scenarios.
    """
    num_plots = len(servers_per_minute_list)
    num_columns = 2
    num_rows = (num_plots + 1) // 2

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axs = axs.flatten() if num_rows > 1 else [axs]

    for i, (ax, servers_per_minute) in enumerate(zip(axs[:num_plots], servers_per_minute_list)):
        time_intervals = list(range(len(servers_per_minute)))
        ax.plot(time_intervals, servers_per_minute, label=f'Scenario {i+1}', linewidth=2, marker='o', markersize=5)
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Number of Servers', fontsize=12)
        ax.set_title(f'Scaling Scenario {i+1}', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    for j in range(num_plots, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path)
            logging.info(f"Saved scaling plot to '{save_path}'.")
        except Exception as e:
            logging.error(f"Failed to save scaling plot to '{save_path}': {e}")
    else:
        plt.show()

def plot_pareto(data):
    """
    Plots a Pareto front based on latency and cost data.
    """
    try:
        if not data:
            logging.warning("No data provided for Pareto plot.")
            return

        latencies = [entry[0] for entry in data]
        costs = [entry[2] for entry in data]

        labels = ['DWaaS', 'DWaaS Autoscaling', 'Elastic Pool', 'QaaS']
        if len(latencies) != len(labels):
            logging.warning("Number of data points does not match number of labels.")
            min_length = min(len(latencies), len(labels))
            latencies = latencies[:min_length]
            costs = costs[:min_length]
            labels = labels[:min_length]

        points = np.array(list(zip(latencies, costs)))

        sorted_indices = points[:, 0].argsort()
        points = points[sorted_indices]
        labels = [labels[i] for i in sorted_indices]

        pareto_front = [points[0]]
        pareto_labels = [labels[0]]
        for point, label in zip(points[1:], labels[1:]):
            if point[1] < pareto_front[-1][1]:
                pareto_front.append(point)
                pareto_labels.append(label)
        pareto_front = np.array(pareto_front)

        plt.figure(figsize=(10, 7))

        plt.scatter(latencies, costs, color='green', label='All Points')
        plt.plot(pareto_front[:, 0], pareto_front[:, 1],
                 color='red', marker='o', linestyle='-', linewidth=2, label='Pareto Front')

        for latency, cost, label in zip(latencies, costs, labels):
            plt.text(latency, cost, label, fontsize=12, ha='right', va='bottom')

        plt.xlabel('Latency', fontsize=14)
        plt.ylabel('Cost', fontsize=14)
        plt.title('Pareto Front: Latency vs Cost', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(left=0, right=max(latencies) * 1.1 if latencies else 10)
        plt.ylim(bottom=0, top=max(costs) * 1.1 if costs else 10)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

        logging.info("Pareto plot generated successfully.")

    except Exception as e:
        logging.error(f"Error in plot_pareto: {e}")

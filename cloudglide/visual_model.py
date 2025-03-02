# visual_model.py

import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import logging
from cloudglide.job import Job
from dataclasses import dataclass, asdict

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visual_model_debug.log"),
        logging.StreamHandler()
    ]
)


def print_info(current_second, nodes, io_jobs, waiting_jobs, cpu_jobs, jobs, buffer_jobs, memory_jobs, disk_jobs, s3_jobs):
    """
    Prints the current state of various job queues every 60 seconds.

    Args:
        current_second (float): Current simulation second.
        nodes (int): Number of nodes.
        io_jobs (deque): Queue of I/O jobs.
        waiting_jobs (deque): Queue of waiting jobs.
        cpu_jobs (deque): Queue of CPU jobs.
        jobs (List[Job]): List of all jobs.
        buffer_jobs (deque): Queue of buffer jobs.
        memory_jobs (List[Job]): List of memory jobs.
        disk_jobs (List[Job]): List of disk jobs.
        s3_jobs (List[Job]): List of S3 jobs.
    """
    if current_second % 60 == 0:
        logging.info(f"Second {current_second} - Remaining: {len(jobs)}, I/O: {len(io_jobs)}, DRAM: {len(memory_jobs)}, "
                     f"SSD: {len(disk_jobs)}, S3: {len(s3_jobs)}, Queued: {len(waiting_jobs)}, "
                     f"Buffered: {len(buffer_jobs)}, CPU: {len(cpu_jobs)}")
        print(f"Second {current_second} - Remaining: {len(jobs)}, I/O: {len(io_jobs)}, DRAM: {len(memory_jobs)}, "
              f"SSD: {len(disk_jobs)}, S3: {len(s3_jobs)}, Queued: {len(waiting_jobs)}, "
              f"Buffered: {len(buffer_jobs)}, CPU: {len(cpu_jobs)}")


def write_to_csv(file_path: str, data: list[Job]):
    """
    Writes a list of Job objects to a CSV file with specific field mappings and selective columns.

    Args:
        file_path (str): Path to the output CSV file.
        data (List[Job]): List of Job objects to write.
    """
    if not data:
        logging.warning("No data provided to write to CSV.")
        return  # Return early if there's no data

    # Define a mapping of Job attributes to desired CSV column names
    field_mapping = {
        'query_exec_time_queueing': 'query_duration_with_queue',
        'query_exec_time': 'query_duration',
        'queueing_delay': 'Queueing Delay',
        'buffer_delay': 'Buffer Delay',
        'io_time': 'I/O',
        'processing_time': 'CPU',
        'shuffle_time': 'Shuffle',
    }

    # Define the list of Job attributes to extract
    desired_attributes = [
        'job_id',
        'query_id',
        'start',
        'start_timestamp',
        'end_timestamp',
        'queueing_delay',
        'buffer_delay',
        'io_time',
        'processing_time',
        'shuffle_time',
        'query_exec_time_queueing',
        'query_exec_time'
    ]

    # Define the order of columns in the CSV
    csv_columns = [
        'job_id',
        'query_id',
        'start',
        'start_timestamp',
        'end_timestamp',
        'Queueing Delay',
        'Buffer Delay',
        'I/O',
        'CPU',
        'Shuffle',
        'query_duration_with_queue',
        'query_duration'
    ]

    # Convert Job objects to dictionaries with mapped field names
    converted_data = []
    for job in data:
        job_dict = asdict(job)  # Convert dataclass to dict

        # Extract only the desired attributes
        filtered_job = {attr: job_dict.get(attr, None) for attr in desired_attributes}

        # Apply field mapping
        for old_field, new_field in field_mapping.items():
            if old_field in filtered_job and filtered_job[old_field] is not None:
                filtered_job[new_field] = filtered_job.pop(old_field)

        converted_data.append(filtered_job)

    try:
        with open(file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns)

            # Write the header row with new fieldnames
            writer.writeheader()

            # Write the modified data rows
            writer.writerows(converted_data)

        logging.info(f"Wrote {len(converted_data)} jobs to CSV at '{file_path}'.")
    except Exception as e:
        logging.error(f"Failed to write to CSV at '{file_path}': {e}")

def analyze_results(filepath):
    """
    Analyzes the CSV results file to extract and print metrics.

    Args:
        filepath (str): Path to the CSV file to analyze.

    Returns:
        str: A summary string of the analysis.
    """
    try:
        df = pd.read_csv(filepath)

        if df.empty:
            logging.warning(f"The CSV file '{filepath}' is empty.")
            print(f"The CSV file '{filepath}' is empty.")
            return ""

        # Calculate metrics
        max_4th_column = (df['Queueing Delay'].max() - 360000) / 1000  # Adjusted value
        average_5th_column = df['query_duration'].mean() / 1000  # Converted to seconds

        # Define thresholds in milliseconds
        thresholds = [180000, 240000, 300000, 360000]
        count_less = [
            (df['Queueing Delay'] < threshold).sum()
            for threshold in thresholds
        ]

        # Print metrics
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

    Args:
        file_paths (List[str]): List of CSV file paths to process.
        num_plots (int, optional): Number of plots to generate. Defaults to 4.
        save_path (str, optional): Path to save the plot image. If None, the plot is shown. Defaults to None.
    """
    # Initialize empty lists for range_labels and range_counts for each file
    all_range_labels = []
    all_range_counts = []

    # Determine the global range_start, range_end, and step_size based on all files
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

    # Adjust global_range_start to the nearest step_size multiple
    global_range_start = int(global_range_start // step_size) * step_size

    # Calculate range_counts for all files based on global range_start and global range_end
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            range_counts = [0] * ((global_range_end - global_range_start) // step_size + 1)

            for _, row in df.iterrows():
                actual_start_time, end_time = float(row['Start_Timestamp']), float(row['End_Timestamp'])

                for i, start in enumerate(range(global_range_start, global_range_end + 1, step_size)):
                    range_lower, range_upper = start, start + step_size - 1

                    if (range_lower <= actual_start_time <= range_upper) or \
                       (range_lower <= end_time <= range_upper) or \
                       (actual_start_time < range_lower and end_time > range_upper):
                        range_counts[i] += 1

            # Create range labels in minutes
            range_labels = [f"{start // 60000}-{(start + step_size - 1) // 60000}" 
                            for start in range(global_range_start, global_range_end + 1, step_size)]
            all_range_labels.append(range_labels)
            all_range_counts.append(range_counts)
            logging.info(f"Processed '{file_path}' for plotting.")
        except Exception as e:
            logging.error(f"Error processing '{file_path}' for plotting: {e}")
            continue

    if not all_range_labels or not all_range_counts:
        logging.error("No data available to plot after processing CSV files.")
        return

    # Determine the number of plots based on the provided file_paths
    num_plots = len(file_paths)
    num_columns = 2
    num_rows = (num_plots + num_columns - 1) // num_columns

    # Create a grid of subplots with two columns
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axs = axs.flatten() if num_rows > 1 else [axs]

    # Plot the data for each file on each subplot
    for i, (ax, range_labels, range_counts, file_path) in enumerate(zip(axs, all_range_labels, all_range_counts, file_paths)):
        ax.bar(range_labels, range_counts, color='skyblue')
        ax.set_xlabel('Minute Range')
        ax.set_ylabel('Number of Active Queries')
        ax.set_title(f'Active Queries per Minute Range\n({file_path})')
        ax.set_xticks(range(len(range_labels)))
        ax.set_xticklabels(range_labels, rotation=45, ha='right')
        ax.set_ylim(0, max(range_counts) * 1.1 if range_counts else 10)
        ax.grid(True, linestyle='--', alpha=0.5)
        logging.debug(f"Plotted data for '{file_path}'.")

    # Remove any unused subplots
    for j in range(num_plots, len(axs)):
        fig.delaxes(axs[j])

    # Adjust layout
    plt.tight_layout()

    # Save the plot to a file if save_path is provided
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

    Args:
        servers_per_minute_list (List[List[int]]): List of server counts per minute for each scenario.
        num_plots (int, optional): Number of plots to generate. Defaults to 4.
        save_path (str, optional): Path to save the plot image. If None, the plot is shown. Defaults to None.
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
        logging.debug(f"Plotted scaling scenario {i+1}.")

    # Remove any unused subplots
    for j in range(num_plots, len(axs)):
        fig.delaxes(axs[j])

    # Adjust layout
    plt.tight_layout()

    # Save the plot to a file if save_path is provided
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

    Args:
        data (List[Tuple[float, Any, float, Any]]): List of data points containing latency and cost.
    """
    try:
        if not data:
            logging.warning("No data provided for Pareto plot.")
            return

        # Extract latencies and costs from the data
        latencies = [entry[0] for entry in data]
        costs = [entry[2] for entry in data]

        labels = ['DWaaS', 'DWaaS Autoscaling', 'Elastic Pool', 'QaaS']

        if len(latencies) != len(labels):
            logging.warning("Number of data points does not match number of labels.")
            # Handle mismatch by truncating or extending labels
            min_length = min(len(latencies), len(labels))
            latencies = latencies[:min_length]
            costs = costs[:min_length]
            labels = labels[:min_length]

        # Combine the latency and cost values into a single array
        points = np.array(list(zip(latencies, costs)))

        # Sort the points by latency, then by cost
        sorted_indices = points[:, 0].argsort()
        points = points[sorted_indices]
        labels = [labels[i] for i in sorted_indices]

        # Find the Pareto front
        pareto_front = [points[0]]
        pareto_labels = [labels[0]]
        for point, label in zip(points[1:], labels[1:]):
            if point[1] < pareto_front[-1][1]:
                pareto_front.append(point)
                pareto_labels.append(label)
        pareto_front = np.array(pareto_front)

        # Plot settings
        plt.figure(figsize=(10, 7))

        # Plot all points
        plt.scatter(latencies, costs, color='green', label='All Points')

        # Plot Pareto front
        plt.plot(pareto_front[:, 0], pareto_front[:, 1], color='red', marker='o', linestyle='-', linewidth=2, label='Pareto Front')

        # Annotate each point
        for latency, cost, label in zip(latencies, costs, labels):
            plt.text(latency, cost, label, fontsize=12, ha='right', va='bottom')

        # Set labels and title
        plt.xlabel('Latency', fontsize=14)
        plt.ylabel('Cost', fontsize=14)
        plt.title('Pareto Front: Latency vs Cost', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Set axis limits with padding
        plt.xlim(left=0, right=max(latencies) * 1.1 if latencies else 10)
        plt.ylim(bottom=0, top=max(costs) * 1.1 if costs else 10)

        # Increase tick label sizes
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Add legend
        plt.legend(fontsize=12)

        # Show the plot
        plt.tight_layout()
        plt.show()

        logging.info("Pareto plot generated successfully.")

    except Exception as e:
        logging.error(f"Error in plot_pareto: {e}")

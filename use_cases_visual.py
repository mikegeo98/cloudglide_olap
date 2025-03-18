import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.cm as cm
from adjustText import adjust_text
import matplotlib.lines as mlines
import seaborn as sns

def get_latency_data_from_files(output_files):
    """
    Reads the given list of CSV files, calculates:
      1) The average query duration ('query_duration')
      2) The average query duration with queueing ('query_duration_with_queue')
      3) The average or single 'mon_cost' value
    and returns a list of [avg_latency, avg_latency_with_queue, mon_cost] for each file.

    Args:
        output_files (list of str): List of paths to the CSV files to read.

    Returns:
        data1 (list of list): Each sub-list is [avg_query_exec_time, avg_query_exec_time_queueing, mon_cost].
    """
    data1 = []

    for file in output_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        
        # Calculate the average query duration and query duration with queueing
        avg_query_exec_time = df['query_duration'].mean()
        avg_query_exec_time_queueing = df['query_duration_with_queue'].mean()

        # If 'mon_cost' is the same in every row, we can just take the first value
        # or do df['mon_cost'].mean() if it might vary a bit.
        if 'mon_cost' in df.columns:
            mon_cost_value = df['mon_cost'].iloc[0]
        else:
            mon_cost_value = None  # or 0, or handle as needed

        # Append the [avg_duration, avg_duration_with_queue, mon_cost] for each file
        data1.append([avg_query_exec_time, avg_query_exec_time_queueing, mon_cost_value])

    return data1

# Function to plot scheduling data
def process_and_plot_scheduling(output_files, title):
    algorithms = ['FCFS', 'SJF', 'LJF', 'MLQ']
    
    # Get latency data
    data1 = get_latency_data_from_files(output_files)

    # Extract the latencies, latencies with queueing, and costs (CPU times)
    latencies = [entry[0] for entry in data1]
    latencies_q = [entry[1] for entry in data1]
    
    # Bar width
    bar_width = 0.35

    # X-axis positions
    index = np.arange(len(algorithms))

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(4,3))

    # Bars for latency with queueing (red)
    bars_latency_q = ax.bar(index, latencies_q, bar_width, label='Latency with\n Queueing', color='red', alpha=0.7)

    # Bars for latency (blue), shifted by bar_width to overlap
    bars_latency = ax.bar(index + bar_width, latencies, bar_width, label='Latency', color='blue')

    # Add labels and title with appropriate font sizes
    ax.set_ylabel('Latency (sec)', fontsize=20)
    if title == 0:
        ax.set_title('Medium Contention\n(4 x ra3.xplus)', fontsize=20)  # Folding title to second line
    elif title == 1:
        ax.set_title('High Contention\n(2 x ra3.xplus)', fontsize=20)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(algorithms, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(fontsize=14, loc="lower left")

    # Add gridlines for both x and y axes
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Improve layout and spacing
    plt.tight_layout()

    # Save the figure as a high-resolution image for use in papers
    plt.savefig(f"cloudglide/output_visual/scheduling_algorithms_{title}.png", dpi=300)

    # Show the plot
    # plt.show()
    print(f"Plot saved as cloudglide/output_visual/scheduling.png")


# Function to process input files and return the data for plotting
def process_and_plot_queueing(output_files):
    
    # List of algorithms
    algorithms = ['FCFS', 'MLQ-12', 'SJF']  # Adjusted to 3 algorithms for your use case

    # Get latency data
    data1 = get_latency_data_from_files(output_files)
    
    # For data5: Calculate the maximum query duration grouped by database_id for each file
    data5 = []
    for file in output_files:
        df = pd.read_csv(file)
        max_query_duration = df.groupby('database_id')['query_duration_with_queue'].max().mean()  # Mean of max per database_id
        data5.append(max_query_duration)

    # Extract the latencies, latencies with queueing, and costs (CPU times)
    latencies = [entry[0] for entry in data1]
    latencies_q = [entry[1] for entry in data1]

    # Bar width
    bar_width = 0.45

    # X-axis positions
    index = np.arange(len(algorithms))

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(4,3))

    # Bars for latency with queueing (red)
    bars_latency_q = ax.bar(index, latencies_q, bar_width, label='Query Latency (incl. Queueing)', color='red', alpha=0.7)

    # Bars for latency (blue), shifted by bar_width to overlap
    bars_latency = ax.bar(index, latencies, bar_width, label='Query Latency', color='blue')

    # Add labels and title

    # Customize axis labels and ticks
    ax.set_ylabel('Latency (sec)', fontsize=20)
    ax.set_xticks(index)
    ax.set_xticklabels(algorithms, fontsize=20, rotation=45, ha='right')  # Rotate xtick labels by 45 degrees and align to the right

    # Set tick parameters for y-axis
    ax.tick_params(axis='y', labelsize=20)

    # Add gridlines for better readability
    ax.yaxis.grid(True, linestyle='--', which='both', color='grey', alpha=0.5)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend to differentiate between latencies
    ax.legend(fontsize=14, loc='upper right')  # Adjust font size and location as needed

    # Improve layout and spacing
    plt.tight_layout()

    # Save the figure as a high-resolution image for use in papers
    plt.savefig('cloudglide/output_visual/queueing_policies.png', dpi=300)

    # Show the plot
    plt.show()
    
# Function to process input files and return the data for plotting
def process_and_plot_scaling_options(output_files):

    results = get_latency_data_from_files(output_files)
    
    # Extract latencies and costs from the data
    latencies1 = [entry[0] for entry in results[0:3]]
    costs1 = [entry[2] for entry in results[0:3]]

    latencies2 = [entry[0] for entry in results[3:6]]
    costs2 = [entry[2] for entry in results[3:6]]

    print(latencies1, costs1, latencies2, costs2)

    # Labels for the points (adjust as needed)
    labels1 = ['N=2', 'N=4', 'N=8']
    labels2 = ['N=2', 'N=4', 'N=8']

    # Combine the latency and cost values into single arrays
    points1 = np.array(list(zip(latencies1, costs1)))
    points2 = np.array(list(zip(latencies2, costs2)))

    # Sort the points by latency, and then by cost
    points1 = points1[points1[:, 0].argsort()]
    points2 = points2[points2[:, 0].argsort()]


    # Set figure size
    plt.figure(figsize=(7, 4))

    # Plot the points for the first dataset (Static DWaaS)
    plt.scatter(latencies1, costs1, color='red', label='Fixed DWaaS', s=100, marker='o')
    plt.plot(latencies1, costs1, color='red', marker='o', markersize=10, linestyle='-', linewidth=2)

    # Plot the points for the second dataset
    plt.scatter(latencies2, costs2, color='blue', label='Autoscaling DWaaS', s=100, marker='o')
    plt.plot(points2[:, 0], points2[:, 1], color='blue', marker='o', markersize=10, linestyle='-', linewidth=2)

    # Add labels for each point
    for i, (latency, cost, label) in enumerate(zip(latencies1, costs1, labels1)):
        plt.text(latency, cost, label, fontsize=20, ha='right', va='bottom', color='red')

    for i, (latency, cost, label) in enumerate(zip(latencies2, costs2, labels2)):
        plt.text(latency, cost, label, fontsize=20, ha='left', va='bottom', color='blue')

    # Add labels and title
    plt.xlabel('Avg. Query Latency (sec)', fontsize=26)
    plt.ylabel('Cost ($)', fontsize=26)
    plt.title('Static vs Autoscaling DWaaS', fontsize=26)
    plt.legend(fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set the axis limits to start from 0
    plt.xlim(left=0, right=15)
    plt.ylim(bottom=0, top=26)

    # Set tick label sizes
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    plt.savefig('cloudglide/output_visual/scaling_options.png', dpi=300)

    # Display the plot
    plt.show()

# Function to process input files and return the data for plotting
def process_and_plot_scaling_algorithms(output_files):   
 
    results = get_latency_data_from_files(output_files)
 
    latencies = [entry[1] for entry in results]
    costs = [entry[2] for entry in results]

    labels = ['Queue', 'Reactive', 'Mov. Average', 'Incr. Trend']

    colors = cm.rainbow(np.linspace(0, 1, len(labels)))

    # Combine the latency and cost values into a single array
    points = np.array(list(zip(latencies, costs)))

    # Sort the points by latency, and then by cost
    points = points[points[:, 0].argsort()]

    # Find the Pareto front
    pareto_front = [points[0]]
    for point in points[1:]:
        if point[1] < pareto_front[-1][1]:
            pareto_front.append(point)
    pareto_front = np.array(pareto_front)

    # Set figure size
    plt.figure(figsize=(6, 4))

    # Plot the Pareto front with larger markers and thicker line
    plt.plot(pareto_front[:, 0], pareto_front[:, 1], color='red', marker='o', markersize=12, linestyle='-', linewidth=3)

    # Plot the individual points with black outline and text labels in red
    for i, (latency, cost, label) in enumerate(zip(latencies, costs, labels)):
        plt.scatter(latency, cost, color='blue', s=200, edgecolor='black', linewidth=1.5)
        plt.text(latency, cost, label, fontsize=26, ha='left', va='bottom', color='red')

    # Labels and Title with enhanced font size and padding
    plt.xlabel('Avg. Latency (sec)', fontsize=26, labelpad=10)
    plt.ylabel('Cost ($)', fontsize=26, labelpad=10)
    plt.title('Scaling Policies Pareto Front', fontsize=26, pad=15)

    # Set grid and adjust transparency for better aesthetics
    plt.grid(True, linestyle='--', alpha=0.6)

    # Set axis limits with some padding
    plt.xlim(left=0, right=max(latencies) * 1.4)
    plt.ylim(bottom=0, top=max(costs) * 1.4)

    # Set tick label sizes
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    # Adjust layout to prevent label clipping
    plt.tight_layout()

    plt.savefig('cloudglide/output_visual/scaling_algorithms.png', dpi=300)

    # Show the plot
    plt.show()
    
def process_and_plot_cold_starts(results):
    
    print(results)

    # Extract latencies and costs from the data
    latencies1 = [entry[1] for entry in results[0:2]]
    costs1 = [entry[2] for entry in results[0:2]]

    latencies2 = [entry[1] for entry in results[2:4]]
    costs2 = [entry[2] for entry in results[2:4]]

    latencies3 = [entry[1] for entry in results[4:6]]
    costs3 = [entry[2] for entry in results[4:6]]

    latencies4 = [entry[1] for entry in results[6:8]]
    costs4 = [entry[2] for entry in results[6:8]]

    # Labels for the points (adjust as needed)
    labels1 = ['10sec', '60sec']
    labels2 = ['10sec', '60sec']
    labels3 = ['10sec', '60sec']
    labels4 = ['10sec', '60sec']
    # Combine the latency and cost values into single arrays
    points1 = np.array(list(zip(latencies1, costs1)))
    points2 = np.array(list(zip(latencies2, costs2)))
    points3 = np.array(list(zip(latencies3, costs3)))
    points4 = np.array(list(zip(latencies4, costs4)))

    # Sort the points by latency, and then by cost
    points1 = points1[points1[:, 0].argsort()]
    points2 = points2[points2[:, 0].argsort()]
    points3 = points3[points3[:, 0].argsort()]
    points4 = points4[points4[:, 0].argsort()]

    # Set figure size for 6x4 ratio
    plt.figure(figsize=(6, 4))

    # Plot the points for the first dataset
    plt.scatter(latencies1, costs1, color='red', label='Queue-Based')
    plt.plot(latencies1, costs1, color='red', marker='o', markersize=9, linestyle='-', linewidth=3.0)

    # Plot the points for the second dataset
    plt.scatter(latencies2, costs2, color='blue', label='Reactive')
    plt.plot(latencies2, costs2, color='blue', marker='o', markersize=9, linestyle='-', linewidth=3.0)

    # Plot the points for the third dataset
    plt.scatter(latencies3, costs3, color='green', label='Mov. Avg.')
    plt.plot(latencies3, costs3, color='green', marker='o', markersize=9, linestyle='-', linewidth=3.0)

    # Plot the points for the fourth dataset
    plt.scatter(latencies4, costs4, color='purple', label='Incr. Trend')
    plt.plot(latencies4, costs4, color='purple', marker='o', markersize=9, linestyle='-', linewidth=3.0)

    # Add labels for each point
    texts = []
    for latency, cost, label in zip(latencies1, costs1, labels1):
        texts.append(plt.text(latency, cost, label, fontsize=26, ha='right', va='top', color='red', bbox=dict(facecolor='white', alpha=0.2, edgecolor='none', pad=2.0)))
    for latency, cost, label in zip(latencies2, costs2, labels2):
        texts.append(plt.text(latency, cost, label, fontsize=26, ha='left', va='top', color='blue', bbox=dict(facecolor='white', alpha=0.2, edgecolor='none', pad=2.0)))
    for latency, cost, label in zip(latencies3, costs3, labels3):
        texts.append(plt.text(latency, cost, label, fontsize=26, ha='left', va='bottom', color='green', bbox=dict(facecolor='white', alpha=0.2, edgecolor='none', pad=2.0)))
    for latency, cost, label in zip(latencies4, costs4, labels4):
        texts.append(plt.text(latency, cost, label, fontsize=26, ha='right', va='bottom', color='purple', bbox=dict(facecolor='white', alpha=0.2, edgecolor='none', pad=2.0)))

    # Adjust text labels to avoid overlapping
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))

    # Add labels and title with larger font size
    plt.xlabel('Avg. Query Latency (sec)', fontsize=26)
    plt.ylabel('Cost ($)', fontsize=26)
    plt.title('Varying Cold Starts', fontsize=26)

    # Set grid style
    plt.grid(True)

    # Set x and y limits with some padding
    # plt.xlim(left=0, right=30)
    # plt.ylim(bottom=0, top=75)

    # Set tick label sizes
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    # Add the legend with proper positioning
    plt.legend(fontsize=22,  loc='upper right', bbox_to_anchor=(0.98, 1.08), labelspacing=0.1, handlelength=1.5, frameon=False)

    # Show plot
    plt.tight_layout()
    
    plt.savefig('cloudglide/output_visual/cold_starts.png', dpi=300)
    plt.show()
    
def process_and_plot_caching(output_files):
    # Extract latencies and costs from the data
    result = get_latency_data_from_files(output_files)

    latencies1 = [entry[0] for entry in result]
    costs1 = [entry[2] for entry in result]

    # Labels for the points (hit rates in percentages)
    labels1 = ['60%', '65%', '70%', '75%', '80%']

    # Combine the latency and cost values into single arrays for easy manipulation
    points1 = np.array(list(zip(latencies1, costs1)))

    # Sort the points by latency, and then by cost
    points1 = points1[points1[:, 0].argsort()]

    # Set figure size
    plt.figure(figsize=(6, 4))

    # Plot the points for the first dataset (DWaaS Cache Hit Rate)
    # Set smaller marker size and add transparency (alpha)
    plt.scatter(latencies1, costs1, color='red', label='DWaaS (Cache Hit Rate)', s=100, marker='o', linewidth=3, alpha=0.7)
    plt.plot(latencies1, costs1, color='red', linestyle='-', linewidth=2.5, alpha=0.7)

    # Add labels for each point with a white background and better alignment
    for latency, cost, label in zip(latencies1, costs1, labels1):
        plt.text(latency + 0.3, cost + 0.3, label, fontsize=12, ha='center', va='bottom', color='red', 
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=3))

    # Add labels and title with appropriate font sizes
    plt.xlabel('Avg. Query Latency (sec)', fontsize=26)
    plt.ylabel('Cost ($)', fontsize=26)
    # plt.title('DWaaS Cache Hit Rates', fontsize=26)

    # Create a custom line for the legend
    dwass_line = mlines.Line2D([], [], color='red', linewidth=2.5, label='DWaaS Cache Hit Rate')

    # Add the custom line to the legend
    plt.legend(handles=[dwass_line], fontsize=22, loc='upper right', bbox_to_anchor=(1.05, 1.15), borderpad=1.0, frameon=False, handlelength=0.7)
    plt.grid(True, alpha=0.3)

    # Set the axis limits and ensure equal scaling
    plt.xlim(left=0, right=24)
    plt.ylim(bottom=0, top=16)

    # Set tick label sizes
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    plt.savefig('cloudglide/output_visual/caching.png', dpi=300)

    # Display the plot
    plt.show()
    
def process_and_plot_workload_pattern_1(output_files):

    result = get_latency_data_from_files(output_files)
    
    # Extract latencies and costs from the data
    latencies1 = [entry[0] for entry in result[0:4]]
    costs1 = [entry[2] for entry in result[0:4]]

    latencies2 = [entry[0] for entry in result[4:8]]
    costs2 = [entry[2] for entry in result[4:8]]

    latencies3 = [entry[0] for entry in result[8:11]]
    costs3 = [entry[2] for entry in result[8:11]]

    qaas = result[11]
    latencies4 = [qaas[0]]
    costs4 = [qaas[2]]
    
    # Labels for the points (adjust as needed)
    labels1 = ['N=2', 'N=3', 'N=4', 'N=8']
    labels2 = ['N=2', 'N=3', 'N=4', 'N=8']
    labels3 = ['VPU=6', 'VPU=12', 'VPU=24']
    labels4 = ['QaaS']
    fig, ax = plt.subplots(figsize=(7,4))

    # Plot dataset1 (red)
    ax.plot(latencies1, costs1,
            color='red', marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')
    ax.scatter(latencies1, costs1, color='red', s=0)  # s=0 used so we rely on plot() markers
    # dataset2 (blue)
    ax.plot(latencies2, costs2,
            color='blue', marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')
    ax.scatter(latencies2, costs2, color='blue', s=0)

    # dataset3 (green)
    ax.plot(latencies3, costs3,
            color='green', marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')
    ax.scatter(latencies3, costs3, color='green', s=0)

    # dataset4 (purple)
    ax.plot(latencies4, costs4,
            color='purple', marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')
    ax.scatter(latencies4, costs4, color='purple', s=0)

    # Annotations
    for (x, y, lab) in zip(latencies1, costs1, labels1):
        ax.text(x, y, lab, fontsize=22, ha='left', va='top', color='red',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))
    for (x, y, lab) in zip(latencies2, costs2, labels2):
        ax.text(x, y, lab, fontsize=22, ha='right', va='top', color='blue',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))
    for (x, y, lab) in zip(latencies3, costs3, labels3):
        ax.text(x, y, lab, fontsize=22, ha='left', va='bottom', color='green',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))
    for (x, y, lab) in zip(latencies4, costs4, labels4):
        ax.text(x, y, lab, fontsize=22, ha='left', va='bottom', color='purple',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    # Axes
    ax.set_xlabel('Avg. Query Latency (sec)', fontsize=22)
    ax.set_ylabel('Cost ($)', fontsize=22)
    ax.tick_params(axis='both', labelsize=22)
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Optional: x/y limits
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 27)

    # Example legend usage (commented out)
    # ax.legend(['DWaaS','DWaaS Autoscaling','Elastic Pool','QaaS'],
    #           fontsize=24, loc='best')

    plt.tight_layout()
    plt.show()
      
def process_and_plot_workload_pattern_2(output_files):

    result = get_latency_data_from_files(output_files)

    # Extract latencies and costs from the data
    latencies1 = [entry[0] for entry in result[0:4]]
    costs1 = [entry[2] for entry in result[0:4]]

    latencies2 = [entry[0] for entry in result[4:8]]
    costs2 = [entry[2] for entry in result[4:8]]

    latencies3 = [entry[0] for entry in result[8:14]]
    costs3 = [entry[2] for entry in result[8:14]]

    qaas = result[14]
    latencies4 = [qaas[0]]
    costs4 = [qaas[2]]
    
    # Labels for the points (adjust as needed)
    labels1 = ['N=8', 'N=16', 'N=24', 'N=32']
    labels2 = ['N=8', 'N=16', 'N=24', 'N=32']
    labels3 = ['VPU=6','VPU=16', 'VPU=24','VPU=32', 'VPU=48', 'VPU=64']
    labels4 = ['QaaS']
    
    fig, ax = plt.subplots(figsize=(7,4))

    # Plot each data set with line+markers
    ax.plot(latencies1, costs1, color='red',   marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')
    ax.plot(latencies2, costs2, color='blue',  marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')
    ax.plot(latencies3, costs3, color='green', marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')
    ax.plot(latencies4, costs4, color='purple',marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')

    # Annotations
    for (x,y,l) in zip(latencies1, costs1, labels1):
        ax.text(x, y, l, fontsize=22, color='red',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))
    for (x,y,l) in zip(latencies2, costs2, labels2):
        ax.text(x, y, l, fontsize=22, ha='right', color='blue',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))
    for (x,y,l) in zip(latencies3, costs3, labels3):
        ax.text(x, y, l, fontsize=22, color='green',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))
    for (x,y,l) in zip(latencies4, costs4, labels4):
        ax.text(x, y, l, fontsize=22, ha='right', color='purple',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))

    # Axes
    ax.set_xlabel('Avg. Query Latency (sec)', fontsize=28)
    ax.set_ylabel('Cost ($)', fontsize=28)
    ax.tick_params(axis='both', labelsize=24)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Example axis limits
    ax.set_xlim(0,30)
    ax.set_ylim(0,85)

    plt.tight_layout()
    plt.show()
    
def process_and_plot_workload_pattern_3(output_files):

    result = get_latency_data_from_files(output_files)

    # Extract latencies and costs from the data
    latencies1 = [entry[0] for entry in result[0:4]]
    costs1 = [entry[2] for entry in result[0:4]]

    latencies2 = [entry[0] for entry in result[4:8]]
    costs2 = [entry[2] for entry in result[4:8]]

    latencies3 = [entry[0] for entry in result[8:14]]
    costs3 = [entry[2] for entry in result[8:14]]

    qaas = result[14]
    latencies4 = [qaas[0]]
    costs4 = [qaas[2]]
    
    # Labels for the points (adjust as needed)
    labels1 = ['N=4', 'N=8', 'N=12', 'N=16']
    labels2 = ['N=4', 'N=8', 'N=12', 'N=16']
    labels3 = ['VPU=6','VPU=16', 'VPU=24','VPU=32', 'VPU=48', 'VPU=64']
    labels4 = ['QaaS']
    
    fig, ax = plt.subplots(figsize=(7,4))

    ax.plot(latencies1, costs1, color='red',   marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')
    ax.plot(latencies2, costs2, color='blue',  marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')
    ax.plot(latencies3, costs3, color='green', marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')
    ax.plot(latencies4, costs4, color='purple',marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')

    # Annotations
    for (x,y,lbl) in zip(latencies1, costs1, labels1):
        ax.text(x, y, lbl, fontsize=22, color='red',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))
    for (x,y,lbl) in zip(latencies2, costs2, labels2):
        ax.text(x, y, lbl, fontsize=22, color='blue',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))
    for (x,y,lbl) in zip(latencies3, costs3, labels3):
        ax.text(x, y, lbl, fontsize=22, ha='center', color='green',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))
    for (x,y,lbl) in zip(latencies4, costs4, labels4):
        ax.text(x, y, lbl, fontsize=22, color='purple',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))

    ax.set_xlabel('Avg. Query Latency (sec)', fontsize=22)
    ax.set_ylabel('Cost ($)', fontsize=22)
    ax.tick_params(axis='both', labelsize=22)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set the axis limits to start from 0
    plt.xlim(left=0, right=30)
    plt.ylim(bottom=0, top=85)

    plt.tight_layout()
    plt.show()
    
def process_and_plot_workload_pattern_4(output_files):

    result = get_latency_data_from_files(output_files)

    # Extract latencies and costs from the data
    latencies1 = [entry[0] for entry in result[0:4]]
    costs1 = [entry[2] for entry in result[0:4]]

    latencies2 = [entry[0] for entry in result[4:8]]
    costs2 = [entry[2] for entry in result[4:8]]

    latencies3 = [entry[0] for entry in result[8:11]]
    costs3 = [entry[2] for entry in result[8:11]]

    qaas = result[11]
    latencies4 = [qaas[0]]
    costs4 = [qaas[2]]
    
    # Labels for the points (adjust as needed)
    labels1 = ['N=2', 'N=3', 'N=4', 'N=8']
    labels2 = ['N=2', 'N=3', 'N=4', 'N=8']
    labels3 = ['VPU=6', 'VPU=12', 'VPU=24']
    labels4 = ['QaaS']
    
    fig, ax = plt.subplots(figsize=(7,4))

    ax.plot(latencies1, costs1,
            color='red', marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')
    ax.scatter(latencies1, costs1, color='red', s=0)

    ax.plot(latencies2, costs2,
            color='blue', marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')
    ax.scatter(latencies2, costs2, color='blue', s=0)

    ax.plot(latencies3, costs3,
            color='green', marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')
    ax.scatter(latencies3, costs3, color='green', s=0)

    ax.plot(latencies4, costs4,
            color='purple', marker='o', markersize=12, linewidth=2,
            markeredgecolor='black', alpha=0.9, linestyle='-')
    ax.scatter(latencies4, costs4, color='purple', s=0)

    for (x,y,lab) in zip(latencies1, costs1, labels1):
        ax.text(x, y, lab, fontsize=22, color='red',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))
    for (x,y,lab) in zip(latencies2, costs2, labels2):
        ax.text(x, y, lab, fontsize=22, ha='right', va='top', color='blue',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))
    for (x,y,lab) in zip(latencies3, costs3, labels3):
        ax.text(x, y, lab, fontsize=22, va='bottom', color='green',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))
    for (x,y,lab) in zip(latencies4, costs4, labels4):
        ax.text(x, y, lab, fontsize=22, ha='right', va='bottom', color='purple',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    ax.set_xlabel('Avg. Query Latency (sec)', fontsize=22)
    ax.set_ylabel('Cost ($)', fontsize=22)
    ax.tick_params(axis='both', labelsize=22)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(0,15)
    ax.set_ylim(0,27)

    plt.tight_layout()
    plt.show()
    
def process_and_plot_workload_pattern_5(output_files):
    
    result = get_latency_data_from_files(output_files)

    # Extract latencies and costs from the data
    latencies1 = [entry[0] for entry in result[0:4]]
    costs1 = [entry[2] for entry in result[0:4]]

    latencies2 = [entry[0] for entry in result[4:8]]
    costs2 = [entry[2] for entry in result[4:8]]

    latencies3 = [entry[0] for entry in result[8:12]]
    costs3 = [entry[2] for entry in result[8:12]]

    qaas = result[12]
    latencies4 = [qaas[0]]
    costs4 = [qaas[2]]
    
    # Labels for the points (adjust as needed)
    labels1 = ['N=2', 'N=3', 'N=4', 'N=8']
    labels2 = ['N=2', '', 'N=4', 'N=8']
    labels3 = ['VPU=4', 'VPU=6', 'VPU=12', 'VPU=24']
    labels4 = ['QaaS']


    # Sort by latency so the line is drawn left -> right
    # (If there's only one point, you'll just see a single marker.)
    def sort_data(xs, ys, lbs):
        pts = sorted(zip(xs, ys, lbs), key=lambda x: x[0])
        return zip(*pts)  # returns (sorted_x, sorted_y, sorted_labels)

    latencies1, costs1, labels1 = sort_data(latencies1, costs1, labels1)
    latencies1 = list(latencies1)
    costs1     = list(costs1)
    labels1    = list(labels1)

    latencies2, costs2, labels2 = sort_data(latencies2, costs2, labels2)
    latencies2 = list(latencies2)
    costs2     = list(costs2)
    labels2    = list(labels2)

    latencies3, costs3, labels3 = sort_data(latencies3, costs3, labels3)
    latencies3 = list(latencies3)
    costs3     = list(costs3)
    labels3    = list(labels3)

    latencies4, costs4, labels4 = sort_data(latencies4, costs4, labels4)
    latencies4 = list(latencies4)
    costs4     = list(costs4)
    labels4    = list(labels4)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot dataset1 (red)
    ax.plot(latencies1, costs1, color='red', marker='o', markersize=12,
            linewidth=2, markeredgecolor='black', alpha=0.9, linestyle='-')
    # dataset2 (blue)
    ax.plot(latencies2, costs2, color='blue', marker='o', markersize=12,
            linewidth=2, markeredgecolor='black', alpha=0.9, linestyle='-')
    # dataset3 (green)
    ax.plot(latencies3, costs3, color='green', marker='o', markersize=12,
            linewidth=2, markeredgecolor='black', alpha=0.9, linestyle='-')
    # dataset4 (purple)
    ax.plot(latencies4, costs4, color='purple', marker='o', markersize=12,
            linewidth=2, markeredgecolor='black', alpha=0.9, linestyle='-')

    # Annotations
    for (x, y, lab) in zip(latencies1, costs1, labels1):
        ax.text(x, y, lab, fontsize=22, va='bottom', color='red',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))
    for (x, y, lab) in zip(latencies2, costs2, labels2):
        ax.text(x, y, lab, fontsize=22, va='top', ha='right', color='blue',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))
    for (x, y, lab) in zip(latencies3, costs3, labels3):
        ax.text(x, y, lab, fontsize=22, va='bottom', color='green',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))
    for (x, y, lab) in zip(latencies4, costs4, labels4):
        ax.text(x, y, lab, fontsize=22, va='bottom', color='purple',
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))

    # Axis labels, ticks, grid
    ax.set_xlabel('Avg. Query Latency (sec)', fontsize=22)
    ax.set_ylabel('Cost ($)', fontsize=22)
    ax.tick_params(axis='both', labelsize=22)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # If you have multiple data sets, you can add a legend:
    # ax.legend(['DWaaS','DWaaS A/S','Elastic Pool','QaaS'], fontsize=20, loc='best')

    ax.set_xlim(0,25)
    ax.set_ylim(0,27)

    plt.tight_layout()
    plt.show()



def tpch_results():
    # Create the output directory if it doesn't exist
    os.makedirs("cloudglide/output_visual/tpch", exist_ok=True)

    # ----
    # Settings
    # ----
    target_scales = [1, 10, 23, 59, 85, 125, 202, 356]
    plot_scales = [23, 59, 85, 125, 202]

    exp_file = "cloudglide/datasets/aws_output.csv"

    sim_files = {
        "N=1": "cloudglide/output_simulation/tpch_1.csv",
        "N=2": "cloudglide/output_simulation/tpch_2.csv",
        "N=4": "cloudglide/output_simulation/tpch_3.csv",
        "N=8": "cloudglide/output_simulation/tpch_4.csv"
    }

    config_colors = {
        "N=1": "red",
        "N=2": "blue",
        "N=4": "green",
        "N=8": "purple"
    }

    # ----
    # 1. Load and reshape experimental data
    # ----
    df_exp = pd.read_csv(exp_file)

    required_cols = ["Query", "Data Size", "N=1 Redshift", "N=2 Redshift", "N=4 Redshift", "N=8 Redshift"]
    for col in required_cols:
        if col not in df_exp.columns:
            raise ValueError(f"Column {col} not found in {exp_file}.")

    df_exp.rename(columns={"Query": "query_id", "Data Size": "scale_factor"}, inplace=True)
    df_exp["query_id"] = pd.to_numeric(df_exp["query_id"], errors="coerce")
    df_exp["scale_factor"] = pd.to_numeric(df_exp["scale_factor"], errors="coerce")

    dwaas_cols = ["N=1 Redshift", "N=2 Redshift", "N=4 Redshift", "N=8 Redshift"]
    df_exp_long = df_exp.melt(
        id_vars=["query_id", "scale_factor"],
        value_vars=dwaas_cols,
        var_name="config",
        value_name="exp_duration"
    )
    df_exp_long["config"] = df_exp_long["config"].str.replace(" Redshift", "")

    # ----
    # 2. Load simulation data
    # ----
    sim_data = {}
    for config, fname in sim_files.items():
        df_sim = pd.read_csv(fname)
        for col in ["query_id", "query_duration"]:
            if col not in df_sim.columns:
                raise ValueError(f"Column {col} not found in {fname}.")
        df_sim["query_id"] = pd.to_numeric(df_sim["query_id"], errors="coerce")
        
        # If scale_factor not in the CSV, assign it via group
        if "scale_factor" not in df_sim.columns:
            def assign_scale(group):
                group = group.copy()
                n = len(target_scales)
                if len(group) != n:
                    raise ValueError(
                        f"Query {group['query_id'].iloc[0]} does not have {n} rows in {fname}."
                    )
                group["scale_factor"] = target_scales
                return group
            df_sim = df_sim.groupby("query_id", group_keys=False).apply(assign_scale)

        df_sim["scale_factor"] = pd.to_numeric(df_sim["scale_factor"], errors="coerce")
        df_sim["config"] = config
        sim_data[config] = df_sim

    df_sim_all = pd.concat(sim_data.values(), ignore_index=True)

    # ----
    # 3. Per-query analysis + stats
    # ----
    queries = sorted(df_exp_long["query_id"].unique())

    stats_results = []
    mre_list = []
    fine_results = []  # row-by-row stats: MRE, QERROR, SRE

    for q in queries:
        plt.figure(figsize=(8,6))
        
        for config in config_colors.keys():
            sub_exp = df_exp_long[
                (df_exp_long["query_id"] == q)
                & (df_exp_long["config"] == config)
                & (df_exp_long["scale_factor"].isin(plot_scales))
            ].sort_values(by="scale_factor")
            
            sub_sim = df_sim_all[
                (df_sim_all["query_id"] == q)
                & (df_sim_all["config"] == config)
                & (df_sim_all["scale_factor"].isin(plot_scales))
            ].sort_values(by="scale_factor")
            
            plt.plot(
                sub_exp["scale_factor"], sub_exp["exp_duration"],
                marker="o", linestyle="-", color=config_colors[config],
                label=f"{config} Exp" if q == queries[0] else None
            )
            plt.plot(
                sub_sim["scale_factor"], sub_sim["query_duration"],
                marker="s", linestyle="--", color=config_colors[config],
                label=f"{config} Sim" if q == queries[0] else None
            )
            
            if not sub_exp.empty and not sub_sim.empty:
                exp_vals = sub_exp["exp_duration"].values
                sim_vals = sub_sim["query_duration"].values
                sc_vals = sub_exp["scale_factor"].values
                
                errors = sim_vals - exp_vals
                mae = np.mean(np.abs(errors))
                rmse = np.sqrt(np.mean(errors**2))
                rel_errors = np.abs(errors) / exp_vals
                mre = np.mean(rel_errors)
                
                stats_results.append({
                    "query_id": q,
                    "config": config,
                    "num_points": len(exp_vals),
                    "MAE": mae,
                    "RMSE": rmse,
                    "MRE": mre
                })
                mre_list.append({"query_id": q, "config": config, "MRE": mre})
                
                # finer row-by-row
                for i in range(len(sc_vals)):
                    e = exp_vals[i]
                    s = sim_vals[i]
                    sc = sc_vals[i]
                    
                    err = s - e
                    mae_ = abs(err)
                    rmse_ = abs(err)  # single point
                    mre_ = mae_ / e if e != 0 else np.inf
                    if e > 0 and s > 0:
                        qerr_ = max(s/e, e/s)
                    else:
                        qerr_ = np.inf
                    sre_ = (err/e) if e !=0 else np.inf
                    
                    fine_results.append({
                        "query_id": q,
                        "config": config,
                        "scale_factor": sc,
                        "MAE": mae_,
                        "RMSE": rmse_,
                        "MRE": mre_,
                        "QERROR": qerr_,
                        "SRE": sre_
                    })

        plt.xlabel("Scale Factor")
        plt.ylabel("Query Duration (s)")
        plt.title(f"Query {int(q)} Performance vs Scale Factor")
        plt.legend(loc="best", fontsize=9)
        plt.grid(True)
        plt.tight_layout()

        # Save figure (no plt.show())
        save_path = f"cloudglide/output_visual/tpch/query_{int(q)}_performance.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    # Aggregated stats
    df_stats = pd.DataFrame(stats_results)
    pd.set_option('display.max_rows', None)
    print(df_stats)

    # We also save the summary CSV in the same directory
    df_stats.to_csv("cloudglide/output_visual/tpch/statistics_summary.csv", index=False)

    # ---- 4. MRE violin
    df_mre = pd.DataFrame(mre_list)

    sns.set(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(8,6))
    sns.violinplot(x="config", y="MRE", data=df_mre, palette=config_colors, inner="quartile")
    plt.xlabel("DWaaS Configuration")
    plt.ylabel("Mean Relative Error (Abs)")
    plt.title("Distribution of Mean Relative Error across Queries per Config")
    plt.tight_layout()

    # Save the violinplot
    plt.savefig("cloudglide/output_visual/tpch/mean_relative_error_violinplot.png", dpi=300)
    plt.close()

    # ---- 5. Boxplot-based MRE outliers
    print("\nMRE Outliers by config (MRE > Q3+1.5*IQR):")
    for config, group in df_mre.groupby("config"):
        Q1 = group["MRE"].quantile(0.25)
        Q3 = group["MRE"].quantile(0.75)
        IQR = Q3 - Q1
        threshold = Q3 + 1.5*IQR
        worst = group[group["MRE"]>threshold]
        if not worst.empty:
            print(f"\nConfiguration {config} MRE > {threshold:.3f} =>")
            print(worst.to_string(index=False))
        else:
            print(f"\nConfiguration {config}: No MRE outliers found.")

    # (New) Print top 10 MRE overall:
    df_mre_sorted = df_mre.sort_values(by="MRE", ascending=False)
    print("\nTop 10 MRE outliers overall:")
    print(df_mre_sorted.head(10).to_string(index=False))

    # ---- 6. Write finer granularity with SRE
    df_fine = pd.DataFrame(fine_results)
    df_fine = df_fine[[
        "query_id", "config", "scale_factor",
        "MAE", "RMSE", "MRE", "QERROR", "SRE"
    ]]
    df_fine_path = "cloudglide/output_visual/tpch/finer_statistics.csv"
    df_fine.to_csv(df_fine_path, index=False)
    print(f"\nWrote finer granularity rows to '{df_fine_path}'.")

    # Set style/context for the next plot
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.4)

    plt.figure(figsize=(6,4))
    ax = sns.violinplot(
        x="config",
        y="SRE",
        data=df_fine,
        palette=config_colors,
        inner="quartile",
        cut=0,       # prevents extreme tails
        linewidth=1
    )
    ax.axhline(0, color='k', linestyle='--', linewidth=1.0)
    ax.set_xlabel("DWaaS Configuration", fontsize=22)
    ax.set_ylabel("Signed Relative Error", fontsize=22)
    ax.tick_params(axis='both', labelsize=22)
    ax.set_ylim([-1, 1])
    sns.despine(trim=True)
    plt.tight_layout()

    # Save the SRE violinplot as PDF
    pdf_path = "cloudglide/output_visual/tpch/signed_relative_error_violinplot.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    # ---- 9. QERROR outliers
    print("\nQERROR Outliers by config (QERROR > Q3+1.5*IQR):")
    for config, group in df_fine.groupby("config"):
        Q1 = group["QERROR"].quantile(0.25)
        Q3 = group["QERROR"].quantile(0.75)
        IQR = Q3 - Q1
        threshold = Q3 + 1.5 * IQR
        worst = group[group["QERROR"] > threshold]
        if not worst.empty:
            print(f"\nConfiguration {config}, QERROR > {threshold:.3f} =>")
            print(worst.to_string(index=False))
        else:
            print(f"\nConfiguration {config}: No QERROR outliers found.")

    # ---- (New) Signed Relative Error outliers
    print("\nSRE Outliers by config using boxplot approach (over & under):")
    df_sre = df_fine[["query_id","config","SRE"]]

    for config, group in df_sre.groupby("config"):
        Q1 = group["SRE"].quantile(0.25)
        Q3 = group["SRE"].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 1*IQR
        lower = Q1 - 1*IQR

        worst_over = group[group["SRE"] > upper]
        worst_under = group[group["SRE"] < lower]

        print(f"\nConfiguration {config}:")
        if worst_over.empty and worst_under.empty:
            print("  No SRE outliers found.")
        else:
            if not worst_over.empty:
                print("  Overest Outliers (SRE > {:.3f}):".format(upper))
                print(worst_over.to_string(index=False))
            if not worst_under.empty:
                print("  Underest Outliers (SRE < {:.3f}):".format(lower))
                print(worst_under.to_string(index=False))

    print("Done.")
    
    
def plot_concurrency():
    
    # Set up Seaborn for a polished look with larger fonts
    sns.set_theme(style='whitegrid', context='paper', font_scale=3)
    
    # Concurrency from 1 to 10
    concurrency_levels = np.arange(1, 11)

    # Approximate per-query times from your figure
    # Ideal (Linear): 3 sec * N
    ideal_linear = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0])
    # Measured times for Q1 SF=10
    measured = np.array([3.1, 5.6, 8.4, 11.2, 13.5, 17.1, 19.2, 22.0, 24.5, 27.0])
    
    # Create figure with dimensions 8x4 inches
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot Ideal (Linear)
    ax.plot(concurrency_levels, ideal_linear, 
            marker='o', markersize=12, linestyle='--', linewidth=3.0, 
            color='magenta', label='CloudGlide')

    # Plot Measured
    ax.plot(concurrency_levels, measured, 
            marker='s', markersize=10, linestyle='-', linewidth=3.0, 
            color='black', label='Measured')

    # Configure axes
    ax.set_xticks(concurrency_levels)
    ax.set_xlabel('Concurrent Queries (N)', fontsize=28)
    ax.set_ylabel('Per-Query \n Ex. Time(sec)', fontsize=28)
#     ax.set_title('TPC-H Q1 Concurrency Model', fontsize=28)

    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=28)

    # Style grid and spines
    ax.grid(True, linestyle='--', linewidth=1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Add legend with bigger fonts
    ax.legend(loc='upper left', fontsize=23, frameon=False)

    plt.tight_layout()
    plt.show()
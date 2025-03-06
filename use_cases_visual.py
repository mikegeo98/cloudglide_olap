import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.cm as cm
from adjustText import adjust_text
import matplotlib.lines as mlines

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
    
def process_and_plot_workload_pattern_1(result):

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

    # Set figure size
    plt.figure(figsize=(7,4))

    # Plot the points for the first dataset
    plt.scatter(latencies1, costs1, color='red', label='DWaaS')
    plt.plot(points1[:, 0], points1[:, 1], color='red', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Plot the points for the second dataset
    plt.scatter(latencies2, costs2, color='blue', label='DWaaS Autoscaling')
    plt.plot(points2[:, 0], points2[:, 1], color='blue', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Plot the points for the third dataset
    plt.scatter(latencies3, costs3, color='green', label='Elastic Pool')
    plt.plot(points3[:, 0], points3[:, 1], color='green', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Plot the points for the fourth dataset
    plt.scatter(latencies4, costs4, color='purple', label='QaaS')
    plt.plot(points4[:, 0], points4[:, 1], color='purple', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Add labels for each point with a white background and better alignment
    for i, (latency, cost, label) in enumerate(zip(latencies1, costs1, labels1)):
        plt.text(latency, cost, label, fontsize=22, ha='right', va='top', color='red', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    for i, (latency, cost, label) in enumerate(zip(latencies2, costs2, labels2)):
        plt.text(latency, cost, label, fontsize=22, ha='left', va='top', color='blue', 
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))

    for i, (latency, cost, label) in enumerate(zip(latencies3, costs3, labels3)):
        plt.text(latency, cost, label, fontsize=22, ha='left', va='bottom', color='green', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    for i, (latency, cost, label) in enumerate(zip(latencies4, costs4, labels4)):
        plt.text(latency, cost, label, fontsize=22, ha='left', va='bottom', color='purple', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    # Add labels and title
    plt.xlabel('Avg. Query Latency (sec)', fontsize=25)
    plt.ylabel('Cost ($)', fontsize=25)
    plt.title('Pattern 1 Architecture Comparison', fontsize=25)
    # plt.legend(fontsize=25)
    plt.grid(True)

    # Set the axis limits to start from 0
    # plt.xlim(left=0, right=15)
    # plt.ylim(bottom=0, top=25)

    # Set tick label sizes
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    
    plt.savefig('cloudglide/output_visual/pattern1.png', dpi=300)

    plt.show()
      
def process_and_plot_workload_pattern_2(result):

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

    # Set figure size
    plt.figure(figsize=(7,4))

    # Plot the points for the first dataset
    plt.scatter(latencies1, costs1, color='red', label='DWaaS')
    plt.plot(points1[:, 0], points1[:, 1], color='red', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Plot the points for the second dataset
    plt.scatter(latencies2, costs2, color='blue', label='DWaaS Autoscaling')
    plt.plot(points2[:, 0], points2[:, 1], color='blue', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Plot the points for the third dataset
    plt.scatter(latencies3, costs3, color='green', label='Elastic Pool')
    plt.plot(points3[:, 0], points3[:, 1], color='green', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Plot the points for the fourth dataset
    plt.scatter(latencies4, costs4, color='purple', label='QaaS')
    plt.plot(points4[:, 0], points4[:, 1], color='purple', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Add labels for each point with a white background and better alignment
    for i, (latency, cost, label) in enumerate(zip(latencies1, costs1, labels1)):
        plt.text(latency, cost, label, fontsize=22, ha='right', va='top', color='red', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    for i, (latency, cost, label) in enumerate(zip(latencies2, costs2, labels2)):
        plt.text(latency, cost, label, fontsize=22, ha='left', va='top', color='blue', 
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))

    for i, (latency, cost, label) in enumerate(zip(latencies3, costs3, labels3)):
        plt.text(latency, cost, label, fontsize=22, ha='left', va='bottom', color='green', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    for i, (latency, cost, label) in enumerate(zip(latencies4, costs4, labels4)):
        plt.text(latency, cost, label, fontsize=22, ha='left', va='bottom', color='purple', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    # Add labels and title
    plt.xlabel('Avg. Query Latency (sec)', fontsize=25)
    plt.ylabel('Cost ($)', fontsize=25)
    plt.title('Pattern 2 Architecture Comparison', fontsize=25)
    # plt.legend(fontsize=25)
    plt.grid(True)

    # Set the axis limits to start from 0
    # plt.xlim(left=0, right=15)
    # plt.ylim(bottom=0, top=25)

    # Set tick label sizes
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    
    plt.savefig('cloudglide/output_visual/pattern2.png', dpi=300)

    plt.show()
    
def process_and_plot_workload_pattern_3(result):

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

    # Set figure size
    plt.figure(figsize=(7,4))

    # Plot the points for the first dataset
    plt.scatter(latencies1, costs1, color='red', label='DWaaS')
    plt.plot(points1[:, 0], points1[:, 1], color='red', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Plot the points for the second dataset
    plt.scatter(latencies2, costs2, color='blue', label='DWaaS Autoscaling')
    plt.plot(points2[:, 0], points2[:, 1], color='blue', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Plot the points for the third dataset
    plt.scatter(latencies3, costs3, color='green', label='Elastic Pool')
    plt.plot(points3[:, 0], points3[:, 1], color='green', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Plot the points for the fourth dataset
    plt.scatter(latencies4, costs4, color='purple', label='QaaS')
    plt.plot(points4[:, 0], points4[:, 1], color='purple', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Add labels for each point with a white background and better alignment
    for i, (latency, cost, label) in enumerate(zip(latencies1, costs1, labels1)):
        plt.text(latency, cost, label, fontsize=22, ha='right', va='top', color='red', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    for i, (latency, cost, label) in enumerate(zip(latencies2, costs2, labels2)):
        plt.text(latency, cost, label, fontsize=22, ha='left', va='top', color='blue', 
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))

    for i, (latency, cost, label) in enumerate(zip(latencies3, costs3, labels3)):
        plt.text(latency, cost, label, fontsize=22, ha='left', va='bottom', color='green', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    for i, (latency, cost, label) in enumerate(zip(latencies4, costs4, labels4)):
        plt.text(latency, cost, label, fontsize=22, ha='left', va='bottom', color='purple', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    # Add labels and title
    plt.xlabel('Avg. Query Latency (sec)', fontsize=25)
    plt.ylabel('Cost ($)', fontsize=25)
    plt.title('Pattern 3 Architecture Comparison', fontsize=25)
    # plt.legend(fontsize=25)
    plt.grid(True)

    # Set tick label sizes
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    
    plt.savefig('cloudglide/output_visual/pattern3.png', dpi=300)

    plt.show()
    
def process_and_plot_workload_pattern_4(result):

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

    # Set figure size
    plt.figure(figsize=(7,4))

    # Plot the points for the first dataset
    plt.scatter(latencies1, costs1, color='red', label='DWaaS')
    plt.plot(points1[:, 0], points1[:, 1], color='red', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Plot the points for the second dataset
    plt.scatter(latencies2, costs2, color='blue', label='DWaaS Autoscaling')
    plt.plot(points2[:, 0], points2[:, 1], color='blue', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Plot the points for the third dataset
    plt.scatter(latencies3, costs3, color='green', label='Elastic Pool')
    plt.plot(points3[:, 0], points3[:, 1], color='green', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Plot the points for the fourth dataset
    plt.scatter(latencies4, costs4, color='purple', label='QaaS')
    plt.plot(points4[:, 0], points4[:, 1], color='purple', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Add labels for each point with a white background and better alignment
    for i, (latency, cost, label) in enumerate(zip(latencies1, costs1, labels1)):
        plt.text(latency, cost, label, fontsize=22, ha='right', va='top', color='red', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    for i, (latency, cost, label) in enumerate(zip(latencies2, costs2, labels2)):
        plt.text(latency, cost, label, fontsize=22, ha='left', va='top', color='blue', 
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))

    for i, (latency, cost, label) in enumerate(zip(latencies3, costs3, labels3)):
        plt.text(latency, cost, label, fontsize=22, ha='left', va='bottom', color='green', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    for i, (latency, cost, label) in enumerate(zip(latencies4, costs4, labels4)):
        plt.text(latency, cost, label, fontsize=22, ha='left', va='bottom', color='purple', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    # Add labels and title
    plt.xlabel('Avg. Query Latency (sec)', fontsize=25)
    plt.ylabel('Cost ($)', fontsize=25)
    plt.title('Pattern 4 Architecture Comparison', fontsize=25)
    # plt.legend(fontsize=25)
    plt.grid(True)

    # Set tick label sizes
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig('cloudglide/output_visual/pattern4.png', dpi=300)
    plt.show()
    
def process_and_plot_workload_pattern_5(result):

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

    # Set figure size
    plt.figure(figsize=(7,4))

    # Plot the points for the first dataset
    plt.scatter(latencies1, costs1, color='red', label='DWaaS')
    plt.plot(points1[:, 0], points1[:, 1], color='red', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Plot the points for the second dataset
    plt.scatter(latencies2, costs2, color='blue', label='DWaaS Autoscaling')
    plt.plot(points2[:, 0], points2[:, 1], color='blue', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Plot the points for the third dataset
    plt.scatter(latencies3, costs3, color='green', label='Elastic Pool')
    plt.plot(points3[:, 0], points3[:, 1], color='green', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Plot the points for the fourth dataset
    plt.scatter(latencies4, costs4, color='purple', label='QaaS')
    plt.plot(points4[:, 0], points4[:, 1], color='purple', marker='o', markersize=6, linestyle='-',linewidth=3.0)

    # Add labels for each point with a white background and better alignment
    for i, (latency, cost, label) in enumerate(zip(latencies1, costs1, labels1)):
        plt.text(latency, cost, label, fontsize=22, ha='right', va='top', color='red', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    for i, (latency, cost, label) in enumerate(zip(latencies2, costs2, labels2)):
        plt.text(latency, cost, label, fontsize=22, ha='left', va='top', color='blue', 
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=2.0))

    for i, (latency, cost, label) in enumerate(zip(latencies3, costs3, labels3)):
        plt.text(latency, cost, label, fontsize=22, ha='left', va='bottom', color='green', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    for i, (latency, cost, label) in enumerate(zip(latencies4, costs4, labels4)):
        plt.text(latency, cost, label, fontsize=22, ha='left', va='bottom', color='purple', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2.0))

    # Add labels and title
    plt.xlabel('Avg. Query Latency (sec)', fontsize=25)
    plt.ylabel('Cost ($)', fontsize=25)
    plt.title('Pattern 5 Architecture Comparison', fontsize=25)
    # plt.legend(fontsize=25)
    plt.grid(True)

    # Set tick label sizes
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig('cloudglide/output_visual/pattern5.png', dpi=300)
    plt.show()
import csv
import math
from cloudglide.event import Event, next_event_counter
import heapq

import logging
from math import ceil
import random
import time
from collections import deque, namedtuple
from typing import List, Tuple
import numpy as np
import pandas as pd
from cloudglide.cost_model import (
    cost_calculator,
    redshift_persecond_cost,
    qaas_total_cost,
)
from cloudglide.scaling_model import Autoscaler
from cloudglide.scheduling_model import io_scheduler, cpu_scheduler
from cloudglide.query_processing_model import (
    simulate_cpu_autoscaling,
    simulate_cpu_qaas,
    simulate_cpu_nodes,
    simulate_io_qaas,
    simulate_io_cached,
    simulate_io_elastic_pool
)
from cloudglide.visual_model import write_to_csv
from cloudglide.job import Job
from cloudglide.config import (
    INTERRUPT_PROBABILITY,
    INTERRUPT_DURATION,
    COST_PER_SECOND_REDSHIFT,
    COST_PER_RPU_HOUR,
    COST_PER_SLOT_HOUR,
)


def handle_interrupt(
    io_jobs: deque,
    cpu_jobs: deque,
    buffer_jobs: deque,
    waiting_jobs: deque,
    shuffled_jobs: List[Job],
    jobs: List[Job],
    current_second: float
):
    """
    Handle spot instance interruptions by resetting job states.

    Args:
        io_jobs (deque): Queue of I/O jobs.
        cpu_jobs (deque): Queue of CPU jobs.
        buffer_jobs (deque): Queue of buffer jobs.
        waiting_jobs (deque): Queue of waiting jobs.
        shuffled_jobs (List[Job]): List of shuffled jobs.
        jobs (List[Job]): List of all jobs.
        current_second (float): Current simulation second.
    """
    logging.warning("INTERRUPT: Spot instance interruption occurred.")

    # Requeue I/O Jobs
    while io_jobs:
        job = io_jobs.popleft()
        job.reset_progress(current_second)
        jobs.append(job)

    # Requeue CPU Jobs
    while cpu_jobs:
        job = cpu_jobs.popleft()
        job.reset_progress(current_second)
        jobs.append(job)

    # Requeue Buffer Jobs
    while buffer_jobs:
        job = buffer_jobs.popleft()
        job.reset_progress(current_second)
        jobs.append(job)

    # Requeue Waiting Jobs
    while waiting_jobs:
        job = waiting_jobs.popleft()
        job.reset_progress(current_second)
        jobs.append(job)

    # Clear Shuffled Jobs
    shuffled_jobs.clear()


def schedule_jobs(
    architecture: int,
    execution_params: List,
    simulation_params: List,
    output_file: str
) -> Tuple[List, List, float, float, float]:
    """
    Schedule and process jobs based on the provided parameters.

    Args:
        architecture (int): Type of architecture.
        execution_params (List): Execution parameters.
        simulation_params (List): Simulation parameters.
        output_file (str): Path to the output CSV file.

    Returns:
        Tuple[List, List, float, float, float]: Scaling observations, memory tracking,
                                                 average query latency, average query with queueing latency,
                                                 and total price.
    """
    
    # Unpack execution parameters
    scheduling, scaling, nodes, cpu_cores, io_bandwidth, max_jobs, vpu, network_bandwidth, memory_bandwidth, memoryz, cold_start, hit_rate = execution_params

    # Initialize Autoscaler if needed
    autoscaler = Autoscaler(cold_start, second_range) if architecture in [1, 2] else None

    # Initialize job queues and tracking variables
    waiting_jobs = deque()
    io_jobs = deque()
    cpu_jobs = deque()
    buffer_jobs = deque()
    finished_jobs = []
    shuffled_jobs = []

    # Initialize memory tracking
    memory = [0]  # Using list for mutable reference

    # Initialize cost monitoring variables
    money, vpu_charge, slots, slots_charge, base_n, base_cores, interrupt = 0.0, 0.0, 0, 0.0, 0, 0, 0
    spot = 0  # Assuming a flag for spot instances; adjust as needed

    # Initialize DRAM nodes
    dram_nodes = [[] for _ in range(nodes)]
    dram_job_counts = [0] * nodes

    # Initialize job queues
    num_queues = 5
    job_queues = [[] for _ in range(num_queues)]

    if architecture < 2:
        cpu_cores_per_node = cpu_cores // nodes
        base_n = nodes
        sec_money = redshift_persecond_cost(cpu_cores / nodes)
    elif architecture == 2:
        base_cores = cpu_cores
        sec_money = redshift_persecond_cost(cpu_cores / nodes)
    else:
        sec_money = COST_PER_SECOND_REDSHIFT  # Adjust based on architecture

    current_second = 0.0
    job_memory_tiers = {}

    # Determine capacity pricing
    capacity_pricing = 1 if architecture == 4 else 0

    # Initialize tracking lists
    scale_observe = []
    mem_track = []

    # Initialize interrupt countdown
    interrupt_countdown = 0

    # Initialize Jobs from CSV
    jobs = []
    costq = 0.0
    try:
        with open(simulation_params[0], 'r') as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                job = Job(
                    job_id=i,
                    database_id=int(row['database_id']),
                    query_id=int(row['query_id']),
                    start=float(row['start']),
                    cpu_time=float(row['cpu_time']),
                    data_scanned=float(row['data_scanned']),
                    scale_factor=float(row['scale_factor'])
                )
                jobs.append(job)
                costq += job.data_scanned
    except FileNotFoundError:
        logging.error(f"Dataset file '{simulation_params[0]}' not found.")
        return [], [], 0.0, 0.0, 0.0
    except Exception as e:
        logging.error(f"Error reading dataset file '{simulation_params[0]}': {e}")
        return [], [], 0.0, 0.0, 0.0

    # Sort jobs by start time
    jobs.sort(key=lambda job: job.start)

    events = []
    # Seed arrival events
    for job in jobs:
        heapq.heappush(events, Event(job.start, next_event_counter(), job, "arrival"))

    # before the loop
    current_second = 0.0  # in seconds, rounded to 0.1s increments
    previous_second = 0.0
    # Main simulation loop
    while events or jobs or io_jobs or waiting_jobs or buffer_jobs or cpu_jobs:
       
        if len(events) > 0:
            ev = heapq.heappop(events)
            
            raw = ev.time
            
            previous_second = current_second
            
            # Round *up* to the next tenth:
            current_second = math.ceil(raw * 10) / 10

            if(previous_second == current_second and current_second!=0.0):
                continue

            second_range = current_second - previous_second
            # time.sleep(0.5)
            print("We start at", current_second, "with", len(io_jobs), len(cpu_jobs))

        # Calculate per-second cost
        if architecture != 3 or capacity_pricing:
            money, vpu_charge, slots_charge = cost_calculator(
                second_range, architecture, money, nodes, base_n,
                sec_money, vpu, vpu_charge,
                spot, interrupt, slots, slots_charge
            )

        # Schedule I/O jobs
        io_scheduler(
            scheduling, architecture, current_second,
            jobs, waiting_jobs, io_jobs,
            cpu_cores, second_range, num_queues, job_queues
        )

        # Schedule CPU jobs
        cpu_scheduler(
            scheduling, architecture, current_second,
            jobs, buffer_jobs, cpu_jobs,
            cpu_cores, memory, second_range
        )

        # print("After scheduling we have", len(io_jobs), len(cpu_jobs))

        # Logging information about running jobs if required
        # if simulation_params[3] == 1:
        #     df.loc[df["query_id"] == 13, "cpu_time"] *= 0.8(
        #         f"Time {current_second}: I/O Jobs: {len(io_jobs)}, CPU Jobs: {len(cpu_jobs)}, "
        #         f"Buffer Jobs: {len(buffer_jobs)}, Waiting Jobs: {len(waiting_jobs)}"
        #     )

        # Simulate I/O operations based on architecture
        if architecture < 2:
            simulate_io_cached(
                current_second, hit_rate, nodes, io_jobs, io_bandwidth,
                memory_bandwidth, network_bandwidth, buffer_jobs, cpu_jobs,
                finished_jobs, job_memory_tiers, dram_nodes,
                dram_job_counts, second_range, events
            )
        elif architecture == 2:
            simulate_io_elastic_pool(
                current_second, hit_rate, base_cores, cpu_cores, io_jobs,
                io_bandwidth, memory_bandwidth, buffer_jobs, {}, second_range
            )
        else:
            simulate_io_qaas(io_jobs, network_bandwidth,
                             buffer_jobs, second_range, cpu_jobs, finished_jobs)

        # Simulate CPU operations based on architecture
        if architecture < 2:
            simulate_cpu_nodes(
                current_second, cpu_jobs, cpu_cores,
                cpu_cores_per_node, network_bandwidth,
                finished_jobs, shuffled_jobs, io_jobs,
                waiting_jobs, {}, memory, second_range, events
            )
        elif architecture == 2:
            simulate_cpu_autoscaling(
                current_second, cpu_jobs, base_cores,
                cpu_cores, network_bandwidth,
                finished_jobs, shuffled_jobs,
                io_jobs, {}, memory, second_range
            )
        else:
            slots = simulate_cpu_qaas(
                current_second, cpu_jobs, cpu_cores, network_bandwidth,
                finished_jobs, shuffled_jobs, waiting_jobs, io_jobs, {}, memory, second_range
            )
        # Check and perform autoscaling if applicable
        if architecture == 1 and autoscaler:
            nodes, cpu_cores, io_bandwidth, memoryz = autoscaler.autoscaling_dw(
                scaling, cpu_jobs, io_jobs, waiting_jobs, buffer_jobs,
                nodes, cpu_cores_per_node, io_bandwidth, cpu_cores,
                base_n, memoryz, current_second, second_range
            )
            if current_second % 60000 == 0:
                scale_observe.append(nodes)
        elif architecture == 2 and autoscaler:
            vpu, cpu_cores = autoscaler.autoscaling_ec(
                scaling, cpu_jobs, io_jobs, waiting_jobs, buffer_jobs,
                vpu, cpu_cores, base_cores, current_second, second_range
            )
            if current_second % 60000 == 0:
                scale_observe.append(vpu)

        # Track memory usage every 60 seconds
        if current_second % 60000 == 0:
            if memoryz != 0:
                mem_track.append(ceil((memory[0] / memoryz) * 100))

        # Simulate spot instance interruptions
        if spot and interrupt_countdown == 0:
            if random.random() < INTERRUPT_PROBABILITY:
                handle_interrupt(io_jobs, cpu_jobs, buffer_jobs, waiting_jobs,
                                 shuffled_jobs, jobs, current_second)
                interrupt_countdown = INTERRUPT_DURATION
        elif interrupt_countdown > 0:
            interrupt_countdown -= 1


        # Sleep to simulate real-time passage if needed
        time.sleep(simulation_params[1])
    
    # Write finished jobs to CSV
    finished_jobs.sort(key=lambda job: job.start)


    # Calculate total price based on architecture
    if architecture in [0, 1]:
        total_price = money
    elif architecture == 2:
        total_price = (second_range * vpu_charge) / 3600000 * COST_PER_RPU_HOUR
    else:
        if architecture > 2:
            if capacity_pricing == 1:
                total_price = slots_charge * second_range / 3600000 * COST_PER_SLOT_HOUR
            else:
                total_price = qaas_total_cost(costq)
        else:
            total_price = 0.0  # Default case

    write_to_csv(output_file, finished_jobs, total_price)
    logging.info(f"Total Price: {total_price}")

    # Process simulation results
    try:
        df = pd.read_csv(output_file)
    except FileNotFoundError:
        logging.error(f"Output file '{output_file}' not found.")
        return [], [], 0.0, 0.0, 0.0
    except Exception as e:
        logging.error(f"Error reading output file '{output_file}': {e}")
        return [], [], 0.0, 0.0, 0.0

    # Calculate metrics
    metrics = {
        "average_queueing_delay": df['Queueing Delay'].mean(),
        "average_buffer_delay": df['Buffer Delay'].mean(),
        "average_io": df['I/O'].mean(),
        "average_cpu": df['CPU'].mean(),
        "average_shuffle": df['Shuffle'].mean(),
        "average_query_latency": df['query_duration'].mean(),
        "median_query_latency": df['query_duration'].median(),
        "percentile_95_query_latency": np.percentile(df['query_duration'], 95),
        "average_query_wq_latency": df['query_duration_with_queue'].mean(),
        "median_query_wq_latency": df['query_duration_with_queue'].median(),
        "percentile_95_query_wq_latency": np.percentile(df['query_duration_with_queue'], 95)
    }

    # Log metrics
    logging.info(f"Queue: {metrics['average_queueing_delay']} | Buffer: {metrics['average_buffer_delay']}")
    logging.info(f"Mean IO: {metrics['average_io']} | Mean CPU: {metrics['average_cpu']} | Mean Shuffle: {metrics['average_shuffle']}")
    logging.info(f"Mean Query Latency: {metrics['average_query_latency']} | "
                 f"Mean Query (with queueing) Latency: {metrics['average_query_wq_latency']}")
    logging.info(f"Median Query Latency: {metrics['median_query_latency']} | "
                 f"Median Query (with queueing) Latency: {metrics['median_query_wq_latency']}")
    logging.info(f"95th Percentile Query Latency: {metrics['percentile_95_query_latency']} | "
                 f"95th Percentile Query (with queueing) Latency: {metrics['percentile_95_query_wq_latency']}")

    # Optionally, print metrics to console
    print(f"Queue: {metrics['average_queueing_delay']} | Buffer: {metrics['average_buffer_delay']}")
    print(f"Mean IO: {metrics['average_io']} | Mean CPU: {metrics['average_cpu']} | Mean Shuffle: {metrics['average_shuffle']}")
    print(f"Mean Query Latency: {metrics['average_query_latency']} | "
          f"Mean Query (with queueing) Latency: {metrics['average_query_wq_latency']}")
    print(f"Median Query Latency: {metrics['median_query_latency']} | "
          f"Median Query (with queueing) Latency: {metrics['median_query_wq_latency']}")
    print(f"95th Percentile Query Latency: {metrics['percentile_95_query_latency']} | "
          f"95th Percentile Query (with queueing) Latency: {metrics['percentile_95_query_wq_latency']}")

    return scale_observe, mem_track, metrics['average_query_latency'], metrics['average_query_wq_latency'], total_price, metrics['median_query_latency'], metrics['percentile_95_query_latency']

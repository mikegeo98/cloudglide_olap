# simulation_runner.py
import cProfile
import itertools
import logging
from collections import deque
from dataclasses import dataclass, field
import pstats
from cloudglide.execution_model import schedule_jobs
from typing import List, Tuple
import numpy as np
from cloudglide.config import (
    INSTANCE_TYPES,
    SCALE_FACTOR_MIN,
    SCALE_FACTOR_MAX,
    SHUFFLE_PERCENTAGE_MIN,
    SHUFFLE_PERCENTAGE_MAX,
    COST_PER_RPU_HOUR,
    COST_PER_SLOT_HOUR,
    DEFAULT_SLEEP_TIME,
    DEFAULT_MAX_DURATION,
    DATASET_FILES
)

from cloudglide.cost_model import qaas_total_cost
from cloudglide.visual_model import write_to_csv


@dataclass
class Job:
    job_id: int
    database_id: int
    query_id: int
    start: float
    cpu_time: float
    data_scanned: float
    scale_factor: float
    data_shuffle: float = field(init=False)
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    queueing_delay: float = 0.0
    io_time: float = 0.0
    buffer_delay: float = 0.0
    cpu_time_progress: float = field(init=False)
    data_scanned_progress: float = field(init=False)
    shuffle_time: float = 0.0
    processing_time: float = 0.0
    query_exec_time: float = 0.0
    query_exec_time_queueing: float = 0.0
    priority_level: int = 0

    def __post_init__(self):
        self.data_shuffle = self.calculate_data_shuffle()
        self.cpu_time_progress = self.cpu_time
        self.data_scanned_progress = self.data_scanned

    def calculate_data_shuffle(self) -> float:
        if self.scale_factor <= SCALE_FACTOR_MIN:
            return SHUFFLE_PERCENTAGE_MIN * self.data_scanned
        elif self.scale_factor >= SCALE_FACTOR_MAX:
            return SHUFFLE_PERCENTAGE_MAX * self.data_scanned
        else:
            percentage = SHUFFLE_PERCENTAGE_MIN + (
                (self.scale_factor - SCALE_FACTOR_MIN) /
                (SCALE_FACTOR_MAX - SCALE_FACTOR_MIN)
            ) * (SHUFFLE_PERCENTAGE_MAX - SHUFFLE_PERCENTAGE_MIN)
            return percentage * self.data_scanned

    def reset_progress(self, current_second: float):
        self.cpu_time_progress = self.cpu_time
        self.data_scanned_progress = self.data_scanned
        self.start = current_second * 1000 + 60000


def get_instance_config(instance_index: int, nodes: int) -> dict:
    if 0 <= instance_index < len(INSTANCE_TYPES):
        config = INSTANCE_TYPES[instance_index]
        # Scale configurations based on the number of nodes
        scaled_config = {
            "cpu_cores": config.cpu_cores * nodes,
            "memory": config.memory * nodes,  # in GB
            "io_bandwidth": config.io_bandwidth * nodes,  # in Mbps
            # Assuming network bandwidth is shared or fixed
            "network_bandwidth": config.network_bandwidth,
            # Assuming memory bandwidth is shared or fixed
            "memory_bandwidth": config.memory_bandwidth
        }
        return scaled_config
    else:
        raise ValueError(f"Invalid instance index: {instance_index}")


def configure_execution_params(
    architecture: int,
    scheduling: int,
    nodes: int,
    vpu: int,
    scaling: int,
    cold_start: bool,
    hit_rate: float,
    instance: int,
    arrival_rate: float,
    network_bandwidth: int,
    io_bandwidth: int,
    memory_bandwidth: int
) -> List:
    """
    Configure execution parameters based on the architecture type.
    """
    if architecture < 2:
        config = get_instance_config(instance, nodes)
        cpu_cores = config["cpu_cores"]
        memory = config["memory"] * 1024  # Convert GB to MB
        io_bw = config["io_bandwidth"]
        memory_bw = config["memory_bandwidth"]
        network_bw = config["network_bandwidth"]

        max_jobs = cpu_cores
        vpu = 0  # Virtual Processing Unit not used in this architecture
        execution_params = [
            scheduling,
            scaling,
            nodes,
            cpu_cores,
            io_bw,
            max_jobs,
            vpu,
            network_bw,
            memory_bw,
            memory,
            cold_start,
            hit_rate
        ]

    elif architecture == 2:
        max_jobs = 100
        cpu_cores = 2 * vpu
        memory = 4 * vpu * 1024  # Convert to MB
        network_bw = network_bandwidth * 1000
        memory_bw = 50000
        io_bw = int(io_bandwidth * vpu / 4)  # Scale based on VPUs
        execution_params = [
            scheduling,
            scaling,
            cpu_cores,
            cpu_cores,
            io_bw,
            max_jobs,
            vpu,
            network_bw,
            memory_bw,
            memory,
            cold_start,
            hit_rate
        ]

    else:
        max_jobs = 100000
        cpu_cores = 0
        memory = 0
        io_bw = 0
        vpu = 0
        network_bw = network_bandwidth * 1000
        memory_bw = 50000
        execution_params = [
            scheduling,
            scaling,
            0,
            0,
            0,
            max_jobs,
            0,
            network_bw,
            memory_bw,
            memory,
            cold_start,
            hit_rate
        ]

    return execution_params


def configure_simulation_params(ds_idx: int) -> List:
    """
    Configure simulation parameters based on the dataset index.
    """
    dataset = DATASET_FILES.get(ds_idx, 'datasets/test.csv')
    sleep_time = DEFAULT_SLEEP_TIME
    max_duration = DEFAULT_MAX_DURATION  # in seconds
    logging_flag = 0
    return [dataset, sleep_time, max_duration, logging_flag]


def handle_interrupt(
    io_jobs: deque,
    cpu_jobs: deque,
    buffer_jobs: deque,
    waiting_jobs: deque,
    shuffled_jobs: List,
    jobs: List[Job],
    current_second: float
):
    """
    Handle spot instance interruptions by resetting job states.
    """
    logging.warning("INTERRUPT: Spot instance interruption occurred.")

    # Requeue I/O Jobs
    for job in list(io_jobs):
        io_jobs.remove(job)
        job.data_scanned_progress = job.data_scanned
        job.start = current_second * 1000 + 60000
        jobs.append(job)

    # Requeue CPU Jobs
    for job in list(cpu_jobs):
        cpu_jobs.remove(job)
        job.cpu_time_progress = job.cpu_time
        job.data_scanned_progress = job.data_scanned
        job.start = current_second * 1000 + 60000
        jobs.append(job)

    # Requeue Buffer Jobs
    for job in list(buffer_jobs):
        buffer_jobs.remove(job)
        job.data_scanned_progress = job.data_scanned
        job.start = current_second * 1000 + 60000
        jobs.append(job)

    # Requeue Waiting Jobs
    for job in list(waiting_jobs):
        waiting_jobs.remove(job)
        job.start = current_second * 1000 + 60000
        jobs.append(job)

    # Clear Shuffled Jobs
    shuffled_jobs.clear()

    # Additional Reset Logic if Necessary
    # ...


def run_simulation(
    architecture: List[int],
    scheduling: List[int],
    nodes: List[int],
    vpu: List[int],
    scaling: List[int],
    cold_starts: List[bool],
    hit_rate: List[float],
    instance: List[int],
    arrival_rate: List[float],
    network_bandwidth: List[int],
    io_bandwidth: List[int],
    memory_bandwidth: List[int],
    dataset_index: List[int],
    output_prefix: str = "output_simulation/simulation"
) -> Tuple[List[str], List[List], List[List], List[List]]:
    """
    Run simulations for all parameter combinations.
    """
    all_configs = itertools.product(
        architecture,
        scheduling,
        nodes,
        vpu,
        scaling,
        cold_starts,
        hit_rate,
        instance,
        arrival_rate,
        network_bandwidth,
        io_bandwidth,
        memory_bandwidth,
        dataset_index
    )
    output_files = []
    scalings = []
    memories = []
    results = []
    file_counter = 1

    for config in all_configs:
        arch, sched, node, v, sc, cs, hr, inst, ar, nb, iob, mb, ds_idx = config
        exec_params = configure_execution_params(
            arch, sched, node, v, sc, cs, hr, inst, ar, nb, iob, mb)
        sim_params = configure_simulation_params(ds_idx)

        output_file_path = f"{output_prefix}_{file_counter}.csv"
        output_files.append(output_file_path)
        file_counter += 1

        logging.info(
            f"---------EXECUTION---------\n"
            f"Architecture: {arch}, Scheduling: {sched}, Nodes: {node}, VPU: {v}, Scaling: {sc}, "
            f"Cold Starts: {cs}, Hit Rate: {hr}, Instance: {inst}, Arrival Rate: {ar}, "
            f"Network Bandwidth: {nb}, I/O Bandwidth: {iob}, Memory Bandwidth: {mb}, Dataset: {ds_idx}"
        )
        
        pr = cProfile.Profile()
        pr.enable()

        scales, mems, lat, lat_queue, money, med, ninetyfive = schedule_jobs(
            arch, exec_params, sim_params, output_file_path)
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('tottime')
        ps.print_stats(20)

        scalings.append(scales)
        memories.append(mems)
        results.append([lat, lat_queue, money, med, ninetyfive])

    return output_files, scalings, memories, results


def calculate_metrics(df) -> dict:
    """
    Calculate various performance metrics from the simulation results.
    """
    return {
        "average_queueing_delay": df['Queueing Delay'].mean(),
        "average_buffer_delay": df['Buffer Delay'].mean(),
        "average_io": df['I/O'].mean(),
        "average_cpu": df['CPU'].mean(),
        "average_shuffle": df['Shuffle'].mean(),
        "average_query_latency": df['query_duration'].mean(),
        "average_query_wq_latency": df['query_duration_with_queue'].mean(),
        "median_query_latency": df['query_duration'].median(),
        "median_query_wq_latency": df['query_duration_with_queue'].median(),
        "percentile_95_query_latency": np.percentile(df['query_duration'], 95),
        "percentile_95_query_wq_latency": np.percentile(df['query_duration_with_queue'], 95)
    }


def print_metrics(metrics: dict):
    """
    Print the calculated performance metrics.
    """
    logging.info(
        f"Queue: {metrics['average_queueing_delay']} Buffer: {metrics['average_buffer_delay']}")
    logging.info(
        f"Mean IO: {metrics['average_io']} Mean CPU: {metrics['average_cpu']} Mean Shuffle: {metrics['average_shuffle']}")
    logging.info(f"Mean Query Latency: {metrics['average_query_latency']} "
                 f"Mean Query (with queueing) Latency: {metrics['average_query_wq_latency']}")
    logging.info(f"Median Query Latency: {metrics['median_query_latency']} "
                 f"Median Query (with queueing) Latency: {metrics['median_query_wq_latency']}")
    logging.info(f"95th Percentile Query Latency: {metrics['percentile_95_query_latency']} "
                 f"95th Percentile Query (with queueing) Latency: {metrics['percentile_95_query_wq_latency']}")


def finalize_simulation(
    architecture: int,
    costq: float,
    second_range: int,
    vpu_charge: float,
    slots_charge: float,
    output_file: str,
    finished_jobs: List[Job],
    scale_observe: List,
    mem_track: List
) -> Tuple[List, List, float]:
    """
    Finalize the simulation by calculating total price and processing results.
    """
    write_to_csv(output_file, sorted(
        finished_jobs, key=lambda job: job.start_timestamp))

    # Calculate total price based on architecture
    if architecture in [0, 1]:
        total_price = costq  # Assuming 'money' is equivalent to 'costq' for these architectures
    elif architecture == 2:
        total_price = (second_range * vpu_charge) / 3600 * COST_PER_RPU_HOUR
    else:
        if architecture > 2:
            if architecture == 4:
                total_price = slots_charge * second_range / 3600 * COST_PER_SLOT_HOUR
            else:
                total_price = qaas_total_cost(costq)
        else:
            total_price = 0  # Default case

    logging.info(f"Total Price: {total_price}")

    # Process CSV results
    import pandas as pd
    df = pd.read_csv(output_file)

    metrics = calculate_metrics(df)
    print_metrics(metrics)

    return scale_observe, mem_track, total_price

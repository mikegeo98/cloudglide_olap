# simulation_runner.py
import cProfile
import itertools
import logging
from collections import deque
from dataclasses import dataclass, field
import os
import pstats

import psutil
from cloudglide.execution_model import schedule_jobs
from typing import List, Tuple
import numpy as np
from cloudglide.config import DATASET_FILES, ArchitectureType, SimulationConfig
from cloudglide.config import (
    INSTANCE_TYPES,
)

from cloudglide.cost_model import qaas_total_cost
from cloudglide.visual_model import write_to_csv


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


def configure_simulation_params(ds_idx: int, config) -> List:
    """
    Configure simulation parameters based on the dataset index.
    """
    dataset = DATASET_FILES.get(ds_idx, 'datasets/test.csv')
    sleep_time = config.default_sleep_time
    max_duration = config.default_max_duration  # in seconds
    logging_flag = 0
    return [dataset, sleep_time, max_duration, logging_flag]



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
    output_prefix: str
) -> Tuple[List[str], List[List], List[List], List[List]]:
    """
    Run simulations for all parameter combinations.
    """
    all_params = itertools.product(
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

    for param in all_params:
        config = SimulationConfig()
        arch, sched, node, v, sc, cs, hr, inst, ar, nb, iob, mb, ds_idx = param
        exec_params = configure_execution_params(
            arch, sched, node, v, sc, cs, hr, inst, ar, nb, iob, mb)
        sim_params = configure_simulation_params(ds_idx, config)

        output_file_path = f"{output_prefix}_{file_counter}.csv"
        output_files.append(output_file_path)
        file_counter += 1

        logging.info(
            f"---------EXECUTION---------\n"
            f"Architecture: {arch}, Scheduling: {sched}, Nodes: {node}, VPU: {v}, Scaling: {sc}, "
            f"Cold Starts: {cs}, Hit Rate: {hr}, Instance: {inst}, Arrival Rate: {ar}, "
            f"Network Bandwidth: {nb}, I/O Bandwidth: {iob}, Memory Bandwidth: {mb}, Dataset: {ds_idx}"
        )

        proc = psutil.Process(os.getpid())
        before = proc.memory_info().rss
        
        arch_enum = ArchitectureType(arch) 
        
        scales, mems, lat, lat_queue, money, med, ninetyfive = schedule_jobs(
            arch_enum, exec_params, sim_params, output_file_path, config
        )
        
        after  = proc.memory_info().rss
        print(f"RSS before: {before/1024**2:.1f} MiB")
        print(f"RSS after:  {after/1024**2:.1f} MiB")
        print(f"Delta:      {(after-before)/1024**2:.1f} MiB")

        scalings.append(scales)
        memories.append(mems)
        results.append([lat, lat_queue, money, med, ninetyfive])

    return output_files, scalings, memories, results

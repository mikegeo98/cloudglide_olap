# simulation_runner.py
import logging
from dataclasses import dataclass
import os

import psutil
from cloudglide.execution_model import schedule_jobs
from typing import List, Tuple
from cloudglide.config import (
    DATASET_FILES,
    ArchitectureType,
    SimulationConfig,
    ExecutionParams,
    SimulationParams,
    INSTANCE_TYPES,
)

from copy import deepcopy


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
    network_bandwidth: int,
    io_bandwidth: int,
    memory_bandwidth: int,
    cpu_cores_override: int = None
) -> ExecutionParams:
    """
    Configure execution parameters based on the architecture type.

    Args:
        architecture: Architecture type (0=DWAAS, 1=DWAAS_AUTOSCALING, 2=ELASTIC_POOL, 3+=QAAS)
        scheduling: Scheduling policy
        nodes: Number of nodes (used for DWaaS, ignored for EP/QaaS)
        vpu: Virtual Processing Units (used for EP only)
        scaling: Scaling policy
        cold_start: Cold start delay
        hit_rate: Cache hit rate
        instance: Instance type index (used for DWaaS only)
        network_bandwidth: Network bandwidth in Gbps
        io_bandwidth: I/O bandwidth (per node for DWaaS, per VPU for EP)
        memory_bandwidth: Memory bandwidth in Mbps
        cpu_cores_override: Optional override for cpu_cores (DWaaS only)

    Returns:
        ExecutionParams configured for the specified architecture
    """
    # DWaaS and DWaaS_AUTOSCALING (architecture 0 and 1)
    if architecture < 2:
        config = get_instance_config(instance, nodes)

        # Allow cpu_cores override, otherwise use calculated value
        if cpu_cores_override is not None:
            cpu_cores = cpu_cores_override
        else:
            cpu_cores = config["cpu_cores"]

        memory = config["memory"] * 1024  # Convert GB to MB
        io_bw = config["io_bandwidth"]
        memory_bw = config["memory_bandwidth"]
        network_bw = config["network_bandwidth"]

        max_jobs = cpu_cores
        vpu_param = 0  # Virtual Processing Unit not used in this architecture
        return ExecutionParams(
            scheduling=scheduling,
            scaling=scaling,
            nodes=nodes,
            cpu_cores=cpu_cores,
            io_bw=io_bw,
            max_jobs=max_jobs,
            vpu=vpu_param,
            network_bw=network_bw,
            memory_bw=memory_bw,
            total_memory_capacity_mb=memory,
            cold_start=cold_start,
            hit_rate=hit_rate
        )

    # Elastic Pool (architecture 2)
    elif architecture == 2:
        max_jobs = 100
        cpu_cores = 2 * vpu
        memory = 4 * vpu * 1024  # Convert to MB
        network_bw = network_bandwidth * 1000
        memory_bw = 50000
        io_bw = int(io_bandwidth * vpu / 4)  # Scale based on VPUs

        # Calculate nodes internally based on cpu_cores for memory distribution
        # User no longer needs to specify nodes for EP
        nodes_internal = cpu_cores

        return ExecutionParams(
            scheduling=scheduling,
            scaling=scaling,
            nodes=nodes_internal,
            cpu_cores=cpu_cores,
            io_bw=io_bw,
            max_jobs=max_jobs,
            vpu=vpu,
            network_bw=network_bw,
            memory_bw=memory_bw,
            total_memory_capacity_mb=memory,
            cold_start=cold_start,
            hit_rate=hit_rate
        )

    # QaaS (architecture 3+)
    else:
        max_jobs = 100000
        cpu_cores = 0
        memory = 0
        io_bw = 0
        vpu_param = 0
        network_bw = network_bandwidth * 1000
        memory_bw = 50000

        # QaaS has no persistent infrastructure, so force these to 0
        nodes_internal = 0
        hit_rate_internal = 0.0  # No persistent cache
        cold_start_internal = 0.0  # No cold start delay

        return ExecutionParams(
            scheduling=scheduling,
            scaling=scaling,
            nodes=nodes_internal,
            cpu_cores=cpu_cores,
            io_bw=io_bw,
            max_jobs=max_jobs,
            vpu=vpu_param,
            network_bw=network_bw,
            memory_bw=memory_bw,
            total_memory_capacity_mb=memory,
            cold_start=cold_start_internal,
            hit_rate=hit_rate_internal
        )


def configure_simulation_params(ds_idx: int, config) -> SimulationParams:
    """
    Configure simulation parameters based on the dataset index.
    """
    dataset = DATASET_FILES.get(ds_idx, 'datasets/test.csv')
    return SimulationParams(dataset_path=dataset)



@dataclass
class SimulationRun:
    name: str
    architecture: ArchitectureType
    scheduling_policy: int
    nodes: int
    vpu: int
    scaling_policy: int
    cold_start: float
    hit_rate: float
    instance: int
    network_bandwidth: int
    io_bandwidth: int
    memory_bandwidth: int
    dataset_index: int
    config: SimulationConfig
    cpu_cores: int = None  # Optional override for DWaaS architectures


def run_simulation(
    runs: List[SimulationRun],
    output_prefix: str
) -> Tuple[List[str], List[List], List[List], List[List]]:
    """
    Run simulations for pre-expanded run configurations.
    """
    output_files = []
    scalings = []
    memories = []
    results = []
    file_counter = 1

    for run in runs:
        config = run.config.copy()
        arch = run.architecture.value if isinstance(run.architecture, ArchitectureType) else run.architecture
        sched = run.scheduling_policy
        node = run.nodes
        v = run.vpu
        sc = run.scaling_policy
        cs = run.cold_start
        hr = run.hit_rate
        inst = run.instance
        nb = run.network_bandwidth
        iob = run.io_bandwidth
        mb = run.memory_bandwidth
        ds_idx = run.dataset_index
        cpu_cores_override = run.cpu_cores
        exec_params = configure_execution_params(
            arch, sched, node, v, sc, cs, hr, inst, nb, iob, mb, cpu_cores_override)
        sim_params = configure_simulation_params(ds_idx, config)

        output_file_path = f"{output_prefix}_{file_counter}.csv"
        output_files.append(output_file_path)
        file_counter += 1

        logging.info(
            f"---------EXECUTION ({run.name})---------\n"
            f"Architecture: {arch}, Scheduling: {sched}, Nodes: {node}, VPU: {v}, Scaling: {sc}, "
            f"Cold Starts: {cs}, Hit Rate: {hr}, Instance: {inst}, "
            f"Network Bandwidth: {nb}, I/O Bandwidth: {iob}, Memory Bandwidth: {mb}, Dataset: {ds_idx}"
        )

        proc = psutil.Process(os.getpid())
        before = proc.memory_info().rss

        arch_enum = ArchitectureType(arch) if not isinstance(run.architecture, ArchitectureType) else run.architecture

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

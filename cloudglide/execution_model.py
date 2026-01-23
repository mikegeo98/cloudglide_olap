import heapq
import math
import logging
import random
from collections import deque
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd

from cloudglide.event import Event, next_event_counter
from cloudglide.cost_model import (
    cost_calculator,
    redshift_persecond_cost,
    qaas_total_cost,
    faas_cost,
)
from cloudglide.scaling_model import Autoscaler
from cloudglide.scheduling_model import io_scheduler, cpu_scheduler
from cloudglide.query_processing_model import simulate_io, simulate_cpu
from cloudglide.visual_model import load_jobs_from_csv, write_to_csv
from cloudglide.job import Job
from cloudglide.config import ArchitectureType, SimulationConfig, ExecutionParams, SimulationParams
from cloudglide.interrupt_model import handle_interrupt

# ----------------------------
# Initialization Helpers
# ----------------------------


def initialize_state(
    architecture: int,
    execution_params: ExecutionParams,
    simulation_params: SimulationParams,
    config: SimulationConfig
):
    """
    Initialize simulation state with structured parameters.

    Args:
        architecture: Architecture type identifier
        execution_params: Structured execution parameters
        simulation_params: Structured simulation parameters
        config: Simulation configuration

    Returns:
        Tuple of initialized state components
    """
    scheduling = execution_params.scheduling
    scaling = execution_params.scaling
    nodes = execution_params.nodes
    cpu_cores = execution_params.cpu_cores
    io_bw = execution_params.io_bw
    vpu = execution_params.vpu
    network_bw = execution_params.network_bw
    memory_bw = execution_params.memory_bw
    total_memory_capacity_mb = execution_params.total_memory_capacity_mb
    cold_start = execution_params.cold_start
    hit_rate = execution_params.hit_rate

    autoscaler = (
        Autoscaler(cold_start, config.scaling_options)
        if architecture in [
            ArchitectureType.DWAAS_AUTOSCALING,
            ArchitectureType.ELASTIC_POOL
        ]
        else None
    )
    state = {
        "waiting_jobs": deque(),
        "io_jobs": deque(),
        "cpu_jobs": deque(),
        "buffer_jobs": deque(),
        "finished_jobs": [],
        "shuffled_jobs": [],
        "memory": [0],
        "accumulated_cost_usd": 0.0,
        "vpu_charge": 0.0,
        "accumulated_slot_hours": 0.0,
        "base_n": nodes,
        "base_cores": cpu_cores,
        "interrupt_countdown": 0,
        "scale_observe": [],
        "mem_track": [],
        "dram_nodes": [[] for _ in range(nodes)],
        "dram_job_counts": [0] * nodes,
        "job_memory_tiers": {},
        "faas_gb_seconds": [0.0],
    }

    # Initial cost model
    if architecture in [
        ArchitectureType.DWAAS,
        ArchitectureType.DWAAS_AUTOSCALING
    ]:
        cpu_cores_per_node = cpu_cores // nodes
        cost_per_sec_usd = redshift_persecond_cost(cpu_cores / nodes)
    elif architecture == ArchitectureType.ELASTIC_POOL:
        cpu_cores_per_node = 4
        cost_per_sec_usd = redshift_persecond_cost(cpu_cores / nodes)
    else:
        cpu_cores_per_node = 0
        cost_per_sec_usd = config.cost_per_second_redshift

    # Load jobs from CSV
    jobs, workload_data_scanned_mb = load_jobs_from_csv(
        simulation_params.dataset_path
    )

    # Seed arrival events
    events = [
        Event(job.start, next_event_counter(), job, "arrival")
        for job in jobs
    ]
    heapq.heapify(events)

    return (
        state, autoscaler, jobs, workload_data_scanned_mb,
        cpu_cores_per_node, cost_per_sec_usd, nodes, cpu_cores, io_bw,
        vpu, network_bw, memory_bw, total_memory_capacity_mb,
        scaling, scheduling, hit_rate, cold_start, events
    )


# ----------------------------
# Core Logic
# ----------------------------


def charge_costs(
    state: Dict,
    architecture: int,
    dt: float,
    nodes: int,
    base_n: int,
    cost_per_sec_usd: float,
    vpu: int,
    slots: int,
    spot: int,
    interrupt: int
) -> None:
    """
    Calculate and accumulate costs for the current timestep.

    Args:
        state: Simulation state dictionary
        architecture: Architecture type
        dt: Time delta for this step
        nodes: Current number of nodes
        base_n: Base number of nodes
        cost_per_sec_usd: Cost per second in USD
        vpu: Virtual processing units
        slots: Number of slots
        spot: Spot instance flag (0 or 1)
        interrupt: Interrupt flag (0 or 1)
    """
    # QaaS handled separately
    if architecture not in [
        ArchitectureType.QAAS,
        ArchitectureType.QAAS_CAPACITY,
        ArchitectureType.FAAS,
    ]:
        result = cost_calculator(
            dt, architecture, state["accumulated_cost_usd"], nodes,
            base_n, cost_per_sec_usd, vpu, state["vpu_charge"],
            spot, interrupt, slots, state["accumulated_slot_hours"]
        )
        (
            state["accumulated_cost_usd"],
            state["vpu_charge"],
            state["accumulated_slot_hours"]
        ) = result


def maybe_autoscale(architecture, autoscaler, scaling, state, nodes, cpu_cores_per_node,
                    io_bw, cpu_cores, base_n, total_memory_capacity_mb, now, dt, events):
    if not autoscaler:
        return nodes, cpu_cores, io_bw, total_memory_capacity_mb, 0

    if architecture == ArchitectureType.DWAAS_AUTOSCALING:
        nodes, cpu_cores, io_bw, total_memory_capacity_mb = autoscaler.autoscaling_dw(
            scaling, state["cpu_jobs"], state["io_jobs"],
            state["waiting_jobs"], state["buffer_jobs"],
            nodes, cpu_cores_per_node, io_bw, cpu_cores, base_n,
            total_memory_capacity_mb, now, dt, events
        )
        return nodes, cpu_cores, io_bw, total_memory_capacity_mb, nodes

    elif architecture == ArchitectureType.ELASTIC_POOL:
        vpu, cpu_cores = autoscaler.autoscaling_ec(
            scaling, state["cpu_jobs"], state["io_jobs"],
            state["waiting_jobs"], state["buffer_jobs"],
            0, cpu_cores, base_n, now, dt, events
        )
        return nodes, cpu_cores, io_bw, total_memory_capacity_mb, vpu

    return nodes, cpu_cores, io_bw, total_memory_capacity_mb, 0


def process_event(ev, state, now, phase):
    job = ev.job

    if phase == "arrival":
        state["waiting_jobs"].append(job)
        state["buffer_jobs"].append(job)
        return

    pass


def maybe_interrupt(state, jobs, current_second, spot, config):
    if not spot:
        return
    if state["interrupt_countdown"] == 0 and random.random() < config.interrupt_probability:
        handle_interrupt(
            state["io_jobs"], state["cpu_jobs"],
            state["buffer_jobs"], state["waiting_jobs"],
            state["shuffled_jobs"], jobs, current_second, config
        )
        state["interrupt_countdown"] = config.interrupt_duration
        logging.warning("Spot instance interruption occurred.")
    elif state["interrupt_countdown"] > 0:
        state["interrupt_countdown"] -= 1


# ----------------------------
# Finalization
# ----------------------------
def _default_metrics(total_price: float) -> Dict[str, Any]:
    """Return default metrics when no jobs finished."""
    return {
        "average_queueing_delay": 0.0,
        "average_buffer_delay": 0.0,
        "average_io": 0.0,
        "average_cpu": 0.0,
        "average_shuffle": 0.0,
        "average_query_latency": 0.0,
        "median_query_latency": 0.0,
        "percentile_95_query_latency": 0.0,
        "average_query_wq_latency": 0.0,
        "median_query_wq_latency": 0.0,
        "percentile_95_query_wq_latency": 0.0,
        "total_price": total_price
    }


def collect_metrics(output_file: str, finished_jobs: List[Job], total_price: float):
    """
    Finalize the simulation by writing results to CSV, reading them back,
    and computing + logging all performance metrics.
    """

    # Write finished jobs to CSV
    write_to_csv(output_file, finished_jobs, total_price)

    # Load results
    try:
        df = pd.read_csv(output_file)
        if df.empty:
            logging.warning(f"Output file '{output_file}' is empty - no finished jobs.")
            return _default_metrics(total_price)
    except FileNotFoundError:
        logging.error(f"Output file '{output_file}' not found.")
        return _default_metrics(total_price)
    except Exception as e:
        logging.error(f"Error reading output file '{output_file}': {e}")
        return _default_metrics(total_price)

    # Compute all metrics (preserve your original ones)
    metrics = {
        "average_queueing_delay": df["queueing_delay"].mean(),
        "average_buffer_delay": df["buffer_delay"].mean(),
        "average_io": df["io"].mean(),
        "average_cpu": df["cpu"].mean(),
        "average_shuffle": df["shuffle"].mean(),
        "average_query_latency": df["query_duration"].mean(),
        "median_query_latency": df["query_duration"].median(),
        "percentile_95_query_latency": np.percentile(df["query_duration"], 95),
        "average_query_wq_latency": df["query_duration_with_queue"].mean(),
        "median_query_wq_latency": df["query_duration_with_queue"].median(),
        "percentile_95_query_wq_latency": np.percentile(df["query_duration_with_queue"], 95),
        "total_price": total_price
    }

    # --- Logging section (identical in spirit to your original) ---
    logging.info("==== Simulation Metrics ====")
    logging.info(f"Queue Delay Avg: {metrics['average_queueing_delay']:.4f} | "
                 f"Buffer Delay Avg: {metrics['average_buffer_delay']:.4f}")
    logging.info(f"Mean I/O: {metrics['average_io']:.4f} | "
                 f"Mean CPU: {metrics['average_cpu']:.4f} | "
                 f"Mean Shuffle: {metrics['average_shuffle']:.4f}")
    logging.info(f"Mean Query Latency: {metrics['average_query_latency']:.4f} | "
                 f"Mean Query (with queueing): {metrics['average_query_wq_latency']:.4f}")
    logging.info(f"Median Query Latency: {metrics['median_query_latency']:.4f} | "
                 f"Median Query (with queueing): {metrics['median_query_wq_latency']:.4f}")
    logging.info(f"95th Percentile Query Latency: {metrics['percentile_95_query_latency']:.4f} | "
                 f"95th Percentile Query (with queueing): {metrics['percentile_95_query_wq_latency']:.4f}")
    logging.info(f"Total Price: ${metrics['total_price']:.4f}")
    logging.info("=============================")

    return metrics

# ----------------------------
# Main Entry Point
# ----------------------------
def schedule_jobs(
    architecture: ArchitectureType,
    execution_params: List,
    simulation_params: List,
    output_file: str,
    config: SimulationConfig,
):
    (
        state, autoscaler, jobs, workload_data_scanned_mb, cpu_cores_per_node, cost_per_sec_usd,
        nodes, cpu_cores, io_bw, vpu, network_bw, memory_bw, total_memory_capacity_mb,
        scaling, scheduling, hit_rate, cold_start, events
    ) = initialize_state(architecture, execution_params, simulation_params, config)

    current_second, prev_second = 0.0, 0.0
    spot = 1 if config.use_spot_instances else 0
    capacity_pricing = 1 if architecture == ArchitectureType.QAAS_CAPACITY else 0
    slots = 0

    while events:
        ev = heapq.heappop(events)
        now = ev.time
        phase = ev.etype

        process_event(ev, state, now, phase)

        prev_second, current_second = current_second, math.ceil(now * 10) / 10
        if current_second == prev_second:
            continue

        dt = current_second - prev_second
        charge_costs(state, architecture, dt, nodes, state["base_n"], cost_per_sec_usd, vpu, slots, spot, 0)

        # Autoscaling
        nodes, cpu_cores, io_bw, total_memory_capacity_mb, current_scale_level = maybe_autoscale(
            architecture, autoscaler, scaling, state, nodes,
            cpu_cores_per_node, io_bw, cpu_cores, state["base_n"], total_memory_capacity_mb,
            current_second, dt, events
        )
        if current_scale_level and current_second % 60 == 0:
            state["scale_observe"].append(current_scale_level)

        # Scheduling
        io_scheduler(
            scheduling,
            architecture,
            current_second,
            dt,
            jobs,
            state["waiting_jobs"],
            state["io_jobs"],
            cpu_cores,
            phase,
            config.scheduling_options,
            config,
            state["cpu_jobs"],
        )

        cpu_scheduler(
            scheduling,
            architecture,
            current_second,
            jobs,
            state["buffer_jobs"],
            state["cpu_jobs"],
            cpu_cores,
            state["memory"],
            dt,
            phase,
            config,
            config.scheduling_options,
        )

        # Simulations
        simulate_io(current_second, hit_rate, nodes, state["io_jobs"], io_bw,
                    memory_bw, phase, state["buffer_jobs"], state["cpu_jobs"],
                    state["finished_jobs"], state["job_memory_tiers"],
                    state["dram_nodes"], state["dram_job_counts"], dt, events, architecture, config)

        slots = simulate_cpu(current_second, state["cpu_jobs"], phase, cpu_cores,
                             cpu_cores_per_node, network_bw, state["finished_jobs"],
                             state["shuffled_jobs"], state["io_jobs"], state["waiting_jobs"],
                             {}, state["memory"], dt, events, architecture, config,
                             faas_gb_seconds=state.get("faas_gb_seconds"))

        # Interrupts
        maybe_interrupt(state, jobs, current_second, spot, config)

        # Track memory every 60s
        if current_second % 60 == 0 and total_memory_capacity_mb > 0:
            usage = math.ceil((state["memory"][0] / total_memory_capacity_mb) * 100)
            state["mem_track"].append(usage)

    # Final pricing
    if architecture in [ArchitectureType.DWAAS, ArchitectureType.DWAAS_AUTOSCALING]:
        total_price = state["accumulated_cost_usd"]
    elif architecture == ArchitectureType.ELASTIC_POOL:
        total_price = (state["vpu_charge"] / 3600) * config.cost_per_rpu_hour
    elif architecture in [ArchitectureType.QAAS, ArchitectureType.QAAS_CAPACITY]:
        total_price = (
            state["accumulated_slot_hours"] / 3600 * config.cost_per_slot_hour
            if capacity_pricing else qaas_total_cost(workload_data_scanned_mb, config.qaas_cost_per_tb)
        )
    elif architecture == ArchitectureType.FAAS:
        total_price = faas_cost(
            len(state["finished_jobs"]),
            state["faas_gb_seconds"][0],
            config.faas_cost_per_invocation,
            config.faas_cost_per_gb_second,
        )
    else:
        total_price = 0.0

    if not state["finished_jobs"]:
        logging.warning("⚠️ No finished jobs detected. Simulation likely not completing any I/O or CPU.")


    finished_jobs = sorted(state["finished_jobs"], key=lambda j: j.start)
    metrics = collect_metrics(output_file, finished_jobs, total_price)

    return (
        state["scale_observe"],
        state["mem_track"],
        metrics["average_query_latency"],
        metrics["average_query_wq_latency"],
        total_price,
        metrics["median_query_latency"],
        metrics["percentile_95_query_latency"],
    )

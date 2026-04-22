import logging
from typing import Tuple
from cloudglide.config import ArchitectureType

def cost_calculator(
    seconds_range: float,
    architecture: int,
    accumulated_cost_usd: float,
    active_nodes: int,
    base_nodes: int,
    cost_per_sec_usd: float,
    vpu_count: int,
    accumulated_vpu_hours: float,
    is_spot_instance: bool,
    is_interrupting: bool,
    active_slots: int,
    accumulated_slot_hours: float,
) -> Tuple[float, float, float]:


    # --- DWaaS / Elastic Pool architectures ---
    if architecture in [ArchitectureType.DWAAS, ArchitectureType.DWAAS_AUTOSCALING]:
        if is_spot_instance and not is_interrupting:
            # Spot instance discount (50%)
            discounted_cost_per_sec = cost_per_sec_usd * 0.5
            accumulated_cost_usd += seconds_range * active_nodes * discounted_cost_per_sec/1000
            logging.debug(f"Spot instance pricing applied: {discounted_cost_per_sec:.6f}/s per node")
        else:
            # Base + scaled nodes cost
            accumulated_cost_usd += (
                seconds_range * base_nodes * cost_per_sec_usd
                + seconds_range * (active_nodes - base_nodes) * cost_per_sec_usd
            )/1000

    # --- Elastic Compute (Architecture 2) ---
    elif architecture == ArchitectureType.ELASTIC_POOL:
        accumulated_vpu_hours += vpu_count

    # --- Serverless / Capacity Pricing (Architecture 4) ---
    elif architecture == ArchitectureType.QAAS_CAPACITY:
        accumulated_slot_hours += active_slots

    return accumulated_cost_usd, accumulated_vpu_hours, accumulated_slot_hours


def redshift_persecond_cost(vcores: float) -> float:

    return (vcores * 0.00030166666) / 4


def faas_cost(
    num_invocations: int,
    total_gb_seconds: float,
    cost_per_invocation: float,
    cost_per_gb_second: float,
) -> float:
    """
    Calculate total FaaS cost using the hybrid pricing model.

    FaaS pricing has two components:
    1. Per-invocation fee: Fixed cost per function invocation
       (e.g., $0.20 per 1M requests on AWS Lambda)
    2. Duration-based fee: Cost per GB-second of compute time
       (e.g., $0.0000166667 per GB-second on AWS Lambda)

    Implementation requirements:
    ----------
    Compute and return the total cost as:
        invocation_cost = num_invocations * cost_per_invocation
        duration_cost   = total_gb_seconds * cost_per_gb_second
        total_cost      = invocation_cost + duration_cost

    Args:
        num_invocations: Total number of function invocations (= finished jobs)
        total_gb_seconds: Total GB-seconds of compute consumed
        cost_per_invocation: Cost per single invocation
        cost_per_gb_second: Cost per GB-second

    Returns:
        Total cost in USD.
    """
    invocation_cost = num_invocations * cost_per_invocation
    duration_cost   = total_gb_seconds * cost_per_gb_second

    total_cost      = invocation_cost + duration_cost

    logging.debug(f"FaaS cost: ${invocation_cost:.3f} + ${duration_cost:.3f} = ${total_cost:.2f}")
    return total_cost


def qaas_total_cost(total_data_scanned_megabytes: float) -> float:

    terabytes_scanned = total_data_scanned_megabytes / (1024 * 1024)  # Convert bytes → TB
    total_cost_usd = terabytes_scanned * 5.0
    logging.debug(f"QaaS cost: {terabytes_scanned:.3f} TB * $5 = ${total_cost_usd:.2f}")
    return total_cost_usd

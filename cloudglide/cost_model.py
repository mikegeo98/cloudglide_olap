def cost_calculator(sec_range, architecture, money, nodes, base_n, sec_money,
                    vpu, vpu_charge, spot, interrupt, slots, slot_charge):
    if architecture < 2:
        if spot == 1 and not interrupt:
            discounted_sec_money = sec_money * 0.5
            money += sec_range * nodes * discounted_sec_money
        else:
            # Original line split into two lines for readability
            money += (sec_range * base_n * sec_money +
                      sec_range * (nodes - base_n) * sec_money)

    if architecture == 2:
        vpu_charge += vpu
    if architecture == 4:

        slot_charge += slots
    return money, vpu_charge, slot_charge


def redshift_persecond_cost(vCores):
    unit = (vCores * 0.00030166666)/4
    return unit


def qaas_total_cost(total_sum):

    # Calculate total cost based on the sum of values in the specified column
    terras = total_sum / (1024 * 1024)  # Convert bytes to TB
    return terras * 5  # $5 per TB

from collections import deque
import heapq

from cloudglide.event import Event, next_event_counter


class Autoscaler:
    def __init__(self, cold_start, scaling_rules=None):
        self.scaling_out_flag = False
        self.scaling_in_flag = False
        self.scaling_counter = 0
        self.cold_start_limit = cold_start * 1000 # Cold start limit in seconds
        self.history = deque(maxlen=10)  # Store history of last 10 values
        self.last_observation_time = 0  # Initialize the observation timer
        self.pending_scale_out = 0
        self.pending_scale_in = 0
        self.rules = scaling_rules or {}

    def _queue_params(self):
        queue_rules = self.rules.get("queue", {})
        thresholds = queue_rules.get("length_thresholds", [5, 10, 15, 20])
        steps = queue_rules.get("scale_steps", [4, 8, 12, 16])
        scale_in_util = queue_rules.get("scale_in_utilization", 0.4)
        scale_in_step = queue_rules.get("scale_in_step", 4)
        return thresholds, steps, scale_in_util, scale_in_step

    def _reactive_params(self):
        reactive_rules = self.rules.get("reactive", {})
        thresholds = reactive_rules.get("cpu_utilization_thresholds", [0.6, 0.7, 0.8])
        steps = reactive_rules.get("scale_steps", [8, 16, 24])
        scale_in_util = reactive_rules.get("scale_in_utilization", 0.1)
        scale_in_step = reactive_rules.get("scale_in_step", 8)
        return thresholds, steps, scale_in_util, scale_in_step

    def _predictive_params(self):
        predictive_rules = self.rules.get("predictive", {})
        growth = predictive_rules.get("growth_factor", 1.2)
        decline = predictive_rules.get("decline_factor", 0.75)
        history = predictive_rules.get("history", 3)
        interval = predictive_rules.get("observation_interval", 10000)
        step = predictive_rules.get("scale_step", 4)
        utilization_ceiling = predictive_rules.get("utilization_ceiling", 0.75)
        return growth, decline, history, interval, step, utilization_ceiling

    def _schedule_scaling_check(self, current_second, events):
        heapq.heappush(
            events,
            Event(
                current_second + self.cold_start_limit,
                next_event_counter(),
                None,
                "scaling_check",
            ),
        )

    def scale_out(self, scale_out_amount):
        self.pending_scale_out = scale_out_amount
        self.scaling_out_flag = True
        self.scaling_counter = 0

    def scale_in(self, scale_in_amount):
        self.pending_scale_in = scale_in_amount
        self.scaling_in_flag = True
        self.scaling_counter = 0

    def reset_scaling_flag(self, nodes, cpu_cores, cpu_cores_per_node,
                           io_bandwidth, memory, second_range):
        if self.scaling_out_flag or self.scaling_in_flag:
            self.scaling_counter += second_range
            if self.scaling_counter >= self.cold_start_limit:
                if self.scaling_out_flag:
                    # print("Scaling out to", nodes+1, "nodes")
                    nodes = nodes + 1
                    cpu_cores = cpu_cores + cpu_cores_per_node
                    io_bandwidth = io_bandwidth + 650
                    memory = 32 * nodes * 1024
                if self.scaling_in_flag:
                    # print("Scaling in to", nodes, "nodes")
                    nodes = nodes - 1
                    cpu_cores = nodes * cpu_cores_per_node
                    memory = 32 * nodes * 1024
                    io_bandwidth = nodes * 650 
                self.scaling_out_flag = False
                self.scaling_in_flag = False
                self.scaling_counter = 0
                # print("Scaling decision applied")
        return nodes, cpu_cores, io_bandwidth, memory

    def reset_scaling_flag_ec(self, vpu, cpu_cores, base_cores, second_range):
        if self.scaling_out_flag or self.scaling_in_flag:
            self.scaling_counter += second_range

            if self.scaling_counter >= self.cold_start_limit:
                if self.scaling_out_flag:
                    scale_out_amount = min(max(4, self.pending_scale_out), 16)  # Dynamic scaling out amount
                    # print(f"Scaling out to {vpu + scale_out_amount} vpus")
                    vpu += scale_out_amount
                    cpu_cores = vpu  
                if self.scaling_in_flag:
                    scale_in_amount = min(max(4, self.pending_scale_in), vpu - base_cores)  # Dynamic scaling in amount
                    # print(f"Scaling in to {vpu - scale_in_amount} vpus")
                    vpu -= scale_in_amount
                    cpu_cores = vpu
                self.scaling_out_flag = False
                self.scaling_in_flag = False
                self.scaling_counter = 0
                # print("Scaling decision applied")
        return vpu, cpu_cores

    def autoscaling_dw(self, strategy, cpu_jobs, io_jobs, waiting_jobs,
                       buffer_jobs, nodes, cpu_cores_per_node, io_bandwidth,
                       cpu_cores, base_n, memory, current_second, second_range, events):
        if strategy == 4:
            strategy = 3
        elif strategy == 5:
            strategy = 2

        if strategy == 1:  # queue-based
            if not self.scaling_out_flag and not self.scaling_in_flag:
                thresholds, steps, scale_in_util, scale_in_step = self._queue_params()
                queue_len = max(len(buffer_jobs), len(waiting_jobs))
                triggered = False
                for threshold, step in sorted(zip(thresholds, steps), reverse=True):
                    if queue_len > threshold:
                        self.scale_out(step)
                        self._schedule_scaling_check(current_second, events)
                        triggered = True
                        break
                if (
                    not triggered
                    and nodes > base_n
                    and cpu_cores > 0
                    and (len(cpu_jobs) / cpu_cores) < scale_in_util
                ):
                    self.scale_in(scale_in_step)
                    self._schedule_scaling_check(current_second, events)

        # reactive
        elif strategy == 2:  # reactive_utilization_threshold
            if not self.scaling_out_flag and not self.scaling_in_flag:
                thresholds, steps, scale_in_util, scale_in_step = self._reactive_params()
                utilization = (len(cpu_jobs) / cpu_cores) if cpu_cores else 0
                triggered = False
                for threshold, step in sorted(zip(thresholds, steps), reverse=True):
                    if utilization > threshold:
                        self.scale_out(step)
                        self._schedule_scaling_check(current_second, events)
                        triggered = True
                        break

                if (
                    not triggered
                    and nodes > base_n
                    and utilization < scale_in_util
                ):
                    self.scale_in(scale_in_step)
                    self._schedule_scaling_check(current_second, events)

        # Predictive Autoscaling
        elif strategy == 3:  # Predictive Autoscaling
            growth, decline, history_len, interval, step, util_ceiling = self._predictive_params()
            if current_second - self.last_observation_time >= interval:  # configurable window
                combined_length = len(buffer_jobs) + len(cpu_jobs)
                self.history.append(combined_length)
                self.last_observation_time = current_second
                # Ensure there is enough data to calculate the moving average
                if not self.scaling_out_flag and not self.scaling_in_flag:
                    if len(self.history) >= history_len:
                        # Calculate the current moving average
                        current_moving_average = self.calculate_moving_average(list(self.history)[-history_len:])
                        previous_moving_average = self.calculate_moving_average(list(self.history)[:history_len])
                        utilization = (len(cpu_jobs) / cpu_cores) if cpu_cores else 0
                        if (current_moving_average > growth * previous_moving_average) or utilization > util_ceiling:
                            self.scale_out(step)
                            self._schedule_scaling_check(current_second, events)
                        elif (
                            previous_moving_average
                            and current_moving_average < decline * previous_moving_average
                            and nodes > base_n
                        ):
                            self.scale_in(step)
                            self._schedule_scaling_check(current_second, events)

        nodes, cpu_cores, io_bandwidth, memory = self.reset_scaling_flag(nodes, cpu_cores, cpu_cores_per_node, io_bandwidth, memory, second_range)
        return nodes, cpu_cores, io_bandwidth, memory

    def calculate_moving_average(self, data):
        return sum(data) / len(data)

    def autoscaling_ec(self, strategy, cpu_jobs, io_jobs, waiting_jobs, buffer_jobs, vpu, cpu_cores, base_cores, current_second, second_range, events):
        if strategy == 4:
            strategy = 3
        elif strategy == 5:
            strategy = 2

        if strategy == 1:  # queue-based
            if not self.scaling_out_flag and not self.scaling_in_flag:
                thresholds, steps, scale_in_util, scale_in_step = self._queue_params()
                queue_len = max(len(buffer_jobs), len(waiting_jobs))
                triggered = False
                for threshold, step in sorted(zip(thresholds, steps), reverse=True):
                    if queue_len > threshold:
                        self.scale_out(step)
                        self._schedule_scaling_check(current_second, events)
                        triggered = True
                        break
                utilization = (len(cpu_jobs) / cpu_cores) if cpu_cores else 0
                if (
                    not triggered
                    and cpu_cores > base_cores
                    and utilization < scale_in_util
                ):
                    self.scale_in(scale_in_step)
                    self._schedule_scaling_check(current_second, events)

        elif strategy == 2:  # reactive_utilization_threshold
            if not self.scaling_out_flag and not self.scaling_in_flag:
                thresholds, steps, scale_in_util, scale_in_step = self._reactive_params()
                utilization = (len(cpu_jobs) / cpu_cores) if cpu_cores else 0
                triggered = False
                for threshold, step in sorted(zip(thresholds, steps), reverse=True):
                    if utilization > threshold:
                        self.scale_out(step)
                        self._schedule_scaling_check(current_second, events)
                        triggered = True
                        break

                if (
                    not triggered
                    and cpu_cores > base_cores
                    and utilization < scale_in_util
                ):
                    self.scale_in(scale_in_step)
                    self._schedule_scaling_check(current_second, events)

        elif strategy == 3:  # Predictive Autoscaling
            growth, decline, history_len, interval, step, util_ceiling = self._predictive_params()
            if current_second - self.last_observation_time >= interval:  # configurable interval
                combined_length = len(buffer_jobs) + len(cpu_jobs)
                self.history.append(combined_length)
                self.last_observation_time = current_second
                if not self.scaling_out_flag and not self.scaling_in_flag:
                    if len(self.history) >= history_len:
                        current_moving_average = self.calculate_moving_average(list(self.history)[-history_len:])
                        previous_moving_average = self.calculate_moving_average(list(self.history)[:history_len])
                        utilization = (len(cpu_jobs) / cpu_cores) if cpu_cores else 0
                        if (current_moving_average > growth * previous_moving_average) or utilization > util_ceiling:
                            self.scale_out(step)
                            self._schedule_scaling_check(current_second, events)
                        elif (
                            previous_moving_average
                            and current_moving_average < decline * previous_moving_average
                            and cpu_cores > base_cores
                        ):
                            self.scale_in(step)
                            self._schedule_scaling_check(current_second, events)
        vpu, cpu_cores = self.reset_scaling_flag_ec(vpu, cpu_cores, base_cores,
                                                    second_range)

        return vpu, cpu_cores

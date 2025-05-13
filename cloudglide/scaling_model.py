from collections import deque
import heapq

from cloudglide.event import Event, next_event_counter


class Autoscaler:
    def __init__(self, cold_start):
        self.scaling_out_flag = False
        self.scaling_in_flag = False
        self.scaling_counter = 0
        self.cold_start_limit = cold_start * 1000 # Cold start limit in seconds
        self.history = deque(maxlen=10)  # Store history of last 10 values
        self.last_observation_time = 0  # Initialize the observation timer
        self.pending_scale_out = 0
        self.pending_scale_in = 0

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
                    # print(f"Scaling out to {vpu + scale_out_amount} vpus", current_second)
                    vpu += scale_out_amount
                    cpu_cores = vpu  
                if self.scaling_in_flag:
                    scale_in_amount = min(max(4, self.pending_scale_in), vpu - base_cores)  # Dynamic scaling in amount
                    # print(f"Scaling in to {vpu - scale_in_amount} vpus", current_second)
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
        if strategy == 1:  # queue-based
            if not self.scaling_out_flag and not self.scaling_in_flag:
                if len(buffer_jobs) > 20 or len(waiting_jobs) > 20:  # Scaling out condition
                    # print("Clause 1: Requesting Scale-Out", current_second)
                    self.scale_out(16)
                    heapq.heappush(events, Event(current_second + self.cold_start_limit, next_event_counter(), None, "scaling_check"))
                elif len(buffer_jobs) > 15 or len(waiting_jobs) > 15:  # Scaling out condition
                    # print("Clause 2: Requesting Scale-Out", current_second)
                    self.scale_out(12)
                    heapq.heappush(events, Event(current_second + self.cold_start_limit, next_event_counter(), None, "scaling_check"))
                elif len(buffer_jobs) > 10 or len(waiting_jobs) > 10:  # Scaling out condition
                    # print("Clause 3: Requesting Scale-Out", current_second)
                    self.scale_out(8)
                    heapq.heappush(events, Event(current_second + self.cold_start_limit, next_event_counter(), None, "scaling_check"))
                elif len(buffer_jobs) > 5 or len(waiting_jobs) > 5:  # Scaling out condition
                    # print("Clause 4: Requesting Scale-Out", current_second)
                    self.scale_out(4)
                    heapq.heappush(events, Event(current_second + self.cold_start_limit, next_event_counter(), None, "scaling_check"))
                elif (len(cpu_jobs) < 0.4 * cpu_cores and nodes > base_n) or (len(io_jobs) < 0.4 * cpu_cores and nodes > base_n):  # Scaling in condition
                    # print("Requesting Scale-In", current_second)
                    self.scale_in(4)
                    heapq.heappush(events, Event(current_second + self.cold_start_limit, next_event_counter(), None, "scaling_check"))

        # reactive
        elif strategy == 2:  # reactive_utilization_threshold
            if not self.scaling_out_flag and not self.scaling_in_flag:
                if len(cpu_jobs) > 0.8 * cpu_cores:  # Scaling out condition
                    # print("Requesting Scale-Out", current_second)
                    # print(len(cpu_jobs), len(buffer_jobs), cpu_cores)
                    self.scale_out(24)
                if len(cpu_jobs) > 0.7 * cpu_cores:  # Scaling out condition
                    # print("Requesting Scale-Out", current_second)
                    # print(len(cpu_jobs), len(buffer_jobs), cpu_cores)
                    self.scale_out(16)
                if len(cpu_jobs) > 0.6 * cpu_cores:  # Scaling out condition
                    # print("Requesting Scale-Out", current_second)
                    # print(len(cpu_jobs), len(buffer_jobs), cpu_cores)
                    self.scale_out(8)
                # if len(cpu_jobs) > 0.5 * cpu_cores:  # Scaling out condition
                    # print("Requesting Scale-Out", current_second)
                    # print(len(cpu_jobs), len(buffer_jobs), cpu_cores)
                    # self.scale_out(4)

                elif len(cpu_jobs) < 0.1 * cpu_cores and nodes > base_n:  # Scaling in condition
                    # print("Requesting Scale-In", current_second)
                    # print(len(cpu_jobs), len(buffer_jobs), cpu_cores)
                    self.scale_in(8)

        # Predictive Autoscaling
        elif strategy == 3:  # Predictive Autoscaling
            if current_second - self.last_observation_time >= 10000:  # Check if 10 seconds have passed
                combined_length = len(buffer_jobs) + len(cpu_jobs)
                self.history.append(combined_length)
                self.last_observation_time = current_second
                # Ensure there is enough data to calculate the moving average
                if not self.scaling_out_flag and not self.scaling_in_flag:
                    if len(self.history) >= 3:
                        # Calculate the current moving average
                        current_moving_average = self.calculate_moving_average(list(self.history)[-3:])
                        # print(list(self.history))
                        # Calculate the previous moving average
                        previous_moving_average = self.calculate_moving_average(list(self.history)[:3])
                        # if previous_moving_average != 0:
                        #     print(current_moving_average/previous_moving_average)
                        # print(current_moving_average, previous_moving_average)
                        # Check if there is a significant increase (e.g., 40%)
                        if (current_moving_average > 1.2 * previous_moving_average) or len(cpu_jobs) > 0.75 * cpu_cores:
                            self.scale_out(4)
                        elif current_moving_average < 0.75 * previous_moving_average and nodes > base_n:
                            self.scale_in(4)

        # Predictive Autoscaling
        elif strategy == 4:  # Predictive Autoscaling
            if current_second - self.last_observation_time >= 10000:  # Check if 10 seconds have passed
                combined_length = len(buffer_jobs) + len(cpu_jobs)
                self.history.append(combined_length)
                self.last_observation_time = current_second  # Reset the timer
            if not self.scaling_out_flag and not self.scaling_in_flag:
                # Trigger scale-out if there is a consistent increasing trend

                if len(self.history) == self.history.maxlen:
                    previous_length = self.history[0]
                    current_length = self.history[-1]
                    # Scale out if current length is at least 25% greater than 10 measurements ago
                    if current_length >= 1.5 * previous_length:
                        self.scale_out(4)
                    # Scale in if current length is at least 25% less than 10 measurements ago and nodes > base_n
                    elif current_length <= 0.75 * previous_length and nodes > base_n:
                        self.scale_in(4)

        # Reactive
        elif strategy == 5:  # reactive_utilization_threshold
            if not self.scaling_out_flag and not self.scaling_in_flag:
                # Scaling out condition
                if len(cpu_jobs) > 0.75 * cpu_cores or len(buffer_jobs) > 0.8 * cpu_cores:
                    self.scale_out(8)
                # elif len(cpu_jobs) > 0.7 * cpu_cores or len(buffer_jobs) > 0.6 * cpu_cores:
                #     self.scale_out(4)
                # Scaling in condition
                elif len(cpu_jobs) < 0.2 * cpu_cores and nodes > base_n:
                    self.scale_in(8)
                # elif len(cpu_jobs) < 0.2 * cpu_cores and cpu_cores > base_cores + 4:
                #     self.scale_in(4)

        nodes, cpu_cores, io_bandwidth, memory = self.reset_scaling_flag(nodes, cpu_cores, cpu_cores_per_node, io_bandwidth, memory, second_range)
        return nodes, cpu_cores, io_bandwidth, memory

    def calculate_moving_average(self, data):
        return sum(data) / len(data)

    def autoscaling_ec(self, strategy, cpu_jobs, io_jobs, waiting_jobs, buffer_jobs, vpu, cpu_cores, base_cores, current_second, second_range):
        if strategy == 1:  # queue-based
            if not self.scaling_out_flag and not self.scaling_in_flag:
                if len(buffer_jobs) > 20 or len(waiting_jobs) > 20:
                    self.scale_out(16)
                elif len(buffer_jobs) > 15 or len(waiting_jobs) > 15:
                    self.scale_out(12)
                elif len(buffer_jobs) > 10 or len(waiting_jobs) > 10:
                    self.scale_out(8)
                elif len(buffer_jobs) > 5 or len(waiting_jobs) > 5:
                    self.scale_out(4)
                elif (len(cpu_jobs) < 0.4 * cpu_cores and cpu_cores > base_cores) or (len(io_jobs) < 0.4 * cpu_cores and cpu_cores > base_cores):  # Scaling in condition
                    self.scale_in(4)

        # Reactive
        elif strategy == 2:  # reactive_utilization_threshold
            if not self.scaling_out_flag and not self.scaling_in_flag:
                if len(cpu_jobs) > 0.8 * cpu_cores:  # Scaling out condition
                    self.scale_out(24)
                if len(cpu_jobs) > 0.7 * cpu_cores:  # Scaling out condition
                    # print("Requesting Scale-Out", current_second)
                    # print(len(cpu_jobs), len(buffer_jobs), cpu_cores)
                    self.scale_out(16)
                if len(cpu_jobs) > 0.6 * cpu_cores:  # Scaling out condition
                    # print("Requesting Scale-Out", current_second)
                    # print(len(cpu_jobs), len(buffer_jobs), cpu_cores)
                    self.scale_out(8)
                # if len(cpu_jobs) > 0.5 * cpu_cores:  # Scaling out condition
                    # print("Requesting Scale-Out", current_second)
                    # print(len(cpu_jobs), len(buffer_jobs), cpu_cores)
                    # self.scale_out(4)

                elif len(cpu_jobs) < 0.1 * cpu_cores and cpu_cores > base_cores:  # Scaling in condition
                    # print("Requesting Scale-In", current_second)
                    # print(len(cpu_jobs), len(buffer_jobs), cpu_cores)
                    self.scale_in(8)

        # Predictive Autoscaling
        elif strategy == 3:  # Predictive Autoscaling
            if current_second - self.last_observation_time >= 10000:  # Check if 10 seconds have passed
                combined_length = len(buffer_jobs) + len(cpu_jobs)
                self.history.append(combined_length)
                self.last_observation_time = current_second
                # Ensure there is enough data to calculate the moving average
                if not self.scaling_out_flag and not self.scaling_in_flag:
                    if len(self.history) >= 3:
                        # Calculate the current moving average
                        current_moving_average = self.calculate_moving_average(list(self.history)[-3:])
                        # print(list(self.history))
                        # Calculate the previous moving average
                        previous_moving_average = self.calculate_moving_average(list(self.history)[:3])
                        # if previous_moving_average != 0:
                        #     print(current_moving_average/previous_moving_average)
                        # print(current_moving_average, previous_moving_average)
                        # Check if there is a significant increase (e.g., 40%)
                        if (current_moving_average > 1.2 * previous_moving_average) or len(cpu_jobs) > 0.75 * cpu_cores:
                            self.scale_out(4)
                        elif current_moving_average < 0.75 * previous_moving_average and cpu_cores > base_cores:
                            self.scale_in(4)

        # Predictive Autoscaling
        elif strategy == 4:  # Predictive Autoscaling
            if current_second - self.last_observation_time >= 10000:  # Check if 10 seconds have passed
                combined_length = len(buffer_jobs) + len(cpu_jobs)
                self.history.append(combined_length)
                self.last_observation_time = current_second  # Reset the timer
            if not self.scaling_out_flag and not self.scaling_in_flag:
                # Trigger scale-out if there is a consistent increasing trend

                if len(self.history) == self.history.maxlen:
                    previous_length = self.history[0]
                    current_length = self.history[-1]
                    # Scale out if current length is at least 25% greater than 10 measurements ago
                    if current_length >= 1.5 * previous_length:
                        self.scale_out(4)
                    # Scale in if current length is at least 25% less than 10 measurements ago and nodes > base_n
                    elif current_length <= 0.75 * previous_length and cpu_cores > base_cores:
                        self.scale_in(4)

        # Reactive
        elif strategy == 5:  # reactive_utilization_threshold
            if not self.scaling_out_flag and not self.scaling_in_flag:
                # Scaling out condition
                if len(cpu_jobs) > 0.75 * cpu_cores or len(buffer_jobs) > 0.8 * cpu_cores:
                    self.scale_out(8)
                # elif len(cpu_jobs) > 0.7 * cpu_cores or len(buffer_jobs) > 0.6 * cpu_cores:
                #     self.scale_out(4)
                # Scaling in condition
                elif len(cpu_jobs) < 0.2 * cpu_cores and cpu_cores > base_cores + 8:
                    self.scale_in(8)
                # elif len(cpu_jobs) < 0.2 * cpu_cores and cpu_cores > base_cores + 4:
                #     self.scale_in(4)
        vpu, cpu_cores = self.reset_scaling_flag_ec(vpu, cpu_cores, base_cores,
                                                    second_range)

        return vpu, cpu_cores

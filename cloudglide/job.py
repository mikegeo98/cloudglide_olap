# job.py

from dataclasses import dataclass, field


@dataclass
class Job:
    # Identity and input parameters
    job_id: int
    database_id: int
    query_id: int
    start: float  # Job arrival time (ms)
    cpu_time: float  # Total CPU time required (ms)
    data_scanned: float  # Total data scanned (bytes)
    scale_factor: float  # Scale factor affecting shuffle size
    
    # Derived fields (initialized after basic init)
    data_shuffle: float = field(init=False)
    cpu_time_progress: float = field(init=False)
    data_scanned_progress: float = field(init=False)
    
    # Progress and timing metrics
    start_timestamp: float = 0.0  # Actual processing start time (ms)
    end_timestamp: float = 0.0  # Completion time (ms)
    queueing_delay: float = 0.0
    io_time: float = 0.0
    buffer_delay: float = 0.0
    shuffle_time: float = 0.0
    processing_time: float = 0.0
    query_exec_time: float = 0.0
    query_exec_time_queueing: float = 0.0

    # Scheduling metadata for event-driven sim
    priority_level: int = 0
    scheduled: bool = field(init=False, default=False)
    next_time: float | None = field(init=False, default=None)
    dram_node_index: int | None = field(init=False, default=None)
    
    def __post_init__(self):
        self.data_shuffle = self.calculate_data_shuffle()
        self.cpu_time_progress = self.cpu_time
        self.data_scanned_progress = self.data_scanned

    def calculate_data_shuffle(self) -> float:
        """
        Calculate the amount of data to shuffle based on the scale factor.
        """
        if self.scale_factor <= 1:
            return 0.1 * self.data_scanned
        elif self.scale_factor >= 100:
            return 0.35 * self.data_scanned
        else:
            # Linear interpolation between 10% and 35%
            percentage = 0.1 + ((self.scale_factor - 1) / 99) * 0.25
            return percentage * self.data_scanned

    def reset_progress(self, current_second: float):
        """
        Reset job progress upon interruption.
        """
        self.cpu_time_progress = self.cpu_time
        self.data_scanned_progress = self.data_scanned
        self.start = current_second * 1000 + 60000  # Adjust start time

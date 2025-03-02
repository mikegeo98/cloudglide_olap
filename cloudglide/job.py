# job.py

from dataclasses import dataclass, field


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

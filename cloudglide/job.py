# job.py
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(slots=True)
class Job:
    # Identity and input parameters
    job_id: int
    database_id: int
    query_id: int
    start: float         # Job arrival time (ms)
    cpu_time: float      # Total CPU time required (ms)
    data_scanned: float  # Total data scanned (bytes)
    scale_factor: float  # Scale factor affecting shuffle size

    # Derived fields (initialized after basic init)
    data_shuffle: float = field(init=False)
    cpu_time_progress: float = field(init=False)
    data_scanned_progress: float = field(init=False)

    # Progress and timing metrics
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    queueing_delay: float = 0.0
    io_time: float = 0.0
    buffer_delay: float = 0.0
    shuffle_time: float = 0.0
    processing_time: float = 0.0
    query_exec_time: float = 0.0
    query_exec_time_queueing: float = 0.0

    # Phase-specific timestamps
    io_start_timestamp: float = 0.0
    io_end_timestamp: float = 0.0
    cpu_start_timestamp: float = 0.0
    cpu_end_timestamp: float = 0.0
    shuffle_start_timestamp: float = 0.0
    shuffle_end_timestamp: float = 0.0

    # Scheduling metadata
    priority_level: int = 0
    scheduled: bool = field(init=False, default=False)
    next_io_done:      float | None = field(init=False, default=None)
    next_shuffle_done: float | None = field(init=False, default=None)
    next_cpu_done:     float | None = field(init=False, default=None)
    dram_node_index: Optional[int] = field(init=False, default=None)
    
    # per-estimator outputs
    estimators: Dict[str, float] = field(default_factory=dict)
    selected_estimator: str = ""
    queue_total: float = 0.0

    def __post_init__(self):
        self.data_shuffle = self.calculate_data_shuffle()
        self.cpu_time_progress = self.cpu_time
        self.data_scanned_progress = self.data_scanned

    def calculate_data_shuffle(self) -> float:

        if self.scale_factor <= 1:
            return 0.1 * self.data_scanned
        elif self.scale_factor >= 100:
            return 0.35 * self.data_scanned
        else:
            # Linear interpolation between 10% and 35%
            percentage = 0.1 + ((self.scale_factor - 1) / 99) * 0.25
            return percentage * self.data_scanned

    def reset_progress(self, current_second: float, config) -> None:
        """
        Reset job progress after interruption or cold start.
        Uses cold_start_delay from the simulation config (seconds).
        """
        self.cpu_time_progress = self.cpu_time
        self.data_scanned_progress = self.data_scanned
        self.start = current_second + config.cold_start_delay

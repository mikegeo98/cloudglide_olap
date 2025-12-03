# logging_utils.py
"""
Structured logging utilities with context managers for CloudGlide.
Provides consistent, filterable logging with contextual information.
"""

import logging
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SimulationContext:
    """
    Context for a simulation run.
    Tracks metadata that should be included in all log messages.
    """
    simulation_id: str
    architecture: str
    dataset: str
    nodes: int
    timestamp: float = field(default_factory=time.time)
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        return {
            "simulation_id": self.simulation_id,
            "architecture": self.architecture,
            "dataset": self.dataset,
            "nodes": self.nodes,
            "timestamp": self.timestamp,
            **self.extra_fields
        }


class StructuredLogger:
    """
    Wrapper around Python logging with structured output.
    Automatically includes simulation context in all log messages.
    """

    def __init__(self, name: str, context: Optional[SimulationContext] = None):
        self.logger = logging.getLogger(name)
        self.context = context
        self._extra: Dict[str, Any] = {}

    def _format_message(self, msg: str, **kwargs) -> str:
        """Format message with context and extra fields."""
        parts = [msg]

        # Add context
        if self.context:
            ctx_str = " | ".join(
                f"{k}={v}" for k, v in self.context.to_dict().items()
            )
            parts.append(f"[{ctx_str}]")

        # Add extra fields
        if kwargs or self._extra:
            combined_extra = {**self._extra, **kwargs}
            extra_str = " | ".join(
                f"{k}={v}" for k, v in combined_extra.items()
            )
            parts.append(f"({extra_str})")

        return " ".join(parts)

    def debug(self, msg: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(self._format_message(msg, **kwargs))

    def info(self, msg: str, **kwargs):
        """Log info message with context."""
        self.logger.info(self._format_message(msg, **kwargs))

    def warning(self, msg: str, **kwargs):
        """Log warning message with context."""
        self.logger.warning(self._format_message(msg, **kwargs))

    def error(self, msg: str, **kwargs):
        """Log error message with context."""
        self.logger.error(self._format_message(msg, **kwargs))

    def critical(self, msg: str, **kwargs):
        """Log critical message with context."""
        self.logger.critical(self._format_message(msg, **kwargs))

    @contextmanager
    def extra_context(self, **kwargs):
        """
        Temporarily add extra fields to all log messages.

        Usage:
            with logger.extra_context(phase="scheduling", second=100):
                logger.info("Scheduling jobs")  # Includes phase and second
        """
        old_extra = self._extra.copy()
        self._extra.update(kwargs)
        try:
            yield self
        finally:
            self._extra = old_extra


@contextmanager
def simulation_phase(logger: StructuredLogger, phase_name: str):
    """
    Context manager for timing and logging simulation phases.

    Usage:
        with simulation_phase(logger, "initialization"):
            # initialization code
    """
    start_time = time.time()
    logger.info(f"Starting phase: {phase_name}")

    try:
        yield
        elapsed = time.time() - start_time
        logger.info(
            f"Completed phase: {phase_name}",
            elapsed_seconds=f"{elapsed:.2f}"
        )
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"Failed phase: {phase_name}",
            error=str(e),
            elapsed_seconds=f"{elapsed:.2f}"
        )
        raise


@contextmanager
def log_performance(logger: StructuredLogger, operation: str):
    """
    Context manager for logging performance metrics.

    Usage:
        with log_performance(logger, "job_scheduling"):
            # expensive operation
    """
    start_time = time.time()
    start_memory = None

    try:
        import psutil
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        pass

    try:
        yield
    finally:
        elapsed = time.time() - start_time
        log_data = {
            "operation": operation,
            "elapsed_ms": f"{elapsed * 1000:.2f}"
        }

        if start_memory is not None:
            try:
                end_memory = process.memory_info().rss / 1024 / 1024
                log_data["memory_delta_mb"] = f"{end_memory - start_memory:.2f}"
            except:
                pass

        logger.debug(f"Performance: {operation}", **log_data)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for CloudGlide simulation.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        format_string: Custom format string (default provides timestamp and level)

    Returns:
        Configured root logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(logging.Formatter(format_string))
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(file_handler)

    return root_logger


# Convenience function for creating structured loggers
def get_logger(
    name: str,
    simulation_id: Optional[str] = None,
    architecture: Optional[str] = None,
    dataset: Optional[str] = None,
    nodes: Optional[int] = None
) -> StructuredLogger:
    """
    Create a structured logger with optional simulation context.

    Args:
        name: Logger name (typically __name__)
        simulation_id: Unique simulation identifier
        architecture: Architecture type
        dataset: Dataset name
        nodes: Number of nodes

    Returns:
        StructuredLogger instance
    """
    context = None
    if simulation_id:
        context = SimulationContext(
            simulation_id=simulation_id,
            architecture=architecture or "unknown",
            dataset=dataset or "unknown",
            nodes=nodes or 0
        )

    return StructuredLogger(name, context)

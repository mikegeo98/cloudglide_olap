import math
from collections import deque

from cloudglide import query_processing_model as qpm
from cloudglide.config import ArchitectureType, SimulationConfig
from cloudglide.event import Event, next_event_counter
from cloudglide.job import Job


def make_job(job_id=1, cpu_time=1000, data_scanned=1000, scale_factor=1):
    job = Job(
        job_id=job_id,
        database_id=0,
        query_id=job_id,
        start=0.0,
        cpu_time=cpu_time,
        data_scanned=data_scanned,
        scale_factor=scale_factor,
    )
    job.start_timestamp = 0.0
    return job


def test_assign_memory_tier_respects_warmup(monkeypatch):
    monkeypatch.setattr(qpm.random, "random", lambda: 0.0)
    tier = qpm.assign_memory_tier(
        hit_rate=1.5,
        architecture=ArchitectureType.ELASTIC_POOL,
        n=10,
        warmup_rate=1.0,
    )
    assert tier == "DRAM"


def test_assign_memory_tier_can_end_up_in_cold_tiers(monkeypatch):
    monkeypatch.setattr(qpm.random, "random", lambda: 0.99)
    tier = qpm.assign_memory_tier(
        hit_rate=0.0,
        architecture=ArchitectureType.DWAAS,
        n=0,
        warmup_rate=1.0,
    )
    assert tier == "S3"


def test_assign_memory_tier_warmup_increases_dram_probability(monkeypatch):
    monkeypatch.setattr(qpm.random, "random", lambda: 0.3)
    cold_tier = qpm.assign_memory_tier(
        hit_rate=0.8,
        architecture=ArchitectureType.ELASTIC_POOL,
        n=0,
        warmup_rate=0.2,
    )
    warm_tier = qpm.assign_memory_tier(
        hit_rate=0.8,
        architecture=ArchitectureType.ELASTIC_POOL,
        n=20,
        warmup_rate=0.2,
    )
    assert cold_tier != "DRAM"
    assert warm_tier == "DRAM"


def test_peek_next_event_time_returns_inf_when_empty():
    job = make_job()
    events = [Event(5.0, next_event_counter(), job, "arrival")]
    assert qpm.peek_next_event_time(events) == 5.0
    assert math.isinf(qpm.peek_next_event_time([]))


def test_update_dram_nodes_handles_resize():
    job = make_job()
    dram_nodes = [[job]]
    dram_job_counts = [1]

    expanded_nodes, expanded_counts = qpm.update_dram_nodes(
        dram_nodes, dram_job_counts, 3
    )
    assert len(expanded_nodes) == 3
    assert expanded_counts[-1] == 0

    reduced_nodes, reduced_counts = qpm.update_dram_nodes(
        expanded_nodes, expanded_counts, 1
    )
    assert len(reduced_nodes) == 1
    assert len(reduced_nodes[0]) == 1
    assert reduced_counts[0] == 1


def test_simulate_io_dwaas_completes_job(monkeypatch):
    monkeypatch.setattr(qpm.random, "random", lambda: 0.0)
    config = SimulationConfig()
    job = make_job(data_scanned=1, cpu_time=100)
    job.data_scanned_progress = 0.1

    io_jobs = deque([job])
    buffer_jobs = deque()
    cpu_jobs = deque()
    finished_jobs = []
    dram_nodes = [[] for _ in range(2)]
    dram_job_counts = [0, 0]
    job_memory_tiers = {}
    events = []

    qpm.simulate_io(
        current_second=5.0,
        hit_rate=1.0,
        num_nodes=2,
        io_jobs=io_jobs,
        io_bandwidth=2000,
        memory_bandwidth=5000,
        phase="arrival",
        buffer_jobs=buffer_jobs,
        cpu_jobs=cpu_jobs,
        finished_jobs=finished_jobs,
        job_memory_tiers=job_memory_tiers,
        dram_nodes=dram_nodes,
        dram_job_counts=dram_job_counts,
        second_range=1.0,
        events=events,
        architecture=ArchitectureType.DWAAS,
        config=config,
    )

    assert job not in io_jobs
    assert finished_jobs == [job]
    assert job_memory_tiers[job.job_id] == "DRAM"


def test_simulate_io_qaas_moves_completed_job():
    config = SimulationConfig()
    job = make_job(cpu_time=2000)
    job.data_scanned_progress = 0

    io_jobs = deque([job])
    buffer_jobs = deque()
    cpu_jobs = deque()
    finished_jobs = []

    qpm.simulate_io(
        current_second=5.0,
        hit_rate=0.5,
        num_nodes=1,
        io_jobs=io_jobs,
        io_bandwidth=1000,
        memory_bandwidth=2000,
        phase="arrival",
        buffer_jobs=buffer_jobs,
        cpu_jobs=cpu_jobs,
        finished_jobs=finished_jobs,
        job_memory_tiers={},
        dram_nodes=[],
        dram_job_counts=[],
        second_range=1.0,
        events=[],
        architecture=ArchitectureType.QAAS,
        config=config,
    )

    assert job not in io_jobs
    assert finished_jobs == [job]


def test_assign_cores_to_jobs_weights_required_work():
    job_a = make_job(job_id=1, cpu_time=1000)
    job_b = make_job(job_id=2, cpu_time=4000)
    cpu_jobs = [job_a, job_b]
    allocation = qpm.assign_cores_to_jobs(
        cpu_jobs, {}, num_cores=4, current_second=10, time_limit=1
    )
    assert sum(allocation) == 4
    assert allocation[0] == 1
    assert allocation[1] == 3


def test_schedule_event_deduplicates_entries():
    job = make_job()
    events = [Event(5.0, next_event_counter(), job, "io_done")]
    qpm.schedule_event(job, 2.0, "io_done", events)
    assert len(events) == 1
    assert events[0].time == 2.0
    assert job.next_io_done == 2.0


def test_assign_cores_to_jobs_respects_shuffle_blocks():
    job = make_job(job_id=3)
    job.data_shuffle = 10
    allocation = qpm.assign_cores_to_jobs(
        cpu_jobs=[job],
        shuffle={job.job_id: 1},
        num_cores=4,
        current_second=5,
        time_limit=1,
    )
    assert allocation == [0]


def test_simulate_cpu_finishes_job_and_updates_finished():
    config = SimulationConfig()
    job = make_job(cpu_time=3)
    job.cpu_time_progress = 3
    job.data_shuffle = 0
    memory = [job.data_scanned * config.materialization_fraction]

    cpu_jobs = [job]
    shuffle_jobs = []
    finished_jobs = []
    io_jobs = deque()
    waiting_jobs = deque()
    shuffle = {}
    events = []

    qpm.simulate_cpu(
        current_second=5.0,
        cpu_jobs=cpu_jobs,
        phase="arrival",
        cpu_cores=4,
        cpu_cores_per_node=4,
        network_bandwidth=1000,
        finished_jobs=finished_jobs,
        shuffle_jobs=shuffle_jobs,
        io_jobs=io_jobs,
        waiting_jobs=waiting_jobs,
        shuffle=shuffle,
        memory=memory,
        second_range=1.0,
        events=events,
        architecture=ArchitectureType.DWAAS,
        config=config,
    )

    assert cpu_jobs == []
    assert finished_jobs == [job]
    assert job.cpu_time_progress == 0


def test_simulate_cpu_returns_total_cores_for_elastic_pool():
    config = SimulationConfig()
    job = make_job(job_id=10, cpu_time=2)
    job.cpu_time_progress = 2
    job.data_shuffle = 0
    cpu_jobs = [job]
    shuffle_jobs = []
    finished_jobs = []
    io_jobs = deque()
    waiting_jobs = deque()
    events = []
    memory = [job.data_scanned * config.materialization_fraction]

    allocated = qpm.simulate_cpu(
        current_second=10.0,
        cpu_jobs=cpu_jobs,
        phase="arrival",
        cpu_cores=4,
        cpu_cores_per_node=4,
        network_bandwidth=1000,
        finished_jobs=finished_jobs,
        shuffle_jobs=shuffle_jobs,
        io_jobs=io_jobs,
        waiting_jobs=waiting_jobs,
        shuffle={},
        memory=memory,
        second_range=1.0,
        events=events,
        architecture=ArchitectureType.ELASTIC_POOL,
        config=config,
    )

    assert allocated == 4
    assert job not in cpu_jobs


def test_job_finalization_updates_estimators():
    job = make_job()
    job.io_time = 2.0
    job.processing_time = 3.0
    job.shuffle_time = 1.0
    job.queueing_delay = 0.5
    job.buffer_delay = 0.25
    memory = [job.data_scanned * 0.5]
    config = SimulationConfig()
    config.materialization_fraction = 0.5

    qpm.job_finalization(
        job=job,
        memory=memory,
        cpu_jobs=[],
        shuffle_jobs=[],
        finished_jobs=[],
        io_jobs=[],
        waiting_jobs=[],
        current_second=10.0,
        config=config,
    )

    assert job.estimators
    expected_total = job.query_exec_time + job.queue_total
    assert job.query_exec_time_queueing == expected_total
    assert memory[0] == 0.0


def test_assign_cores_to_job_qaas_interpolates():
    job = make_job(cpu_time=5_000_000)
    cores = qpm.assign_cores_to_job_qaas(job, time_limit=2)
    assert cores == math.ceil(5_000_000 / (12_000))


def test_assign_cores_to_jobs_qaas_vectorizes():
    jobs = [
        make_job(job_id=1, cpu_time=1_000_000),
        make_job(job_id=2, cpu_time=5_000_000),
    ]
    cores = qpm.assign_cores_to_jobs_qaas(jobs, time_limit=2)
    assert cores[0] == qpm.assign_cores_to_job_qaas(jobs[0], 2)
    assert cores[1] == qpm.assign_cores_to_job_qaas(jobs[1], 2)


def test_simulate_io_schedules_completion_event(monkeypatch):
    config = SimulationConfig()
    job = make_job(job_id=50, data_scanned=5_000, cpu_time=100)
    io_jobs = deque([job])
    buffer_jobs = deque()
    cpu_jobs = deque()
    finished_jobs = []
    dram_nodes = [[]]
    dram_job_counts = [0]
    scheduled = []

    monkeypatch.setattr(qpm.random, "random", lambda: 0.99)

    def _record_event(job_arg, timestamp, etype, events):
        scheduled.append((job_arg.job_id, timestamp, etype))

    monkeypatch.setattr(qpm, "schedule_event", _record_event)

    qpm.simulate_io(
        current_second=10.0,
        hit_rate=0.0,
        num_nodes=1,
        io_jobs=io_jobs,
        io_bandwidth=100,
        memory_bandwidth=100,
        phase="arrival",
        buffer_jobs=buffer_jobs,
        cpu_jobs=cpu_jobs,
        finished_jobs=finished_jobs,
        job_memory_tiers={},
        dram_nodes=dram_nodes,
        dram_job_counts=dram_job_counts,
        second_range=1.0,
        events=[],
        architecture=ArchitectureType.DWAAS,
        config=config,
    )

    assert io_jobs  # job still queued because not enough bandwidth
    assert any(evt[2] == "io_done" for evt in scheduled)


def test_simulate_cpu_qaas_completes_job():
    config = SimulationConfig()
    job = make_job(cpu_time=1)
    job.cpu_time_progress = 1
    job.data_shuffle = 0
    memory = [job.data_scanned * config.materialization_fraction]

    cpu_jobs = [job]
    finished_jobs = []
    shuffle_jobs = []
    io_jobs = deque()
    waiting_jobs = deque()
    shuffle = {}
    events = []

    qpm.simulate_cpu_qaas(
        current_second=5.0,
        cpu_jobs=cpu_jobs,
        network_bandwidth=1000,
        finished_jobs=finished_jobs,
        shuffle_jobs=shuffle_jobs,
        waiting_jobs=waiting_jobs,
        io_jobs=io_jobs,
        shuffle=shuffle,
        memory=memory,
        second_range=1.0,
        events=events,
        config=config,
    )

    assert cpu_jobs == []
    assert finished_jobs == [job]
    assert job.cpu_time_progress == 0

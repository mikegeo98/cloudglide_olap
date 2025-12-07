export interface InstanceConfig {
    cpu_cores: number;
    memory: number;       // in GB
    io_bandwidth: number; // in Mbps
    network_bandwidth: number; // in Mbps
    memory_bandwidth: number;  // in Mbps
}

export interface DefaultConfig {
    simulation: {
        scale_factor_min: number,
        scale_factor_max: number,
        shuffle_percentage_min: number,
        shuffle_percentage_max: number,
        interrupt_probability: number,
        interrupt_duration: number,
        logging_interval: number,
        default_max_duration: number,
        default_estimator: string,
        pm_p: number,
        delta: number,
        queue_agg: string,
        materialization_fraction: number,
        parallelizable_portion: number,
        s3_bandwidth: number,
    },
    dwaas: {
        cost_per_second_redshift: number,
        spot_discount: number,
        use_spot_instances: boolean,
        cold_start_delay: number,
    },
    elastic_pool: {
        cost_per_rpu_hour: number,
        cache_warmup_gamma: number,
        cold_start_delay: number,
    },
    qaas: {
        cost_per_slot_hour: number,
        qaas_cost_per_tb: number,
        qaas_io_per_core_bw: number,
        qaas_shuffle_bw_per_core: number,
        qaas_base_cores: number,
        qaas_base_time_limit: number,
        core_alloc_window: number,
    },
}

export const instanceTypes: InstanceConfig[] = [
    { cpu_cores: 4, memory: 32, io_bandwidth: 650, network_bandwidth: 1000, memory_bandwidth: 40000 },   // ra3.xlplus
    { cpu_cores: 12, memory: 96, io_bandwidth: 2000, network_bandwidth: 10000, memory_bandwidth: 40000 },  // ra3.4xlarge
    { cpu_cores: 48, memory: 384, io_bandwidth: 8000, network_bandwidth: 10000, memory_bandwidth: 40000 },  // ra3.16xlarge
    { cpu_cores: 4, memory: 8, io_bandwidth: 143, network_bandwidth: 10000, memory_bandwidth: 40000 },     // c5d.xlarge
    { cpu_cores: 8, memory: 16, io_bandwidth: 287, network_bandwidth: 10000, memory_bandwidth: 40000 },    // c5d.2xlarge
    { cpu_cores: 16, memory: 32, io_bandwidth: 575, network_bandwidth: 10000, memory_bandwidth: 40000 },   // c5d.4xlarge
    { cpu_cores: 4, memory: 32, io_bandwidth: 650, network_bandwidth: 1000, memory_bandwidth: 2500 },   // ra3.xlplus (alt)
];

export const defaultConfig: DefaultConfig = {
    simulation: {
        scale_factor_min: 1,
        scale_factor_max: 1000,
        shuffle_percentage_min: 0.1,
        shuffle_percentage_max: 0.35,
        interrupt_probability: 0.01,
        interrupt_duration: 6000,
        logging_interval: 60,
        default_max_duration: 108000,
        default_estimator: "pm",
        pm_p: 4.0,
        delta: 0.3,
        queue_agg: "sum",
        materialization_fraction: 0.25,
        parallelizable_portion: 0.9,
        s3_bandwidth: 1000,
    },
    dwaas: {
        cost_per_second_redshift: 0.00030166666,
        spot_discount: 0.5,
        use_spot_instances: false,
        cold_start_delay: 60.0,
    },
    elastic_pool: {
        cost_per_rpu_hour: 0.375,
        cache_warmup_gamma: 0.05,
        cold_start_delay: 120.0,
    },
    qaas: {
        cost_per_slot_hour: 0.04,
        qaas_cost_per_tb: 5.0,
        qaas_io_per_core_bw: 150,
        qaas_shuffle_bw_per_core: 50,
        qaas_base_cores: 4,
        qaas_base_time_limit: 2,
        core_alloc_window: 10.0,
    },
}
export interface InstanceConfig {
    cpu_cores: number;
    memory: number;       // in GB
    io_bandwidth: number; // in Mbps
    network_bandwidth: number; // in Mbps
    memory_bandwidth: number;  // in Mbps
}

export const instanceTypes: InstanceConfig[] = [
    { cpu_cores: 4, memory: 32, io_bandwidth: 650, network_bandwidth: 1000, memory_bandwidth: 40000 },   // ra3.xlplus
    { cpu_cores: 12, memory: 96, io_bandwidth: 2000, network_bandwidth: 10000, memory_bandwidth: 40000 },  // ra3.4xlarge
    { cpu_cores: 48, memory: 384, io_bandwidth: 8000, network_bandwidth: 10000, memory_bandwidth: 40000 },  // ra3.16xlarge
    { cpu_cores: 4, memory: 8, io_bandwidth: 143, network_bandwidth: 10000, memory_bandwidth: 40000 },     // c5d.xlarge
    { cpu_cores: 8, memory: 16, io_bandwidth: 287, network_bandwidth: 10000, memory_bandwidth: 40000 },    // c5d.2xlarge
    { cpu_cores: 16, memory: 32, io_bandwidth: 575, network_bandwidth: 10000, memory_bandwidth: 40000 },   // c5d.4xlarge
    { cpu_cores: 4, memory: 32, io_bandwidth: 650, network_bandwidth: 1000, memory_bandwidth: 2500 },   // ra3.xlplus
];
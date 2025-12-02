"use client"

import React from "react";

export type InputData = {
    // step 1
    input_csv?: File,
    dataset?: number,

    // step 2
    architecture?: string,
    nodes?: number,
    hit_rate?: number,
    instance_index?: number,
    vpu?: number,
    memory?: number,
    network_bandwidth?: number,
    io_bandwidth?: number,
    memory_bandwidth?: number,
    scheduling?: {
        policy: string,
        max_io_concurrency: number,
        max_cpu_concurrency: number,
    },

    // step 3
    defaults: {
        simulation: {
            interrupt_probability: number,
            interrupt_duration: number,
            spot_discount: number,
            cost_per_second_redshift: number,
            cost_per_rpu_hour: number,
            cost_per_slot_hour: number,
            qaas_cost_per_tb: number,
            logging_interval: number,
            default_max_duration: number,
            default_estimator: string,
            pm_p: number,
            delta: number,
            queue_agg: string,
            materialization_fraction: number,
            parallelizable_portion: number,
            cold_start_delay: number,
            cache_warmup_gamma: number,
            qaas_io_per_core_bw: number,
            qaas_shuffle_bw_per_core: number,
            qaas_base_cores: number,
            qaas_base_time_limit: number,
            core_alloc_window: number,
            s3_bandwidth: number
        },
        scheduling: {
            policy: string,
            max_io_concurrency: number,
            max_cpu_concurrency: number,
        },
        scaling: {
            policy: string,
            queue: {
                length_thresholds: number[],
                scale_steps: number[],
                scale_in_utilization: number,
                scale_in_step: number
            },
            reactive: {
                cpu_utilization_thresholds: number[],
                scale_steps: number[],
                scale_in_utilization: number,
                scale_in_step: number
            },
            predictive: {
                growth_factor: number,
                decline_factor: number,
                history: number,
                observation_interval: number,
                scale_step: number,
                utilization_ceiling: number
            }
        }
    },
}

export const InputContext = React.createContext({
    stage: 0,
    increaseStage: (n: number) => { },
    data: {} as InputData,
    setData: (d: InputData) => { },
    dataToJson: () => ("" as string),
})

export default function Provider({
    children,
}: {
    children: React.ReactNode
}) {
    const [stage, increaseStage] = React.useState<number>(0)
    const [data, setData] = React.useState<InputData>({
        // step 3
        defaults: {
            simulation: {
                interrupt_probability: 0.01,
                interrupt_duration: 6000,
                spot_discount: 0.5,
                cost_per_second_redshift: 0.00030166666,
                cost_per_rpu_hour: 0.375,
                cost_per_slot_hour: 0.04,
                qaas_cost_per_tb: 5.0,
                logging_interval: 60,
                default_max_duration: 108000,
                default_estimator: "pm",
                pm_p: 4.0,
                delta: 0.3,
                queue_agg: "sum",
                materialization_fraction: 0.25,
                parallelizable_portion: 0.9,
                cold_start_delay: 60.0,
                cache_warmup_gamma: 0.05,
                qaas_io_per_core_bw: 150,
                qaas_shuffle_bw_per_core: 50,
                qaas_base_cores: 4,
                qaas_base_time_limit: 2,
                core_alloc_window: 10.0,
                s3_bandwidth: 1000
            },
            scheduling: {
                policy: "fcfs",
                max_io_concurrency: 64,
                max_cpu_concurrency: 64
            },
            scaling: {
                policy: "queue",
                queue: {
                    length_thresholds: [5, 10, 15, 20],
                    scale_steps: [4, 8, 12, 16],
                    scale_in_utilization: 0.4,
                    scale_in_step: 4
                },
                reactive: {
                    cpu_utilization_thresholds: [0.6, 0.7, 0.8],
                    scale_steps: [8, 12, 16],
                    scale_in_utilization: 0.2,
                    scale_in_step: 8
                },
                predictive: {
                    growth_factor: 1.2,
                    decline_factor: 0.75,
                    history: 3,
                    observation_interval: 10000,
                    scale_step: 4,
                    utilization_ceiling: 0.75
                }
            },
        },
    })

    function dataToJson() {
        const j = {
            defaults: data.defaults,
            scenarios: [
                {
                    name: data.architecture + "1",
                    architecture: data.architecture,
                    dataset: data.input_csv ? 1001 : data.dataset,
                    nodes: data.nodes,
                    hit_rate: data.hit_rate,
                    vpu: data.vpu,
                    memory: data.memory,
                    network_bandwidth: data.network_bandwidth,
                    io_bandwidth: data.io_bandwidth,
                    memory_bandwidth: data.memory_bandwidth,
                    scheduling: data.scheduling ? data.scheduling : data.defaults.scheduling,
                }
            ]
        }
        return JSON.stringify(j, null, "\t")
    }

    return (
        <InputContext.Provider value={{ stage, increaseStage, data, setData, dataToJson }}>
            {children}
        </InputContext.Provider>
    )
}
import z from "zod";
import { ArchitectureType, instanceTypes } from "./config";

// First page: Input Schema
export const inputSchema = z.object({
    file: z.file().optional(),
    dataset: z.number().optional(),
})
    .superRefine((data, ctx) => {
        if (data.file === undefined && (data.dataset === undefined || data.dataset === undefined)) {
            ctx.addIssue({
                code: "custom",
                message: "Either upload a CSV file or use an alternative",
                path: ["file"]
            })
        }
    });


// Second page: Architecture Schema
export const createArchitectureSchema = (archType: typeof ArchitectureType[keyof typeof ArchitectureType]): z.ZodObject<z.ZodRawShape> => {
    switch (archType) {
        case ArchitectureType.DWAAS:
            return z.object({
                architecture: z.literal(archType),
                dataset: z.number(),
                nodes: z.number(),
                instance: z.number().min(0).max(instanceTypes.length - 1),
                hit_rate: z.number().min(0.0).max(1.0),

                cpu_cores: z.number().optional(),
                network_bandwidth: z.number().optional(),
                io_bandwidth: z.number().optional(),
                memory_bandwidth: z.number().optional(),
                total_memory_capacity_mb: z.number().optional(),
                scheduling: z.object({
                    policy: z.enum(["fcfs", "sjf", "ljf", "multi_level"]),
                    max_io_concurrency: z.number(),
                    max_cpu_concurrency: z.number(),
                }).optional(),
            });
        case ArchitectureType.DWAAS_AUTOSCALING:
            return z.object({
                architecture: z.literal(archType),
                dataset: z.number(),
                nodes: z.number(),
                instance: z.number().min(0).max(instanceTypes.length - 1),
                hit_rate: z.number().min(0.0).max(1.0),
                cold_start: z.number(),
                scaling: z.object({
                    policy: z.enum(["queue", "reactive", "predictive"]),
                }),

                cpu_cores: z.number().optional(),
                network_bandwidth: z.number().optional(),
                io_bandwidth: z.number().optional(),
                memory_bandwidth: z.number().optional(),
                total_memory_capacity_mb: z.number().optional(),
                scheduling: z.object({
                    policy: z.enum(["fcfs", "sjf", "ljf", "multi_level"]),
                    max_io_concurrency: z.number(),
                    max_cpu_concurrency: z.number(),
                }).optional(),
                use_spot_instances: z.boolean().optional(),
            });
        case ArchitectureType.ELASTIC_POOL:
            return z.object({
                architecture: z.literal(archType),
                dataset: z.number(),
                vpu: z.number(),
                hit_rate: z.number().min(0.0).max(1.0),
                cold_start: z.number(),
                scaling: z.object({
                    policy: z.enum(["queue", "reactive", "predictive"]),
                }),

                network_bandwidth: z.number().optional(),
                io_bandwidth: z.number().optional(),
                memory_bandwidth: z.number().optional(),
                total_memory_capacity_mb: z.number().optional(),
                scheduling: z.object({
                    policy: z.enum(["fcfs", "sjf", "ljf", "multi_level"]),
                    max_io_concurrency: z.number(),
                    max_cpu_concurrency: z.number(),
                }).optional(),
            });
        case ArchitectureType.QAAS:
            return z.object({
                architecture: z.literal(archType),
                dataset: z.number(),

                network_bandwidth: z.number().optional(),
            });
        case ArchitectureType.FAAS:
            return z.object({
                architecture: z.literal(archType),
                dataset: z.number(),
            });
        default:
            throw new Error(`Unknown architecture type: ${archType}`);
    }
};

// Third page: System Parameters Schema
export const sysParamsSchema = z.object({
    simulation: z.object({
        scale_factor_min: z.number(),
        scale_factor_max: z.number(),
        shuffle_percentage_min: z.number(),
        shuffle_percentage_max: z.number(),
        interrupt_probability: z.number(),
        interrupt_duration: z.number(),
        logging_interval: z.number(),
        default_max_duration: z.number(),
        default_estimator: z.string(),
        pm_p: z.number(),
        delta: z.number(),
        queue_agg: z.string(),
        materialization_fraction: z.number(),
        parallelizable_portion: z.number(),
        s3_bandwidth: z.number(),
    }),
    dwaas: z.object({
        cost_per_second_redshift: z.number(),
        spot_discount: z.number(),
        use_spot_instances: z.boolean(),
        cold_start_delay: z.number(),
    }),
    elastic_pool: z.object({
        cost_per_rpu_hour: z.number(),
        cache_warmup_gamma: z.number(),
        cold_start_delay: z.number(),
    }),
    qaas: z.object({
        cost_per_slot_hour: z.number(),
        qaas_cost_per_tb: z.number(),
        qaas_io_per_core_bw: z.number(),
        qaas_shuffle_bw_per_core: z.number(),
        qaas_base_cores: z.number(),
        qaas_base_time_limit: z.number(),
        core_alloc_window: z.number(),
    }),
});

export type ZodDWAAS = {
    architecture: z.ZodLiteral<string>;
    dataset: z.ZodNumber;
    nodes: z.ZodNumber;
    instance: z.ZodNumber;
    hit_rate: z.ZodNumber;
    cpu_cores: z.ZodOptional<z.ZodNumber>;
    network_bandwidth: z.ZodOptional<z.ZodNumber>;
    io_bandwidth: z.ZodOptional<z.ZodNumber>;
    memory_bandwidth: z.ZodOptional<z.ZodNumber>;
    total_memory_capacity_mb: z.ZodOptional<z.ZodNumber>;
    scheduling: z.ZodOptional<z.ZodObject<{
        policy: z.ZodEnum<{
            sjf: "sjf";
            fcfs: "fcfs";
            ljf: "ljf";
            multi_level: "multi_level";
        }>;
        max_io_concurrency: z.ZodNumber;
        max_cpu_concurrency: z.ZodNumber;
    }, z.core.$strip>>;
}

export type ZodDWAASAutoscaling = {
    architecture: z.ZodLiteral<string>;
    dataset: z.ZodNumber;
    nodes: z.ZodNumber;
    instance: z.ZodNumber;
    hit_rate: z.ZodNumber;
    cold_start: z.ZodNumber;
    scaling: z.ZodObject<{
        policy: z.ZodEnum<{
            queue: "queue";
            reactive: "reactive";
            predictive: "predictive";
        }>;
    }, z.core.$strip>;
    cpu_cores: z.ZodOptional<z.ZodNumber>;
    network_bandwidth: z.ZodOptional<z.ZodNumber>;
    io_bandwidth: z.ZodOptional<z.ZodNumber>;
    memory_bandwidth: z.ZodOptional<z.ZodNumber>;
    total_memory_capacity_mb: z.ZodOptional<z.ZodNumber>;
    scheduling: z.ZodOptional<z.ZodObject<{
        policy: z.ZodEnum<{
            sjf: "sjf";
            fcfs: "fcfs";
            ljf: "ljf";
            multi_level: "multi_level";
        }>;
        max_io_concurrency: z.ZodNumber;
        max_cpu_concurrency: z.ZodNumber;
    }, z.core.$strip>>;
    use_spot_instances: z.ZodOptional<z.ZodBoolean>;
}

export type ZodElasticPool = {
    architecture: z.ZodLiteral<string>;
    dataset: z.ZodNumber;
    vpu: z.ZodNumber;
    hit_rate: z.ZodNumber;
    cold_start: z.ZodNumber;
    scaling: z.ZodObject<{
        policy: z.ZodEnum<{
            queue: "queue";
            reactive: "reactive";
            predictive: "predictive";
        }>;
    }, z.core.$strip>;
    network_bandwidth: z.ZodOptional<z.ZodNumber>;
    io_bandwidth: z.ZodOptional<z.ZodNumber>;
    memory_bandwidth: z.ZodOptional<z.ZodNumber>;
    total_memory_capacity_mb: z.ZodOptional<z.ZodNumber>;
    scheduling: z.ZodOptional<z.ZodObject<{
        policy: z.ZodEnum<{
            sjf: "sjf";
            fcfs: "fcfs";
            ljf: "ljf";
            multi_level: "multi_level";
        }>;
        max_io_concurrency: z.ZodNumber;
        max_cpu_concurrency: z.ZodNumber;
    }, z.core.$strip>>;
}

export type ZodQAAS = {
    architecture: z.ZodLiteral<string>;
    dataset: z.ZodNumber;
    network_bandwidth: z.ZodNumber;
}

export type ZodFAAS = {
    architecture: z.ZodLiteral<string>;
    dataset: z.ZodNumber;
}
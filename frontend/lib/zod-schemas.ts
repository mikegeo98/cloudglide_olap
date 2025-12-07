import z from "zod";
import { instanceTypes } from "./config";

export const ArchitectureType = {
    DWAAS: "DWAAS",
    DWAAS_AUTOSCALING: "DWAAS_AUTOSCALING",
    ELASTIC_POOL: "ELASTIC_POOL",
    QAAS: "QAAS",
};

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
export const creatArchitectureSchema = (archType: typeof ArchitectureType[keyof typeof ArchitectureType]): z.ZodObject<ZodDWAAS | ZodQAAS> | z.ZodObject<z.ZodRawShape> => {
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
        case ArchitectureType.ELASTIC_POOL:
        case ArchitectureType.QAAS:
            return z.object({
                architecture: z.literal(archType),
                dataset: z.number(),

                network_bandwidth: z.number().default(10),
            });
        default:
            throw new Error(`Unknown architecture type: ${archType}`);
    }
};

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
        policy: z.ZodString;
        max_io_concurrency: z.ZodNumber;
        max_cpu_concurrency: z.ZodNumber;
    }, z.core.$strip>>;
}

export type ZodQAAS = {
    architecture: z.ZodLiteral<string>;
    dataset: z.ZodNumber;
    network_bandwidth: z.ZodNumber;
}
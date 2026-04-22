"use client"

import { instanceTypes } from "@/lib/config"

import { ColumnDef } from "@tanstack/react-table"

export type ArchitectureConfig = {
    architecture: string,
    nodes: number | undefined,
    hit_rate: number | undefined,
    instance: number | undefined,
    scaling_policy: string | undefined,
    cold_start: number | undefined,
    vpu: number | undefined,
}

export const diffColumns: ColumnDef<ArchitectureConfig>[] = [
    {
        id: "architecture",
        accessorKey: "architecture",
        header: "Architecture",
    },
    {
        id: "nodes",
        accessorKey: "nodes",
        header: "Nodes",
    },
    {
        id: "hit_rate",
        accessorKey: "hit_rate",
        header: "Hit Rate",
    },
    {
        id: "instance",
        accessorKey: "instance",
        header: "Base Instance",
        cell: ({ row }) => row.original.instance ? instanceTypes[row.original.instance].name : null
    },
    {
        id: "scaling_policy",
        accessorKey: "scaling_policy",
        header: "Scaling Policy",
    },
    {
        id: "cold_start",
        accessorKey: "cold_start",
        header: "Cold Start Delay",
    },
    {
        id: "vpu",
        accessorKey: "vpu",
        header: "Virtual Processing Units",
    },
    {
        id: "filename",
        accessorKey: "filename",
        header: "Filename",
    },
]
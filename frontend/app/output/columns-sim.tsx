"use client"

import { Checkbox } from "@/components/ui/checkbox"

import { ColumnDef } from "@tanstack/react-table"

export type Simulation = {
    job_id: number
    query_id: number
    database_id: number
    start: number
    start_timestamp: number
    end_timestamp: number
    queueing_delay: number
    buffer_delay: number
    io: number
    cpu: number
    shuffle: number
    query_duration_with_queue: number
    query_duration: number
    mon_cost: number
}

export const columns: ColumnDef<Simulation>[] = [
    {
        accessorKey: "job_id",
        header: "Job ID",
    },
    {
        accessorKey: "query_id",
        header: "Query ID",
    },
    {
        accessorKey: "database_id",
        header: "Database ID",
    },
    {
        accessorKey: "start",
        header: "Start",
    },
    {
        accessorKey: "start_timestamp",
        header: "Start Timestamp",
    },
    {
        accessorKey: "end_timestamp",
        header: "End Timestamp",
    },
    {
        accessorKey: "queueing_delay",
        header: "Queue Delay",
    },
    {
        accessorKey: "buffer_delay",
        header: "Buffer Delay",
    },
    {
        accessorKey: "io",
        header: "IO",
    },
    {
        accessorKey: "cpu",
        header: "CPU",
    },
    {
        accessorKey: "shuffle",
        header: "Shuffle",
    },
    {
        accessorKey: "query_duration_with_queue",
        header: "Query Duration with Queue",
    },
    {
        accessorKey: "query_duration",
        header: "Query Duration",
    },
    {
        accessorKey: "mon_cost",
        header: "Monetary Cost",
    },
]
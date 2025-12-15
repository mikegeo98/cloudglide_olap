"use client"

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
        cell: ({ row }) => row.original.io.toFixed(4)
    },
    {
        accessorKey: "cpu",
        header: "CPU",
        cell: ({ row }) => row.original.cpu.toFixed(4)
    },
    {
        accessorKey: "shuffle",
        header: "Shuffle",
        cell: ({ row }) => row.original.shuffle.toFixed(4)
    },
    {
        accessorKey: "query_duration_with_queue",
        header: "Query Duration with queueing",
        cell: ({ row }) => row.original.query_duration_with_queue.toFixed(4)
    },
    {
        accessorKey: "query_duration",
        header: "Query Duration",
        cell: ({ row }) => row.original.query_duration.toFixed(4)
    },
    {
        accessorKey: "mon_cost",
        header: "Monetary Cost",
        cell: ({ row }) => {
            return "$" + row.original.mon_cost.toFixed(4)
        },
    },
]
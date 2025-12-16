"use client"

import { Checkbox } from "@/components/ui/checkbox"
import { instanceTypes } from "@/lib/config"

import { ColumnDef } from "@tanstack/react-table"

export type SimRun = {
    filename: string,
    architecture: string | undefined,
    nodes: number | undefined,
    hit_rate: number | undefined,
    instance: number | undefined,
    scaling_policy: string | undefined,
    cold_start: number | undefined,
    vpu: number | undefined,
}

export const columns: ColumnDef<SimRun>[] = [
    {
        id: "select",
        header: ({ table }) => (
            <Checkbox
                checked={
                    table.getIsAllPageRowsSelected() ||
                    (table.getIsSomePageRowsSelected() && "indeterminate")
                }
                onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
                aria-label="Select all"
            />
        ),
        cell: ({ row }) => (
            <Checkbox
                checked={row.getIsSelected()}
                onCheckedChange={(value) => row.toggleSelected(!!value)}
                aria-label="Select row"
            />
        ),
        enableSorting: false,
        enableHiding: false,
    },
    {
        accessorKey: "filename",
        header: "Filename",
    },
    {
        accessorKey: "architecture",
        header: "Architecture",
    },
    {
        accessorKey: "nodes",
        header: "Nodes",
    },
    {
        accessorKey: "hit_rate",
        header: "Hit Rate",
    },
    {
        accessorKey: "instance",
        header: "Base Instance",
        cell: ({ row }) => row.original.instance ? instanceTypes[row.original.instance].name : null
    },
    {
        accessorKey: "scaling_policy",
        header: "Scaling Policy",
    },
    {
        accessorKey: "cold_start",
        header: "Cold Start Delay",
    },
    {
        accessorKey: "vpu",
        header: "Virtual Processing Units",
    },
]
"use client"

import {
    Select,
    SelectContent,
    SelectGroup,
    SelectItem,
    SelectLabel,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";

import React from "react";
import { columns, Simulation } from "./columns-sim";
import { DataTable } from "./data-table";

export default function Visualization({ data }: { data: Simulation[][] }) {
    const [sim, setSim] = React.useState<number>(0)
    const [rowSelection, setRowSelection] = React.useState<Record<number, boolean>>({})

    return (
        <div className="flex flex-col w-full h-full max-h-full gap-8 items-center overflow-hidden">
            <h1>Dashboard</h1>
            <div className="flex flex-col w-full h-full max-h-full gap-6 items-center overflow-hidden">
                <div className="flex justify-start items-start w-full">
                    <Select defaultValue="0" onValueChange={(e) => setSim(Number.parseInt(e, 10))}>
                        <SelectTrigger className="w-[180px]">
                            <SelectValue placeholder="Select a simulation" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectGroup>
                                <SelectLabel>Simulations</SelectLabel>
                                {data.map((_, index) => (
                                    <SelectItem key={"select_" + index} value={index + ""}>simulation_{index + 1}.csv</SelectItem>
                                ))}
                            </SelectGroup>
                        </SelectContent>
                    </Select>
                </div>
                <DataTable className="w-full max-h-full overflow-y-auto" columns={columns} data={data[sim]} rowSelection={rowSelection} setRowSelection={setRowSelection} />
            </div>
        </div>
    )
}
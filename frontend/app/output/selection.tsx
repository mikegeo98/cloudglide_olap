"use client"

import { Button } from "@/components/ui/button";

import React from "react";
import { columns, SimRun } from "./columns-select";
import { DataTable } from "./data-table";
import { Simulation } from "./columns-sim";
import Visualization from "./visualization";

export default function Selection({ data }: { data: { data: Simulation[][], files: string[] } }) {
    const [visualize, setVisualize] = React.useState(false)
    const [visualizedData, setVisualizedData] = React.useState(data.data)
    const [rowSelection, setRowSelection] = React.useState<Record<number, boolean>>({})

    function handleVisualize() {
        setVisualizedData(data.data.filter((_, index) => rowSelection[index]))
        setVisualize(true)
    }

    if (visualize) {
        return <Visualization data={visualizedData} />
    } else {
        return (
            <div className="flex flex-col gap-6 items-center">
                <DataTable className="w-[500px]" columns={columns} data={
                    data.files.map(f => {
                        return { filename: f } as SimRun
                    })
                } rowSelection={rowSelection} setRowSelection={setRowSelection} />
                <Button disabled={Object.keys(rowSelection).length === 0} variant="outline" className="bg-foreground text-background w-[180px]" onClick={handleVisualize}>
                    Visualize
                </Button>
            </div>
        )
    }
}
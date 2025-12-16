"use client"

import { Button } from "@/components/ui/button";
import { InputContext } from "@/components/provider";

import React from "react";
import { columns, SimRun } from "./columns-select";
import { DataTable } from "./data-table";
import { Simulation } from "./columns-sim";
import Visualization from "./visualization";

export default function Selection({ data }: { data: { data: Simulation[][], files: string[] } }) {
    const { data: input } = React.useContext(InputContext)
    const [visualize, setVisualize] = React.useState(false)
    const [visualizedData, setVisualizedData] = React.useState(data.data)
    const [filenameData, setFilenameData] = React.useState(data.files)
    const [rowSelection, setRowSelection] = React.useState<Record<number, boolean>>({})
    const [tableData, setTableData] = React.useState<SimRun[]>()

    function handleVisualize() {
        setVisualizedData(data.data.filter((_, index) => rowSelection[index]))
        setFilenameData(data.files.filter((_, index) => rowSelection[index]))
        setVisualize(true)
    }

    React.useEffect(() => {
        const srs = data.files.map(f => {
            const scenario = input.scenarios.find(s => s.name === f.split("_")[0])
            if (scenario) {
                return {
                    filename: f,
                    architecture: scenario.architecture,
                    nodes: scenario.nodes,
                    hit_rate: scenario.hit_rate,
                    instance: scenario.instance,
                    scaling_policy: scenario.scaling?.policy,
                    vpu: scenario.vpu,
                    cold_start: scenario.cold_start,
                } as SimRun
            } else {
                return { filename: f } as SimRun
            }
        })
        setTableData(srs)
    }, [])

    if (visualize) {
        return <Visualization data={visualizedData} filenames={filenameData} />
    } else {
        return (
            <div className="max-w-full flex flex-col gap-6 items-center overflow-hidden">
                {tableData ? <DataTable className="w-fit max-w-full overflow-hidden" columns={columns} data={tableData} rowSelection={rowSelection} setRowSelection={setRowSelection} /> : null}
                <Button disabled={Object.keys(rowSelection).length === 0} variant="outline" className="bg-foreground text-background w-[180px]" onClick={handleVisualize}>
                    Visualize
                </Button>
            </div>
        )
    }
}
import { InputContext } from "@/components/provider";
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";

import { Simulation } from "./columns-sim";
import React from "react";
import { DataTable } from "./data-table";
import { columns, SimRun } from "./columns-select";
import { CartesianGrid, Label, LabelList, Line, LineChart, XAxis, YAxis } from "recharts";
import { ChartConfig, ChartContainer, ChartLegend, ChartLegendContent } from "@/components/ui/chart";

const latencyCostChartConfig = {
    avgQueryLatency: {
        label: "Avg. Query Latency",
        color: "var(--chart-3)",
    },
    cost: {
        label: "CPU Phase",
        color: "var(--chart-4)",
    },
} satisfies ChartConfig

export default function MultipleRuns({ data, filenames }: { data: Simulation[][], filenames: string[] }) {
    const { data: input } = React.useContext(InputContext)
    const [rowSelection, setRowSelection] = React.useState<Record<number, boolean>>({})
    const [tableData, setTableData] = React.useState<SimRun[]>()

    React.useEffect(() => {
        const srs = filenames.map(f => {
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

    return (
        <div className="flex flex-col w-full h-full max-h-full gap-4 items-center overflow-hidden">
            {tableData ? <DataTable className="w-full max-w-full max-h-[200px] overflow-x-hidden overflow-y-auto" columns={columns} data={tableData} rowSelection={rowSelection} setRowSelection={setRowSelection} /> : null}
            <div className="grid grid-cols-2 gap-4 w-full">
                <Card className="w-full">
                    <CardHeader>
                        <CardTitle>Latency–Cost Scatter Plot</CardTitle>
                        <CardDescription>
                            Each point represents a simulation run.<br />
                            x-axis: average query latency (sec)<br />
                            y-axis: total cost (USD)
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ChartContainer config={latencyCostChartConfig}>
                            <LineChart
                                accessibilityLayer
                                data={[]}
                                margin={{
                                    top: 30,
                                    right: 12,
                                    bottom: 10,
                                }}
                            >
                                <CartesianGrid vertical={false} />
                                <XAxis
                                    dataKey="avgQueryLatency"
                                    tickLine={true}
                                    axisLine={false}
                                    label={<Label position="middle" dy={15}>Avg. Query Latency (sec)</Label>}
                                />
                                <YAxis
                                    axisLine={false}
                                    tickLine={false}
                                    label={<Label position="middle" angle={270}>Cost ($)</Label>}
                                />
                                <Line
                                    dataKey="arrivalsCount"
                                    type="linear"
                                    stroke="var(--chart-3)"
                                    strokeWidth={2}
                                    dot={{
                                        fill: "var(--chart-3)",
                                    }}
                                    activeDot={{
                                        r: 6,
                                    }}
                                    isAnimationActive={false}
                                >
                                    <LabelList
                                        position="top"
                                        offset={12}
                                        className="fill-foreground"
                                        fontSize={12}
                                        formatter={(value: number) => {
                                            if (value !== 0) {
                                                return value
                                            }
                                        }}
                                    />
                                </Line>
                                <Line
                                    dataKey="activesCount"
                                    type="linear"
                                    stroke="var(--chart-4)"
                                    strokeWidth={2}
                                    dot={{
                                        fill: "var(--chart-4)",
                                    }}
                                    activeDot={{
                                        r: 6,
                                    }}
                                    isAnimationActive={false}
                                >
                                    <LabelList
                                        position="top"
                                        offset={12}
                                        className="fill-foreground"
                                        fontSize={12}
                                        formatter={(value: number) => {
                                            if (value !== 0) {
                                                return value
                                            }
                                        }}
                                    />
                                </Line>
                                <ChartLegend content={<ChartLegendContent />} />
                            </LineChart>
                        </ChartContainer>
                    </CardContent>
                </Card>
                <Card className="w-full">
                    <CardHeader>
                        <CardTitle>Variable Sensitivity Plot</CardTitle>
                        <CardDescription>
                            Shows how changing a specific variable affects latency and cost.<br />
                            x-axis: varying input parameter (e.g., hit rate, number of nodes, arrival rate)<br />
                            y-axis: latency (sec) or cost (USD)
                        </CardDescription>
                    </CardHeader>
                    <CardContent>

                    </CardContent>
                </Card>
            </div>
        </div>
    )
}
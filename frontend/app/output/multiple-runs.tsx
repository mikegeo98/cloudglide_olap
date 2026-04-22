import {
    Card,
    CardAction,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";

import { Simulation } from "./columns-sim";
import React from "react";
import { DataTable } from "./data-table";
import { columns, SimRun } from "./columns-select";
import { Bar, BarChart, CartesianGrid, ComposedChart, Label, Line, Scatter, XAxis, YAxis } from "recharts";
import { ChartContainer, ChartLegend, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ArrowLeftRight } from "lucide-react";
import { ButtonGroup } from "@/components/ui/button-group";

export default function MultipleRuns({ data, filenames }: { data: Simulation[][], filenames: string[] }) {
    const [rowSelection, setRowSelection] = React.useState<Record<number, boolean>>({})
    const [tableData, setTableData] = React.useState<SimRun[]>()
    const [latencyCostDots, setLatencyCostDots] = React.useState<{ "avgLatency": number, "cost": number }[]>()
    const [pareto, setPareto] = React.useState<{ "avgLatency": number, "cost": number }[]>()
    const [latencyBreakdown, setLatencyBreakdown] = React.useState<{ filename: string, io: number, cpu: number, shuffle: number }[]>()
    const [sensitivityData, setSensitivityData] = React.useState<{ [key: string]: string | number | undefined; avgLatency: number; cost: number }[]>()
    const [showLatencyInSens, setShowLatencyInSens] = React.useState(false)
    const [sweepVariable, setSweepVariable] = React.useState<string>("nodes")

    React.useEffect(() => {
        async function fetchScenariosAndBuildTable() {
            try {
                const response = await fetch("/api/scenarios");
                const customJson = await response.json();
                
                const srs = filenames.map(f => {
                    const scenario = customJson.scenarios?.find((s: { name: string }) => s.name === f.split("_")[0])
                    if (scenario) {
                        return {
                            filename: f,
                            architecture: scenario.architecture,
                            nodes: scenario.nodes,
                            hit_rate: scenario.hit_rate,
                            instance: scenario.instance,
                            scaling_policy: scenario.scaling?.policy,
                            scheduling_policy: scenario.scheduling?.policy,
                            network_bandwidth: scenario.network_bandwidth,
                            vpu: scenario.vpu,
                            cold_start: scenario.cold_start,
                        } as SimRun
                    } else {
                        return { filename: f } as SimRun
                    }
                })
                setTableData(srs)
            } catch (error) {
                console.error("Error fetching scenarios:", error);
                setTableData(filenames.map(f => ({ filename: f } as SimRun)))
            }
        }
        
        fetchScenariosAndBuildTable();
    }, [filenames])

    React.useEffect(() => {
        if (rowSelection && Object.keys(rowSelection).length > 0) {
            const selectedDots = Object.entries(rowSelection)
                .filter(([_, isSelected]) => isSelected)
                .map(([idx]) => {
                    const simData = data[parseInt(idx)]
                    if (simData && simData.length > 0) {
                        const avgLatency = simData.reduce((prev, curr) => prev + curr.query_duration, 0) / simData.length
                        const cost = simData.reduce((prev, curr) => prev + curr.mon_cost, 0) / simData.length
                        return { avgLatency, cost }
                    }
                    return null
                })
                .filter((dot): dot is { avgLatency: number, cost: number } => dot !== null)
                .toSorted((a, b) => a.avgLatency - b.avgLatency)
            setLatencyCostDots(selectedDots)

            // Calculate Pareto front - a point is on the Pareto front if no other point has both lower latency and lower cost
            const paretoFront = selectedDots.filter((dot, _, allDots) => {
                return !allDots.some(other => other.avgLatency < dot.avgLatency && other.cost < dot.cost)
            })
            setPareto(paretoFront)

            // Detect if there's a single sweep variable across selected runs
            const selectedTableData = Object.entries(rowSelection)
                .filter(([_, isSelected]) => isSelected)
                .map(([idx]) => tableData ? tableData[parseInt(idx)] : null)
                .filter((sr): sr is SimRun => sr !== null)

            // Check which variables differ across selected runs
            const varyingVariables: { key: keyof SimRun; values: (string | number | undefined)[] }[] = []
            const keysToCheck: (keyof SimRun)[] = ["nodes", "hit_rate", "vpu", "cold_start", "scaling_policy", "scheduling_policy", "network_bandwidth"]

            keysToCheck.forEach(key => {
                const uniqueValues = [...new Set(selectedTableData.map(sr => sr[key]))]
                if (uniqueValues.length > 1) {
                    varyingVariables.push({ key, values: uniqueValues })
                }
            })

            // If exactly one variable varies, we can auto-generate sensitivity plot data
            if (varyingVariables.length >= 1) {
                setSweepVariable(varyingVariables[0].key)
                const sweepVar = varyingVariables[0].key
                const sensitivityPlotData = Object.entries(rowSelection)
                    .filter(([_, isSelected]) => isSelected)
                    .map(([idx]) => {
                        const simData = data[parseInt(idx)]
                        const tableEntry = tableData ? tableData[parseInt(idx)] : null
                        if (simData && simData.length > 0 && tableEntry) {
                            const avgLatency = simData.reduce((prev, curr) => prev + curr.query_duration, 0) / simData.length
                            const cost = simData.reduce((prev, curr) => prev + curr.mon_cost, 0) / simData.length
                            const dataPoint: { [key: string]: string | number | undefined; avgLatency: number; cost: number } = { avgLatency, cost }
                            varyingVariables.forEach(vv => {
                                dataPoint[vv.key] = tableEntry[vv.key]
                            })
                            return dataPoint
                        }
                        return null
                    })
                    .filter((entry): entry is { [key: string]: string | number | undefined; avgLatency: number; cost: number } => entry !== null)
                    .sort((a, b) => Number(a[sweepVar]) - Number(b[sweepVar]))

                const colors = ['var(--color-chart-1)', 'var(--color-chart-2)', 'var(--color-chart-3)', 'var(--color-chart-4)', 'var(--color-chart-5)', 'var(--color-chart-6)']
                setSensitivityData(sensitivityPlotData.map((point, index) => ({
                    ...point,
                    name: tableData?.[parseInt(Object.entries(rowSelection).filter(([_, isSelected]) => isSelected)[index][0])]?.filename,
                    color: colors[index % colors.length],
                })))
            }

            const selectedResourceUsage = Object.entries(rowSelection)
                .filter(([_, isSelected]) => isSelected)
                .map(([idx]) => {
                    const simData = data[parseInt(idx)]

                    const io = parseFloat((simData.reduce((prev, curr) => prev + curr.io, 0) / simData.length).toFixed(4))
                    const cpu = parseFloat((simData.reduce((prev, curr) => prev + curr.cpu, 0) / simData.length).toFixed(4))
                    const shuffle = parseFloat((simData.reduce((prev, curr) => prev + curr.shuffle, 0) / simData.length).toFixed(4))
                    return { filename: filenames[parseInt(idx)], io, cpu, shuffle }
                })
            setLatencyBreakdown(selectedResourceUsage)
        } else {
            setLatencyCostDots([])
            setPareto([])
            setLatencyBreakdown([])
        }
    }, [data, rowSelection])

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
                        <ChartContainer config={{}}>
                            <ComposedChart
                                data={latencyCostDots}
                                margin={{
                                    top: 30,
                                    right: 12,
                                    left: 20,
                                    bottom: 10,
                                }}
                            >
                                <CartesianGrid vertical={false} />
                                <XAxis
                                    dataKey="avgLatency"
                                    unit={" s"}
                                    tickLine={true}
                                    axisLine={false}
                                    label={<Label position="middle" dy={15}>Avg. Query Latency</Label>}
                                    tickFormatter={(value: number) => (value).toFixed(2)}
                                    allowDuplicatedCategory={false}
                                    type="number"
                                    domain={[0, 'dataMax']}
                                />
                                <YAxis
                                    dataKey="cost"
                                    unit={" $"}
                                    axisLine={false}
                                    tickLine={false}
                                    type="number"
                                    domain={[0, 'dataMax']}
                                    label={<Label position="middle" angle={270} dx={-30}>Cost</Label>}
                                    tickFormatter={(value: number) => (value).toFixed(2)}
                                />
                                <ChartTooltip
                                    cursor={false}
                                    content={<ChartTooltipContent hideLabel />}
                                />
                                {latencyCostDots && latencyCostDots.map((dot, index) => (
                                    <Scatter key={index} isAnimationActive={false} data={[dot]} dataKey="cost" fill="red" shape="wye" />
                                ))}
                                <Line type="monotone" isAnimationActive={false} data={pareto} dataKey="cost" stroke="black" name="paretoFront" />
                                <ChartLegend payload={[
                                    { value: "Simulation Run", type: "wye", color: "red" },
                                    { value: "Pareto Front", type: "line", color: "black" },
                                ]} />
                            </ComposedChart>
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
                        <CardAction>
                            <ButtonGroup orientation="vertical">
                                <Button className="w-full" variant="outline" onClick={() => setShowLatencyInSens(!showLatencyInSens)}>
                                    <ArrowLeftRight />
                                    {showLatencyInSens ? "Cost" : "Latency"}
                                </Button>
                                <Select value={sweepVariable} onValueChange={(e) => setSweepVariable(e)}>
                                    <SelectTrigger className="w-full">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectGroup>
                                            <SelectItem value="nodes">Number of Nodes</SelectItem>
                                            <SelectItem value="vpu">Number of VPUs</SelectItem>
                                            <SelectItem value="hit_rate">Hit Rate</SelectItem>
                                            <SelectItem value="network_bandwidth">Network Bandwidth</SelectItem>
                                            <SelectItem value="cold_start">Cold Start Delay</SelectItem>
                                        </SelectGroup>
                                    </SelectContent>
                                </Select>
                            </ButtonGroup>
                        </CardAction>
                    </CardHeader>
                    <CardContent>
                        <ChartContainer config={{}} className="min-h-[200px] w-full">
                            <ComposedChart
                                data={sensitivityData}
                                margin={{
                                    top: 30,
                                    right: 12,
                                    left: 20,
                                    bottom: 10,
                                }}
                            >
                                <CartesianGrid />
                                <XAxis
                                    dataKey={sweepVariable}
                                    tickLine={false}
                                    axisLine={false}
                                    type="number"
                                    domain={[0, 'dataMax']}
                                    label={<Label position="middle" dy={15}>{sweepVariable}</Label>}
                                />
                                {showLatencyInSens ? (
                                    <YAxis
                                        dataKey={"avgLatency"}
                                        unit={" s"}
                                        axisLine={false}
                                        tickLine={false}
                                        type="number"
                                        domain={[0, 'dataMax']}
                                        label={<Label position="middle" angle={270} dx={-30}>Avg. Query Latency</Label>}
                                        tickFormatter={(value: number) => value.toFixed(1)}
                                    />
                                ) : (
                                    <YAxis
                                        dataKey={"cost"}
                                        unit={" $"}
                                        axisLine={false}
                                        tickLine={false}
                                        type="number"
                                        domain={[0, 'dataMax']}
                                        label={<Label position="middle" angle={270} dx={-30}>Cost</Label>}
                                        tickFormatter={(value: number) => value.toFixed(2)}
                                    />
                                )}
                                {sensitivityData && sensitivityData.map((dot, index) => (
                                    <Scatter key={index} isAnimationActive={false} data={[dot]} dataKey={showLatencyInSens ? "avgLatency" : "cost"} fill={dot.color as string} shape="wye" />
                                ))}
                                <ChartLegend payload={
                                    sensitivityData && sensitivityData.length > 0 ? sensitivityData.map(dot => ({
                                        value: dot.name as string,
                                        type: "wye",
                                        color: dot.color as string,
                                    })) : []
                                } />
                            </ComposedChart>
                        </ChartContainer>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader>
                        <CardTitle>Component Cost Decomposition</CardTitle>
                        <CardDescription>
                            Average resource usage breakdown across selected simulation runs.<br />
                            Shows the contribution of I/O, CPU, and Shuffle to overall query latency.
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ChartContainer config={{}}>
                            <BarChart
                                accessibilityLayer
                                data={latencyBreakdown}
                                margin={{ top: 20, right: 30, bottom: 5 }}
                            >
                                <CartesianGrid vertical={false} />
                                <XAxis
                                    dataKey="filename"
                                    tickLine={false}
                                    tickMargin={10}
                                    axisLine={false}
                                />
                                <YAxis
                                    tickLine={false}
                                    tickMargin={10}
                                    axisLine={false}
                                />
                                <ChartTooltip
                                    cursor={false}
                                    content={<ChartTooltipContent hideLabel />}
                                />
                                <Bar dataKey="io" stackId="a" fill="var(--color-chart-1)" />
                                <Bar dataKey="cpu" stackId="a" fill="var(--color-chart-2)" />
                                <Bar dataKey="shuffle" stackId="a" fill="var(--color-chart-3)" />
                                <ChartLegend payload={[
                                    { value: "I/O", type: "square", color: "var(--color-chart-1)" },
                                    { value: "CPU", type: "square", color: "var(--color-chart-2)" },
                                    { value: "Shuffle", type: "square", color: "var(--color-chart-3)" },
                                ]} />
                            </BarChart>
                        </ChartContainer>
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}
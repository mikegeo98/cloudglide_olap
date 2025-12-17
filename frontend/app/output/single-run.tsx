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
import {
    Card,
    CardAction,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import {
    ChartConfig,
    ChartContainer,
    ChartLegend,
    ChartLegendContent,
    ChartTooltip,
    ChartTooltipContent,
} from "@/components/ui/chart";
import { Button } from "@/components/ui/button";

import { columns, Simulation } from "./columns-sim";
import { DataTable } from "./data-table";
import { Bar, BarChart, CartesianGrid, Line, LineChart, LabelList, XAxis } from "recharts";
import React from "react";

const simChartConfig = {
    io: {
        label: "I/O Phase",
        color: "var(--chart-3)",
    },
    cpu: {
        label: "CPU Phase",
        color: "var(--chart-4)",
    },
    shuffle: {
        label: "Shuffle Phase",
        color: "var(--chart-5)",
    },
} satisfies ChartConfig

export default function SingleRun({ data, filenames }: { data: Simulation[][], filenames: string[] }) {
    const [sim, setSim] = React.useState<number>(0)
    const [buckets, setBuckets] = React.useState<number>(10)
    const [rowSelection, setRowSelection] = React.useState<Record<number, boolean>>({})
    const [simData, setSimData] = React.useState<{ simTime: number, io: number, cpu: number, shuffle: number }[]>([])
    const [timelineData, setTimelineData] = React.useState<{ simTime: number, arrivalsCount: number }[]>([])
    const [histoData, setHistoData] = React.useState<{ sec: number, finishedCount: number }[]>([])
    const [summaryTable, setSummaryTable] = React.useState<{
        queueDelayAvg: number,
        bufferDelayAvg: number,
        meanIO: number,
        meanCPU: number,
        meanShuffle: number,
        meanQueryLatency: number,
        meanQuery: number,
        medianQueryLatency: number,
        medianQuery: number,
        percQueryLatency: number,
        percQuery: number,
        totalPrice: number,
    }>()
    const [showSimCard, setShowSimCard] = React.useState(false)

    React.useEffect(() => {
        const maxTime = data[sim].reduce((max, item) => item.query_duration > max ? item.query_duration : max, 0)
        setHistoData(Array.from({ length: Math.ceil(maxTime) + 1 }, (_, i) => {
            const count = data[sim].filter(item => Math.ceil(item.query_duration) === i).length
            return { sec: i, finishedCount: count }
        }))

        const lastTimestamp = data[sim][data[sim].length - 1].start_timestamp
        let lastBucket = 0
        const timelineArray = []
        for (let i = 0; i <= lastTimestamp; i += lastTimestamp / buckets) {
            const count = data[sim].filter(item => item.start_timestamp <= i && item.start_timestamp > lastBucket).length
            lastBucket = i
            timelineArray.push({ simTime: Math.ceil(i), arrivalsCount: count })
        }
        setTimelineData(timelineArray)

        setSimData(data[sim].map(s => {
            return {
                simTime: s.start_timestamp,
                io: s.io,
                cpu: s.cpu,
                shuffle: s.shuffle,
            }
        }))

        setSummaryTable({
            queueDelayAvg: parseFloat((data[sim].reduce((prev, curr) => prev + curr.queueing_delay, 0) / data[sim].length).toFixed(4)),
            bufferDelayAvg: parseFloat((data[sim].reduce((prev, curr) => prev + curr.buffer_delay, 0) / data[sim].length).toFixed(4)),
            meanIO: parseFloat((data[sim].reduce((prev, curr) => prev + curr.io, 0) / data[sim].length).toFixed(4)),
            meanCPU: parseFloat((data[sim].reduce((prev, curr) => prev + curr.cpu, 0) / data[sim].length).toFixed(4)),
            meanShuffle: parseFloat((data[sim].reduce((prev, curr) => prev + curr.shuffle, 0) / data[sim].length).toFixed(4)),
            meanQueryLatency: parseFloat((data[sim].reduce((prev, curr) => prev + curr.query_duration, 0) / data[sim].length).toFixed(4)),
            meanQuery: parseFloat((data[sim].reduce((prev, curr) => prev + curr.query_duration_with_queue, 0) / data[sim].length).toFixed(4)),
            medianQueryLatency: parseFloat(data[sim].toSorted((a, b) => b.query_duration - a.query_duration)[Math.floor(data[sim].length / 2)].query_duration.toFixed(4)),
            medianQuery: parseFloat(data[sim].toSorted((a, b) => b.query_duration_with_queue - a.query_duration_with_queue)[Math.floor(data[sim].length / 2)].query_duration_with_queue.toFixed(4)),
            percQueryLatency: parseFloat(data[sim].toSorted((a, b) => b.query_duration - a.query_duration)[Math.floor(data[sim].length * 0.05)].query_duration.toFixed(4)),
            percQuery: parseFloat(data[sim].toSorted((a, b) => b.query_duration_with_queue - a.query_duration_with_queue)[Math.floor(data[sim].length * 0.05)].query_duration_with_queue.toFixed(4)),
            totalPrice: parseFloat((data[sim].reduce((prev, curr) => prev + curr.mon_cost, 0)).toFixed(4)),
        })
    }, [sim, data, buckets])

    return (
        <div className="flex flex-col w-full h-full max-h-full gap-4 items-center overflow-hidden">
            <div className="flex justify-between items-center gap-4 w-full">
                <Select defaultValue={filenames[0]} onValueChange={(e) => setSim(filenames.indexOf(e))}>
                    <SelectTrigger className="w-[180px]">
                        <SelectValue placeholder="Select a simulation" />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectGroup>
                            <SelectLabel>Simulations</SelectLabel>
                            {filenames.map((name) => (
                                <SelectItem key={"select_" + name} value={name}>{name}</SelectItem>
                            ))}
                        </SelectGroup>
                    </SelectContent>
                </Select>
                <Button variant="default" onClick={() => setShowSimCard(!showSimCard)}>
                    {showSimCard ? "Hide Simulation" : "Show Simulation"}
                </Button>
            </div>
            {
                showSimCard ? (
                    <Card className="w-full">
                        <CardHeader>
                            <CardTitle>Simulation</CardTitle>
                            <CardAction>
                                <Button variant="default" onClick={() => {
                                    setShowSimCard(false)
                                    setTimeout(() => { // little hack to restart the animation
                                        setShowSimCard(true)
                                    }, 1)
                                }}>
                                    Rerun
                                </Button>
                            </CardAction>
                        </CardHeader>
                        <CardContent>
                            <ChartContainer config={simChartConfig} className="h-[300px] w-full">
                                <LineChart
                                    accessibilityLayer
                                    data={simData}
                                    margin={{
                                        left: 12,
                                        right: 12,
                                    }}
                                >
                                    <CartesianGrid strokeDasharray="10 10" />
                                    <XAxis
                                        dataKey="simTime"
                                        tickLine={true}
                                        axisLine={false}
                                        tickMargin={8}
                                    />
                                    <ChartTooltip content={<ChartTooltipContent className="w-[175px]" labelFormatter={() => "Phases"} />} />
                                    <Line
                                        dataKey="cpu"
                                        type="monotone"
                                        stroke="var(--chart-4)"
                                        strokeWidth={2}
                                        animationDuration={8000}
                                        animationBegin={200}
                                        dot={false}
                                    />
                                    <Line
                                        dataKey="io"
                                        type="monotone"
                                        stroke="var(--chart-3)"
                                        strokeWidth={2}
                                        animationDuration={8000}
                                        animationBegin={0}
                                        dot={false}
                                    />
                                    <Line
                                        dataKey="shuffle"
                                        type="monotone"
                                        stroke="var(--chart-5)"
                                        strokeWidth={2}
                                        animationDuration={8000}
                                        animationBegin={400}
                                        dot={false}
                                    />
                                    <ChartLegend content={<ChartLegendContent />} />
                                </LineChart>
                            </ChartContainer>
                        </CardContent>
                    </Card>
                ) : null
            }
            <div className="grid grid-cols-2 gap-4">
                <DataTable className="w-full max-h-[300px] overflow-auto" columns={columns} data={data[sim]} rowSelection={rowSelection} setRowSelection={setRowSelection} />
                <Card className="w-full">
                    <CardHeader>
                        <CardTitle>Summary Table</CardTitle>
                    </CardHeader>
                    <CardContent className="text-sm">
                        <div className="grid grid-cols-2">
                            <p><span className="text-muted-foreground">Queue Delay Avg:</span> {summaryTable?.queueDelayAvg}</p>
                            <p><span className="text-muted-foreground">Buffer Delay Avg:</span> {summaryTable?.bufferDelayAvg}</p>
                        </div>
                        <div className="grid grid-cols-3">
                            <p><span className="text-muted-foreground">Mean I/O:</span> {summaryTable?.meanIO}</p>
                            <p><span className="text-muted-foreground">Mean CPU:</span> {summaryTable?.meanCPU}</p>
                            <p><span className="text-muted-foreground">Mean Shuffle:</span> {summaryTable?.meanShuffle}</p>
                        </div>
                        <div className="grid grid-cols-2">
                            <p><span className="text-muted-foreground">Mean Query Latency:</span> {summaryTable?.meanQueryLatency}</p>
                            <p><span className="text-muted-foreground">Mean Query (with queueing):</span> {summaryTable?.meanQuery}</p>
                            <p><span className="text-muted-foreground">Median Query Latency:</span> {summaryTable?.medianQueryLatency}</p>
                            <p><span className="text-muted-foreground">Median Query (with queueing):</span> {summaryTable?.medianQuery}</p>
                            <p><span className="text-muted-foreground">95th Percentile Query Latency:</span> {summaryTable?.percQueryLatency}</p>
                            <p><span className="text-muted-foreground">95th Percentile Query (with queueing):</span> {summaryTable?.percQuery}</p>
                            <p><span className="text-muted-foreground">Total Price:</span> ${summaryTable?.totalPrice}</p>
                        </div>
                    </CardContent>
                </Card>
            </div>
            <div className="w-full grid grid-cols-2 gap-4">
                <Card className="w-full">
                    <CardHeader>
                        <CardTitle>Timeline or Activity Chart</CardTitle>
                        <CardDescription>
                            x-axis: simulation time<br />
                            y-axis: number of arrivals per minute
                        </CardDescription>
                        <CardAction className="flex items-center gap-3">
                            Buckets:
                            <Select defaultValue={"10"} onValueChange={(e) => setBuckets(Number.parseInt(e))}>
                                <SelectTrigger className="w-fit">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value={"10"}>10</SelectItem>
                                    <SelectItem value={"20"}>20</SelectItem>
                                    <SelectItem value={"50"}>50</SelectItem>
                                    <SelectItem value={"100"}>100</SelectItem>
                                </SelectContent>
                            </Select>
                        </CardAction>
                    </CardHeader>
                    <CardContent>
                        <ChartContainer config={{}}>
                            <LineChart
                                accessibilityLayer
                                data={timelineData}
                                margin={{
                                    top: 30,
                                    left: 12,
                                    right: 12,
                                }}
                            >
                                <CartesianGrid vertical={false} />
                                <XAxis
                                    dataKey="simTime"
                                    tickLine={false}
                                    axisLine={false}
                                    tickMargin={8}
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
                            </LineChart>
                        </ChartContainer>
                    </CardContent>
                </Card>
                <Card className="w-full">
                    <CardHeader>
                        <CardTitle>Histogram of per-query latencies</CardTitle>
                        <CardDescription>
                            x-axis: query duration in seconds<br />
                            y-axis: number of queries finished under the given duration
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ChartContainer config={{}}>
                            <BarChart
                                accessibilityLayer
                                data={histoData}
                                margin={{
                                    top: 30,
                                }}
                            >
                                <CartesianGrid vertical={false} />
                                <XAxis
                                    dataKey="sec"
                                    tickLine={false}
                                    tickMargin={10}
                                    axisLine={false}
                                />
                                <Bar dataKey="finishedCount" fill="var(--chart-3)" radius={8}>
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
                                </Bar>
                            </BarChart>
                        </ChartContainer>
                    </CardContent>
                </Card>
            </div>
        </div >
    )
}
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
import {
    InputGroup,
    InputGroupAddon,
    InputGroupText,
} from "@/components/ui/input-group";
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";

import { columns, Simulation } from "./columns-sim";
import { Bar, BarChart, CartesianGrid, Line, LineChart, LabelList, XAxis, Label, YAxis } from "recharts";
import React from "react";
import { ChartColumn, ChartSpline, Download } from "lucide-react";
import BottleneckPanel from "@/components/bottleneck-panel";
import { downloadCSV, downloadJSON } from "@/lib/export-utils";

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

const timelineChartConfig = {
    arrivalsCount: {
        label: "Arrivals per Bucket",
        color: "var(--chart-3)",
    },
    activesCount: {
        label: "Active Queries per Bucket",
        color: "var(--chart-4)",
    },
} satisfies ChartConfig

export default function SingleRun({ data, filenames }: { data: Simulation[][], filenames: string[] }) {
    const [sim, setSim] = React.useState<number>(0)
    const [buckets, setBuckets] = React.useState<number>(10)
    const [simData, setSimData] = React.useState<{ simTime: number, io: number, cpu: number, shuffle: number }[]>([])
    const [timelineData, setTimelineData] = React.useState<{ simTime: number, arrivalsCount: number, activesCount: number }[]>([])
    const [histoData, setHistoData] = React.useState<{ sec: number, finishedCount: number, accFinishedCount: number }[]>([])
    const [isCDF, setCDF] = React.useState<boolean>(false)
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
            const acc = data[sim].filter(item => Math.ceil(item.query_duration) < i).length
            return { sec: i, finishedCount: count, accFinishedCount: acc }
        }))

        const lastTimestamp = data[sim][data[sim].length - 1].start_timestamp
        let lastBucket = 0
        const timelineArray = []
        for (let i = 0; i <= lastTimestamp; i += lastTimestamp / buckets) {
            const arrivalsCount = data[sim].filter(item => item.start_timestamp <= i && item.start_timestamp > lastBucket).length
            const sillActiveFromPrev = data[sim].filter(item => item.start_timestamp <= lastBucket && item.end_timestamp >= i).length
            lastBucket = i
            timelineArray.push({ simTime: Math.ceil(i), arrivalsCount: arrivalsCount, activesCount: arrivalsCount + sillActiveFromPrev })
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
            // mon_cost is the total simulation cost repeated for each row, so use first row
            totalPrice: parseFloat((data[sim].length > 0 ? data[sim][0].mon_cost : 0).toFixed(4)),
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
                <div className="flex gap-2">
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={() => downloadCSV(data[sim], `${filenames[sim]}_data.csv`)}
                    >
                        <Download className="h-4 w-4 mr-2" />
                        Export CSV
                    </Button>
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={() => downloadJSON({
                            filename: filenames[sim],
                            summary: summaryTable,
                            data: data[sim]
                        }, `${filenames[sim]}_full.json`)}
                    >
                        <Download className="h-4 w-4 mr-2" />
                        Export JSON
                    </Button>
                    <Button variant="default" onClick={() => setShowSimCard(!showSimCard)}>
                        {showSimCard ? "Hide Simulation" : "Show Simulation"}
                    </Button>
                </div>
            </div>

            {/* Performance Analysis + Summary Table side by side */}
            <div className="grid grid-cols-2 gap-4 w-full">
                <BottleneckPanel data={data[sim]} compact />
                <Card className="w-full">
                    <CardHeader className="pb-2">
                        <CardTitle className="text-lg">Summary Statistics</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <Table className="text-sm">
                            <TableHeader>
                                <TableRow className="bg-secondary">
                                    <TableHead className="py-1.5">Metric</TableHead>
                                    <TableHead className="text-right py-1.5">Value</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                <TableRow>
                                    <TableCell className="py-1.5">Mean Query Latency</TableCell>
                                    <TableCell className="text-right py-1.5 font-mono">{summaryTable?.meanQueryLatency}s</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="py-1.5">Median Query Latency</TableCell>
                                    <TableCell className="text-right py-1.5 font-mono">{summaryTable?.medianQueryLatency}s</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="py-1.5">P95 Query Latency</TableCell>
                                    <TableCell className="text-right py-1.5 font-mono">{summaryTable?.percQueryLatency}s</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="py-1.5">Mean I/O Time</TableCell>
                                    <TableCell className="text-right py-1.5 font-mono">{summaryTable?.meanIO}s</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="py-1.5">Mean CPU Time</TableCell>
                                    <TableCell className="text-right py-1.5 font-mono">{summaryTable?.meanCPU}s</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="py-1.5">Mean Shuffle Time</TableCell>
                                    <TableCell className="text-right py-1.5 font-mono">{summaryTable?.meanShuffle}s</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="py-1.5">Queue Delay Avg</TableCell>
                                    <TableCell className="text-right py-1.5 font-mono">{summaryTable?.queueDelayAvg}s</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="py-1.5">Buffer Delay Avg</TableCell>
                                    <TableCell className="text-right py-1.5 font-mono">{summaryTable?.bufferDelayAvg}s</TableCell>
                                </TableRow>
                                <TableRow className="bg-muted/50">
                                    <TableCell className="py-1.5 font-semibold">Total Cost</TableCell>
                                    <TableCell className="text-right py-1.5 font-mono font-semibold">${summaryTable?.totalPrice}</TableCell>
                                </TableRow>
                            </TableBody>
                        </Table>
                    </CardContent>
                </Card>
            </div>

            {showSimCard && (
                <Card className="w-full">
                    <CardHeader>
                        <CardTitle>Simulation</CardTitle>
                        <CardDescription>
                            Simulating the process of query execution over time, showing the resource usage phases (I/O, CPU, Shuffle) as they occur.
                        </CardDescription>
                        <CardAction>
                            <Button variant="default" onClick={() => {
                                setShowSimCard(false)
                                setTimeout(() => {
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
                                    right: 12,
                                }}
                            >
                                <CartesianGrid strokeDasharray="10 10" />
                                <XAxis
                                    dataKey="simTime"
                                    tickLine={true}
                                    axisLine={false}
                                    tickFormatter={(value: number) => (value / 1000).toFixed(1)}
                                    label={<Label position="middle" dy={15}>Time (sec)</Label>}
                                />
                                <YAxis
                                    tickLine={false}
                                    axisLine={false}
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
            )}

            <div className="w-full grid grid-cols-2 gap-4">
                <Card className="w-full">
                    <CardHeader>
                        <CardTitle>Timeline or Activity Chart</CardTitle>
                        <CardDescription>
                            x-axis: simulation time<br />
                            y-axis: number of active queries and arrivals
                        </CardDescription>
                        <CardAction className="flex items-center gap-3">
                            <InputGroup>
                                <Select defaultValue={"10"} onValueChange={(e) => setBuckets(Number.parseInt(e))}>
                                    <SelectTrigger className="w-fit border-none">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent className="border-none">
                                        <SelectItem value={"5"}>5</SelectItem>
                                        <SelectItem value={"10"}>10</SelectItem>
                                        <SelectItem value={"20"}>20</SelectItem>
                                        <SelectItem value={"50"}>50</SelectItem>
                                        <SelectItem value={"100"}>100</SelectItem>
                                    </SelectContent>
                                </Select>
                                <InputGroupAddon align="inline-end">
                                    <InputGroupText>Buckets</InputGroupText>
                                </InputGroupAddon>
                            </InputGroup>
                        </CardAction>
                    </CardHeader>
                    <CardContent>
                        <ChartContainer config={timelineChartConfig}>
                            <LineChart
                                accessibilityLayer
                                data={timelineData}
                                margin={{
                                    top: 30,
                                    right: 12,
                                    bottom: 10,
                                }}
                            >
                                <CartesianGrid vertical={false} />
                                <XAxis
                                    dataKey="simTime"
                                    tickLine={true}
                                    axisLine={false}
                                    tickFormatter={(value: number) => (value / 1000 / 60).toFixed(1)}
                                    label={<Label position="middle" dy={15}>Time (min)</Label>}
                                />
                                <YAxis
                                    axisLine={false}
                                    tickLine={false}
                                    label={<Label position="middle" angle={270}>Arrived and active queries</Label>}
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
                        <CardTitle>{isCDF ? "CDF" : "Histogram"} of per-query latencies</CardTitle>
                        <CardDescription>
                            x-axis: query duration<br />
                            y-axis: number of queries finished under the given duration
                        </CardDescription>
                        <CardAction>
                            <Button variant="outline" onClick={() => setCDF(!isCDF)}>
                                {isCDF ? <ChartColumn /> : <ChartSpline />}
                            </Button>
                        </CardAction>
                    </CardHeader>
                    <CardContent>
                        <ChartContainer config={{}}>
                            {isCDF ? (
                                <LineChart
                                    accessibilityLayer
                                    data={histoData}
                                    margin={{
                                        top: 30,
                                        bottom: 10,
                                    }}
                                >
                                    <CartesianGrid vertical={false} />
                                    <XAxis
                                        dataKey="sec"
                                        tickLine={false}
                                        axisLine={false}
                                        label={<Label position="middle" dy={15}>Query Duration (sec)</Label>}
                                    />
                                    <YAxis
                                        axisLine={false}
                                        tickLine={false}
                                        label={<Label position="middle" angle={270}>Number of finished queries</Label>}
                                    />
                                    <ChartTooltip
                                        cursor={false}
                                        content={<ChartTooltipContent hideLabel />}
                                    />
                                    <Line
                                        dataKey="accFinishedCount"
                                        type="step"
                                        stroke="var(--chart-3)"
                                        strokeWidth={2}
                                        dot={false}
                                    />
                                </LineChart>
                            ) : (
                                <BarChart
                                    accessibilityLayer
                                    data={histoData}
                                    margin={{
                                        top: 30,
                                        bottom: 10,
                                    }}
                                >
                                    <CartesianGrid vertical={false} />
                                    <XAxis
                                        dataKey="sec"
                                        tickLine={false}
                                        axisLine={false}
                                        label={<Label position="middle" dy={15}>Query Duration (sec)</Label>}
                                    />
                                    <YAxis
                                        axisLine={false}
                                        tickLine={false}
                                        label={<Label position="middle" angle={270}>Number of finished queries</Label>}
                                    />
                                    <Bar dataKey="finishedCount" fill="var(--chart-3)" radius={8} isAnimationActive={false}>
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
                            )}
                        </ChartContainer>
                    </CardContent>
                </Card>
            </div>
        </div >
    )
}

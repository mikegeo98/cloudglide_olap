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
    TableCaption,
    TableCell,
    TableFooter,
    TableHead,
    TableHeader,
    TableRow,
    UnscrollableTable,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";

import { columns, Simulation } from "./columns-sim";
import { DataTable } from "./data-table";
import { Bar, BarChart, CartesianGrid, Line, LineChart, LabelList, XAxis, Label, YAxis } from "recharts";
import React from "react";
import { ChartColumn, ChartSpline } from "lucide-react";

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
    const [simBuckets, setSimBuckets] = React.useState<number>(10)
    const [rowSelection, setRowSelection] = React.useState<Record<number, boolean>>({})
    const [simData, setSimData] = React.useState<{ simTime: number, io: number, cpu: number, shuffle: number }[]>([])
    const [timelineData, setTimelineData] = React.useState<{ simTime: number, arrivalsCount: number, activesCount: number }[]>([])
    const [histoData, setHistoData] = React.useState<{ sec: number, finishedCount: number, accFinishedCount: number }[]>([])
    const [isCDF, setCDF] = React.useState<boolean>(false)
    const [insights, setInsights] = React.useState<Record<number, boolean>>({})
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
    const [showCSVCard, setShowCSVCard] = React.useState(false)

    React.useEffect(() => {
        const maxTime = data[sim].reduce((max, item) => item.query_duration > max ? item.query_duration : max, 0)
        setHistoData(Array.from({ length: Math.ceil(maxTime) + 1 }, (_, i) => {
            const count = data[sim].filter(item => Math.ceil(item.query_duration) === i).length
            const acc = data[sim].filter(item => Math.ceil(item.query_duration) < i).length
            return { sec: i, finishedCount: count, accFinishedCount: acc }
        }))

        const summary = {
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
        }
        setSummaryTable(summary)

        const insights: Record<number, boolean> = {}
        insights[0] = summary.queueDelayAvg > summary.meanQueryLatency
        const lastQueriesCount = Math.ceil(data[sim].length * 0.05)
        const lastQueries = data[sim].slice(-lastQueriesCount)
        const shuffleRatio = lastQueries.reduce((sum, q) => sum + ((q.shuffle_end_timestamp - q.shuffle_start_timestamp) / q.query_duration), 0)
        insights[1] = shuffleRatio > 0.3
        setInsights(insights)
    }, [data, sim])

    React.useEffect(() => {
        const lastTimestamp = data[sim][data[sim].length - 1].end_timestamp
        let lastBucket = 0
        const timelineArray = []
        for (let i = 0; i <= lastTimestamp; i += lastTimestamp / buckets) {
            const arrivalsCount = data[sim].filter(item => item.start_timestamp <= i && item.start_timestamp > lastBucket).length
            const sillActiveFromPrev = data[sim].filter(item => item.start_timestamp <= lastBucket && item.end_timestamp >= i).length
            lastBucket = i
            timelineArray.push({ simTime: Math.ceil(i), arrivalsCount: arrivalsCount, activesCount: arrivalsCount + sillActiveFromPrev })
        }
        setTimelineData(timelineArray)
    }, [data, sim, buckets])

    React.useEffect(() => {
        const lastTimestamp = data[sim][data[sim].length - 1].end_timestamp
        let lastBucket = 0
        const simArray = []
        for (let i = 0; i <= lastTimestamp; i += lastTimestamp / simBuckets) {
            const io = data[sim].filter(item => (item.io_start_timestamp <= i && item.io_start_timestamp > lastBucket) || (item.io_end_timestamp <= i && item.io_end_timestamp > lastBucket)).reduce((sum) => sum + 1, 0)
            const cpu = data[sim].filter(item => (item.cpu_start_timestamp <= i && item.cpu_start_timestamp > lastBucket) || (item.cpu_end_timestamp <= i && item.cpu_end_timestamp > lastBucket)).reduce((sum) => sum + 1, 0)
            const shuffle = data[sim].filter(item => (item.shuffle_start_timestamp <= i && item.shuffle_start_timestamp > lastBucket) || (item.shuffle_end_timestamp <= i && item.shuffle_end_timestamp > lastBucket)).reduce((sum) => sum + 1, 0)
            lastBucket = i
            simArray.push({ simTime: Math.ceil(i), io: io, cpu: cpu, shuffle: shuffle })
        }
        setSimData(simArray)
    }, [data, sim, simBuckets])

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
                <div className="flex gap-4">
                    <Button variant={showCSVCard ? "outline" : "default"} onClick={() => setShowCSVCard(!showCSVCard)}>
                        {showCSVCard ? "Hide CSV" : "Show CSV"}
                    </Button>
                    <Button variant={showSimCard ? "outline" : "default"} onClick={() => setShowSimCard(!showSimCard)}>
                        {showSimCard ? "Hide Simulation" : "Show Simulation"}
                    </Button>
                </div>
            </div>
            {showCSVCard ? (
                <DataTable className="w-full max-h-[300px] overflow-auto" columns={columns} data={data[sim]} rowSelection={rowSelection} setRowSelection={setRowSelection} />
            ) : null}
            {showSimCard ? (
                <Card className="w-full">
                    <CardHeader>
                        <CardTitle>Simulation</CardTitle>
                        <CardDescription>
                            Simulating the process of query execution over time, showing the resource usage phases (I/O, CPU, Shuffle) as they occur.
                        </CardDescription>
                        <CardAction className="flex items-center gap-3">
                            <InputGroup>
                                <Select defaultValue={simBuckets.toString()} onValueChange={(e) => setSimBuckets(Number.parseInt(e))}>
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
                                    label={<Label position="middle" angle={270}>Active queries</Label>}
                                />
                                <ChartTooltip content={<ChartTooltipContent className="w-[175px]" labelFormatter={() => "Active queries"} />} />
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
            <div className="w-full grid grid-cols-2 gap-4">
                <Card className="w-full">
                    <CardHeader>
                        <CardTitle>Insight Panel</CardTitle>
                    </CardHeader>
                    <CardContent>
                        {summaryTable ? (
                            <UnscrollableTable>
                                <TableBody>
                                    <TableRow className={insights[0] ? "text-destructive" : "text-muted-foreground opacity-25"}>
                                        <TableCell className="whitespace-normal">Queue delay dominates latency</TableCell>
                                        <TableCell>→</TableCell>
                                        <TableCell className="whitespace-normal">reduce arrival rate or add capacity</TableCell>
                                    </TableRow>
                                    <TableRow className={insights[1] ? "text-destructive" : "text-muted-foreground opacity-25"}>
                                        <TableCell className="whitespace-normal">Tail latency driven by shuffle</TableCell>
                                        <TableCell>→</TableCell>
                                        <TableCell className="whitespace-normal">increase network BW or reduce parallelism threshold</TableCell>
                                    </TableRow>
                                </TableBody>
                            </UnscrollableTable>
                        ) : null}
                    </CardContent>
                </Card>
                <Card className="w-full max-h-[300px] overflow-auto">
                    <CardHeader>
                        <CardTitle>Summary Table</CardTitle>
                    </CardHeader>
                    <CardContent>
                        {summaryTable ? (
                        <Table className="border">
                            <TableHeader>
                                <TableRow className="bg-secondary">
                                    <TableHead className="w-[100px]">Metric</TableHead>
                                    <TableHead className="text-right">Value</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                <TableRow>
                                    <TableCell className="font-medium">Queue Delay Avg</TableCell>
                                    <TableCell className="text-right">{summaryTable.queueDelayAvg}</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="font-medium">Buffer Delay Avg</TableCell>
                                    <TableCell className="text-right">{summaryTable.bufferDelayAvg}</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="font-medium">Mean I/O</TableCell>
                                    <TableCell className="text-right">{summaryTable.meanIO}</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="font-medium">Mean CPU</TableCell>
                                    <TableCell className="text-right">{summaryTable.meanCPU}</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="font-medium">Mean Shuffle</TableCell>
                                    <TableCell className="text-right">{summaryTable.meanShuffle}</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="font-medium">Mean Query Latency</TableCell>
                                    <TableCell className="text-right">{summaryTable.meanQueryLatency}</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="font-medium">Mean Query (with queueing)</TableCell>
                                    <TableCell className="text-right">{summaryTable.meanQuery}</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="font-medium">Median Query Latency</TableCell>
                                    <TableCell className="text-right">{summaryTable.medianQueryLatency}</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="font-medium">Median Query (with queueing)</TableCell>
                                    <TableCell className="text-right">{summaryTable.medianQuery}</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="font-medium">95th Percentile Query Latency</TableCell>
                                    <TableCell className="text-right">{summaryTable.percQueryLatency}</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="font-medium">95th Percentile Query (with queueing)</TableCell>
                                    <TableCell className="text-right">{summaryTable.percQuery}</TableCell>
                                </TableRow>
                                <TableRow>
                                    <TableCell className="font-medium">Total Price</TableCell>
                                    <TableCell className="text-right">${summaryTable.totalPrice}</TableCell>
                                </TableRow>
                            </TableBody>
                        </Table>
                        ) : null}
                    </CardContent>
                </Card>
            </div>
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
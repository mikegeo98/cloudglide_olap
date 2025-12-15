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
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import {
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
    type ChartConfig,
} from "@/components/ui/chart";
import { columns, Simulation } from "./columns-sim";
import { DataTable } from "./data-table";

import { Bar, BarChart, CartesianGrid, LabelList, XAxis } from "recharts";
import React, { Suspense } from "react";
import { Spinner } from "@/components/ui/spinner";

const chartData = [
    { month: "January", desktop: 186 },
    { month: "February", desktop: 305 },
    { month: "March", desktop: 237 },
    { month: "April", desktop: 73 },
    { month: "May", desktop: 209 },
    { month: "June", desktop: 214 },
]

const chartConfig = {
    desktop: {
        label: "Desktop",
        color: "var(--chart-1)",
    },
} satisfies ChartConfig

export default function SingleRun({ data }: { data: Simulation[][] }) {
    const [sim, setSim] = React.useState<number>(0)
    const [rowSelection, setRowSelection] = React.useState<Record<number, boolean>>({})
    const [timelineData, setTimelineData] = React.useState<{ simTime: number, activeQueriesCount: number }[]>([])
    const [histoData, setHistoData] = React.useState<{ sec: number, finishedCount: number }[]>([])
    const [loading, setLoading] = React.useState(false)

    function handleSelectChange(e: string) {
        setLoading(true)
        setSim(Number.parseInt(e, 10))

        const maxTime = data[sim].reduce((max, item) => item.query_duration > max ? item.query_duration : max, 0)
        setHistoData(Array.from({ length: Math.ceil(maxTime) }, (_, i) => {
            const count = data[sim].filter(item => Math.ceil(item.query_duration) === i).length
            return { sec: i, finishedCount: count }
        }))

        setTimelineData(Array.from({ length: Math.ceil(maxTime) }, (_, i) => {
            const count = data[sim].filter(item => Math.ceil(item.query_duration) === i).length
            return { simTime: i, activeQueriesCount: count }
        }))
        setLoading(false)
    }

    return (
        <div className="flex flex-col w-full h-full max-h-full gap-6 items-center overflow-hidden">
            <div className="flex justify-start items-start w-full">
                <Select defaultValue="0" onValueChange={(e) => handleSelectChange(e)}>
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
            <DataTable className="w-full max-h-[300px] overflow-auto" columns={columns} data={data[sim]} rowSelection={rowSelection} setRowSelection={setRowSelection} />
            <div className="w-full grid grid-cols-2 gap-6">
                <Card className="w-full">
                    <CardHeader>
                        <CardTitle>Timeline or Activity Chart</CardTitle>
                        <CardDescription>
                            x-axis: simulation time<br />
                            y-axis: number of active queries or arrivals per minute
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        {loading ? <Spinner /> : (
                            <ChartContainer config={chartConfig}>
                                <BarChart
                                    accessibilityLayer
                                    data={timelineData}
                                    margin={{
                                        top: 30,
                                    }}
                                >
                                    <CartesianGrid vertical={false} />
                                    <XAxis
                                        dataKey="simTime"
                                        tickLine={false}
                                        tickMargin={10}
                                        axisLine={false}
                                    />
                                    <Bar dataKey="activeQueriesCount" fill="var(--chart-3)" radius={8}>
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
                        )}
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
                        <ChartContainer config={chartConfig}>
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
        </div>
    )
}
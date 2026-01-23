"use client"

import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";

import { Simulation } from "./columns-sim";
import React from "react";
import { Bar, BarChart, CartesianGrid, Label, LabelList, Scatter, XAxis, YAxis, Tooltip, Cell, ComposedChart, ReferenceLine } from "recharts";
import { ChartConfig, ChartContainer } from "@/components/ui/chart";
import { Download, Check } from "lucide-react";
import { downloadCSV, downloadJSON } from "@/lib/export-utils";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// Distinct color palette for scenarios
const SCENARIO_COLORS = [
    '#2E86AB', // Blue
    '#A23B72', // Magenta
    '#F18F01', // Orange
    '#06A77D', // Green
    '#9B59B6', // Purple
    '#E74C3C', // Red
    '#3498DB', // Light Blue
    '#1ABC9C', // Teal
    '#F39C12', // Yellow
    '#8E44AD', // Dark Purple
]

type ComparisonMetrics = {
    filename: string
    shortName: string
    color: string
    avgQueryLatency: number
    medianQueryLatency: number
    p95QueryLatency: number
    totalCost: number
    avgQueueDelay: number
    avgBufferDelay: number
    avgIO: number
    avgCPU: number
    avgShuffle: number
    queryCount: number
}

function calculateMetrics(data: Simulation[], filename: string, color: string): ComparisonMetrics {
    const avgQueryLatency = data.reduce((sum, s) => sum + s.query_duration, 0) / data.length
    const sortedByDuration = [...data].sort((a, b) => a.query_duration - b.query_duration)
    const medianQueryLatency = sortedByDuration[Math.floor(data.length / 2)].query_duration
    const sortedDesc = [...data].sort((a, b) => b.query_duration - a.query_duration)
    const p95QueryLatency = sortedDesc[Math.floor(data.length * 0.05)].query_duration
    // mon_cost is the total simulation cost repeated for each row, so use first row
    const totalCost = data.length > 0 ? data[0].mon_cost : 0
    const avgQueueDelay = data.reduce((sum, s) => sum + s.queueing_delay, 0) / data.length
    const avgBufferDelay = data.reduce((sum, s) => sum + s.buffer_delay, 0) / data.length
    const avgIO = data.reduce((sum, s) => sum + s.io, 0) / data.length
    const avgCPU = data.reduce((sum, s) => sum + s.cpu, 0) / data.length
    const avgShuffle = data.reduce((sum, s) => sum + s.shuffle, 0) / data.length

    // Create a more distinguishable short name
    // Remove common prefixes/suffixes and file extension, keep distinctive parts
    const baseName = filename.replace(/\.csv$/i, '').replace(/^output_?/i, '')
    // If name has underscores, try to create a meaningful abbreviation
    const parts = baseName.split('_').filter(p => p.length > 0)
    let shortName: string
    if (parts.length <= 2) {
        shortName = baseName
    } else {
        // Keep first part + last 1-2 distinctive parts, limit total length
        const firstPart = parts[0].substring(0, 8)
        const lastParts = parts.slice(-2).join('-').substring(0, 12)
        shortName = `${firstPart}_${lastParts}`
    }
    // Truncate if still too long
    if (shortName.length > 20) {
        shortName = shortName.substring(0, 18) + '..'
    }

    return {
        filename,
        shortName,
        color,
        avgQueryLatency,
        medianQueryLatency,
        p95QueryLatency,
        totalCost,
        avgQueueDelay,
        avgBufferDelay,
        avgIO,
        avgCPU,
        avgShuffle,
        queryCount: data.length
    }
}

/**
 * Calculate Pareto frontier for cost-latency trade-off
 * A point is on the Pareto frontier if no other point has both lower cost AND lower latency
 */
function calculateParetoFrontier(points: { name: string, latency: number, cost: number, color: string }[]) {
    // Sort by latency (ascending)
    const sorted = [...points].sort((a, b) => a.latency - b.latency)

    const paretoPoints: typeof points = []
    let minCost = Infinity

    for (const point of sorted) {
        // A point is on the frontier if its cost is lower than all points with lower latency
        if (point.cost <= minCost) {
            paretoPoints.push(point)
            minCost = point.cost
        }
    }

    return paretoPoints
}

export default function MultipleRuns({ data, filenames }: { data: Simulation[][], filenames: string[] }) {
    const [selectedScenarios, setSelectedScenarios] = React.useState<Set<number>>(new Set(filenames.map((_, i) => i)))
    const [allMetrics, setAllMetrics] = React.useState<ComparisonMetrics[]>([])

    // Toggle scenario selection
    const toggleScenario = (index: number) => {
        setSelectedScenarios(prev => {
            const newSet = new Set(prev)
            if (newSet.has(index)) {
                newSet.delete(index)
            } else {
                newSet.add(index)
            }
            return newSet
        })
    }

    // Select all / clear all
    const selectAll = () => setSelectedScenarios(new Set(filenames.map((_, i) => i)))
    const clearAll = () => setSelectedScenarios(new Set())

    React.useEffect(() => {
        // Calculate metrics for all simulations with colors
        const metrics = data.map((simData, idx) => {
            return calculateMetrics(simData, filenames[idx], SCENARIO_COLORS[idx % SCENARIO_COLORS.length])
        })
        setAllMetrics(metrics)
    }, [data, filenames])

    // Filter metrics based on selection
    const comparisonMetrics = allMetrics.filter((_, idx) => selectedScenarios.has(idx))

    // Prepare chart data
    const latencyCostData = comparisonMetrics.map(m => ({
        name: m.shortName,
        latency: parseFloat(m.avgQueryLatency.toFixed(2)),
        cost: parseFloat(m.totalCost.toFixed(2)),
        color: m.color
    }))

    // Calculate Pareto frontier
    const paretoFrontier = calculateParetoFrontier(latencyCostData)

    const phaseBreakdownData = comparisonMetrics.map(m => ({
        name: m.shortName,
        io: parseFloat(m.avgIO.toFixed(2)),
        cpu: parseFloat(m.avgCPU.toFixed(2)),
        shuffle: parseFloat(m.avgShuffle.toFixed(2)),
        color: m.color
    }))

    const costComparisonData = comparisonMetrics.map(m => ({
        name: m.shortName,
        cost: parseFloat(m.totalCost.toFixed(2)),
        costPerQuery: parseFloat((m.totalCost / m.queryCount).toFixed(4)),
        color: m.color
    }))

    const delayComparisonData = comparisonMetrics.map(m => ({
        name: m.shortName,
        queue: parseFloat(m.avgQueueDelay.toFixed(2)),
        buffer: parseFloat(m.avgBufferDelay.toFixed(2)),
        color: m.color
    }))

    const chartConfig = {
        latency: {
            label: "Latency (s)",
            color: "hsl(var(--chart-1))",
        },
        cost: {
            label: "Cost ($)",
            color: "hsl(var(--chart-2))",
        },
        avg: {
            label: "Average",
            color: "hsl(var(--chart-1))",
        },
        median: {
            label: "Median",
            color: "hsl(var(--chart-2))",
        },
        p95: {
            label: "95th Percentile",
            color: "hsl(var(--chart-3))",
        },
        io: {
            label: "I/O",
            color: "hsl(var(--chart-3))",
        },
        cpu: {
            label: "CPU",
            color: "hsl(var(--chart-4))",
        },
        shuffle: {
            label: "Shuffle",
            color: "hsl(var(--chart-5))",
        },
        costPerQuery: {
            label: "Cost per Query",
            color: "hsl(var(--chart-2))",
        }
    } satisfies ChartConfig

    return (
        <div className="flex flex-col w-full h-full max-h-full gap-3 items-center overflow-y-auto overflow-x-hidden pb-6">
            {/* Header with scenario selection inline */}
            <div className="flex flex-wrap items-center justify-between gap-2 w-full">
                <div className="flex items-center gap-3">
                    <h2 className="text-xl font-semibold">Multiple Runs Comparison</h2>
                    <span className="text-sm text-muted-foreground">
                        {selectedScenarios.size}/{filenames.length} selected
                    </span>
                    <Button variant="outline" size="sm" onClick={selectAll}>All</Button>
                    <Button variant="outline" size="sm" onClick={clearAll}>None</Button>
                </div>
                <div className="flex gap-2">
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={() => downloadCSV(comparisonMetrics, 'comparison_metrics.csv')}
                        disabled={comparisonMetrics.length === 0}
                    >
                        <Download className="h-4 w-4 mr-1" />
                        CSV
                    </Button>
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={() => downloadJSON({
                            paretoFrontier,
                            metrics: comparisonMetrics,
                            summary: {
                                bestLatency: comparisonMetrics.length > 0 ? Math.min(...comparisonMetrics.map(m => m.avgQueryLatency)) : 0,
                                lowestCost: comparisonMetrics.length > 0 ? Math.min(...comparisonMetrics.map(m => m.totalCost)) : 0
                            }
                        }, 'comparison_full.json')}
                        disabled={comparisonMetrics.length === 0}
                    >
                        <Download className="h-4 w-4 mr-1" />
                        JSON
                    </Button>
                </div>
            </div>

            {/* Scenario Selection - compact chips */}
            <div className="flex flex-wrap gap-1.5 w-full">
                {allMetrics.map((m, idx) => {
                    const isSelected = selectedScenarios.has(idx)
                    return (
                        <button
                            key={idx}
                            onClick={() => toggleScenario(idx)}
                            className={cn(
                                "flex items-center gap-1.5 px-2 py-1 rounded-md border transition-all text-sm",
                                isSelected
                                    ? "border-current shadow-sm"
                                    : "border-transparent bg-muted/50 opacity-50 hover:opacity-75"
                            )}
                            style={{
                                borderColor: isSelected ? m.color : undefined,
                                backgroundColor: isSelected ? `${m.color}15` : undefined
                            }}
                        >
                            <div
                                className="w-3 h-3 rounded-full flex items-center justify-center"
                                style={{ backgroundColor: m.color }}
                            >
                                {isSelected && <Check className="w-2 h-2 text-white" />}
                            </div>
                            <span className="font-medium">{m.shortName}</span>
                        </button>
                    )
                })}
            </div>

            {comparisonMetrics.length > 0 ? (
                <>
                    {/* Summary Stats - inline compact */}
                    <div className="grid grid-cols-4 gap-3 w-full">
                        <div className="p-2 border rounded-lg bg-card">
                            <div className="text-xs text-muted-foreground">Comparing</div>
                            <div className="text-xl font-bold">{comparisonMetrics.length} <span className="text-sm font-normal text-muted-foreground">scenarios</span></div>
                        </div>
                        <div className="p-2 border rounded-lg bg-card">
                            <div className="text-xs text-muted-foreground">Best Latency</div>
                            <div className="text-xl font-bold">
                                {Math.min(...comparisonMetrics.map(m => m.avgQueryLatency)).toFixed(2)}s
                                <span className="text-sm font-normal text-muted-foreground ml-1">
                                    ({comparisonMetrics.find(m => m.avgQueryLatency === Math.min(...comparisonMetrics.map(m => m.avgQueryLatency)))?.shortName})
                                </span>
                            </div>
                        </div>
                        <div className="p-2 border rounded-lg bg-card">
                            <div className="text-xs text-muted-foreground">Lowest Cost</div>
                            <div className="text-xl font-bold">
                                ${Math.min(...comparisonMetrics.map(m => m.totalCost)).toFixed(2)}
                                <span className="text-sm font-normal text-muted-foreground ml-1">
                                    ({comparisonMetrics.find(m => m.totalCost === Math.min(...comparisonMetrics.map(m => m.totalCost)))?.shortName})
                                </span>
                            </div>
                        </div>
                        <div className="p-2 border rounded-lg bg-card">
                            <div className="text-xs text-muted-foreground">Best Value</div>
                            <div className="text-xl font-bold">
                                {comparisonMetrics.reduce((best, m) => {
                                    const score = m.totalCost / (1 / m.avgQueryLatency)
                                    const bestScore = best.totalCost / (1 / best.avgQueryLatency)
                                    return score < bestScore ? m : best
                                }).shortName}
                                <span className="text-sm font-normal text-muted-foreground ml-1">(cost/perf)</span>
                            </div>
                        </div>
                    </div>

                    {/* Charts Grid - 2x2 layout with larger charts */}
                    <div className="grid grid-cols-2 gap-4 w-full">
                        {/* Latency vs Cost Scatter with Pareto Frontier */}
                        <Card className="w-full">
                            <CardHeader className="py-3 px-4">
                                <CardTitle className="text-base">Latency vs Cost Trade-off</CardTitle>
                                <CardDescription className="text-xs">
                                    Green shaded area shows Pareto frontier (optimal configs)
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="px-4 pb-3 pt-0">
                                <ChartContainer config={chartConfig} className="h-[240px]">
                                    <ComposedChart
                                        margin={{ top: 20, right: 30, bottom: 30, left: 30 }}
                                        data={latencyCostData}
                                    >
                                        <defs>
                                            <linearGradient id="paretoGradient" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="hsl(142, 76%, 36%)" stopOpacity={0.3}/>
                                                <stop offset="95%" stopColor="hsl(142, 76%, 36%)" stopOpacity={0.05}/>
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis
                                            type="number"
                                            dataKey="latency"
                                            name="Latency"
                                            domain={[0, 'auto']}
                                            tick={{ fontSize: 12 }}
                                        >
                                            <Label value="Avg Query Latency (s)" position="bottom" offset={5} style={{ fontSize: 13, fontWeight: 'bold' }} />
                                        </XAxis>
                                        <YAxis
                                            type="number"
                                            dataKey="cost"
                                            name="Cost"
                                            domain={[0, 'auto']}
                                            tick={{ fontSize: 12 }}
                                        >
                                            <Label value="Total Cost ($)" angle={-90} position="insideLeft" style={{ fontSize: 13, fontWeight: 'bold' }} />
                                        </YAxis>

                                        {/* Pareto frontier lines connecting optimal points */}
                                        {paretoFrontier.length > 1 && paretoFrontier.map((point, index) => {
                                            if (index === 0) return null
                                            const prevPoint = paretoFrontier[index - 1]
                                            return (
                                                <ReferenceLine
                                                    key={`pareto-line-${index}`}
                                                    segment={[
                                                        { x: prevPoint.latency, y: prevPoint.cost },
                                                        { x: point.latency, y: point.cost }
                                                    ]}
                                                    stroke="hsl(142, 76%, 36%)"
                                                    strokeWidth={2}
                                                    strokeDasharray="5 5"
                                                />
                                            )
                                        })}

                                        {/* Scatter points */}
                                        <Scatter
                                            data={latencyCostData}
                                            name="Scenarios"
                                            isAnimationActive={false}
                                        >
                                            {latencyCostData.map((entry, index) => {
                                                const isPareto = paretoFrontier.some(p => p.name === entry.name)
                                                return (
                                                    <Cell
                                                        key={`cell-${index}`}
                                                        fill={entry.color}
                                                        stroke={isPareto ? "hsl(142, 76%, 36%)" : "transparent"}
                                                        strokeWidth={isPareto ? 3 : 0}
                                                        r={isPareto ? 8 : 6}
                                                    />
                                                )
                                            })}
                                            <LabelList
                                                dataKey="name"
                                                position="top"
                                                offset={10}
                                                style={{ fontSize: 10, fontWeight: 500 }}
                                                content={({ x, y, value, index }) => {
                                                    const entry = latencyCostData[index as number]
                                                    return (
                                                        <text
                                                            x={x as number}
                                                            y={(y as number) - 10}
                                                            textAnchor="middle"
                                                            fill={entry?.color || '#666'}
                                                            style={{ fontSize: 10, fontWeight: 600 }}
                                                        >
                                                            {value}
                                                        </text>
                                                    )
                                                }}
                                            />
                                        </Scatter>

                                        <Tooltip
                                            cursor={{ strokeDasharray: '3 3' }}
                                            content={({ active, payload }) => {
                                                if (active && payload && payload.length) {
                                                    const data = payload[0].payload
                                                    const isPareto = paretoFrontier.some(p => p.name === data.name)
                                                    return (
                                                        <div className="rounded-lg border bg-background p-2 shadow-sm">
                                                            <div className="flex items-center gap-2">
                                                                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: data.color }} />
                                                                <span className="font-semibold">{data.name}</span>
                                                            </div>
                                                            <div className="text-sm">Latency: {data.latency}s</div>
                                                            <div className="text-sm">Cost: ${data.cost}</div>
                                                            {isPareto && (
                                                                <div className="text-xs text-green-600 font-semibold mt-1">
                                                                    ✓ Pareto Optimal
                                                                </div>
                                                            )}
                                                        </div>
                                                    )
                                                }
                                                return null
                                            }}
                                        />
                                    </ComposedChart>
                                </ChartContainer>

                                {/* Pareto legend */}
                                {paretoFrontier.length > 0 && (
                                    <div className="mt-2 flex items-center gap-4 text-sm">
                                        <div className="flex items-center gap-2">
                                            <div className="w-4 h-4 rounded-full bg-[hsl(142,76%,36%)] border-2 border-[hsl(142,76%,36%)]" />
                                            <span className="text-muted-foreground">Pareto optimal ({paretoFrontier.length})</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <div className="w-4 h-4 rounded-full bg-muted border" />
                                            <span className="text-muted-foreground">Sub-optimal ({latencyCostData.length - paretoFrontier.length})</span>
                                        </div>
                                    </div>
                                )}
                            </CardContent>
                        </Card>

                        {/* Per-Phase Latency Breakdown - grouped bar chart */}
                        <Card className="w-full">
                            <CardHeader className="py-3 px-4">
                                <CardTitle className="text-base">Latency by Phase</CardTitle>
                                <CardDescription className="text-xs">
                                    I/O, CPU, Shuffle breakdown per scenario
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="px-4 pb-3 pt-0">
                                <ChartContainer config={chartConfig} className="h-[240px]">
                                    <BarChart data={phaseBreakdownData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                                        <YAxis tick={{ fontSize: 12 }}>
                                            <Label value="Time (s)" angle={-90} position="insideLeft" style={{ fontSize: 13, fontWeight: 'bold' }} />
                                        </YAxis>
                                        <Tooltip
                                            content={({ active, payload }) => {
                                                if (active && payload && payload.length) {
                                                    const data = payload[0].payload
                                                    const total = data.io + data.cpu + data.shuffle
                                                    return (
                                                        <div className="rounded-lg border bg-background p-2 shadow-sm">
                                                            <div className="flex items-center gap-2 mb-1">
                                                                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: data.color }} />
                                                                <span className="font-semibold">{data.name}</span>
                                                            </div>
                                                            <div className="text-sm text-blue-600">I/O: {data.io}s ({(data.io/total*100).toFixed(0)}%)</div>
                                                            <div className="text-sm text-orange-600">CPU: {data.cpu}s ({(data.cpu/total*100).toFixed(0)}%)</div>
                                                            <div className="text-sm text-purple-600">Shuffle: {data.shuffle}s ({(data.shuffle/total*100).toFixed(0)}%)</div>
                                                            <div className="text-xs text-muted-foreground mt-1">Total: {total.toFixed(2)}s</div>
                                                        </div>
                                                    )
                                                }
                                                return null
                                            }}
                                        />
                                        <Bar dataKey="io" fill="#3498DB" name="I/O" />
                                        <Bar dataKey="cpu" fill="#F39C12" name="CPU" />
                                        <Bar dataKey="shuffle" fill="#9B59B6" name="Shuffle" />
                                    </BarChart>
                                </ChartContainer>
                            </CardContent>
                        </Card>

                        {/* Queue & Buffer Delays */}
                        <Card className="w-full">
                            <CardHeader className="py-3 px-4">
                                <CardTitle className="text-base">Queue & Buffer Delays</CardTitle>
                                <CardDescription className="text-xs">
                                    Overhead from queueing and buffer contention
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="px-4 pb-3 pt-0">
                                <ChartContainer config={chartConfig} className="h-[240px]">
                                    <BarChart data={delayComparisonData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                                        <YAxis tick={{ fontSize: 12 }}>
                                            <Label value="Delay (s)" angle={-90} position="insideLeft" style={{ fontSize: 13, fontWeight: 'bold' }} />
                                        </YAxis>
                                        <Tooltip
                                            content={({ active, payload }) => {
                                                if (active && payload && payload.length) {
                                                    const data = payload[0].payload
                                                    return (
                                                        <div className="rounded-lg border bg-background p-2 shadow-sm">
                                                            <div className="flex items-center gap-2 mb-1">
                                                                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: data.color }} />
                                                                <span className="font-semibold">{data.name}</span>
                                                            </div>
                                                            <div className="text-sm text-red-600">Queue: {data.queue}s</div>
                                                            <div className="text-sm text-amber-600">Buffer: {data.buffer}s</div>
                                                            <div className="text-xs text-muted-foreground mt-1">Total overhead: {(data.queue + data.buffer).toFixed(2)}s</div>
                                                        </div>
                                                    )
                                                }
                                                return null
                                            }}
                                        />
                                        <Bar dataKey="queue" fill="#E74C3C" name="Queue Delay" />
                                        <Bar dataKey="buffer" fill="#F39C12" name="Buffer Delay" />
                                    </BarChart>
                                </ChartContainer>
                            </CardContent>
                        </Card>

                        {/* Cost Comparison */}
                        <Card className="w-full">
                            <CardHeader className="py-3 px-4">
                                <CardTitle className="text-base">Cost Analysis</CardTitle>
                                <CardDescription className="text-xs">
                                    Total cost per scenario
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="px-4 pb-3 pt-0">
                                <ChartContainer config={chartConfig} className="h-[240px]">
                                    <BarChart data={costComparisonData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                                        <YAxis tick={{ fontSize: 12 }}>
                                            <Label value="Total Cost ($)" angle={-90} position="insideLeft" style={{ fontSize: 13, fontWeight: 'bold' }} />
                                        </YAxis>
                                        <Tooltip
                                            content={({ active, payload }) => {
                                                if (active && payload && payload.length) {
                                                    const data = payload[0].payload
                                                    return (
                                                        <div className="rounded-lg border bg-background p-2 shadow-sm">
                                                            <div className="flex items-center gap-2">
                                                                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: data.color }} />
                                                                <span className="font-semibold">{data.name}</span>
                                                            </div>
                                                            <div className="text-sm">Total Cost: ${data.cost}</div>
                                                            <div className="text-sm">Per Query: ${data.costPerQuery}</div>
                                                        </div>
                                                    )
                                                }
                                                return null
                                            }}
                                        />
                                        <Bar dataKey="cost" name="Total Cost">
                                            {costComparisonData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.color} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ChartContainer>
                            </CardContent>
                        </Card>
                    </div>
                </>
            ) : (
                <Card className="w-full">
                    <CardContent className="pt-6">
                        <p className="text-center text-muted-foreground">
                            Select at least one scenario to view comparison charts
                        </p>
                    </CardContent>
                </Card>
            )}
        </div>
    )
}

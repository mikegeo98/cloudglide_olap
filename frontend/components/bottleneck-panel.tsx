"use client"

import React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"
import {
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { AlertCircle, AlertTriangle, CheckCircle, Info, TrendingUp, Settings2, ChevronDown, Target, DollarSign } from "lucide-react"
import { Simulation } from "@/app/output/columns-sim"

type BottleneckInsight = {
    severity: "critical" | "warning" | "info" | "success"
    category: string
    title: string
    description: string
    value?: string
    recommendation?: string
}

type SLAConfig = {
    latencyPercentile: number // e.g., 95 for P95
    latencyTarget: number // in seconds
    budgetLimit: number | null // in dollars, null = no limit
}

type BottleneckAnalysis = {
    insights: BottleneckInsight[]
    performanceScore: number
    slaCompliance: {
        latencyMet: boolean
        latencyActual: number
        budgetMet: boolean
        budgetActual: number
    } | null
}

function getPercentileLatency(data: Simulation[], percentile: number): number {
    const sorted = [...data].sort((a, b) => a.query_duration - b.query_duration)
    const index = Math.floor(sorted.length * (percentile / 100))
    return sorted[Math.min(index, sorted.length - 1)].query_duration
}

function analyzeBottlenecks(data: Simulation[], sla: SLAConfig | null): BottleneckAnalysis {
    const insights: BottleneckInsight[] = []

    // Calculate metrics
    const avgQueueDelay = data.reduce((sum, s) => sum + s.queueing_delay, 0) / data.length
    const avgBufferDelay = data.reduce((sum, s) => sum + s.buffer_delay, 0) / data.length
    const avgIO = data.reduce((sum, s) => sum + s.io, 0) / data.length
    const avgCPU = data.reduce((sum, s) => sum + s.cpu, 0) / data.length
    const avgShuffle = data.reduce((sum, s) => sum + s.shuffle, 0) / data.length
    const avgQueryDuration = data.reduce((sum, s) => sum + s.query_duration, 0) / data.length
    const avgQueryWithQueue = data.reduce((sum, s) => sum + s.query_duration_with_queue, 0) / data.length
    // Note: mon_cost in CSV is the total simulation cost repeated for each row,
    // so we just take the first row's value instead of summing
    const totalCost = data.length > 0 ? data[0].mon_cost : 0
    const p95Duration = getPercentileLatency(data, 95)

    // Calculate phase proportions
    const totalPhaseTime = avgIO + avgCPU + avgShuffle
    const ioPercent = (avgIO / totalPhaseTime) * 100
    const cpuPercent = (avgCPU / totalPhaseTime) * 100
    const shufflePercent = (avgShuffle / totalPhaseTime) * 100

    // SLA compliance calculation
    let slaCompliance: BottleneckAnalysis["slaCompliance"] = null
    if (sla) {
        const targetLatency = getPercentileLatency(data, sla.latencyPercentile)
        const latencyMet = targetLatency <= sla.latencyTarget
        const budgetMet = sla.budgetLimit === null || totalCost <= sla.budgetLimit

        slaCompliance = {
            latencyMet,
            latencyActual: targetLatency,
            budgetMet,
            budgetActual: totalCost
        }

        // Add SLA-based insights
        if (latencyMet) {
            insights.push({
                severity: "success",
                category: "SLA",
                title: `P${sla.latencyPercentile} Latency Target Met`,
                description: `P${sla.latencyPercentile} latency (${targetLatency.toFixed(2)}s) is within target (${sla.latencyTarget}s)`,
                value: `${((1 - targetLatency / sla.latencyTarget) * 100).toFixed(0)}% headroom`
            })
        } else {
            const overage = ((targetLatency - sla.latencyTarget) / sla.latencyTarget) * 100
            insights.push({
                severity: overage > 50 ? "critical" : "warning",
                category: "SLA",
                title: `P${sla.latencyPercentile} Latency Target Missed`,
                description: `P${sla.latencyPercentile} latency (${targetLatency.toFixed(2)}s) exceeds target (${sla.latencyTarget}s)`,
                value: `${overage.toFixed(0)}% over target`,
                recommendation: overage > 50
                    ? "Consider scaling out, adding cache tiers, or relaxing SLA targets"
                    : "Minor optimization or cache tuning may help meet targets"
            })
        }

        if (sla.budgetLimit !== null) {
            if (budgetMet) {
                insights.push({
                    severity: "success",
                    category: "SLA",
                    title: "Budget Target Met",
                    description: `Total cost ($${totalCost.toFixed(2)}) is within budget ($${sla.budgetLimit.toFixed(2)})`,
                    value: `$${(sla.budgetLimit - totalCost).toFixed(2)} remaining`
                })
            } else {
                const overage = ((totalCost - sla.budgetLimit) / sla.budgetLimit) * 100
                insights.push({
                    severity: overage > 30 ? "critical" : "warning",
                    category: "SLA",
                    title: "Budget Target Exceeded",
                    description: `Total cost ($${totalCost.toFixed(2)}) exceeds budget ($${sla.budgetLimit.toFixed(2)})`,
                    value: `${overage.toFixed(0)}% over budget`,
                    recommendation: "Consider using spot instances, reducing node count, or optimizing query efficiency"
                })
            }
        }
    }

    // Queue delay analysis
    const queueOverhead = ((avgQueryWithQueue - avgQueryDuration) / avgQueryDuration) * 100
    if (avgQueueDelay > 1.0) {
        insights.push({
            severity: "critical",
            category: "Queueing",
            title: "High Queue Delays Detected",
            description: `Queries are spending significant time waiting in queue (avg: ${avgQueueDelay.toFixed(2)}s)`,
            value: `${queueOverhead.toFixed(1)}% overhead`,
            recommendation: "Consider increasing compute capacity or implementing auto-scaling"
        })
    } else if (avgQueueDelay > 0.5) {
        insights.push({
            severity: "warning",
            category: "Queueing",
            title: "Moderate Queue Delays",
            description: `Queue delays are impacting performance (avg: ${avgQueueDelay.toFixed(2)}s)`,
            value: `${queueOverhead.toFixed(1)}% overhead`,
            recommendation: "Monitor workload patterns and consider scaling during peak times"
        })
    } else {
        insights.push({
            severity: "success",
            category: "Queueing",
            title: "Low Queue Delays",
            description: `Queueing is well-managed (avg: ${avgQueueDelay.toFixed(2)}s)`,
            value: `${queueOverhead.toFixed(1)}% overhead`
        })
    }

    // Buffer delay analysis
    if (avgBufferDelay > 0.5) {
        insights.push({
            severity: "warning",
            category: "Buffering",
            title: "Buffer Delays Present",
            description: `Data buffering is causing delays (avg: ${avgBufferDelay.toFixed(2)}s)`,
            recommendation: "Optimize buffer sizes or increase memory allocation"
        })
    }

    // I/O bottleneck detection
    if (ioPercent > 50) {
        insights.push({
            severity: "critical",
            category: "I/O",
            title: "I/O Bottleneck Detected",
            description: `I/O operations dominate execution time (${ioPercent.toFixed(1)}% of total)`,
            value: `Avg I/O: ${avgIO.toFixed(2)}s`,
            recommendation: "Consider adding cache tiers (SSD, Memory) or optimizing data layout"
        })
    } else if (ioPercent > 35) {
        insights.push({
            severity: "warning",
            category: "I/O",
            title: "Significant I/O Time",
            description: `I/O is a major component (${ioPercent.toFixed(1)}% of total)`,
            value: `Avg I/O: ${avgIO.toFixed(2)}s`,
            recommendation: "Evaluate caching strategies to reduce remote storage access"
        })
    } else {
        insights.push({
            severity: "info",
            category: "I/O",
            title: "I/O Performance Good",
            description: `I/O time is reasonable (${ioPercent.toFixed(1)}% of total)`,
            value: `Avg I/O: ${avgIO.toFixed(2)}s`
        })
    }

    // CPU bottleneck detection
    if (cpuPercent > 60) {
        insights.push({
            severity: "warning",
            category: "CPU",
            title: "CPU-Intensive Workload",
            description: `CPU operations dominate (${cpuPercent.toFixed(1)}% of total)`,
            value: `Avg CPU: ${avgCPU.toFixed(2)}s`,
            recommendation: "Consider using higher CPU instance types or parallel processing"
        })
    } else {
        insights.push({
            severity: "info",
            category: "CPU",
            title: "CPU Usage Normal",
            description: `CPU time is balanced (${cpuPercent.toFixed(1)}% of total)`,
            value: `Avg CPU: ${avgCPU.toFixed(2)}s`
        })
    }

    // Shuffle bottleneck detection
    if (shufflePercent > 40) {
        insights.push({
            severity: "warning",
            category: "Shuffle",
            title: "High Shuffle Overhead",
            description: `Data shuffling is expensive (${shufflePercent.toFixed(1)}% of total)`,
            value: `Avg Shuffle: ${avgShuffle.toFixed(2)}s`,
            recommendation: "Optimize data partitioning or reduce shuffle operations"
        })
    } else if (shufflePercent > 20) {
        insights.push({
            severity: "info",
            category: "Shuffle",
            title: "Moderate Shuffle Time",
            description: `Shuffle time is acceptable (${shufflePercent.toFixed(1)}% of total)`,
            value: `Avg Shuffle: ${avgShuffle.toFixed(2)}s`
        })
    }

    // P95 latency check (only if no SLA, otherwise SLA check handles it)
    if (!sla) {
        const p95Ratio = p95Duration / avgQueryDuration
        if (p95Ratio > 3) {
            insights.push({
                severity: "critical",
                category: "Latency",
                title: "High Latency Variance",
                description: `P95 latency is ${p95Ratio.toFixed(1)}x higher than average`,
                value: `P95: ${p95Duration.toFixed(2)}s vs Avg: ${avgQueryDuration.toFixed(2)}s`,
                recommendation: "Investigate outlier queries and optimize tail latencies"
            })
        } else if (p95Ratio > 2) {
            insights.push({
                severity: "warning",
                category: "Latency",
                title: "Latency Variance Present",
                description: `P95 latency is ${p95Ratio.toFixed(1)}x higher than average`,
                value: `P95: ${p95Duration.toFixed(2)}s vs Avg: ${avgQueryDuration.toFixed(2)}s`
            })
        }
    }

    // Calculate performance score (0-100)
    let score = 100

    // SLA-based scoring (weighted heavily if SLA is defined)
    if (sla && slaCompliance) {
        // Latency SLA: 40 points
        if (slaCompliance.latencyMet) {
            // Bonus for headroom
            const headroom = 1 - slaCompliance.latencyActual / sla.latencyTarget
            score += Math.min(5, headroom * 10) // Up to 5 bonus points
        } else {
            const overage = (slaCompliance.latencyActual - sla.latencyTarget) / sla.latencyTarget
            score -= Math.min(40, overage * 40) // Up to 40 points penalty
        }

        // Budget SLA: 30 points (if defined)
        if (sla.budgetLimit !== null) {
            if (slaCompliance.budgetMet) {
                const remaining = (sla.budgetLimit - slaCompliance.budgetActual) / sla.budgetLimit
                score += Math.min(3, remaining * 10) // Up to 3 bonus points
            } else {
                const overage = (slaCompliance.budgetActual - sla.budgetLimit) / sla.budgetLimit
                score -= Math.min(30, overage * 30) // Up to 30 points penalty
            }
        }
    }

    // Infrastructure-based scoring (reduced weight if SLA is defined)
    const infraWeight = sla ? 0.5 : 1.0

    if (avgQueueDelay > 1.0) score -= 25 * infraWeight
    else if (avgQueueDelay > 0.5) score -= 10 * infraWeight

    if (ioPercent > 50) score -= 20 * infraWeight
    else if (ioPercent > 35) score -= 10 * infraWeight

    if (cpuPercent > 60) score -= 10 * infraWeight
    if (shufflePercent > 40) score -= 10 * infraWeight

    if (!sla) {
        const p95Ratio = p95Duration / avgQueryDuration
        if (p95Ratio > 3) score -= 15
        else if (p95Ratio > 2) score -= 8
    }

    return { insights, performanceScore: Math.max(0, Math.min(100, Math.round(score))), slaCompliance }
}

function getSeverityIcon(severity: BottleneckInsight["severity"]) {
    switch (severity) {
        case "critical":
            return <AlertCircle className="h-5 w-5 text-red-500" />
        case "warning":
            return <AlertTriangle className="h-5 w-5 text-yellow-500" />
        case "success":
            return <CheckCircle className="h-5 w-5 text-green-500" />
        case "info":
            return <Info className="h-5 w-5 text-blue-500" />
    }
}

function getSeverityBadgeVariant(severity: BottleneckInsight["severity"]): "destructive" | "default" | "secondary" | "outline" {
    switch (severity) {
        case "critical":
            return "destructive"
        case "warning":
            return "default"
        case "success":
            return "secondary"
        case "info":
            return "outline"
    }
}

function getScoreColor(score: number): string {
    if (score >= 80) return "text-green-600"
    if (score >= 60) return "text-yellow-600"
    return "text-red-600"
}

function getScoreLabel(score: number): string {
    if (score >= 80) return "Excellent"
    if (score >= 60) return "Good"
    if (score >= 40) return "Fair"
    return "Needs Attention"
}

export default function BottleneckPanel({ data, compact = false }: { data: Simulation[], compact?: boolean }) {
    const [slaEnabled, setSlaEnabled] = React.useState(false)
    const [slaOpen, setSlaOpen] = React.useState(false)
    const [slaConfig, setSlaConfig] = React.useState<SLAConfig>({
        latencyPercentile: 95,
        latencyTarget: 5,
        budgetLimit: null
    })
    const [budgetEnabled, setBudgetEnabled] = React.useState(false)

    const activeSla = slaEnabled ? slaConfig : null
    const analysis = React.useMemo(() => analyzeBottlenecks(data, activeSla), [data, activeSla])

    // Group insights by category
    const slaInsights = analysis.insights.filter(i => i.category === "SLA")
    const criticalInsights = analysis.insights.filter(i => i.severity === "critical" && i.category !== "SLA")
    const warningInsights = analysis.insights.filter(i => i.severity === "warning" && i.category !== "SLA")
    const otherInsights = analysis.insights.filter(i => (i.severity === "info" || i.severity === "success") && i.category !== "SLA")

    // For compact mode, show critical, warning, and a few key observations
    const displaySla = slaInsights
    const displayCritical = compact ? criticalInsights.slice(0, 2) : criticalInsights
    const displayWarning = compact ? warningInsights.slice(0, 2) : warningInsights
    // In compact mode, show up to 2 key observations (prioritize success items like queue status)
    const displayOther = compact ? otherInsights.slice(0, 2) : otherInsights

    return (
        <Card className="w-full">
            <CardHeader className={compact ? "pb-2" : ""}>
                <div className="flex items-center justify-between">
                    <div>
                        <CardTitle className={compact ? "text-lg" : ""}>Performance Analysis</CardTitle>
                        {!compact && <CardDescription>Automated bottleneck detection and optimization hints</CardDescription>}
                    </div>
                    <div className="text-right">
                        <div className={`${compact ? 'text-3xl' : 'text-4xl'} font-bold ${getScoreColor(analysis.performanceScore)}`}>
                            {analysis.performanceScore}
                        </div>
                        <div className="text-sm text-muted-foreground">
                            {getScoreLabel(analysis.performanceScore)}
                        </div>
                    </div>
                </div>
            </CardHeader>
            <CardContent className={compact ? "pt-0" : ""}>
                <div className={compact ? "space-y-2" : "space-y-4"}>
                    {/* SLA Configuration */}
                    <Collapsible open={slaOpen} onOpenChange={setSlaOpen}>
                        <CollapsibleTrigger asChild>
                            <Button variant="outline" size="sm" className="w-full justify-between">
                                <span className="flex items-center gap-2">
                                    <Settings2 className="h-4 w-4" />
                                    SLA Targets
                                    {slaEnabled && (
                                        <Badge variant="secondary" className="text-xs">
                                            P{slaConfig.latencyPercentile} &lt; {slaConfig.latencyTarget}s
                                            {budgetEnabled && slaConfig.budgetLimit && ` | $${slaConfig.budgetLimit}`}
                                        </Badge>
                                    )}
                                </span>
                                <ChevronDown className={`h-4 w-4 transition-transform ${slaOpen ? 'rotate-180' : ''}`} />
                            </Button>
                        </CollapsibleTrigger>
                        <CollapsibleContent className="pt-3">
                            <div className="p-3 border rounded-lg bg-muted/30 space-y-3">
                                <div className="flex items-center justify-between">
                                    <Label className="text-sm font-medium">Enable SLA-based scoring</Label>
                                    <Button
                                        variant={slaEnabled ? "default" : "outline"}
                                        size="sm"
                                        onClick={() => setSlaEnabled(!slaEnabled)}
                                    >
                                        {slaEnabled ? "Enabled" : "Disabled"}
                                    </Button>
                                </div>

                                {slaEnabled && (
                                    <>
                                        <Separator />
                                        <div className="space-y-3">
                                            {/* Latency SLA */}
                                            <div className="flex items-center gap-2">
                                                <Target className="h-4 w-4 text-muted-foreground" />
                                                <Label className="text-sm min-w-[60px]">Latency:</Label>
                                                <Select
                                                    value={slaConfig.latencyPercentile.toString()}
                                                    onValueChange={(v) => setSlaConfig({ ...slaConfig, latencyPercentile: parseInt(v) })}
                                                >
                                                    <SelectTrigger className="w-[80px] h-8">
                                                        <SelectValue />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="50">P50</SelectItem>
                                                        <SelectItem value="90">P90</SelectItem>
                                                        <SelectItem value="95">P95</SelectItem>
                                                        <SelectItem value="99">P99</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                                <span className="text-sm text-muted-foreground">&lt;</span>
                                                <Input
                                                    type="number"
                                                    min="0.1"
                                                    step="0.1"
                                                    value={slaConfig.latencyTarget}
                                                    onChange={(e) => setSlaConfig({ ...slaConfig, latencyTarget: parseFloat(e.target.value) || 1 })}
                                                    className="w-[70px] h-8"
                                                />
                                                <span className="text-sm text-muted-foreground">seconds</span>
                                            </div>

                                            {/* Budget SLA */}
                                            <div className="flex items-center gap-2">
                                                <DollarSign className="h-4 w-4 text-muted-foreground" />
                                                <Label className="text-sm min-w-[60px]">Budget:</Label>
                                                <Button
                                                    variant={budgetEnabled ? "default" : "outline"}
                                                    size="sm"
                                                    className="h-8"
                                                    onClick={() => {
                                                        setBudgetEnabled(!budgetEnabled)
                                                        if (!budgetEnabled) {
                                                            setSlaConfig({ ...slaConfig, budgetLimit: 10 })
                                                        } else {
                                                            setSlaConfig({ ...slaConfig, budgetLimit: null })
                                                        }
                                                    }}
                                                >
                                                    {budgetEnabled ? "On" : "Off"}
                                                </Button>
                                                {budgetEnabled && (
                                                    <>
                                                        <span className="text-sm text-muted-foreground">&lt;</span>
                                                        <span className="text-sm">$</span>
                                                        <Input
                                                            type="number"
                                                            min="0.01"
                                                            step="0.01"
                                                            value={slaConfig.budgetLimit ?? 10}
                                                            onChange={(e) => setSlaConfig({ ...slaConfig, budgetLimit: parseFloat(e.target.value) || 1 })}
                                                            className="w-[80px] h-8"
                                                        />
                                                    </>
                                                )}
                                            </div>
                                        </div>
                                    </>
                                )}
                            </div>
                        </CollapsibleContent>
                    </Collapsible>

                    {/* SLA Insights (always show first if present) */}
                    {displaySla.length > 0 && (
                        <div className={compact ? "space-y-1" : "space-y-2"}>
                            <h4 className="text-sm font-semibold text-purple-600 flex items-center gap-2">
                                <Target className="h-4 w-4" />
                                SLA Compliance
                            </h4>
                            {displaySla.map((insight, idx) => (
                                <InsightCard key={`sla-${idx}`} insight={insight} compact={compact} />
                            ))}
                        </div>
                    )}

                    {displayCritical.length > 0 && (
                        <div className={compact ? "space-y-1" : "space-y-2"}>
                            <h4 className="text-sm font-semibold text-red-600 flex items-center gap-2">
                                <AlertCircle className="h-4 w-4" />
                                Critical Issues
                            </h4>
                            {displayCritical.map((insight, idx) => (
                                <InsightCard key={`critical-${idx}`} insight={insight} compact={compact} />
                            ))}
                        </div>
                    )}

                    {displayWarning.length > 0 && (
                        <div className={compact ? "space-y-1" : "space-y-2"}>
                            <h4 className="text-sm font-semibold text-yellow-600 flex items-center gap-2">
                                <AlertTriangle className="h-4 w-4" />
                                Warnings
                            </h4>
                            {displayWarning.map((insight, idx) => (
                                <InsightCard key={`warning-${idx}`} insight={insight} compact={compact} />
                            ))}
                        </div>
                    )}

                    {displayOther.length > 0 && (
                        <div className={compact ? "space-y-1" : "space-y-2"}>
                            {!compact && (
                                <h4 className="text-sm font-semibold text-muted-foreground flex items-center gap-2">
                                    <Info className="h-4 w-4" />
                                    Observations
                                </h4>
                            )}
                            {displayOther.map((insight, idx) => (
                                <InsightCard key={`other-${idx}`} insight={insight} compact={compact} />
                            ))}
                        </div>
                    )}

                    {compact && criticalInsights.length === 0 && warningInsights.length === 0 && slaInsights.filter(s => s.severity === "critical" || s.severity === "warning").length === 0 && (
                        <div className="flex items-center gap-2 text-green-600">
                            <CheckCircle className="h-4 w-4" />
                            <span className="text-sm">No critical issues detected</span>
                        </div>
                    )}
                </div>
            </CardContent>
        </Card>
    )
}

function InsightCard({ insight, compact = false }: { insight: BottleneckInsight, compact?: boolean }) {
    if (compact) {
        return (
            <div className="flex items-center gap-2 p-2 rounded-lg border bg-card text-sm">
                {getSeverityIcon(insight.severity)}
                <Badge variant={getSeverityBadgeVariant(insight.severity)} className="text-xs">
                    {insight.category}
                </Badge>
                <span className="font-medium truncate">{insight.title}</span>
                {insight.value && (
                    <span className="text-xs font-mono text-muted-foreground ml-auto">{insight.value}</span>
                )}
            </div>
        )
    }

    return (
        <div className="flex items-start gap-3 p-3 rounded-lg border bg-card">
            <div className="pt-0.5">
                {getSeverityIcon(insight.severity)}
            </div>
            <div className="flex-1 space-y-1">
                <div className="flex items-center gap-2">
                    <Badge variant={getSeverityBadgeVariant(insight.severity)} className="text-xs">
                        {insight.category}
                    </Badge>
                    <span className="font-semibold text-sm">{insight.title}</span>
                </div>
                <p className="text-sm text-muted-foreground">{insight.description}</p>
                {insight.value && (
                    <p className="text-sm font-mono text-foreground">{insight.value}</p>
                )}
                {insight.recommendation && (
                    <div className="flex items-start gap-2 mt-2 p-2 rounded bg-muted/50">
                        <TrendingUp className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                        <p className="text-xs text-muted-foreground italic">{insight.recommendation}</p>
                    </div>
                )}
            </div>
        </div>
    )
}

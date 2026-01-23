"use client"

import React from "react"
import NavHeader from "@/components/nav-header"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
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
    ChartConfig,
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
} from "@/components/ui/chart"
import { Bar, BarChart, CartesianGrid, XAxis, YAxis, Label as RechartsLabel, LabelList, Cell, Legend } from "recharts"
import { Plus, Trash2, Edit2, Save, X, Bookmark, TrendingUp, Activity, Zap, Cloud } from "lucide-react"

type CacheTier = {
    id: string
    name: string
    hitRate: number // 0-100
    storageType: "DRAM" | "SSD" | "Remote"
    bandwidthGBps: number
    volatility: number // 0-1: sensitivity to updates (cache invalidation rate)
}

type CacheConfig = {
    name: string
    tiers: CacheTier[]
    // Workload parameters
    warmupRate: number // queries/sec to warm up cache (cold-start behavior)
    readUpdateRatio: number // reads per update (higher = more read-heavy)
    convergenceQueries: number // queries needed to reach steady-state hit rate
    degradationPerTier: number // 0-1: interference/overhead per additional tier
}

type PresetConfig = {
    name: string
    description: string
    tiers: CacheTier[]
    warmupRate?: number
    readUpdateRatio?: number
    convergenceQueries?: number
    degradationPerTier?: number
}

const presetConfigs: PresetConfig[] = [
    {
        name: "No Cache",
        description: "Baseline with no caching",
        tiers: []
    },
    {
        name: "Single-Tier SSD",
        description: "Single SSD cache tier",
        tiers: [
            { id: "ssd1", name: "SSD Cache", hitRate: 35, storageType: "SSD", bandwidthGBps: 4, volatility: 0.1 }
        ]
    },
    {
        name: "Multi-Tier (Mem+SSD)",
        description: "DRAM and SSD hierarchical cache",
        tiers: [
            { id: "mem1", name: "Memory Cache", hitRate: 40, storageType: "DRAM", bandwidthGBps: 40, volatility: 0.2 },
            { id: "ssd1", name: "SSD Cache", hitRate: 35, storageType: "SSD", bandwidthGBps: 4, volatility: 0.1 }
        ]
    },
    {
        name: "High-Churn BI",
        description: "BI dashboard with frequent ingestions",
        tiers: [
            { id: "mem1", name: "Hot Data", hitRate: 30, storageType: "DRAM", bandwidthGBps: 40, volatility: 0.4 },
            { id: "ssd1", name: "Warm Data", hitRate: 40, storageType: "SSD", bandwidthGBps: 4, volatility: 0.2 }
        ],
        readUpdateRatio: 5,
        warmupRate: 50,
        convergenceQueries: 200
    }
]

const defaultBandwidthByType = {
    "DRAM": 40,
    "SSD": 4,
    "Remote": 1
}

type CacheMetrics = {
    name: string
    overallHitRate: number
    effectiveHitRate: number // after accounting for volatility and updates
    dramPct: number
    ssdPct: number
    remotePct: number
    avgBandwidthGBps: number
    latencyMs: number
    throughputQps: number
    warmupTimeQueries: number // queries to reach steady state
    tierOverhead: number // degradation from multi-tier interference
    p99LatencyMs: number // tail latency estimate
    estimatedCostFactor: number // relative cost (1.0 = baseline)
}

function calculateMetrics(config: CacheConfig): CacheMetrics {
    const { name, tiers, warmupRate, readUpdateRatio, convergenceQueries, degradationPerTier } = config

    // Calculate cumulative hit rate (steady-state)
    let cumulativeMissRate = 1.0
    for (const tier of tiers) {
        cumulativeMissRate *= (1 - tier.hitRate / 100)
    }
    const overallHitRate = (1 - cumulativeMissRate) * 100

    // Calculate effective hit rate accounting for volatility and read/update ratio
    // Higher volatility + lower read/update ratio = more cache invalidation
    let effectiveMissRate = 1.0
    for (const tier of tiers) {
        const volatilityImpact = tier.volatility / Math.max(1, readUpdateRatio / 10)
        const effectiveTierHitRate = tier.hitRate * (1 - volatilityImpact)
        effectiveMissRate *= (1 - effectiveTierHitRate / 100)
    }
    const effectiveHitRate = (1 - effectiveMissRate) * 100

    // Calculate tier distribution
    let dramPct = 0
    let ssdPct = 0
    let remainingQueries = 100

    for (const tier of tiers) {
        const hitPct = (tier.hitRate / 100) * remainingQueries

        if (tier.storageType === "DRAM") {
            dramPct += hitPct
        } else if (tier.storageType === "SSD") {
            ssdPct += hitPct
        }

        remainingQueries -= hitPct
    }
    const remotePct = remainingQueries

    // Calculate tier overhead (interference from additional tiers)
    const tierOverhead = tiers.length > 1 ? (tiers.length - 1) * degradationPerTier : 0

    // Calculate weighted average bandwidth with tier overhead
    const rawBandwidth = (dramPct * 40 + ssdPct * 4 + remotePct * 1) / 100
    const avgBandwidthGBps = rawBandwidth * (1 - tierOverhead)

    // Calculate latency and throughput
    const latencyMs = 1000 / avgBandwidthGBps
    const throughputQps = avgBandwidthGBps * 100

    // Estimate warmup time based on convergence queries and warmup rate
    const warmupTimeQueries = convergenceQueries

    // Estimate p99 latency (tail latency affected by remote reads and volatility)
    const avgVolatility = tiers.length > 0
        ? tiers.reduce((sum, t) => sum + t.volatility, 0) / tiers.length
        : 0
    const tailLatencyMultiplier = 2 + (remotePct / 100) * 3 + avgVolatility * 2
    const p99LatencyMs = latencyMs * tailLatencyMultiplier

    // Estimate relative cost (more tiers = more infrastructure cost, but offset by fewer remote reads)
    const tierCost = 1 + tiers.length * 0.1 // each tier adds 10% infra cost
    const remoteCostSavings = (100 - remotePct) / 100 * 0.5 // up to 50% savings from avoiding remote
    const estimatedCostFactor = tierCost - remoteCostSavings

    return {
        name,
        overallHitRate,
        effectiveHitRate,
        dramPct,
        ssdPct,
        remotePct,
        avgBandwidthGBps,
        latencyMs,
        throughputQps,
        warmupTimeQueries,
        tierOverhead,
        p99LatencyMs,
        estimatedCostFactor
    }
}

// Colors matching generate_demo_figure.py
const chartColors = {
    configs: ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#9B59B6'],
    memory: '#06A77D',
    ssd: '#F18F01',
    remote: '#D32F2F'
}

// Default workload parameters
const defaultWorkloadParams = {
    warmupRate: 100,
    readUpdateRatio: 20,
    convergenceQueries: 100,
    degradationPerTier: 0.05
}

export default function CacheExplorer() {
    const [configurations, setConfigurations] = React.useState<CacheConfig[]>([
        { name: "No Cache", tiers: [], ...defaultWorkloadParams }
    ])
    const [activeConfigIndex, setActiveConfigIndex] = React.useState(0)
    const [editingId, setEditingId] = React.useState<string | null>(null)
    const [newTier, setNewTier] = React.useState<Partial<CacheTier>>({
        name: "",
        hitRate: 50,
        storageType: "SSD",
        bandwidthGBps: 4,
        volatility: 0.1
    })
    const [newConfigName, setNewConfigName] = React.useState("")

    // Calculate metrics for all configurations
    const allMetrics = React.useMemo(() => {
        return configurations.map(config => calculateMetrics(config))
    }, [configurations])

    const baseline = allMetrics.find(m => m.name === "No Cache") || allMetrics[0]

    const activeConfig = configurations[activeConfigIndex]
    const activeTiers = activeConfig?.tiers || []

    const handlePresetSelect = (preset: PresetConfig) => {
        // Add as new configuration if not exists
        const exists = configurations.some(c => c.name === preset.name)
        if (!exists) {
            const newConfig: CacheConfig = {
                name: preset.name,
                tiers: preset.tiers.map(t => ({ ...t, id: Date.now().toString() + Math.random() })),
                warmupRate: preset.warmupRate ?? defaultWorkloadParams.warmupRate,
                readUpdateRatio: preset.readUpdateRatio ?? defaultWorkloadParams.readUpdateRatio,
                convergenceQueries: preset.convergenceQueries ?? defaultWorkloadParams.convergenceQueries,
                degradationPerTier: preset.degradationPerTier ?? defaultWorkloadParams.degradationPerTier
            }
            setConfigurations([...configurations, newConfig])
            setActiveConfigIndex(configurations.length)
        } else {
            setActiveConfigIndex(configurations.findIndex(c => c.name === preset.name))
        }
    }

    const handleAddConfiguration = () => {
        if (!newConfigName.trim()) return
        const newConfig: CacheConfig = {
            name: newConfigName,
            tiers: [],
            ...defaultWorkloadParams
        }
        setConfigurations([...configurations, newConfig])
        setActiveConfigIndex(configurations.length)
        setNewConfigName("")
    }

    const handleUpdateConfigParams = (updates: Partial<CacheConfig>) => {
        const updatedConfigs = [...configurations]
        updatedConfigs[activeConfigIndex] = {
            ...updatedConfigs[activeConfigIndex],
            ...updates
        }
        setConfigurations(updatedConfigs)
    }

    const handleRemoveConfiguration = (index: number) => {
        if (configurations.length <= 1) return
        const newConfigs = configurations.filter((_, i) => i !== index)
        setConfigurations(newConfigs)
        if (activeConfigIndex >= newConfigs.length) {
            setActiveConfigIndex(newConfigs.length - 1)
        }
    }

    const handleAddTier = () => {
        if (!newTier.name || !newTier.name.trim()) return

        const tier: CacheTier = {
            id: Date.now().toString(),
            name: newTier.name,
            hitRate: newTier.hitRate || 50,
            storageType: newTier.storageType || "SSD",
            bandwidthGBps: newTier.bandwidthGBps || 4,
            volatility: newTier.volatility || 0.1
        }

        const updatedConfigs = [...configurations]
        updatedConfigs[activeConfigIndex] = {
            ...updatedConfigs[activeConfigIndex],
            tiers: [...updatedConfigs[activeConfigIndex].tiers, tier]
        }
        setConfigurations(updatedConfigs)
        setNewTier({
            name: "",
            hitRate: 50,
            storageType: "SSD",
            bandwidthGBps: 4,
            volatility: 0.1
        })
    }

    const handleRemoveTier = (id: string) => {
        const updatedConfigs = [...configurations]
        updatedConfigs[activeConfigIndex] = {
            ...updatedConfigs[activeConfigIndex],
            tiers: updatedConfigs[activeConfigIndex].tiers.filter(t => t.id !== id)
        }
        setConfigurations(updatedConfigs)
    }

    const handleEditTier = (id: string, updates: Partial<CacheTier>) => {
        const updatedConfigs = [...configurations]
        updatedConfigs[activeConfigIndex] = {
            ...updatedConfigs[activeConfigIndex],
            tiers: updatedConfigs[activeConfigIndex].tiers.map(t => t.id === id ? { ...t, ...updates } : t)
        }
        setConfigurations(updatedConfigs)
    }

    const handleStorageTypeChange = (storageType: "DRAM" | "SSD" | "Remote") => {
        setNewTier({
            ...newTier,
            storageType,
            bandwidthGBps: defaultBandwidthByType[storageType]
        })
    }

    // Prepare chart data for all configurations
    const hitRateData = allMetrics.map((m, i) => ({
        name: m.name,
        hitRate: parseFloat(m.effectiveHitRate.toFixed(1)),
        idealHitRate: parseFloat(m.overallHitRate.toFixed(1)),
        fill: chartColors.configs[i % chartColors.configs.length]
    }))

    const tierDistributionData = allMetrics.map(m => ({
        name: m.name,
        memory: parseFloat(m.dramPct.toFixed(1)),
        ssd: parseFloat(m.ssdPct.toFixed(1)),
        remote: parseFloat(m.remotePct.toFixed(1))
    }))

    const latencyData = allMetrics.map((m, i) => {
        const reduction = ((baseline.latencyMs - m.latencyMs) / baseline.latencyMs * 100)
        return {
            name: m.name,
            reduction: parseFloat(reduction.toFixed(0)),
            latency: parseFloat(m.latencyMs.toFixed(0)),
            p99: parseFloat(m.p99LatencyMs.toFixed(0)),
            fill: chartColors.configs[i % chartColors.configs.length]
        }
    })

    const throughputData = allMetrics.map((m, i) => ({
        name: m.name,
        speedup: parseFloat((m.throughputQps / baseline.throughputQps).toFixed(1)),
        qps: parseFloat(m.throughputQps.toFixed(0)),
        costFactor: parseFloat(m.estimatedCostFactor.toFixed(2)),
        fill: chartColors.configs[i % chartColors.configs.length]
    }))

    return (
        <div className="w-full h-screen overflow-hidden bg-zinc-50 dark:bg-black">
            <NavHeader />
            <div className="h-[calc(100vh-56px)] overflow-y-auto">
                <div className="container mx-auto px-4 py-3 space-y-3">
                {/* Header + Presets inline */}
                <div className="flex items-center gap-4 flex-wrap">
                    <h1 className="text-2xl font-bold whitespace-nowrap">Cache Tier Builder</h1>
                    <div className="flex flex-wrap gap-1.5 flex-1">
                        {presetConfigs.map((preset) => {
                            const exists = configurations.some(c => c.name === preset.name)
                            return (
                                <Button
                                    key={preset.name}
                                    size="sm"
                                    variant={exists ? "secondary" : "outline"}
                                    onClick={() => handlePresetSelect(preset)}
                                    disabled={exists}
                                    className="h-7 px-2"
                                >
                                    {preset.name}
                                </Button>
                            )
                        })}
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-3">
                    {/* Left: Configuration Management */}
                    <Card className="lg:col-span-4">
                        <CardHeader className="py-2 px-4">
                            <CardTitle className="text-base">Configurations</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-3 px-4 pb-4 pt-0">
                            {/* Configuration Tabs + Add inline */}
                            <div className="flex flex-wrap gap-1.5 items-center">
                                {configurations.map((config, index) => (
                                    <div key={index} className="flex items-center">
                                        <Button
                                            size="sm"
                                            variant={activeConfigIndex === index ? "default" : "outline"}
                                            onClick={() => setActiveConfigIndex(index)}
                                            className="h-7 px-2"
                                        >
                                            {config.name}
                                        </Button>
                                        {config.name !== "No Cache" && (
                                            <Button
                                                size="sm"
                                                variant="ghost"
                                                onClick={() => handleRemoveConfiguration(index)}
                                                className="h-5 w-5 p-0 ml-0.5"
                                            >
                                                <X className="h-3 w-3" />
                                            </Button>
                                        )}
                                    </div>
                                ))}
                                <div className="flex gap-1">
                                    <Input
                                        placeholder="New config"
                                        value={newConfigName}
                                        onChange={(e) => setNewConfigName(e.target.value)}
                                        className="h-7 w-24 text-sm"
                                    />
                                    <Button size="sm" onClick={handleAddConfiguration} disabled={!newConfigName.trim()} className="h-7 px-2">
                                        <Plus className="h-3 w-3" />
                                    </Button>
                                </div>
                            </div>

                            <Separator />

                            {/* Active Config Tiers */}
                            <div className="space-y-1.5">
                                <Label className="text-sm font-semibold">Cache Tiers: {activeConfig?.name}</Label>
                                {activeConfig?.name === "No Cache" ? (
                                    <p className="text-sm text-muted-foreground">Baseline - no tiers</p>
                                ) : (
                                    <>
                                        {activeTiers.length > 0 && (
                                            <div className="space-y-1.5">
                                                {activeTiers.map((tier) => (
                                                    <div key={tier.id}>
                                                        {editingId === tier.id ? (
                                                            <div className="p-2 border rounded-lg space-y-1.5 bg-muted/50">
                                                                <div className="flex gap-1.5">
                                                                    <Input value={tier.name} onChange={(e) => handleEditTier(tier.id, { name: e.target.value })} className="h-7 text-sm flex-1" placeholder="Tier name" />
                                                                    <Input type="number" min="0" max="100" value={tier.hitRate} onChange={(e) => { const val = parseFloat(e.target.value); if (!isNaN(val)) handleEditTier(tier.id, { hitRate: val }) }} className="h-7 text-sm w-16" placeholder="Hit%" />
                                                                    <Select value={tier.storageType} onValueChange={(value: "DRAM" | "SSD" | "Remote") => handleEditTier(tier.id, { storageType: value, bandwidthGBps: defaultBandwidthByType[value] })}>
                                                                        <SelectTrigger className="h-7 text-sm w-20"><SelectValue /></SelectTrigger>
                                                                        <SelectContent>
                                                                            <SelectItem value="DRAM">DRAM</SelectItem>
                                                                            <SelectItem value="SSD">SSD</SelectItem>
                                                                            <SelectItem value="Remote">Remote</SelectItem>
                                                                        </SelectContent>
                                                                    </Select>
                                                                </div>
                                                                <div className="flex items-center gap-1.5">
                                                                    <Label className="text-xs text-muted-foreground">Volatility:</Label>
                                                                    <Input type="number" min="0" max="1" step="0.1" value={tier.volatility ?? 0.1} onChange={(e) => { const val = parseFloat(e.target.value); if (!isNaN(val)) handleEditTier(tier.id, { volatility: val }) }} className="h-7 text-sm w-16" />
                                                                    <Button size="sm" onClick={() => setEditingId(null)} className="h-7 ml-auto"><Save className="h-3 w-3 mr-1" /> Done</Button>
                                                                </div>
                                                            </div>
                                                        ) : (
                                                            <div className="flex items-center gap-2 px-2 py-1.5 border rounded-lg">
                                                                <span className="font-medium">{tier.name}</span>
                                                                <span className="text-muted-foreground text-sm">{tier.hitRate}% · {tier.storageType} · v{(tier.volatility ?? 0.1).toFixed(1)}</span>
                                                                <div className="ml-auto flex">
                                                                    <Button size="sm" variant="ghost" className="h-6 w-6 p-0" onClick={() => setEditingId(tier.id)}><Edit2 className="h-3 w-3" /></Button>
                                                                    <Button size="sm" variant="ghost" className="h-6 w-6 p-0" onClick={() => handleRemoveTier(tier.id)}><Trash2 className="h-3 w-3 text-red-500" /></Button>
                                                                </div>
                                                            </div>
                                                        )}
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                        {/* Add Tier Form */}
                                        <div className="p-2 border rounded-lg bg-muted/20 space-y-1.5">
                                            <div className="flex gap-1.5 text-xs text-muted-foreground">
                                                <span className="flex-1">Tier Name</span>
                                                <span className="w-14 text-center">Hit %</span>
                                                <span className="w-20 text-center">Type</span>
                                            </div>
                                            <div className="flex gap-1.5">
                                                <Input placeholder="e.g. Hot Data" value={newTier.name} onChange={(e) => setNewTier({ ...newTier, name: e.target.value })} className="h-7 text-sm flex-1" />
                                                <Input type="number" min="0" max="100" value={newTier.hitRate ?? 50} onChange={(e) => { const val = parseFloat(e.target.value); if (!isNaN(val)) setNewTier({ ...newTier, hitRate: val }) }} className="h-7 text-sm w-14 text-center" />
                                                <Select value={newTier.storageType} onValueChange={handleStorageTypeChange}>
                                                    <SelectTrigger className="h-7 text-sm w-20"><SelectValue /></SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="DRAM">DRAM</SelectItem>
                                                        <SelectItem value="SSD">SSD</SelectItem>
                                                        <SelectItem value="Remote">Remote</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                            <div className="flex items-center gap-1.5">
                                                <Label className="text-xs text-muted-foreground">Volatility:</Label>
                                                <Input type="number" min="0" max="1" step="0.1" value={newTier.volatility ?? 0.1} onChange={(e) => { const val = parseFloat(e.target.value); if (!isNaN(val)) setNewTier({ ...newTier, volatility: val }) }} className="h-7 text-sm w-16" />
                                                <Button size="sm" onClick={handleAddTier} disabled={!newTier.name?.trim()} className="h-7 ml-auto"><Plus className="h-3 w-3 mr-1" /> Add</Button>
                                            </div>
                                        </div>
                                    </>
                                )}
                            </div>

                            <Separator />

                            {/* Workload Parameters */}
                            <div className="space-y-1.5">
                                <Label className="text-sm font-semibold flex items-center gap-1.5"><Activity className="h-4 w-4" />Workload Parameters</Label>
                                <div className="grid grid-cols-2 gap-2">
                                    <div className="space-y-0.5">
                                        <Label className="text-xs text-muted-foreground">Read/Update Ratio</Label>
                                        <Input type="number" min="1" max="1000" value={activeConfig?.readUpdateRatio ?? defaultWorkloadParams.readUpdateRatio} onChange={(e) => { const val = parseFloat(e.target.value); if (!isNaN(val)) handleUpdateConfigParams({ readUpdateRatio: val }) }} className="h-7 text-sm" />
                                    </div>
                                    <div className="space-y-0.5">
                                        <Label className="text-xs text-muted-foreground">Warmup Rate (q/s)</Label>
                                        <Input type="number" min="1" max="10000" value={activeConfig?.warmupRate ?? defaultWorkloadParams.warmupRate} onChange={(e) => { const val = parseFloat(e.target.value); if (!isNaN(val)) handleUpdateConfigParams({ warmupRate: val }) }} className="h-7 text-sm" />
                                    </div>
                                    <div className="space-y-0.5">
                                        <Label className="text-xs text-muted-foreground">Convergence Queries</Label>
                                        <Input type="number" min="10" max="10000" value={activeConfig?.convergenceQueries ?? defaultWorkloadParams.convergenceQueries} onChange={(e) => { const val = parseFloat(e.target.value); if (!isNaN(val)) handleUpdateConfigParams({ convergenceQueries: val }) }} className="h-7 text-sm" />
                                    </div>
                                    <div className="space-y-0.5">
                                        <Label className="text-xs text-muted-foreground">Tier Degradation</Label>
                                        <Input type="number" min="0" max="0.5" step="0.01" value={activeConfig?.degradationPerTier ?? defaultWorkloadParams.degradationPerTier} onChange={(e) => { const val = parseFloat(e.target.value); if (!isNaN(val)) handleUpdateConfigParams({ degradationPerTier: val }) }} className="h-7 text-sm" />
                                    </div>
                                </div>
                            </div>
                        </CardContent>
                    </Card>

                    {/* Right: Performance Comparison with Hit Rate Boxes and 2x2 Charts */}
                    <div className="lg:col-span-8 space-y-3">
                        {/* Hit Rate Summary Boxes */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                            {allMetrics.map((m, i) => (
                                <div
                                    key={m.name}
                                    className="p-2 border rounded-lg bg-card"
                                    style={{ borderLeftColor: chartColors.configs[i % chartColors.configs.length], borderLeftWidth: 4 }}
                                >
                                    <div className="text-sm text-muted-foreground truncate">{m.name}</div>
                                    <div className="text-lg font-bold leading-tight">
                                        {m.effectiveHitRate.toFixed(1)}%
                                        {m.effectiveHitRate < m.overallHitRate && (
                                            <span className="text-xs font-normal text-amber-600 ml-1">({m.overallHitRate.toFixed(0)}% ideal)</span>
                                        )}
                                    </div>
                                    <div className="text-sm text-muted-foreground">{m.throughputQps.toFixed(0)} QPS · P99: {m.p99LatencyMs.toFixed(0)}ms</div>
                                </div>
                            ))}
                        </div>

                        {/* 2x2 Performance Charts Grid */}
                        <Card>
                            <CardContent className="p-4">
                                <div className="grid grid-cols-2 gap-3">
                                    {/* Chart 1: Cache Hit Rates */}
                                    <div>
                                        <h3 className="text-sm font-semibold mb-1">(a) Effective Hit Rate</h3>
                                        <ChartContainer config={{ hitRate: { label: "Hit Rate (%)" } }} className="h-[180px]">
                                            <BarChart data={hitRateData} margin={{ top: 22, right: 8, bottom: 25, left: 12 }}>
                                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                                <XAxis dataKey="name" tick={{ fontSize: 12 }} interval={0} angle={-15} textAnchor="end" height={45} />
                                                <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} width={40}>
                                                    <RechartsLabel value="Hit Rate (%)" angle={-90} position="insideLeft" style={{ fontWeight: 'bold', fontSize: 12 }} />
                                                </YAxis>
                                                <Bar dataKey="hitRate" radius={[4, 4, 0, 0]}>
                                                    {hitRateData.map((entry, index) => (<Cell key={`cell-${index}`} fill={entry.fill} />))}
                                                    <LabelList dataKey="hitRate" position="top" formatter={(v: number) => `${v}%`} style={{ fontWeight: 'bold', fontSize: 12 }} />
                                                </Bar>
                                            </BarChart>
                                        </ChartContainer>
                                    </div>

                                    {/* Chart 2: Storage Tier Distribution */}
                                    <div>
                                        <h3 className="text-sm font-semibold mb-1">(b) Tier Distribution</h3>
                                        <ChartContainer config={{ memory: { label: "DRAM", color: chartColors.memory }, ssd: { label: "SSD", color: chartColors.ssd }, remote: { label: "Remote", color: chartColors.remote } }} className="h-[180px]">
                                            <BarChart data={tierDistributionData} margin={{ top: 12, right: 8, bottom: 25, left: 12 }}>
                                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                                <XAxis dataKey="name" tick={{ fontSize: 12 }} interval={0} angle={-15} textAnchor="end" height={45} />
                                                <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} width={40}>
                                                    <RechartsLabel value="%" angle={-90} position="insideLeft" style={{ fontWeight: 'bold', fontSize: 12 }} />
                                                </YAxis>
                                                <Legend wrapperStyle={{ fontSize: 11, paddingTop: 0 }} />
                                                <Bar dataKey="memory" stackId="a" fill={chartColors.memory} name="DRAM" />
                                                <Bar dataKey="ssd" stackId="a" fill={chartColors.ssd} name="SSD" />
                                                <Bar dataKey="remote" stackId="a" fill={chartColors.remote} name="Remote" />
                                            </BarChart>
                                        </ChartContainer>
                                    </div>

                                    {/* Chart 3: Latency Reduction with P99 */}
                                    <div>
                                        <h3 className="text-sm font-semibold mb-1">(c) Latency Reduction & P99</h3>
                                        <ChartContainer config={{ reduction: { label: "Latency Reduction (%)" } }} className="h-[180px]">
                                            <BarChart data={latencyData} margin={{ top: 30, right: 8, bottom: 25, left: 12 }}>
                                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                                <XAxis dataKey="name" tick={{ fontSize: 12 }} interval={0} angle={-15} textAnchor="end" height={45} />
                                                <YAxis domain={[-10, 100]} tick={{ fontSize: 12 }} width={40}>
                                                    <RechartsLabel value="Reduction (%)" angle={-90} position="insideLeft" style={{ fontWeight: 'bold', fontSize: 12 }} />
                                                </YAxis>
                                                <Bar dataKey="reduction" radius={[4, 4, 0, 0]}>
                                                    {latencyData.map((entry, index) => (<Cell key={`cell-${index}`} fill={entry.fill} />))}
                                                    <LabelList dataKey="reduction" position="top" content={({ x, y, width, index }) => {
                                                        const d = latencyData[index as number]
                                                        return (<text x={(x as number) + (width as number) / 2} y={(y as number) - 4} textAnchor="middle" style={{ fontWeight: 'bold', fontSize: 11 }}>
                                                            <tspan x={(x as number) + (width as number) / 2} dy="0">{d.reduction}%</tspan>
                                                            <tspan x={(x as number) + (width as number) / 2} dy="11" style={{ fontSize: 10 }}>P99: {d.p99}ms</tspan>
                                                        </text>)
                                                    }} />
                                                </Bar>
                                            </BarChart>
                                        </ChartContainer>
                                    </div>

                                    {/* Chart 4: Throughput & Cost */}
                                    <div>
                                        <h3 className="text-sm font-semibold mb-1">(d) Throughput & Cost Factor</h3>
                                        <ChartContainer config={{ speedup: { label: "Throughput Speedup" } }} className="h-[180px]">
                                            <BarChart data={throughputData} margin={{ top: 30, right: 8, bottom: 25, left: 12 }}>
                                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                                <XAxis dataKey="name" tick={{ fontSize: 12 }} interval={0} angle={-15} textAnchor="end" height={45} />
                                                <YAxis domain={[0, Math.max(...throughputData.map(d => d.speedup)) * 1.3]} tick={{ fontSize: 12 }} width={40}>
                                                    <RechartsLabel value="Speedup (x)" angle={-90} position="insideLeft" style={{ fontWeight: 'bold', fontSize: 12 }} />
                                                </YAxis>
                                                <Bar dataKey="speedup" radius={[4, 4, 0, 0]}>
                                                    {throughputData.map((entry, index) => (<Cell key={`cell-${index}`} fill={entry.fill} />))}
                                                    <LabelList dataKey="speedup" position="top" content={({ x, y, width, index }) => {
                                                        const d = throughputData[index as number]
                                                        const costColor = d.costFactor < 1 ? '#06A77D' : d.costFactor > 1.1 ? '#E74C3C' : '#666'
                                                        return (<text x={(x as number) + (width as number) / 2} y={(y as number) - 4} textAnchor="middle" style={{ fontWeight: 'bold', fontSize: 11 }}>
                                                            <tspan x={(x as number) + (width as number) / 2} dy="0">{d.speedup}x</tspan>
                                                            <tspan x={(x as number) + (width as number) / 2} dy="11" style={{ fontSize: 10, fill: costColor }}>${d.costFactor.toFixed(2)}</tspan>
                                                        </text>)
                                                    }} />
                                                </Bar>
                                            </BarChart>
                                        </ChartContainer>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </div>
                </div>
            </div>
        </div>
    )
}

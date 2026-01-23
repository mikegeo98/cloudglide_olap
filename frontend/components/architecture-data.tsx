"use client"

import NextButton from "@/components/next-btn";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { InputContext } from "@/components/provider";
import {
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Spinner } from "@/components/ui/spinner";

import React from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { createArchitectureSchema, ZodDWAAS, ZodDWAASAutoscaling, ZodElasticPool, ZodQAAS } from "@/lib/zod-schemas";
import { ArchitectureType, instanceTypes } from "@/lib/config";
import { ChevronsUpDown, Plus, Copy, Trash2, Info } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function ArchitectureData() {
    const { data, stage, setStage, setData } = React.useContext(InputContext)
    const [arch, setArch] = React.useState<string>()
    const [add, setAdd] = React.useState(false)

    function duplicateArchitecture(scenario: object) {
        setData({
            ...data,
            scenarios: [
                ...data.scenarios,
                JSON.parse(JSON.stringify(scenario)), // copy by value instead of reference
            ]
        })
    }

    function removeScenario(index: number) {
        setData({
            ...data,
            scenarios: data.scenarios.filter((_, i) => i !== index)
        })
    }

    return (
        <div className="flex flex-col items-center gap-6 w-[850px] px-6 max-h-full overflow-y-auto pb-8">
            <div className="w-full">
                <h1 className="text-2xl font-bold text-center mb-2">Architecture Configuration</h1>
                <p className="text-sm text-muted-foreground text-center">
                    Add multiple scenarios to compare different configurations in a single simulation run
                </p>
            </div>

            {/* Scenarios Summary Card */}
            {data.scenarios.length > 0 && (
                <Card className="w-full">
                    <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                            <div>
                                <CardTitle className="text-lg">Scenarios ({data.scenarios.length})</CardTitle>
                                <CardDescription>
                                    All scenarios will run when you click "Run Simulation"
                                </CardDescription>
                            </div>
                            <div className="flex gap-2">
                                <Button variant="outline" size="sm" onClick={() => { setAdd(true); setArch(undefined) }}>
                                    <Plus className="h-4 w-4 mr-1" />
                                    Add New
                                </Button>
                                <NextButton rightOnClick={() => setStage(stage + 1)} />
                            </div>
                        </div>
                    </CardHeader>
                    <CardContent className="space-y-3">
                        {data.scenarios.map((scenario, index) => {
                            let form = null
                            switch (scenario.architecture) {
                                case ArchitectureType.DWAAS:
                                    form = <DWAASForm key={index} scenario={scenario} />
                                    break
                                case ArchitectureType.DWAAS_AUTOSCALING:
                                    form = <DWAASAutoscalingForm key={index} scenario={scenario} />
                                    break
                                case ArchitectureType.ELASTIC_POOL:
                                    form = <ElasticPoolForm key={index} scenario={scenario} />
                                    break
                                case ArchitectureType.QAAS:
                                    form = <QAASForm key={index} scenario={scenario} />
                                    break
                                default:
                                    return null;
                            }

                            // Get a brief summary of the scenario
                            const summary = scenario.architecture === ArchitectureType.QAAS
                                ? `BW: ${scenario.network_bandwidth}`
                                : scenario.architecture === ArchitectureType.ELASTIC_POOL
                                    ? `${scenario.vpu} RPUs, ${(scenario.hit_rate * 100).toFixed(0)}% hit`
                                    : `${scenario.nodes} nodes, ${(scenario.hit_rate * 100).toFixed(0)}% hit`

                            return (
                                <Collapsible key={index} className="border rounded-lg">
                                    <CollapsibleTrigger asChild>
                                        <div className="flex items-center justify-between p-3 cursor-pointer hover:bg-muted/50 rounded-t-lg">
                                            <div className="flex items-center gap-3">
                                                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-sm font-bold">
                                                    {index + 1}
                                                </div>
                                                <div>
                                                    <h4 className="text-sm font-semibold">{scenario.architecture}</h4>
                                                    <p className="text-xs text-muted-foreground">{summary}</p>
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-1">
                                                <Button
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={(e) => { e.stopPropagation(); duplicateArchitecture(scenario) }}
                                                    title="Duplicate scenario"
                                                >
                                                    <Copy className="h-4 w-4" />
                                                </Button>
                                                <Button
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={(e) => { e.stopPropagation(); removeScenario(index) }}
                                                    title="Remove scenario"
                                                    className="text-destructive hover:text-destructive"
                                                >
                                                    <Trash2 className="h-4 w-4" />
                                                </Button>
                                                <Button variant="ghost" size="icon" className="size-8">
                                                    <ChevronsUpDown className="h-4 w-4" />
                                                </Button>
                                            </div>
                                        </div>
                                    </CollapsibleTrigger>
                                    <CollapsibleContent className="p-4 pt-0 border-t">
                                        {form}
                                    </CollapsibleContent>
                                </Collapsible>
                            )
                        })}
                    </CardContent>
                </Card>
            )}

            {/* Add New Scenario Section */}
            {(add || data.scenarios.length === 0) && (
                <Card className="w-full">
                    <CardHeader>
                        <CardTitle className="text-lg">
                            {data.scenarios.length === 0 ? "Create Your First Scenario" : "Add New Scenario"}
                        </CardTitle>
                        <CardDescription>
                            Select an architecture type to configure
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-3">
                            {Object.values(ArchitectureType).map((archType) => (
                                <Button
                                    key={archType}
                                    variant={arch === archType ? "default" : "outline"}
                                    className="h-auto py-3 justify-start"
                                    onClick={() => setArch(archType)}
                                >
                                    <div className="text-left">
                                        <div className="font-semibold">{archType}</div>
                                        <div className="text-xs opacity-80">
                                            {archType === ArchitectureType.DWAAS && "Data Warehouse as a Service"}
                                            {archType === ArchitectureType.DWAAS_AUTOSCALING && "DWAAS with autoscaling"}
                                            {archType === ArchitectureType.ELASTIC_POOL && "Serverless elastic pool"}
                                            {archType === ArchitectureType.QAAS && "Query as a Service"}
                                        </div>
                                    </div>
                                </Button>
                            ))}
                        </div>

                        {arch && (
                            <>
                                <Separator />
                                {(() => {
                                    const onAdded = () => { setAdd(false); setArch(undefined) }
                                    switch (arch) {
                                        case ArchitectureType.DWAAS:
                                            return <DWAASForm onAdded={onAdded} />;
                                        case ArchitectureType.DWAAS_AUTOSCALING:
                                            return <DWAASAutoscalingForm onAdded={onAdded} />;
                                        case ArchitectureType.ELASTIC_POOL:
                                            return <ElasticPoolForm onAdded={onAdded} />;
                                        case ArchitectureType.QAAS:
                                            return <QAASForm onAdded={onAdded} />;
                                        default:
                                            return null;
                                    }
                                })()}
                            </>
                        )}

                        {data.scenarios.length > 0 && (
                            <Button variant="ghost" className="w-full" onClick={() => { setAdd(false); setArch(undefined) }}>
                                Cancel
                            </Button>
                        )}
                    </CardContent>
                </Card>
            )}

            {/* Help tip */}
            {data.scenarios.length === 0 && (
                <Alert>
                    <Info className="h-4 w-4" />
                    <AlertDescription>
                        <strong>Tip:</strong> You can add multiple scenarios with different configurations (e.g., varying hit rates, node counts, or architectures) to compare them side-by-side in the results dashboard.
                    </AlertDescription>
                </Alert>
            )}
        </div>
    )
}

function DWAASForm({ scenario, onAdded }: { scenario?: z.infer<z.ZodObject<ZodDWAAS>>, onAdded?: () => void }) {
    const { data, setData } = React.useContext(InputContext)
    const formSchema: z.ZodObject<ZodDWAAS> = createArchitectureSchema(ArchitectureType.DWAAS) as z.ZodObject<ZodDWAAS>
    const [nodes, setNodes] = React.useState<number | undefined>()
    const [instance, setInstance] = React.useState<number | undefined>()
    const [loading, setLoading] = React.useState(false)

    // Use -1 as a marker for custom CSV uploads when no dataset is selected
    const datasetValue = data.dataset ?? (data.input_csv ? -1 : undefined)

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            ...(scenario ? scenario : {
                architecture: ArchitectureType.DWAAS,
                dataset: datasetValue,
                scheduling: {
                    policy: "sjf",
                },
            }),
        }
    })

    function onSubmit(values: z.infer<typeof formSchema>) {
        if (scenario) {
            setLoading(true)
            setData({
                ...data,
                scenarios: data.scenarios.map((s) =>
                    s === scenario ? values : s
                ),
            })

            // Simulate a delay for loading
            // So the user can be sure the data is saved
            setTimeout(() => {
                setLoading(false)
            }, 300)
        } else {
            setData({
                ...data,
                scenarios: [
                    ...data.scenarios,
                    values
                ]
            })
            // Stay on this page, just close the add form
            onAdded?.()
        }
    }

    React.useEffect(() => {
        if (instance !== undefined) {
            form.setValue("network_bandwidth", instanceTypes[instance].network_bandwidth)
            form.setValue("io_bandwidth", instanceTypes[instance].io_bandwidth)
            form.setValue("memory_bandwidth", instanceTypes[instance].memory_bandwidth)

            form.setValue("scheduling.max_io_concurrency", instanceTypes[instance].cpu_cores)
            form.setValue("scheduling.max_cpu_concurrency", instanceTypes[instance].cpu_cores)
        }

        // TODO: maybe refactor this to a more generic solution
        if (nodes !== undefined || instance !== undefined) {
            if (nodes !== undefined && instance === undefined && scenario) {
                form.setValue("cpu_cores", instanceTypes[scenario.instance].cpu_cores * nodes)
                form.setValue("total_memory_capacity_mb", (instanceTypes[scenario.instance].memory * 1024) * nodes)
            } else if (nodes === undefined && instance !== undefined && scenario) {
                form.setValue("cpu_cores", instanceTypes[instance].cpu_cores * scenario.nodes)
                form.setValue("total_memory_capacity_mb", (instanceTypes[instance].memory * 1024) * scenario.nodes)
            } else if (nodes !== undefined && instance !== undefined) {
                form.setValue("cpu_cores", instanceTypes[instance].cpu_cores * nodes)
                form.setValue("total_memory_capacity_mb", (instanceTypes[instance].memory * 1024) * nodes)
            }
        }
    }, [nodes, instance])

    return (
        <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="w-full space-y-6">
                <div className="grid grid-cols-2 gap-3">
                    <FormField
                        control={form.control}
                        name="nodes"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Number of Nodes</FormLabel>
                                <FormControl>
                                    <Input type="number" defaultValue={field.value} onChange={(e) => {
                                        field.onChange(Number.parseInt(e.target.value))
                                        setNodes(Number.parseInt(e.target.value))
                                    }} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <FormField
                        control={form.control}
                        name="hit_rate"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Hit Rate</FormLabel>
                                <FormControl>
                                    <Input type="number" defaultValue={field.value} step="any" onChange={(e) => field.onChange(Number.parseFloat(e.target.value))} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <FormField
                        control={form.control}
                        name="instance"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Instance Type</FormLabel>
                                <FormControl>
                                    <Select
                                        defaultValue={field.value?.toString()}
                                        onValueChange={(e) => {
                                            field.onChange(Number.parseInt(e))
                                            setInstance(Number.parseInt(e))
                                        }}>
                                        <SelectTrigger className="w-full">
                                            <SelectValue placeholder="Dropdown" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="0">ra3.xlplus</SelectItem>
                                            <SelectItem value="1">ra3.4xlarge</SelectItem>
                                            <SelectItem value="2">ra3.16xlarge</SelectItem>
                                            <SelectItem value="3">c5d.xlarge</SelectItem>
                                            <SelectItem value="4">c5d.2xlarge</SelectItem>
                                            <SelectItem value="5">c5d.4xlarge</SelectItem>
                                            <SelectItem value="6">ra3.xlplus (alt)</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                </div>
                {(nodes !== undefined && instance !== undefined) || scenario ?
                    <Collapsible className="space-y-6">
                        <CollapsibleTrigger asChild>
                            <div className="flex items-center justify-between gap-3 px-4 bg-secondary rounded-md">
                                <h4 className="text-sm font-semibold">
                                    Optional Parameters
                                </h4>
                                <Button variant="ghost" size="icon" className="size-8">
                                    <ChevronsUpDown />
                                    <span className="sr-only">Toggle</span>
                                </Button>
                            </div>
                        </CollapsibleTrigger>
                        <CollapsibleContent className="space-y-3">
                            <div className="grid grid-cols-3 gap-3">
                                <FormField
                                    control={form.control}
                                    name="cpu_cores"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>CPU Cores</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="network_bandwidth"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Network Bandwidth</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="io_bandwidth"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>I/O Bandwidth</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="memory_bandwidth"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Memory Bandwidth</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="total_memory_capacity_mb"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Total Memory Capacity</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                            </div>
                            <div className="grid grid-cols-3 gap-3">
                                <FormField
                                    control={form.control}
                                    name="scheduling.policy"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Scheduling policy</FormLabel>
                                            <FormControl>
                                                <Select
                                                    defaultValue={field.value}
                                                    onValueChange={field.onChange}>
                                                    <SelectTrigger className="w-full">
                                                        <SelectValue placeholder="Dropdown" />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="fcfs">First Come First Serve</SelectItem>
                                                        <SelectItem value="sjf">Shortest Job First</SelectItem>
                                                        <SelectItem value="ljf">Longest Job First</SelectItem>
                                                        <SelectItem value="multi_level">Multi Level</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="scheduling.max_io_concurrency"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Max concurrent I/O jobs</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="scheduling.max_cpu_concurrency"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Max concurrent CPU jobs</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                            </div>
                        </CollapsibleContent>
                    </Collapsible>
                    : null}
                {scenario ? (
                    <Button type="submit" variant="default">
                        {loading ? <Spinner /> : "save"}
                    </Button>
                ) : onAdded ? (
                    <Button type="submit" variant="default">
                        <Plus className="h-4 w-4 mr-2" />
                        Add Scenario
                    </Button>
                ) : <NextButton />}
            </form>
        </Form>
    )
}

function DWAASAutoscalingForm({ scenario, onAdded }: { scenario?: z.infer<z.ZodObject<ZodDWAAS>>, onAdded?: () => void }) {
    const { data, setData } = React.useContext(InputContext)
    const formSchema: z.ZodObject<ZodDWAASAutoscaling> = createArchitectureSchema(ArchitectureType.DWAAS_AUTOSCALING) as z.ZodObject<ZodDWAASAutoscaling>
    const [nodes, setNodes] = React.useState<number | undefined>()
    const [instance, setInsance] = React.useState<number | undefined>()
    const [loading, setLoading] = React.useState(false)

    // Use -1 as a marker for custom CSV uploads when no dataset is selected
    const datasetValue = data.dataset ?? (data.input_csv ? -1 : undefined)

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            ...(scenario ? scenario : {
                architecture: ArchitectureType.DWAAS_AUTOSCALING,
                dataset: datasetValue,
                use_spot_instances: false,
                scheduling: {
                    policy: "sjf",
                },
            }),
        }
    })

    function onSubmit(values: z.infer<typeof formSchema>) {
        if (scenario) {
            setLoading(true)
            setData({
                ...data,
                scenarios: data.scenarios.map((s) =>
                    s === scenario ? values : s
                ),
            })

            // Simulate a delay for loading
            // So the user can be sure the data is saved
            setTimeout(() => {
                setLoading(false)
            }, 300)
        } else {
            setData({
                ...data,
                scenarios: [
                    ...data.scenarios,
                    values
                ]
            })
            // Stay on this page, just close the add form
            onAdded?.()
        }
    }

    React.useEffect(() => {
        if (instance !== undefined) {
            form.setValue("network_bandwidth", instanceTypes[instance].network_bandwidth)
            form.setValue("io_bandwidth", instanceTypes[instance].io_bandwidth)
            form.setValue("memory_bandwidth", instanceTypes[instance].memory_bandwidth)

            form.setValue("scheduling.max_io_concurrency", instanceTypes[instance].cpu_cores)
            form.setValue("scheduling.max_cpu_concurrency", instanceTypes[instance].cpu_cores)
        }

        // TODO: maybe refactor this to a more generic solution
        if (nodes !== undefined || instance !== undefined) {
            if (nodes !== undefined && instance === undefined && scenario) {
                form.setValue("cpu_cores", instanceTypes[scenario.instance].cpu_cores * nodes)
                form.setValue("total_memory_capacity_mb", (instanceTypes[scenario.instance].memory * 1024) * nodes)
            } else if (nodes === undefined && instance !== undefined && scenario) {
                form.setValue("cpu_cores", instanceTypes[instance].cpu_cores * scenario.nodes)
                form.setValue("total_memory_capacity_mb", (instanceTypes[instance].memory * 1024) * scenario.nodes)
            } else if (nodes !== undefined && instance !== undefined) {
                form.setValue("cpu_cores", instanceTypes[instance].cpu_cores * nodes)
                form.setValue("total_memory_capacity_mb", (instanceTypes[instance].memory * 1024) * nodes)
            }
        }
    }, [nodes, instance])

    return (
        <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="w-full space-y-6">
                <div className="grid grid-cols-2 gap-3">
                    <FormField
                        control={form.control}
                        name="nodes"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Nodes</FormLabel>
                                <FormControl>
                                    <Input type="number" defaultValue={field.value} onChange={(e) => {
                                        field.onChange(Number.parseInt(e.target.value))
                                        setNodes(Number.parseInt(e.target.value))
                                    }} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <FormField
                        control={form.control}
                        name="hit_rate"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Cache Hit Rate</FormLabel>
                                <FormControl>
                                    <Input type="number" defaultValue={field.value} step="any" onChange={(e) => field.onChange(Number.parseFloat(e.target.value))} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <FormField
                        control={form.control}
                        name="cold_start"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Cold Start Delay in seconds</FormLabel>
                                <FormControl>
                                    <Input type="number" defaultValue={field.value} step="any" onChange={(e) => field.onChange(Number.parseFloat(e.target.value))} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <FormField
                        control={form.control}
                        name="instance"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Instance Type</FormLabel>
                                <FormControl>
                                    <Select
                                        defaultValue={field.value?.toString()}
                                        onValueChange={(e) => {
                                            field.onChange(Number.parseInt(e))
                                            setInsance(Number.parseInt(e))
                                        }}>
                                        <SelectTrigger className="w-full">
                                            <SelectValue placeholder="Dropdown" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="0">ra3.xlplus</SelectItem>
                                            <SelectItem value="1">ra3.4xlarge</SelectItem>
                                            <SelectItem value="2">ra3.16xlarge</SelectItem>
                                            <SelectItem value="3">c5d.xlarge</SelectItem>
                                            <SelectItem value="4">c5d.2xlarge</SelectItem>
                                            <SelectItem value="5">c5d.4xlarge</SelectItem>
                                            <SelectItem value="6">ra3.xlplus (alt)</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <FormField
                        control={form.control}
                        name="scaling.policy"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Scaling Policy</FormLabel>
                                <FormControl>
                                    <Select defaultValue={field.value} onValueChange={field.onChange}>
                                        <SelectTrigger className="w-full">
                                            <SelectValue placeholder="Dropdown" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="queue">queue</SelectItem>
                                            <SelectItem value="reactive">reactive</SelectItem>
                                            <SelectItem value="predictive">predictive</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                </div>
                {(nodes !== undefined && instance !== undefined) || scenario ?
                    <Collapsible className="space-y-6">
                        <CollapsibleTrigger asChild>
                            <div className="flex items-center justify-between gap-3 px-4 bg-secondary rounded-md">
                                <h4 className="text-sm font-semibold">
                                    Optional Parameters
                                </h4>
                                <Button variant="ghost" size="icon" className="size-8">
                                    <ChevronsUpDown />
                                    <span className="sr-only">Toggle</span>
                                </Button>
                            </div>
                        </CollapsibleTrigger>
                        <CollapsibleContent className="space-y-3">
                            <div className="grid grid-cols-3 gap-3">
                                <FormField
                                    control={form.control}
                                    name="network_bandwidth"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Network Bandwidth</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="io_bandwidth"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>I/O Bandwidth</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="memory_bandwidth"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Memory Bandwidth</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="total_memory_capacity_mb"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Total Memory Capacity</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="use_spot_instances"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Use Spot Instances (50% discount)</FormLabel>
                                            <FormControl>
                                                <Switch defaultChecked={field.value} onCheckedChange={field.onChange} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                            </div>
                            <div className="grid grid-cols-3 gap-3">
                                <FormField
                                    control={form.control}
                                    name="scheduling.policy"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Scheduling policy</FormLabel>
                                            <FormControl>
                                                <Select
                                                    defaultValue={field.value}
                                                    onValueChange={(e) => field.onChange(e)}>
                                                    <SelectTrigger className="w-full">
                                                        <SelectValue placeholder="Dropdown" />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="fcfs">First Come First Serve</SelectItem>
                                                        <SelectItem value="sjf">Shortest Job First</SelectItem>
                                                        <SelectItem value="ljf">Longest Job First</SelectItem>
                                                        <SelectItem value="multi_level">Multi Level</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="scheduling.max_io_concurrency"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Max concurrent I/O jobs</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="scheduling.max_cpu_concurrency"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Max concurrent CPU jobs</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                            </div>
                        </CollapsibleContent>
                    </Collapsible>
                    : null}
                {scenario ? (
                    <Button type="submit" variant="default">
                        {loading ? <Spinner /> : "save"}
                    </Button>
                ) : onAdded ? (
                    <Button type="submit" variant="default">
                        <Plus className="h-4 w-4 mr-2" />
                        Add Scenario
                    </Button>
                ) : <NextButton />}
            </form>
        </Form>
    )
}

function ElasticPoolForm({ scenario, onAdded }: { scenario?: z.infer<z.ZodObject<ZodDWAAS>>, onAdded?: () => void }) {
    const { data, setData } = React.useContext(InputContext)
    const formSchema: z.ZodObject<ZodElasticPool> = createArchitectureSchema(ArchitectureType.ELASTIC_POOL) as z.ZodObject<ZodElasticPool>
    const [vpu, setVPUs] = React.useState<number | undefined>()
    const [loading, setLoading] = React.useState(false)

    // Use -1 as a marker for custom CSV uploads when no dataset is selected
    const datasetValue = data.dataset ?? (data.input_csv ? -1 : undefined)

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            ...(scenario ? scenario : {
                architecture: ArchitectureType.ELASTIC_POOL,
                dataset: datasetValue,

                network_bandwidth: 10000,
                io_bandwidth: 1200,
                memory_bandwidth: 40000,
                total_memory_capacity_mb: 96000,
                scheduling: {
                    policy: "sjf",
                    max_io_concurrency: 32,
                    max_cpu_concurrency: 32,
                }
            }),
        }
    })

    function onSubmit(values: z.infer<typeof formSchema>) {
        if (scenario) {
            setLoading(true)
            setData({
                ...data,
                scenarios: data.scenarios.map((s) =>
                    s === scenario ? values : s
                ),
            })

            // Simulate a delay for loading
            // So the user can be sure the data is saved
            setTimeout(() => {
                setLoading(false)
            }, 300)
        } else {
            setData({
                ...data,
                scenarios: [
                    ...data.scenarios,
                    values
                ]
            })
            // Stay on this page, just close the add form
            onAdded?.()
        }
    }

    React.useEffect(() => {
        if (vpu !== undefined) {
            form.setValue("scheduling.max_io_concurrency", vpu * 2)
            form.setValue("scheduling.max_cpu_concurrency", vpu * 2)
        }
    }, [vpu])

    return (
        <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="w-full space-y-6">
                <div className="grid grid-cols-2 gap-3">
                    <FormField
                        control={form.control}
                        name="vpu"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Virtual Processing Units (RPUs)</FormLabel>
                                <FormControl>
                                    <Input type="number" defaultValue={field.value} onChange={(e) => {
                                        field.onChange(Number.parseInt(e.target.value))
                                        setVPUs(Number.parseInt(e.target.value))
                                    }} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <FormField
                        control={form.control}
                        name="hit_rate"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Cache Hit Rate</FormLabel>
                                <FormControl>
                                    <Input type="number" defaultValue={field.value} step="any" onChange={(e) => field.onChange(Number.parseFloat(e.target.value))} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <FormField
                        control={form.control}
                        name="cold_start"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Cold Start Delay in seconds</FormLabel>
                                <FormControl>
                                    <Input type="number" defaultValue={field.value} step="any" onChange={(e) => field.onChange(Number.parseFloat(e.target.value))} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <FormField
                        control={form.control}
                        name="scaling.policy"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Scaling Policy</FormLabel>
                                <FormControl>
                                    <Select defaultValue={field.value} onValueChange={field.onChange}>
                                        <SelectTrigger className="w-full">
                                            <SelectValue placeholder="Dropdown" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="queue">queue</SelectItem>
                                            <SelectItem value="reactive">reactive</SelectItem>
                                            <SelectItem value="predictive">predictive</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                </div>
                {vpu !== undefined || scenario ?
                    <Collapsible className="space-y-6">
                        <CollapsibleTrigger asChild>
                            <div className="flex items-center justify-between gap-3 px-4 bg-secondary rounded-md">
                                <h4 className="text-sm font-semibold">
                                    Optional Parameters
                                </h4>
                                <Button variant="ghost" size="icon" className="size-8">
                                    <ChevronsUpDown />
                                    <span className="sr-only">Toggle</span>
                                </Button>
                            </div>
                        </CollapsibleTrigger>
                        <CollapsibleContent className="space-y-3">
                            <div className="grid grid-cols-3 gap-3">
                                <FormField
                                    control={form.control}
                                    name="network_bandwidth"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Network Bandwidth</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="io_bandwidth"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>I/O Bandwidth</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="memory_bandwidth"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Memory Bandwidth</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="total_memory_capacity_mb"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Total Memory Capacity</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                            </div>
                            <div className="grid grid-cols-3 gap-3">
                                <FormField
                                    control={form.control}
                                    name="scheduling.policy"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Scheduling policy</FormLabel>
                                            <FormControl>
                                                <Select
                                                    defaultValue={field.value}
                                                    onValueChange={field.onChange}>
                                                    <SelectTrigger className="w-full">
                                                        <SelectValue placeholder="Dropdown" />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="fcfs">First Come First Serve</SelectItem>
                                                        <SelectItem value="sjf">Shortest Job First</SelectItem>
                                                        <SelectItem value="ljf">Longest Job First</SelectItem>
                                                        <SelectItem value="multi_level">Multi Level</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="scheduling.max_io_concurrency"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Max concurrent I/O jobs</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="scheduling.max_cpu_concurrency"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Max concurrent CPU jobs</FormLabel>
                                            <FormControl>
                                                <Input type="number" value={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                            </FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                            </div>
                        </CollapsibleContent>
                    </Collapsible>
                    : null}
                {scenario ? (
                    <Button type="submit" variant="default">
                        {loading ? <Spinner /> : "save"}
                    </Button>
                ) : onAdded ? (
                    <Button type="submit" variant="default">
                        <Plus className="h-4 w-4 mr-2" />
                        Add Scenario
                    </Button>
                ) : <NextButton />}
            </form>
        </Form>
    )
}

function QAASForm({ scenario, onAdded }: { scenario?: z.infer<z.ZodObject<ZodDWAAS>>, onAdded?: () => void }) {
    const { data, setData } = React.useContext(InputContext)
    const formSchema: z.ZodObject<ZodQAAS> = createArchitectureSchema(ArchitectureType.QAAS) as z.ZodObject<ZodQAAS>
    const [loading, setLoading] = React.useState(false)

    // Use -1 as a marker for custom CSV uploads when no dataset is selected
    const datasetValue = data.dataset ?? (data.input_csv ? -1 : undefined)

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            ...(scenario ? scenario : {
                architecture: ArchitectureType.QAAS,
                dataset: datasetValue,
                network_bandwidth: 10,
            }),
        }
    })

    function onSubmit(values: z.infer<typeof formSchema>) {
        if (scenario) {
            setLoading(true)
            setData({
                ...data,
                scenarios: data.scenarios.map((s) =>
                    s === scenario ? values : s
                ),
            })

            // Simulate a delay for loading
            // So the user can be sure the data is saved
            setTimeout(() => {
                setLoading(false)
            }, 300)
        } else {
            setData({
                ...data,
                scenarios: [
                    ...data.scenarios,
                    values
                ]
            })
            // Stay on this page, just close the add form
            onAdded?.()
        }
    }

    return (
        <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                <FormField
                    control={form.control}
                    name="network_bandwidth"
                    render={({ field }) => (
                        <FormItem>
                            <FormLabel>Network bandwidth</FormLabel>
                            <FormControl>
                                <Input type="number" defaultValue={field.value} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                            </FormControl>
                            <FormMessage />
                        </FormItem>
                    )}
                />
                {scenario ? (
                    <Button type="submit" variant="default">
                        {loading ? <Spinner /> : "save"}
                    </Button>
                ) : onAdded ? (
                    <Button type="submit" variant="default">
                        <Plus className="h-4 w-4 mr-2" />
                        Add Scenario
                    </Button>
                ) : <NextButton />}
            </form>
        </Form>
    )
}
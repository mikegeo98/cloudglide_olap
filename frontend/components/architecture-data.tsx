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
import { ChevronsUpDown, Plus } from "lucide-react";
import InteractiveCluster from "./interactive-cluster";
import ClusterView from "./cluster-view";

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

    return (
        <div className="flex flex-col items-center gap-8 w-[1000px] px-6 max-h-full overflow-y-auto">
            <h1>Architecture</h1>
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

                return (
                    <div key={index} className="flex w-full gap-3">
                        <Collapsible className="w-full space-y-6">
                            <CollapsibleTrigger asChild>
                                <div className="flex items-center justify-between gap-3 mb-0 rounded-md">
                                    <h4 className="text-sm font-semibold">
                                        {`Scenario ${index + 1}: ${scenario.architecture}`}
                                    </h4>
                                    <Button variant="ghost" size="icon" className="size-8">
                                        <ChevronsUpDown />
                                        <span className="sr-only">Toggle</span>
                                    </Button>
                                </div>
                            </CollapsibleTrigger>
                            <CollapsibleContent className="mt-3 space-y-3">
                                {form}
                            </CollapsibleContent>
                        </Collapsible>
                        <Button variant="default" onClick={() => duplicateArchitecture(scenario)}>
                            Duplicate
                        </Button>
                    </div>
                )
            })}
            {data.scenarios.length > 0 ? (
                <>
                    <Separator />
                    <div className="w-full flex justify-between">
                        <Button variant="default" disabled={add} onClick={() => setAdd(true)}>
                            <Plus />
                        </Button>
                        <NextButton rightOnClick={() => setStage(stage + 1)} />
                    </div>
                </>
            ) : null}
            {add || data.scenarios.length === 0 ? (
                <>
                    <RadioGroup className="flex w-full justify-between" onValueChange={(value) => setArch(value)}>
                        {Object.values(ArchitectureType).map((archType) => (
                            <div key={archType} className="flex items-center gap-3">
                                <RadioGroupItem value={archType} id={archType} />
                                <Label htmlFor={archType}>{archType}</Label>
                            </div>
                        ))}
                    </RadioGroup>
                    {arch ? (
                        (() => {
                            switch (arch) {
                                case ArchitectureType.DWAAS:
                                    return <DWAASForm />;
                                case ArchitectureType.DWAAS_AUTOSCALING:
                                    return <DWAASAutoscalingForm />;
                                case ArchitectureType.ELASTIC_POOL:
                                    return <ElasticPoolForm />;
                                case ArchitectureType.QAAS:
                                    return <QAASForm />;
                                default:
                                    return null;
                            }
                        })()
                    ) : null}
                </>
            ) : null}
        </div>
    )
}

function DWAASForm({ scenario }: { scenario?: z.infer<z.ZodObject<ZodDWAAS>> }) {
    const { stage, setStage, data, setData } = React.useContext(InputContext)
    const formSchema: z.ZodObject<ZodDWAAS> = createArchitectureSchema(ArchitectureType.DWAAS) as z.ZodObject<ZodDWAAS>
    const [nodes, setNodes] = React.useState<number | undefined>()
    const [instance, setInstance] = React.useState<number | undefined>()
    const [loading, setLoading] = React.useState(false)

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            ...(scenario ? scenario : {
                architecture: ArchitectureType.DWAAS,
                dataset: data.dataset,
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
            setStage(stage + 1)
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
        <div className="flex w-full justify-center gap-6">
            <div className="flex-1 max-w-1/2">
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
                                            <Input type="number" defaultValue={field.value ?? ""} onChange={(e) => {
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
                                    <div className="grid grid-cols-2 gap-3">
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
                                    <div className="grid grid-cols-2 gap-3">
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
                        ) : <NextButton />}
                    </form>
                </Form>
            </div>
            {nodes !== undefined && instance !== undefined
                ? <div className="flex-1">
                    <ClusterView nodes={nodes} instanceType={instanceTypes[instance]} />
                </div>
                : null}
        </div>
    )
}

function DWAASAutoscalingForm({ scenario }: { scenario?: z.infer<z.ZodObject<ZodDWAAS>> }) {
    const { stage, setStage, data, setData } = React.useContext(InputContext)
    const formSchema: z.ZodObject<ZodDWAASAutoscaling> = createArchitectureSchema(ArchitectureType.DWAAS_AUTOSCALING) as z.ZodObject<ZodDWAASAutoscaling>
    const [nodes, setNodes] = React.useState<number | undefined>()
    const [instance, setInsance] = React.useState<number | undefined>()
    const [loading, setLoading] = React.useState(false)

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            ...(scenario ? scenario : {
                architecture: ArchitectureType.DWAAS_AUTOSCALING,
                dataset: data.dataset,
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
            setStage(stage + 1)
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
        <div className="flex w-full justify-center gap-6">
            <div className="flex-1 max-w-1/2">
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
                                    <div className="grid grid-cols-2 gap-3">
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
                                    <div className="grid grid-cols-2 gap-3">
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
                        ) : <NextButton />}
                    </form>
                </Form>
            </div>
            {nodes !== undefined && instance !== undefined
                ? <div className="flex-1">
                    <ClusterView nodes={nodes} instanceType={instanceTypes[instance]} />
                </div>
                : null}
        </div>
    )
}

function ElasticPoolForm({ scenario }: { scenario?: z.infer<z.ZodObject<ZodDWAAS>> }) {
    const { stage, setStage, data, setData } = React.useContext(InputContext)
    const formSchema: z.ZodObject<ZodElasticPool> = createArchitectureSchema(ArchitectureType.ELASTIC_POOL) as z.ZodObject<ZodElasticPool>
    const [vpu, setVPUs] = React.useState<number | undefined>()
    const [loading, setLoading] = React.useState(false)

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            ...(scenario ? scenario : {
                architecture: ArchitectureType.ELASTIC_POOL,
                dataset: data.dataset,

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
            setStage(stage + 1)
        }
    }

    React.useEffect(() => {
        if (vpu !== undefined) {
            form.setValue("scheduling.max_io_concurrency", vpu * 2)
            form.setValue("scheduling.max_cpu_concurrency", vpu * 2)
        }
    }, [vpu])

    return (
        <div className="flex w-full justify-center gap-6">
            <div className="flex-1 max-w-1/2">
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
                                    <div className="grid grid-cols-2 gap-3">
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
                                    <div className="grid grid-cols-2 gap-3">
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
                        ) : <NextButton />}
                    </form>
                </Form>
            </div>
        </div>
    )
}

function QAASForm({ scenario }: { scenario?: z.infer<z.ZodObject<ZodDWAAS>> }) {
    const { stage, setStage, data, setData } = React.useContext(InputContext)
    const formSchema: z.ZodObject<ZodQAAS> = createArchitectureSchema(ArchitectureType.QAAS) as z.ZodObject<ZodQAAS>
    const [loading, setLoading] = React.useState(false)

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            ...(scenario ? scenario : {
                architecture: ArchitectureType.QAAS,
                dataset: data.dataset,
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
            setStage(stage + 1)
        }
    }

    return (
        <div className="flex w-full justify-center gap-6">
            <div className="flex-1 max-w-1/2">
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
                        ) : <NextButton />}
                    </form>
                </Form>
            </div>
        </div>
    )
}
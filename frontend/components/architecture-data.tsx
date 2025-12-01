"use client"

import NextButton from "@/components/next-btn";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
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
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

import React from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";

const formSchema = z.object({
    architecture: z.object({
        name: z.string(),
        dwaas_options: z.object({
            name: z.string(),
            spot_discount: z.float32(),
            interruption_freq: z.float32(),
        }).optional(),
        elastic_pool_options: z.object({
            name: z.string(),
            vpus: z.number(),
            io_bandwidth: z.number(),
            memory_bandwidth: z.number(),
            cold_start: z.number(),
            warmup_rate: z.float32(),
            cost: z.float32(),
        }).optional(),
    }),
    nodes: z.number(),
    instanceType: z.string(),
    hitRate: z.number(),
    schedulingPolicy: z.string(),
    maxConcurrency: z.number().optional(),
})

export default function ArchitectureData() {
    const { stage, increaseStage, data } = React.useContext(InputContext)
    const [isDwaas, setDwaas] = React.useState(false)
    const [isEditMaxConc, setEditMaxConc] = React.useState(false)

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        // values: {
        //     architecture: data.architecture,
        //     dwaas_options: data.dwaas_options,
        //     nodes: data.nodes,
        //     instanceType: data.instanceType,
        //     hitRate: data.hitRate,
        //     schedulingPolicy: data.schedulingPolicy
        // }
    })

    function onSubmit(values: z.infer<typeof formSchema>) {
        console.log(values)
        increaseStage(stage + 1)
    }

    return (
        <>
            <h1>Architecture</h1>
            <br />
            <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
                    <div className="grid grid-cols-2 gap-3">
                        <FormField
                            control={form.control}
                            name="nodes"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Number of Nodes</FormLabel>
                                    <FormControl>
                                        <Input type="number" onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                    </FormControl>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                        <FormField
                            control={form.control}
                            name="hitRate"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Hit Rate</FormLabel>
                                    <FormControl>
                                        <Input type="number" step="any" onChange={(e) => field.onChange(Number.parseFloat(e.target.value))} />
                                    </FormControl>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                        <FormField
                            control={form.control}
                            name="instanceType"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Instance Type</FormLabel>
                                    <FormControl>
                                        <Select onValueChange={field.onChange}>
                                            <SelectTrigger className="w-full">
                                                <SelectValue placeholder="Dropdown" />
                                            </SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="ra3.xlplus">ra3.xlplus</SelectItem>
                                                <SelectItem value="ra3.4xlarge">ra3.4xlarge</SelectItem>
                                                <SelectItem value="ra3.16xlarge">ra3.16xlarge</SelectItem>
                                                <SelectItem value="c5d.xlarge">c5d.xlarge</SelectItem>
                                                <SelectItem value="c5d.2xlarge">c5d.2xlarge</SelectItem>
                                                <SelectItem value="c5d.4xlarge">c5d.4xlarge</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </FormControl>
                                    <FormDescription>
                                        <Button variant={"ghost"}>
                                            <span className="text-blue-500 underline">Create new instance</span>
                                        </Button>
                                    </FormDescription>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                        <FormField
                            control={form.control}
                            name="schedulingPolicy"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Scheduling Policy</FormLabel>
                                    <FormControl>
                                        <Select onValueChange={(e) => {
                                            if (e !== "mlq") {
                                                setEditMaxConc(true)
                                            } else {
                                                setEditMaxConc(false)
                                            }
                                            field.onChange(e)
                                        }}>
                                            <SelectTrigger className="w-full">
                                                <SelectValue placeholder="Dropdown" />
                                            </SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="fcfs">First-Come-First-Serve</SelectItem>
                                                <SelectItem value="sjf">Shortest-Job-First</SelectItem>
                                                <SelectItem value="ljf">Longest-Job-First</SelectItem>
                                                <SelectItem value="mlq">Multi-Level Queue</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </FormControl>
                                    <FormDescription>
                                        <Button variant={"ghost"}>
                                            <span className="text-blue-500 underline">Create new policy</span>
                                        </Button>
                                    </FormDescription>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                    </div>
                    {isEditMaxConc ? (
                        <>
                            <Separator />
                            <h4>Scheduling Policy Configuration</h4>
                            <FormField
                                control={form.control}
                                name="maxConcurrency"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Maximum allowed query concurrency</FormLabel>
                                        <FormControl>
                                            <Input type="number" onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                        </FormControl>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                        </>
                    ) : null}
                    <Separator />
                    <div className="grid grid-cols-2">
                        <FormField
                            control={form.control}
                            name="architecture.name"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Architecture</FormLabel>
                                    <FormControl>
                                        <RadioGroup onValueChange={(e) => {
                                            if (e === "dwaas") {
                                                setDwaas(true)
                                            } else {
                                                setDwaas(false)
                                            }
                                            field.onChange(e)
                                        }}>
                                            <div className="flex items-center space-x-2">
                                                <RadioGroupItem value="dwaas" id="dwaas" />
                                                <Label htmlFor="dwaas">DWaaS</Label>
                                            </div>
                                            <div className="flex items-center space-x-2">
                                                <RadioGroupItem value="elastic_pool" id="elastic_pool" />
                                                <Label htmlFor="elastic_pool">Elastic Pool</Label>
                                            </div>
                                            <div className="flex items-center space-x-2">
                                                <RadioGroupItem value="qaas" id="qaas" />
                                                <Label htmlFor="qaas">QaaS</Label>
                                            </div>
                                        </RadioGroup>
                                    </FormControl>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                        {isDwaas ? (
                            <FormField
                                control={form.control}
                                name="architecture.dwaas_options.name"
                                render={({ field }) => (
                                    <FormItem className="h-fit">
                                        <FormLabel className="h-fit">DWaaS Options</FormLabel>
                                        <FormControl>
                                            <RadioGroup onValueChange={field.onChange}>
                                                <div className="flex items-center space-x-2">
                                                    <RadioGroupItem value="auto_dwaas" id="auto_dwaas" />
                                                    <Label htmlFor="auto_dwaas">Autoscaling DWaaS</Label>
                                                </div>
                                                <div className="flex items-center space-x-2">
                                                    <RadioGroupItem value="spot" id="spot" />
                                                    <Label htmlFor="spot">Spot instances</Label>
                                                </div>
                                            </RadioGroup>
                                        </FormControl>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                        ) : null}
                    </div>
                    <NextButton />
                </form>
            </Form>
            <br />
        </>
    )
}
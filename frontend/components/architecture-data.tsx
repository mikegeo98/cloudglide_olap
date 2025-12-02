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

import React from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { instanceTypes } from "@/lib/config";

const formSchema = z.object({
    architecture: z.string(),
    nodes: z.number(),
    hit_rate: z.number(),
    // arrival_rate: z.number(),
    // cold_start: z.number(),
    instance: z.object({
        index: z.number(),
        vpu: z.number(),
        memory: z.number(),
        network_bandwidth: z.number(),
        io_bandwidth: z.number(),
        memory_bandwidth: z.number(),
    }),
    scheduling: z.object({
        policy: z.string(),
        max_io_concurrency: z.number(),
        max_cpu_concurrency: z.number(),
    }),
})

export default function ArchitectureData() {
    const { stage, increaseStage, data, setData } = React.useContext(InputContext)

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            architecture: data.architecture,
            nodes: data.nodes,
            hit_rate: data.hit_rate,
            instance: {
                index: data.instance_index,
                vpu: data.vpu,
                memory: data.memory,
                network_bandwidth: data.network_bandwidth,
                io_bandwidth: data.io_bandwidth,
                memory_bandwidth: data.memory_bandwidth,
            },
            scheduling: data.scheduling,
        },
    })

    function onSubmit(values: z.infer<typeof formSchema>) {
        setData({
            ...data,
            architecture: values.architecture,
            nodes: values.nodes,
            hit_rate: values.hit_rate,
            instance_index: values.instance.index,
            vpu: values.instance.vpu,
            memory: values.instance.memory,
            network_bandwidth: values.instance.network_bandwidth,
            io_bandwidth: values.instance.io_bandwidth,
            memory_bandwidth: values.instance.memory_bandwidth,
            scheduling: {
                policy: values.scheduling.policy,
                max_io_concurrency: values.scheduling.max_io_concurrency,
                max_cpu_concurrency: values.scheduling.max_cpu_concurrency,
            }
        })
        increaseStage(stage + 1)
    }

    return (
        <div className="flex flex-col items-center gap-8">
            <h1>Architecture</h1>
            <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                    <div className="grid grid-cols-2">
                        <FormField
                            control={form.control}
                            name="architecture"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Architecture</FormLabel>
                                    <FormControl>
                                        <RadioGroup defaultValue={field.value ? field.value : undefined} onValueChange={(e) => {
                                            field.onChange(e)
                                        }}>
                                            <div className="flex items-center space-x-2">
                                                <RadioGroupItem value="DWAAS" id="dwaas" />
                                                <Label htmlFor="dwaas">DWaaS</Label>
                                            </div>
                                            <div className="flex items-center space-x-2">
                                                <RadioGroupItem value="ELASTIC_POOL" id="elastic_pool" />
                                                <Label htmlFor="elastic_pool">Elastic Pool</Label>
                                            </div>
                                            <div className="flex items-center space-x-2">
                                                <RadioGroupItem value="QAAS" id="qaas" />
                                                <Label htmlFor="qaas">QaaS</Label>
                                            </div>
                                        </RadioGroup>
                                    </FormControl>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                        <FormField
                            control={form.control}
                            name="nodes"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Number of Nodes</FormLabel>
                                    <FormControl>
                                        <Input type="number" defaultValue={field.value ? field.value : undefined} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
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
                                        <Input type="number" defaultValue={field.value ? field.value : undefined} step="any" onChange={(e) => field.onChange(Number.parseFloat(e.target.value))} />
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
                                        <Select defaultValue={field.value && field.value.index ? field.value?.index.toString() : undefined} onValueChange={(e) => {
                                            const instance = instanceTypes[Number.parseInt(e)]
                                            if (instance) {
                                                field.onChange({
                                                    index: Number.parseInt(e),
                                                    vpu: instance.cpu_cores,
                                                    memory: instance.memory,
                                                    network_bandwidth: instance.network_bandwidth,
                                                    io_bandwidth: instance.io_bandwidth,
                                                    memory_bandwidth: instance.memory_bandwidth,
                                                })
                                            }
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
                                                <SelectItem value="6">ra3.xlplus</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </FormControl>
                                    <FormDescription>
                                        <Button type="button" variant={"ghost"}>
                                            <span className="text-blue-500 underline">Custom instance</span>
                                        </Button>
                                    </FormDescription>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                        <FormField
                            control={form.control}
                            name="scheduling"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Scheduling Policy</FormLabel>
                                    <FormControl>
                                        <Select defaultValue={field.value?.policy} onValueChange={(e) => {
                                            field.onChange({
                                                policy: e,
                                                max_io_concurrency: 0,
                                                max_cpu_concurrency: 0,
                                            })
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
                                        <Button type="button" variant={"ghost"}>
                                            <span className="text-blue-500 underline">Custom policy</span>
                                        </Button>
                                    </FormDescription>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                    </div>
                    <NextButton />
                </form>
            </Form>
        </div>
    )
}
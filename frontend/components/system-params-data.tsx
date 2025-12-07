import { Input } from "@/components/ui/input";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import NextButton from "@/components/next-btn";
import { InputContext } from "@/components/provider";

import React from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";

const formSchema = z.object({
    parallelizable_portion: z.float32(),
    materialization_fraction: z.float32(),
    default_estimator: z.string().min(1),
})

export default function SystemParametersData() {
    const { stage, setStage, data, setData } = React.useContext(InputContext)

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            parallelizable_portion: data.defaults.simulation.parallelizable_portion,
            materialization_fraction: data.defaults.simulation.materialization_fraction,
            default_estimator: data.defaults.simulation.default_estimator,
        }
    })

    function onSubmit(values: z.infer<typeof formSchema>) {
        setData({
            ...data,
            defaults: {
                ...data.defaults,
                simulation: {
                    ...data.defaults.simulation,
                    parallelizable_portion: values.parallelizable_portion,
                    materialization_fraction: values.materialization_fraction,
                    default_estimator: values.default_estimator,
                },
            },
        })
        setStage(stage + 1)
    }

    return (
        <div className="flex flex-col items-center gap-8">
            <h1>System Parameters (Optional)</h1>
            <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="w-[300px] space-y-6">
                    <FormField
                        control={form.control}
                        name="parallelizable_portion"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Parallizable Fraction Average</FormLabel>
                                <FormControl>
                                    <Input type="number" defaultValue={field.value} step="any" onChange={(e) => field.onChange(Number.parseFloat(e.target.value))} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <FormField
                        control={form.control}
                        name="materialization_fraction"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Materialization Fraction Average</FormLabel>
                                <FormControl>
                                    <Input type="number" defaultValue={field.value} step="any" onChange={(e) => field.onChange(Number.parseFloat(e.target.value))} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <FormField
                        control={form.control}
                        name="default_estimator"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Default Estimator</FormLabel>
                                <FormControl>
                                    <Input type="text" defaultValue={field.value} onChange={field.onChange} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <NextButton />
                </form >
            </Form >
        </div>
    )
}
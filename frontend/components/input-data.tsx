"use client"

import { Input } from "@/components/ui/input";
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
    Tooltip,
    TooltipContent,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import NextButton from "@/components/next-btn";
import { Button } from "@/components/ui/button";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { InputContext } from "@/components/provider";

import React from "react";
import { HelpCircle, Trash2 } from "lucide-react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";

const formSchema = z.object({
    file: z.file().optional(),
    dataset: z.string().optional(),
})
    .superRefine((data, ctx) => {
        if (data.file === undefined && (data.dataset === undefined || data.dataset.length === 0)) {
            ctx.addIssue({
                code: "custom",
                message: "Either upload a CSV file or use an alternative",
                path: ["file"]
            })
        }
    })

export default function InputData() {
    const { stage, increaseStage, data, setData } = React.useContext(InputContext)
    const [filename, setFilename] = React.useState<string | undefined>()

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        values: {
            file: data.input_csv,
            dataset: data.default_csv?.name
        }
    })

    async function onSubmit(values: z.infer<typeof formSchema>) {
        if (values.dataset) {
            const response = await fetch(values.dataset)
            const resData = await response.blob()

            setData({ ...data, input_csv: values.file ? values.file : undefined, default_csv: new File([resData], values.dataset, { type: ".csv" }) })
        } else {
            setData({ ...data, input_csv: values.file ? values.file : undefined })
        }

        increaseStage(stage + 2)
    }

    React.useEffect(() => {
        setFilename(form.getValues("file")?.name)
    }, [form])

    return (
        <div className="flex flex-col items-center gap-8">
            <h1>Input Data</h1>
            <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                    <FormField
                        control={form.control}
                        name="file"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>
                                    Upload CSV
                                    <Tooltip>
                                        <TooltipTrigger>
                                            <HelpCircle size={20} className="text-muted-foreground" />
                                        </TooltipTrigger>
                                        <TooltipContent className="p-3">
                                            <h4>Required columns:</h4>
                                            <ul>
                                                <li className="text-sm"><code>start</code> - query arrival time</li>
                                                <li className="text-sm"><code>cpu_time</code> - CPU demand per query</li>
                                                <li className="text-sm"><code>data_scanned</code> - scanned data volume per query</li>
                                            </ul>
                                            <h4>Optional columns:</h4>
                                            <ul>
                                                <li className="text-sm"><code>shuffle</code> - shuffle volume (if applicable)</li>
                                            </ul>
                                        </TooltipContent>
                                    </Tooltip>
                                </FormLabel>
                                <FormControl>
                                    <div className="flex gap-3">
                                        {filename ? (
                                            <Button type="button" variant={"default"} className="w-[280px]">
                                                {filename}
                                            </Button>
                                        ) : (
                                            <Input id="csv" type="file" accept=".csv" className="w-[280px] bg-primary text-muted-foreground file:text-background" onChange={
                                                (e) => {
                                                    if (e.target.files && e.target.files.length > 0) {
                                                        setFilename(e.target.files[0].name)
                                                        field.onChange(e.target.files[0])
                                                    }
                                                }} />
                                        )}
                                        <Button type="reset" variant={"default"} onClick={() => {
                                            setFilename(undefined)
                                            field.onChange(undefined)
                                        }}>
                                            <Trash2 />
                                        </Button>
                                    </div>
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <FormField
                        control={form.control}
                        name="dataset"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>alternatively, select default trace</FormLabel>
                                <FormControl>
                                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                                        <SelectTrigger className="w-[180px]">
                                            <SelectValue placeholder="Dropdown" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectGroup>
                                                <SelectLabel>Datasets</SelectLabel>
                                                <SelectItem value="/concurrency.csv">Concurrency</SelectItem>
                                                <SelectItem value="/tenk.csv">Tenk</SelectItem>
                                                <SelectItem value="/tpch_all_runs.csv">TPC_H all runs</SelectItem>
                                            </SelectGroup>
                                        </SelectContent>
                                    </Select>
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <NextButton />
                </form>
            </Form >
        </div>
    )
}
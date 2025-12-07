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
import { inputSchema } from "@/lib/zod-schemas";

const formSchema = inputSchema

export default function InputData() {
    const { stage, setStage, data, setData } = React.useContext(InputContext)
    const [filename, setFilename] = React.useState<string | undefined>()

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            file: data.input_csv,
            dataset: data.dataset
        }
    })

    async function onSubmit(values: z.infer<typeof formSchema>) {
        if (values.file) {
            setData({ ...data, input_csv: values.file ? values.file : undefined })
        } else {
            setData({ ...data, dataset: values.dataset ? values.dataset : undefined })
        }

        setStage(stage + 1)
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
                                    <Select onValueChange={(e) => field.onChange(Number.parseInt(e))} defaultValue={field.value ? field.value + "" : undefined}>
                                        <SelectTrigger className="w-[180px]">
                                            <SelectValue placeholder="Dropdown" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectGroup>
                                                <SelectLabel>Datasets</SelectLabel>
                                                <SelectItem value="998">Concurrency</SelectItem>
                                                <SelectItem value="22">Tenk</SelectItem>
                                                <SelectItem value="999">TPC_H all runs</SelectItem>
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
import { Input } from "@/components/ui/input";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import NextButton from "@/components/next-btn";
import { InputContext } from "@/components/provider";
import { Separator } from "@/components/ui/separator";

import React from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { sysParamsSchema } from "@/lib/zod-schemas";
import { defaultConfig } from "@/lib/config";
import { Switch } from "./ui/switch";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "./ui/collapsible";
import { Button } from "./ui/button";
import { ChevronsUpDown } from "lucide-react";

export default function SystemParametersData() {
    const { stage, setStage, data, setData } = React.useContext(InputContext)

    const form = useForm<z.infer<typeof sysParamsSchema>>({
        resolver: zodResolver(sysParamsSchema),
        defaultValues: data.defaults
    })

    function onSubmit(values: z.infer<typeof sysParamsSchema>) {
        setData({
            ...data,
            defaults: {
                ...values,
            },
        })
        setStage(stage + 1)
    }

    return (
        <div className="flex flex-col items-center gap-8 w-[600px] px-6 max-h-dvh overflow-y-auto">
            <h1>System Parameters (Optional)</h1>
            <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="w-full space-y-6">
                    {Object.keys(defaultConfig).map((section) => {
                        return (
                            <React.Fragment key={section}>
                                <Collapsible className="space-y-6">
                                    <CollapsibleTrigger asChild>
                                        <div className="flex items-center justify-between gap-3 px-4 bg-secondary rounded-md">
                                            <h4 className="text-sm font-semibold">
                                                <span className="uppercase">{section}</span> defaults
                                            </h4>
                                            <Button variant="ghost" size="icon" className="size-8">
                                                <ChevronsUpDown />
                                                <span className="sr-only">Toggle</span>
                                            </Button>
                                        </div>
                                    </CollapsibleTrigger>
                                    <CollapsibleContent className="space-y-3">
                                        <div className="grid grid-cols-2 gap-6">
                                            {Object.keys(defaultConfig[section as keyof typeof defaultConfig]).map((key) => {
                                                const value = defaultConfig[section as keyof typeof defaultConfig][key as keyof typeof defaultConfig[keyof typeof defaultConfig]];
                                                return (
                                                    <FormField
                                                        key={key}
                                                        control={form.control}
                                                        name={(section + "." + key) as keyof z.infer<typeof sysParamsSchema>}
                                                        render={({ field }) => (
                                                            <FormItem>
                                                                <FormLabel>{key}</FormLabel>
                                                                <FormControl>
                                                                    {(typeof value === "number")
                                                                        ? String(value).includes(".")
                                                                            ? <Input type="number" defaultValue={Number(field.value)} step="any" onChange={(e) => field.onChange(Number.parseFloat(e.target.value))} />
                                                                            : <Input type="number" defaultValue={Number(field.value)} onChange={(e) => field.onChange(Number.parseInt(e.target.value))} />
                                                                        : (typeof value === "string")
                                                                            ? <Input type="text" defaultValue={String(field.value)} onChange={(e) => field.onChange(e.target.value)} />
                                                                            : (typeof value === "boolean")
                                                                                ? <Switch defaultChecked={Boolean(field.value)} onCheckedChange={field.onChange} />
                                                                                : null
                                                                    }
                                                                </FormControl>
                                                                <FormMessage />
                                                            </FormItem>
                                                        )}
                                                    />
                                                )
                                            })}
                                        </div>
                                    </CollapsibleContent>
                                </Collapsible>
                                <Separator />
                            </React.Fragment>
                        )
                    })}
                    <NextButton />
                </form >
            </Form >
        </div>
    )
}
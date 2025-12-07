"use client"

import { DefaultConfig, defaultConfig } from "@/lib/config";
import React from "react";

export type InputData = {
    // step 1
    input_csv?: File,
    dataset?: number,

    // step 2
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    scenarios: any[],

    // step 3
    defaults: DefaultConfig,
}

export const InputContext = React.createContext({
    stage: 0,
    setStage: (n: number) => { },
    data: {} as InputData,
    setData: (d: InputData) => { },
    dataToJson: () => ("" as string),
})

export default function Provider({
    children,
}: {
    children: React.ReactNode
}) {
    const [stage, setStage] = React.useState<number>(0)
    const [data, setData] = React.useState<InputData>({
        defaults: defaultConfig,
        scenarios: [],
    })

    function dataToJson() {
        const j = {
            defaults: data.defaults,
            scenarios: data.scenarios.map((scenario, index) => {
                scenario.name = "scenario_" + index
                return scenario
            }),
        }
        return JSON.stringify(j, null, "\t")
    }

    return (
        <InputContext.Provider value={{ stage, setStage, data, setData, dataToJson }}>
            {children}
        </InputContext.Provider>
    )
}
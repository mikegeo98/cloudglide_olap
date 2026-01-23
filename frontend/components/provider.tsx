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
        const nameCounts: Record<string, number> = {}

        const j = {
            defaults: data.defaults,
            scenarios: data.scenarios.map((scenario) => {
                let baseName: string
                switch (scenario.architecture) {
                    case 'DWAAS':
                        baseName = `dwaas_${scenario.nodes}n`
                        break
                    case 'DWAAS_AUTOSCALING':
                        baseName = `dwaas_as_${scenario.nodes}n`
                        break
                    case 'ELASTIC_POOL':
                        baseName = `ep_${scenario.vpu}rpu`
                        break
                    case 'QAAS':
                        baseName = 'qaas'
                        break
                    default:
                        baseName = `scenario`
                }

                if (nameCounts[baseName] !== undefined) {
                    nameCounts[baseName]++
                    scenario.name = `${baseName}_${nameCounts[baseName]}`
                } else {
                    nameCounts[baseName] = 1
                    scenario.name = baseName
                }

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
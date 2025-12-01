"use client"

import React from "react";

export type InputData = {
    // step 1
    input_csv?: File,
    default_csv?: File,

    // step 2
    nodes: number,
    instanceType: string,
    hitRate: number,
    schedulingPolicy: string,
    max_concurrency: number,
    architecture: string,
    dwaas_options?: string,

    // step 3
    parallelizable_portion: number,
    materialization_fraction: number,
    default_estimator: string,
}

export const InputContext = React.createContext({
    stage: 0,
    increaseStage: (n: number) => { },
    data: {} as InputData,
    setData: (d: InputData) => { },
})

export default function Provider({
    children,
}: {
    children: React.ReactNode
}) {
    const [stage, increaseStage] = React.useState<number>(0)
    const [data, setData] = React.useState<InputData>({
        // step 2
        architecture: "",
        nodes: 0,
        instanceType: "",
        hitRate: 0,
        schedulingPolicy: "",
        max_concurrency: 0,

        // step 3
        parallelizable_portion: 0.9,
        materialization_fraction: 0.25,
        default_estimator: "pm",
    })

    return (
        <InputContext.Provider value={{ stage, increaseStage, data, setData }}>
            {children}
        </InputContext.Provider>
    )
}
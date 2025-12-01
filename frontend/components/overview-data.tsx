import { Button } from "@/components/ui/button";
import { InputContext } from "@/components/provider";
import { Spinner } from "@/components/ui/spinner"

import React from "react";
import { ArrowLeft } from "lucide-react";
import runSimulation from "@/lib/exec_shell_cmd";
import { useRouter } from "next/navigation";

export default function OverviewData() {
    const router = useRouter()
    const { stage, increaseStage, data } = React.useContext(InputContext)
    const [loading, setLoading] = React.useState(false)

    function handleRunSimulation() {
        setLoading(true)
        runSimulation(data)
            .then(() => {
                router.push("/output")
            })
    }

    return (
        <>
            <h1>Overview</h1>
            <br />
            <p>{JSON.stringify(data)}</p>
            <br />
            <div className="flex gap-3">
                <Button variant="outline" size="icon" className="rounded-full" onClick={() => increaseStage(stage - 1)}>
                    <ArrowLeft />
                </Button>
                <Button variant="outline" className="bg-foreground text-background w-[180px]" onClick={handleRunSimulation}>
                    {loading ? <Spinner /> : "Run Simulation"}
                </Button>
            </div>
        </>
    )
}
import { Button } from "@/components/ui/button";
import { InputContext } from "@/components/provider";

import React from "react";
import { ArrowLeft, ArrowRight } from "lucide-react";

export default function NextButton() {
    const { stage, increaseStage } = React.useContext(InputContext)

    const switchStage = (n: -1 | 1) => {
        if (stage === 0 && n === -1) return
        
        increaseStage(stage + n)
    }

    return (
        <div className="flex gap-3">
            <Button variant="outline" disabled={stage === 0} size="icon" className="rounded-full" onClick={() => switchStage(-1)}>
                <ArrowLeft />
            </Button>
            <Button type="submit" variant="outline" size="icon" className="bg-primary text-background rounded-full">
                <ArrowRight />
            </Button>
        </div>
    )
}
import { Button } from "@/components/ui/button";
import { InputContext } from "@/components/provider";

import React from "react";
import { ArrowLeft, ArrowRight } from "lucide-react";

export default function NextButton({ rightOnClick }: { rightOnClick?: () => void }) {
    const { stage, setStage } = React.useContext(InputContext)

    const switchStage = (n: -1 | 1) => {
        if (stage === 0 && n === -1) return

        setStage(stage + n)
    }

    return (
        <div className="flex gap-3">
            <Button variant="outline" disabled={stage === 0} size="icon" className="rounded-full" onClick={() => switchStage(-1)}>
                <ArrowLeft />
            </Button>
            <Button type="submit" variant="default" size="icon" className="text-background rounded-full" onClick={rightOnClick}>
                <ArrowRight />
            </Button>
        </div>
    )
}
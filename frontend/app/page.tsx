"use client"

import { InputContext } from "@/components/provider";
import InputData from "@/components/input-data";
import ArchitectureData from "@/components/architecture-data";
import SystemParametersData from "@/components/system-params-data";
import OverviewData from "@/components/overview-data";

import React from "react";
import { cn } from "@/lib/utils";
import { notFound } from "next/navigation";
import { Check } from "lucide-react";

export default function Page() {
  const { stage } = React.useContext(InputContext)

  return (
    <div className="flex flex-col w-full min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <div className="fixed top-0 flex w-full max-w-7xl justify-evenly items-center py-4 m-0">
        <p className={cn(stage === 0 ? "border-b-2 border-blue-700" : "", "m-0 w-1/5 text-center")}><b>1</b> - Input Data {stage > 0 ? <Check size={17} className="inline text-green-700" /> : null}</p>
        <p className={cn(stage === 1 ? "border-b-2 border-blue-700" : stage < 1 ? "text-muted-foreground" : "", "m-0 w-1/5 text-center")}><b>2</b> - Architecture {stage > 1 ? <Check size={17} className="inline text-green-700" /> : null}</p>
        <p className={cn(stage === 2 ? "border-b-2 border-blue-700" : stage < 2 ? "text-muted-foreground" : "", "m-0 w-1/5 text-center")}><b>3</b> - System Parameters {stage > 2 ? <Check size={17} className="inline text-green-700" /> : null}</p>
        <p className={cn(stage === 3 ? "border-b-2 border-blue-700" : stage < 3 ? "text-muted-foreground" : "", "m-0 w-1/5 text-center")}><b>4</b> - Overview {stage > 3 ? <Check size={17} className="inline text-green-700" /> : null}</p>
      </div>
      <main className="flex min-h-screen flex-col items-center justify-center py-32 px-16 bg-zinc-50 dark:bg-black">
        <InnerPage stage={stage} />
      </main>
    </div>
  );
}

function InnerPage({ stage }: { stage: number }) {
  switch (stage) {
    case 0:
      return <InputData />
    case 1:
      return <ArchitectureData />
    case 2:
      return <SystemParametersData />
    case 3:
      return <OverviewData />
    default: return notFound()
  }
}

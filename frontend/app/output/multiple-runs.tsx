import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";

import { ChevronDown } from "lucide-react";
import { Simulation } from "./columns-sim";
import React from "react";

export default function MultipleRuns({ data }: { data: Simulation[][] }) {
    const [visibleColumns, setVisibleColumns] = React.useState<Record<number, boolean>>({})

    return (
        <div className="flex flex-col w-full h-full max-h-full gap-6 items-center overflow-hidden">
            <div className="flex justify-start items-start w-full">
                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <Button variant="outline">
                            Simulations <ChevronDown />
                        </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                        {data.map((_, index) => (
                            <DropdownMenuCheckboxItem key={"dropdown_" + index} checked={visibleColumns[index] ?? false} onCheckedChange={(checked) => {
                                setVisibleColumns((prev) => ({
                                    ...prev,
                                    [index]: checked ?? false,
                                }))
                            }}>
                                simulation_{index + 1}.csv
                            </DropdownMenuCheckboxItem>
                        ))}
                    </DropdownMenuContent>
                </DropdownMenu>
            </div>
        </div>
    )
}
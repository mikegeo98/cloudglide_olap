import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import SingleRun from "./single-run";
import MultipleRuns from "./multiple-runs";

import { Simulation } from "./columns-sim";

export default function Visualization({ data, filenames }: { data: Simulation[][], filenames: string[] }) {
    return (
        <div className="flex flex-col max-w-[1600px] w-full h-full max-h-full gap-8 items-center px-6 overflow-x-hidden">
            <h1>Dashboard</h1>
            <Tabs defaultValue="single" className="w-full space-y-3">
                <TabsList>
                    <TabsTrigger value="single">Single Run</TabsTrigger>
                    <TabsTrigger value="multiple">Multiple Runs</TabsTrigger>
                </TabsList>
                <TabsContent value="single">
                    <SingleRun data={data} filenames={filenames} />
                </TabsContent>
                <TabsContent value="multiple">
                    <MultipleRuns data={data} filenames={filenames} />
                </TabsContent>
            </Tabs>
        </div>
    )
}
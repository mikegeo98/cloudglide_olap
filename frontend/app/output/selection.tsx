"use client"

import { Button } from "@/components/ui/button";
import { InputContext } from "@/components/provider";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { RefreshCw, BarChart3, FileText, Info, ChevronLeft } from "lucide-react";

import React from "react";
import { columns, SimRun } from "./columns-select";
import { DataTable } from "./data-table";
import { Simulation } from "./columns-sim";
import Visualization from "./visualization";
import { useRouter } from "next/navigation";

export default function Selection({ data }: { data: { data: Simulation[][], files: string[] } }) {
    const router = useRouter()
    const { data: input } = React.useContext(InputContext)
    const [visualize, setVisualize] = React.useState(false)
    const [visualizedData, setVisualizedData] = React.useState(data.data)
    const [filenameData, setFilenameData] = React.useState(data.files)
    const [rowSelection, setRowSelection] = React.useState<Record<number, boolean>>({})
    const [tableData, setTableData] = React.useState<SimRun[]>()
    const [isRefreshing, setIsRefreshing] = React.useState(false)

    function handleVisualize() {
        setVisualizedData(data.data.filter((_, index) => rowSelection[index]))
        setFilenameData(data.files.filter((_, index) => rowSelection[index]))
        setVisualize(true)
    }

    function handleRefresh() {
        setIsRefreshing(true)
        router.refresh()
        setTimeout(() => setIsRefreshing(false), 1000)
    }

    function handleBack() {
        setVisualize(false)
        setRowSelection({})
    }

    function parseFilename(filename: string): Partial<SimRun> {
        const parts = filename.replace('.csv', '').split('_')
        const scenarioName = parts[0] || filename

        const scenario = input.scenarios.find(s => s.name === scenarioName)

        if (scenario) {
            return {
                filename,
                architecture: scenario.architecture,
                nodes: scenario.nodes,
                hit_rate: scenario.hit_rate,
                instance: scenario.instance,
                scaling_policy: scenario.scaling?.policy,
                scheduling_policy: scenario.scheduling?.policy,
                network_bandwidth: scenario.network_bandwidth,
                vpu: scenario.vpu,
                cold_start: scenario.cold_start,
            }
        }

        return { filename }
    }

    React.useEffect(() => {
        const srs = data.files.map(f => parseFilename(f) as SimRun)
        setTableData(srs)
    }, [data.files, input.scenarios])

    if (visualize) {
        return (
            <div className="w-full h-full flex flex-col">
                <div className="mb-4">
                    <Button variant="ghost" size="sm" onClick={handleBack}>
                        <ChevronLeft className="h-4 w-4 mr-1" />
                        Back to file selection
                    </Button>
                </div>
                <Visualization data={visualizedData} filenames={filenameData} />
            </div>
        )
    }

    return (
        <div className="w-full max-w-5xl flex flex-col gap-6 items-center">
            <Card className="w-full">
                <CardHeader>
                    <div className="flex items-center justify-between">
                        <div>
                            <CardTitle className="flex items-center gap-2">
                                <FileText className="h-5 w-5" />
                                Simulation Results
                            </CardTitle>
                            <CardDescription>
                                {data.files.length} simulation file{data.files.length !== 1 ? 's' : ''} found
                            </CardDescription>
                        </div>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={handleRefresh}
                            disabled={isRefreshing}
                        >
                            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
                            Refresh
                        </Button>
                    </div>
                </CardHeader>
                <CardContent>
                    {data.files.length === 0 ? (
                        <Alert>
                            <Info className="h-4 w-4" />
                            <AlertDescription>
                                No simulation files found in the output directory.
                                Run a simulation from the Simulation Wizard to generate results.
                            </AlertDescription>
                        </Alert>
                    ) : (
                        <>
                            {tableData && (
                                <DataTable
                                    className="min-w-[180px] w-full max-w-full"
                                    columns={columns}
                                    data={tableData}
                                    rowSelection={rowSelection}
                                    setRowSelection={setRowSelection}
                                />
                            )}

                            <div className="flex items-center justify-between mt-4 pt-4 border-t">
                                <p className="text-sm text-muted-foreground">
                                    {Object.keys(rowSelection).length} of {data.files.length} selected
                                </p>
                                <Button
                                    disabled={Object.keys(rowSelection).length === 0}
                                    onClick={handleVisualize}
                                >
                                    <BarChart3 className="h-4 w-4 mr-2" />
                                    Visualize Selected
                                </Button>
                            </div>
                        </>
                    )}
                </CardContent>
            </Card>

            {data.files.length > 0 && Object.keys(rowSelection).length === 0 && (
                <Alert>
                    <Info className="h-4 w-4" />
                    <AlertDescription>
                        <strong>Tip:</strong> Select one or more simulation files using the checkboxes, then click &quot;Visualize Selected&quot; to compare results.
                        Select multiple files to see side-by-side comparisons.
                    </AlertDescription>
                </Alert>
            )}
        </div>
    )
}

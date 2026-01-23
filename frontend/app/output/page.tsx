import { csvToSimulation } from "@/lib/utils";
import { promises as fs } from "fs";
import { Simulation } from "./columns-sim";
import Selection from "./selection";
import NavHeader from "@/components/nav-header";

async function getData(): Promise<{ data: Simulation[][], files: string[] }> {
    const sims: Simulation[][] = []
    try {
        const allFiles = await fs.readdir(process.cwd() + "/../cloudglide/output_simulation")
        const files = allFiles.filter(f => f.endsWith('.csv'))

        for (const file of files) {
            try {
                const content = await fs.readFile(process.cwd() + "/../cloudglide/output_simulation/" + file, "utf8");
                const obj = csvToSimulation(content)
                sims.push(obj)
            } catch (err) {
                console.error(`Error reading file ${file}:`, err)
            }
        }
        return { data: sims, files: files }
    } catch (err) {
        console.error('Error reading output directory:', err)
        return { data: [], files: [] }
    }
}

export const dynamic = 'force-dynamic'
export const revalidate = 0


export default async function OutputPage() {
    const data = await getData()

    return (
        <div className="w-full h-screen overflow-hidden bg-zinc-50 dark:bg-black">
            <NavHeader />
            <div className="h-[calc(100vh-56px)] overflow-y-scroll">
                <main className="container mx-auto py-8 px-6 pb-24">
                    <Selection data={data} />
                </main>
            </div>
        </div>
    )
}
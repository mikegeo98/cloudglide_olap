import { csvToSimulation } from "@/lib/utils";
import { promises as fs } from "fs";
import { Simulation } from "./columns-sim";
import Selection from "./selection";

async function getData(): Promise<{ data: Simulation[][], files: string[] }> {
    const sims: Simulation[][] = []
    const files = await fs.readdir(process.cwd() + "/../cloudglide/output_simulation")
    for (const file of files) {
        const content = await fs.readFile(process.cwd() + "/../cloudglide/output_simulation/" + file, "utf8");
        const obj = csvToSimulation(content)
        sims.push(obj)
    }
    return { data: sims, files: files }
}

export default async function OutputPage() {
    const data = await getData()

    return (
        <div className="flex flex-col w-full min-h-screen max-h-screen items-center justify-start bg-zinc-50 font-sans dark:bg-black">
            <main className="flex-auto flex flex-col h-full max-h-full w-full items-center justify-center overflow-hidden py-16 px-16 bg-zinc-50 dark:bg-black">
                <Selection data={data} />
            </main>
        </div>
    )
}
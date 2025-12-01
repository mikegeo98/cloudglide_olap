import { csvJSON } from "@/lib/utils";
import { promises as fs } from "fs";
import { Simulation } from "./columns-sim";
import Selection from "./selection";

async function getData(): Promise<{ data: Simulation[][], files: string[] }> {
    const sims: Simulation[][] = []
    const files = await fs.readdir(process.cwd() + "../../cloudglide/output_simulation")
    for (const file of files) {
        const content = await fs.readFile(process.cwd() + "../../cloudglide/output_simulation/" + file, "utf8");
        const obj = csvJSON(content)
        sims.push(JSON.parse(obj))
    }
    return { data: sims, files: files }
}

export default async function OutputPage() {
    const data = await getData()

    return (
        <div className="flex flex-col w-full min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
            <main className="flex min-h-screen flex-col w-full max-w-[1200px] items-center justify-center gap-6 p-20 bg-zinc-50 dark:bg-black">
                <Selection data={data} />
            </main>
        </div>
    )
}
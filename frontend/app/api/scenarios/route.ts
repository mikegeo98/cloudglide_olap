import { promises as fs } from "fs";
import { NextResponse } from "next/server";

export async function GET() {
    try {
        const customJsonContent = await fs.readFile(
            process.cwd() + "/../cloudglide/simulations/custom.json",
            "utf8"
        );
        const customJson = JSON.parse(customJsonContent);
        return NextResponse.json(customJson);
    } catch (error) {
        console.error("Error reading custom.json:", error);
        return NextResponse.json({ scenarios: [] }, { status: 500 });
    }
}

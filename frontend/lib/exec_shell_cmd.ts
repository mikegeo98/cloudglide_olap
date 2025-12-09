"use server"

import util from "node:util";
import child_process from "node:child_process";
import fs from "fs";

const exec = util.promisify(child_process.exec);

export default async function runSimulation(input_csv: File | undefined, input_json: string) {
    // Store uploaded csv file
    if (input_csv) {
        fs.writeFileSync("../cloudglide/datasets/custom.csv", await input_csv.text())
    }

    // Save input.json file in simulations folder
    const path = "cloudglide/simulations/custom.json"
    fs.writeFileSync("../" + path, input_json) // starting point of this call is frontend/

    // Execute the recently saved input.json file
    const scenarios = JSON.parse(input_json).scenarios
    for (const scenario of scenarios) {
        const { stdout, stderr } = await exec("cd .. && python main.py " + scenario.name + " " + path + " --output_prefix cloudglide/output_simulation/" + scenario.name)
        console.log("stdout:", stdout)
        console.error("stderr:", stderr)
    }
}
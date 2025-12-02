"use server"

import util from "node:util";
import child_process from "node:child_process";
import fs from "fs";

const exec = util.promisify(child_process.exec);

export default async function runSimulation(input_json: string) {
    const path = "cloudglide/simulations/custom.json"
    fs.writeFileSync("../" + path, input_json) // starting point of this call is frontend/

    const architecture = JSON.parse(input_json).scenarios[0].name
    const { stdout, stderr } = await exec("cd .. && python main.py " + architecture + " " + path)
    console.log("stdout:", stdout)
    console.error("stderr:", stderr)
}
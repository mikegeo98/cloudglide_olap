"use server"

import { InputData } from "@/components/provider";

import util from "node:util";
import child_process from "node:child_process";

const exec = util.promisify(child_process.exec);

export default async function runSimulation(data: InputData) {
    const architecture = data.architecture ? data.architecture : "dwaas_case"
    const { stdout, stderr } = await exec("cd .. && python main.py " + architecture + " ./frontend/lib/mixed_arch.json")
    console.log("stdout:", stdout)
    console.error("stderr:", stderr)
}
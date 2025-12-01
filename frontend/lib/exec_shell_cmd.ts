"use server"

import { InputData } from "@/components/provider";
import { exec } from "child_process";

export default async function runSimulation(data: InputData) {
    const architecture = data.architecture ? data.architecture : "dwaas_case"
    exec("cd .. && python main.py " + architecture + " ./frontend/lib/mixed_arch.json", (err, stdout, stderr) => {
        if (err) {
            console.error(err)
            return
        }
        console.log(stdout)
    })
}
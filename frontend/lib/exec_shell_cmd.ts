"use server"

import { InputData } from "@/components/provider";
import { exec } from "child_process";

export default async function runSimulation(data: InputData) {
    exec("cd .. && python main.py " + data.architecture + " ./frontend/lib/mixed_arch.json", (err, stdout, stderr) => {
        if (err) {
            console.error(err)
            return
        }
        console.log(stdout)
    })
}
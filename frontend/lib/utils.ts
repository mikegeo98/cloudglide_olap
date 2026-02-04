/* eslint-disable @typescript-eslint/no-explicit-any */
import { Simulation } from "@/app/output/columns-sim"
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Source - https://stackoverflow.com/a
// Posted by Wesley Smith, modified by community. See post 'Timeline' for change history
// Retrieved 2025-11-30, License - CC BY-SA 4.0

//var csv is the CSV file with headers
export function csvToSimulation(csv: any): Simulation[] {
  const lines: string[] = csv.split("\r\n");
  const result: Simulation[] = [];
  const headers: string[] = lines[0].split(",");
  for (let i = 1; i < lines.length; i++) {
    const obj: Simulation = {} as Simulation;
    const currentline = lines[i].split(",");
    if (currentline.length > 1) { // avoid empty lines
      for (let j = 0; j < headers.length; j++) {
        obj[headers[j] as keyof Simulation] = Number.parseFloat(currentline[j]);
      }
      result.push(obj);
    }
  }

  return result;
}


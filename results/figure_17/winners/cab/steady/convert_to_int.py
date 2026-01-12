import csv
import os
import sys
import math

def convert_float_durations_to_int(input_path, output_path=None):
    """
    Reads a CSV and rounds down the 'query_duration' and
    'query_duration_with_queue' columns to integers.
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_int" + ext

    with open(input_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        rows = []

        for row in reader:
            for col in ["query_duration", "query_duration_with_queue"]:
                if col in row and row[col].strip() != "":
                    try:
                        row[col] = str(int(math.floor(float(row[col]))))
                    except ValueError:
                        # Keep as-is if not numeric
                        pass
            rows.append(row)

    with open(output_path, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Converted '{input_path}' → '{output_path}'")
    print(f"   Floats in query_duration & query_duration_with_queue rounded down to int.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_durations_to_int.py input.csv [output.csv]")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) >= 3 else None
    convert_float_durations_to_int(input_csv, output_csv)

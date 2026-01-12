import csv
import random
import sys
import os

def reduce_csv_durations(input_path, output_path=None):
    """
    Reads a CSV file, randomly picks a reduction percentage between 1% and 20%
    for each row, and scales down the query_duration and
    query_duration_with_queue columns accordingly.

    Args:
        input_path: path to input CSV
        output_path: path to save new CSV (default: input file with _reduced suffix)
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_reduced" + ext

    with open(input_path, "r") as infile, open(output_path, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        modified_rows = 0
        for row in reader:
            try:
                # Random percentage between 1% and 20%
                reduction = random.uniform(0.01, 0.2)
                factor = 1.0 - reduction

                # Apply to both duration columns
                for col in ["query_duration", "query_duration_with_queue"]:
                    if col in row and row[col].strip():
                        row[col] = str(float(row[col]) * factor)

                modified_rows += 1
            except ValueError:
                # Skip rows with invalid numbers
                continue

            writer.writerow(row)

    print(f"✅ Processed {modified_rows} rows from {input_path}")
    print(f"💾 Saved modified CSV to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reduce_durations.py input.csv [output.csv]")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    reduce_csv_durations(input_csv, output_csv)

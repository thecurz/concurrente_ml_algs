import csv
import sys
from pathlib import Path

def multiply_csv_rows(input_file, output_file, target_rows=1_000_000):
    try:
        if input_file == output_file:
            raise ValueError("Input and output files must be different to avoid data loss.")

        with open(input_file, 'r', newline='') as infile:
            reader = csv.reader(infile)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError("The input file is empty.")

            all_rows = list(reader)
            original_row_count = len(all_rows)

            if original_row_count == 0:
                raise ValueError("The input file has no data rows.")

        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)

            repeat_count = (target_rows - 1) // original_row_count
            remaining_rows = (target_rows - 1) % original_row_count

            rows_written = 1
            for _ in range(repeat_count):
                writer.writerows(all_rows)
                rows_written += original_row_count
                print(f"Progress: {rows_written:,} / {target_rows:,} rows written", end='\r')

            writer.writerows(all_rows[:remaining_rows])
            rows_written += remaining_rows
            print(f"Progress: {rows_written:,} / {target_rows:,} rows written")
            print(f"\nDone! {rows_written:,} rows (including header) written to {output_file}")

    except IOError as e:
        print(f"Error reading or writing file: {e}")
    except csv.Error as e:
        print(f"CSV error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    input_files = ["./datasets/adult.csv", "./datasets/Reviews.csv"]
    
    for input_file in input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Error: Input file '{input_file}' does not exist.")
            continue

        output_file = input_path.with_name(f"{input_path.stem}_1m{input_path.suffix}")
        print(f"Processing {input_file}...")
        multiply_csv_rows(input_file, str(output_file))
        print("\n")

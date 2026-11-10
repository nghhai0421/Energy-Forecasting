"""Export the six raw data subsets and holiday workbook to CSV."""

import csv
from pathlib import Path

from openpyxl import load_workbook


def convert_raw_to_csv():
    """Export missing CSVs from the seven hardcoded raw Excel inputs.

    Returns
    -------
    dict of pathlib.Path to int
        New output paths mapped to exported data-row counts; existing CSVs are skipped.

    Raises
    ------
    FileNotFoundError
        If a configured workbook is missing.
    ValueError
        If a workbook has multiple sheets, incomplete headers, or formula cells.

    Notes
    -----
    Reads Dataset/raw/xlsx and writes Dataset/raw/csv relative to the repository.
    Metadata preambles are omitted; values use comma-delimited UTF-8 with BOM.
    """
    # Set Paths
    raw_dir = Path(__file__).resolve().parents[1] / "Dataset" / "raw"
    input_dir = raw_dir / "xlsx"
    output_dir = raw_dir / "csv"
    inputs = [
        ("Actual_consumption_2023_2024_Quarterhour.xlsx", 10),
        ("Actual_consumption_2025_Quarterhour.xlsx", 10),
        ("Day-ahead_prices_2023_2024_Quarterhour.xlsx", 10),
        ("Day-ahead_prices_2025_Quarterhour.xlsx", 10),
        ("open-meteo-48.83N9.15E293m_2023_2024.xlsx", 4),
        ("open-meteo-48.83N9.15E293m_2025.xlsx", 4),
        ("df_holiday.xlsx", 1),
    ]

    # Check Files
    for filename, header_row in inputs:
        source = input_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export Tables
    results = {}
    for filename, header_row in inputs:
        source = input_dir / filename
        destination = output_dir / Path(filename).with_suffix(".csv")
        if destination.exists():
            print(f"Skipped existing CSV: {destination.name}")
            continue
        workbook = load_workbook(source, read_only=True, data_only=False)
        try:
            if len(workbook.worksheets) != 1:
                raise ValueError(f"Expected one worksheet: {source}")
            sheet = workbook.worksheets[0]
            rows = sheet.iter_rows(min_row=header_row)
            header = [cell.value for cell in next(rows)]
            if not header or any(value is None for value in header):
                raise ValueError(f"Expected complete column headers in row {header_row}: {source}")

            row_count = 0
            with destination.open("x", encoding="utf-8-sig", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(header)
                for row in rows:
                    if any(cell.data_type == "f" for cell in row):
                        raise ValueError(f"Formula requires evaluation before CSV export: {source}")
                    writer.writerow([cell.value for cell in row])
                    row_count += 1
        finally:
            workbook.close()

        results[destination] = row_count
        print(f"{destination.name}: {row_count:,} rows, {len(header)} columns")

    return results


if __name__ == "__main__":
    convert_raw_to_csv()

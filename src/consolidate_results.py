#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Script to consolidate benchmark results from multiple Excel files into a single file.

This script reads benchmark results from Excel files in a directory, extracts data from
"Summary" and "Summary MLPERF" sheets, adds model name and source columns, and creates
a consolidated Excel file with all results.

Usage:
  # Use default directory (benchmark_results) and output file
  python -m src.consolidate_results
  
  # Specify custom input directory
  python -m src.consolidate_results --input-dir /path/to/results
  
  # Specify custom output file
  python -m src.consolidate_results --output consolidated_benchmark_results.xlsx
  
  # Use a specific file pattern
  python -m src.consolidate_results --pattern "BM_cuda_*.xlsx"
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def extract_model_name(filename: str) -> str:
    """
    Extract model name from benchmark result filename.
    
    Expected format: BM_{device}_{model_name}_{other_info}
    Example: BM_cuda_Cfa_transistor_runs-5_seed-42_20251106-173024
    Returns: Cfa (3rd word after splitting by underscores)
    
    Args:
        filename: Name of the benchmark result file
        
    Returns:
        Extracted model name
    """
    # Remove file extension
    name_without_ext = Path(filename).stem
    
    # Split by underscore
    parts = name_without_ext.split('_')
    
    # The model name is the 3rd element (index 2) after splitting
    # Format: BM_{device}_{model_name}_{...}
    if len(parts) >= 3:
        return parts[2]
    
    # Fallback: return the filename without extension
    return name_without_ext


def read_sheet_with_model_info(
    file_path: Path,
    sheet_name: str,
    model_name: str,
    source_filename: str
) -> Optional[pd.DataFrame]:
    """
    Read a sheet from an Excel file and add model name and source columns.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Name of the sheet to read
        model_name: Model name to add as first column
        source_filename: Source filename to add as last column
        
    Returns:
        DataFrame with added columns, or None if sheet doesn't exist
    """
    try:
        # Read the sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Add model name as first column
        df.insert(0, 'Model', model_name)
        
        # Add source filename as last column
        df['Source'] = source_filename
        
        return df
    except ValueError:
        # Sheet doesn't exist
        print(f"  Warning: Sheet '{sheet_name}' not found in {file_path.name}")
        return None
    except Exception as e:
        print(f"  Error reading sheet '{sheet_name}' from {file_path.name}: {e}")
        return None


def consolidate_benchmark_results(
    input_dir: Path,
    output_file: Path,
    pattern: str = "*.xlsx"
) -> None:
    """
    Consolidate benchmark results from multiple Excel files into a single file.
    
    Args:
        input_dir: Directory containing benchmark result Excel files
        output_file: Path for the consolidated output Excel file
        pattern: File pattern to match (default: "*.xlsx")
    """
    # Sheets to process
    sheets_to_consolidate = {
        'Summary': [],
        'Summary MLPERF': []
    }
    
    # Find all Excel files matching the pattern
    excel_files = sorted(input_dir.glob(pattern))
    
    if not excel_files:
        print(f"No Excel files found in {input_dir} matching pattern '{pattern}'")
        return
    
    print(f"Found {len(excel_files)} Excel file(s) to process:")
    
    # Process each Excel file
    for excel_file in excel_files:
        print(f"\nProcessing: {excel_file.name}")
        
        # Extract model name from filename
        model_name = extract_model_name(excel_file.name)
        print(f"  Model name: {model_name}")
        
        # Read each sheet and add to the consolidation lists
        for sheet_name in sheets_to_consolidate.keys():
            df = read_sheet_with_model_info(
                excel_file,
                sheet_name,
                model_name,
                excel_file.name
            )
            
            if df is not None:
                sheets_to_consolidate[sheet_name].append(df)
                print(f"  Added {len(df)} rows from '{sheet_name}' sheet")
    
    # Consolidate and write results
    print(f"\n{'='*60}")
    print("Creating consolidated Excel file...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, dataframes in sheets_to_consolidate.items():
            if dataframes:
                # Concatenate all dataframes for this sheet
                consolidated_df = pd.concat(dataframes, ignore_index=True)
                
                # Write to Excel
                consolidated_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                print(f"  Sheet '{sheet_name}': {len(consolidated_df)} total rows from {len(dataframes)} file(s)")
            else:
                print(f"  Sheet '{sheet_name}': No data found (skipping)")
    
    print(f"\n{'='*60}")
    print(f"Consolidated results saved to: {output_file}")
    print(f"{'='*60}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Consolidate benchmark results from multiple Excel files into a single file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default directory (benchmark_results) and output file
  python -m src.consolidate_results
  
  # Specify custom input directory
  python -m src.consolidate_results --input-dir /path/to/results
  
  # Specify custom output file
  python -m src.consolidate_results --output consolidated_benchmark_results.xlsx
  
  # Specify both input directory and output file
  python -m src.consolidate_results -i /path/to/results -o my_results.xlsx
  
  # Use a specific file pattern
  python -m src.consolidate_results --pattern "BM_cuda_*.xlsx"
        """
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        type=Path,
        default=Path('benchmark_results'),
        help='Directory containing benchmark result Excel files (default: benchmark_results)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('consolidated_benchmark_results.xlsx'),
        help='Output file path for consolidated results (default: consolidated_benchmark_results.xlsx)'
    )
    
    parser.add_argument(
        '-p', '--pattern',
        type=str,
        default='*.xlsx',
        help='File pattern to match (default: *.xlsx)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    if not args.input_dir.is_dir():
        print(f"Error: '{args.input_dir}' is not a directory")
        return 1
    
    # Create output directory if it doesn't exist
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Run consolidation
    try:
        consolidate_benchmark_results(
            input_dir=args.input_dir,
            output_file=args.output,
            pattern=args.pattern
        )
        return 0
    except Exception as e:
        print(f"\nError during consolidation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

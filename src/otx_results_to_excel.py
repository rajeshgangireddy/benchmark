# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Standalone tool to convert OTX training results from CSV to Excel.

This is a convenience wrapper around src.utils.metrics for direct use from command line.

Usage:
  # Auto-find latest run
  python -m src.otx_results_to_excel

  # Specify metrics CSV
  python -m src.otx_results_to_excel --metrics-csv /path/to/metrics.csv

  # Specify workspace and output
  python -m src.otx_results_to_excel --workspace ./otx-workspace --output results.xlsx
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.utils.metrics import (
    find_latest_otx_workspace,
    load_otx_metrics_csv,
    create_metrics_summary,
    parse_otx_hparams,
)


def convert_otx_results_to_excel(
    metrics_csv_path: Path,
    output_path: Path,
    hparams_path: Path | None = None
) -> None:
    """Convert OTX metrics CSV to Excel with multiple sheets."""
    
    print(f"Loading metrics from: {metrics_csv_path}")
    df = load_otx_metrics_csv(metrics_csv_path)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary statistics
        summary_data = create_metrics_summary(df)
        if summary_data:
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            print("  - Created 'Summary' sheet")
        
        # Epoch metrics (last row per epoch)
        epoch_metrics = df[df['epoch'].notna()].copy()
        if not epoch_metrics.empty:
            epoch_metrics = epoch_metrics.groupby('epoch').last().reset_index()
            epoch_metrics.to_excel(writer, sheet_name='Epoch_Metrics', index=False)
            print(f"  - Created 'Epoch_Metrics' sheet with {len(epoch_metrics)} epochs")
        
        # All raw metrics
        df.to_excel(writer, sheet_name='All_Metrics', index=False)
        print(f"  - Created 'All_Metrics' sheet with {len(df)} rows")
        
        # Hyperparameters
        if hparams_path and hparams_path.exists():
            hparams_dict = parse_otx_hparams(hparams_path)
            if hparams_dict:
                hparams_df = pd.DataFrame(
                    list(hparams_dict.items()), 
                    columns=['Parameter', 'Value']
                )
                hparams_df.to_excel(writer, sheet_name='Hyperparameters', index=False)
                print("  - Created 'Hyperparameters' sheet")
    
    print(f"\nExcel file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert OTX training results (CSV) to Excel format with summary statistics."
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        help="Path to OTX metrics.csv file. If not provided, searches for latest run.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("../training_extensions/library/otx-workspace"),
        help="Path to OTX workspace directory (default: ../training_extensions/library/otx-workspace)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output Excel file path. If not provided, creates in current directory.",
    )
    
    args = parser.parse_args()
    
    # Find metrics CSV
    if args.metrics_csv:
        metrics_csv_path = args.metrics_csv
        if not metrics_csv_path.exists():
            print(f"Error: Metrics file not found: {metrics_csv_path}")
            sys.exit(1)
        work_dir = metrics_csv_path.parent.parent.parent  # csv/version_0/metrics.csv
    else:
        # Find latest run
        latest_run = find_latest_otx_workspace(args.workspace)
        if not latest_run:
            print(f"Error: No runs found in {args.workspace}")
            sys.exit(1)
        
        metrics_csv_path = latest_run / "csv" / "version_0" / "metrics.csv"
        work_dir = latest_run
        
        if not metrics_csv_path.exists():
            print(f"Error: Metrics file not found at {metrics_csv_path}")
            sys.exit(1)
        
        print(f"Found latest run: {latest_run.name}")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        run_name = work_dir.name
        output_path = Path(f"otx_results_{run_name}.xlsx")
    
    # Find hyperparameters file
    hparams_path = work_dir / "csv" / "version_0" / "hparams.yaml"
    
    # Convert to Excel
    convert_otx_results_to_excel(metrics_csv_path, output_path, hparams_path)


if __name__ == "__main__":
    main()

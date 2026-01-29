# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Consolidated OTX metrics loading and export utilities.
Provides shared functionality for loading OTX training metrics and exporting to Excel.
"""

import logging
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


def find_latest_otx_workspace(workspace_path: str | Path) -> Path | None:
    """
    Find the most recent OTX run directory in the workspace.
    
    Args:
        workspace_path: Path to the OTX workspace directory.
        
    Returns:
        Path to the latest run directory, or None if not found.
    """
    workspace_path = Path(workspace_path)
    
    if not workspace_path.exists():
        return None
    
    run_dirs = [d for d in workspace_path.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    
    # Sort by directory name (timestamp format: YYYYMMDD_HHMMSS)
    return max(run_dirs, key=lambda x: x.name)


def load_otx_metrics_csv(metrics_csv_path: Path) -> pd.DataFrame:
    """
    Load and clean OTX metrics CSV.
    
    Args:
        metrics_csv_path: Path to the metrics.csv file.
        
    Returns:
        Cleaned pandas DataFrame.
    """
    df = pd.read_csv(metrics_csv_path)
    
    # Remove rows with all NaN values except epoch and step
    non_index_cols = [col for col in df.columns if col not in ['epoch', 'step']]
    df = df.dropna(how='all', subset=non_index_cols)
    
    return df


def create_metrics_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Create summary statistics from OTX metrics DataFrame.
    
    Args:
        df: Metrics DataFrame from OTX CSV.
        
    Returns:
        Dictionary with summary statistics.
    """
    summary_data = {}
    
    # Get final epoch metrics
    if 'epoch' in df.columns:
        epoch_data = df[df['epoch'].notna()]
        if not epoch_data.empty:
            final_epoch = epoch_data['epoch'].max()
            final_metrics = epoch_data[epoch_data['epoch'] == final_epoch].iloc[-1]
            summary_data['Final Epoch'] = int(final_epoch)
        else:
            final_metrics = {}
    else:
        final_metrics = {}
    
    # Validation accuracy
    if 'val/accuracy' in df.columns:
        summary_data['Best Validation Accuracy'] = df['val/accuracy'].max()
        if 'val/accuracy' in final_metrics:
            summary_data['Final Validation Accuracy'] = final_metrics.get('val/accuracy')
    
    # Training loss
    if 'train/loss' in df.columns:
        train_losses = df['train/loss'].dropna()
        if len(train_losses) > 0:
            summary_data['Best Train Loss'] = train_losses.min()
            if 'train/loss' in final_metrics:
                summary_data['Final Train Loss'] = final_metrics.get('train/loss')
    
    # GPU memory
    if 'gpu_mem' in df.columns:
        gpu_mem = df['gpu_mem'].dropna()
        if len(gpu_mem) > 0:
            summary_data['Peak GPU Memory (GB)'] = gpu_mem.max()
            summary_data['Avg GPU Memory (GB)'] = gpu_mem.mean()
    
    # Iteration time
    if 'train/iter_time' in df.columns:
        iter_time = df['train/iter_time'].dropna()
        if len(iter_time) > 0:
            summary_data['Avg Iteration Time (s)'] = iter_time.mean()
    
    return summary_data


def parse_otx_hparams(hparams_path: Path) -> dict[str, str]:
    """
    Parse OTX hyperparameters YAML file safely.
    
    OTX hparams.yaml may contain custom YAML tags that can't be parsed
    with yaml.safe_load. This function extracts key-value pairs directly.
    
    Args:
        hparams_path: Path to hparams.yaml file.
        
    Returns:
        Dictionary of hyperparameter key-value pairs.
    """
    hparams_dict = {}
    
    try:
        with open(hparams_path, 'r') as f:
            hparams_raw = f.read()
        
        for line in hparams_raw.split('\n'):
            if ':' in line and not line.strip().startswith('#'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    # Skip YAML tags and empty values
                    if key and not key.startswith('!') and value:
                        hparams_dict[key] = value
    except Exception:
        pass
    
    return hparams_dict


def export_detailed_metrics_to_excel(
    workspace_dir: str | Path,
    output_dir: Path,
    timestamp: str,
    logger: logging.Logger | None = None
) -> Path | None:
    """
    Export detailed OTX metrics CSV to Excel with summary statistics.
    
    Args:
        workspace_dir: Path to OTX workspace directory.
        output_dir: Directory to save Excel file.
        timestamp: Timestamp string for filename.
        logger: Optional logger for status messages.
        
    Returns:
        Path to the exported Excel file, or None if export failed.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # Find latest OTX workspace
        latest_run = find_latest_otx_workspace(workspace_dir)
        if not latest_run:
            logger.warning(f"No OTX workspace found at {workspace_dir} - skipping detailed metrics export")
            return None
        
        metrics_csv = latest_run / "csv" / "version_0" / "metrics.csv"
        if not metrics_csv.exists():
            logger.warning(f"OTX metrics CSV not found at {metrics_csv}")
            return None
        
        # Load metrics
        logger.info(f"Loading detailed OTX metrics from: {metrics_csv}")
        df = load_otx_metrics_csv(metrics_csv)
        
        # Create summary
        summary_data = create_metrics_summary(df)
        
        # Create Excel file
        output_path = output_dir / f"otx_detailed_metrics_{timestamp}.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary statistics
            if summary_data:
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Epoch metrics (last row per epoch)
            epoch_metrics = df[df['epoch'].notna()].copy()
            if not epoch_metrics.empty:
                epoch_metrics = epoch_metrics.groupby('epoch').last().reset_index()
                epoch_metrics.to_excel(writer, sheet_name='Epoch_Metrics', index=False)
            
            # All raw metrics
            df.to_excel(writer, sheet_name='All_Metrics', index=False)
            
            # Hyperparameters
            hparams_path = latest_run / "csv" / "version_0" / "hparams.yaml"
            if hparams_path.exists():
                hparams_dict = parse_otx_hparams(hparams_path)
                if hparams_dict:
                    hparams_df = pd.DataFrame(
                        list(hparams_dict.items()), 
                        columns=['Parameter', 'Value']
                    )
                    hparams_df.to_excel(writer, sheet_name='Hyperparameters', index=False)
        
        logger.info(f"Detailed OTX metrics exported to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to export detailed OTX metrics: {e}")
        logger.error(traceback.format_exc())
        return None

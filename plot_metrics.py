import json
import os
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd

def collect_metrics_from_jsons(directory: str) -> pd.DataFrame:
    """
    Collects r2, mae, and mse metrics for each model across multiple JSON files in a directory.

    Args:
        directory (str): Path to directory containing JSON files.

    Returns:
        pd.DataFrame: Long-form dataframe with columns: 'filename', 'model', 'metric', 'value'
    """
    records = []
    json_files = Path(directory).glob("*.json")

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            for model_name, metrics in data.get("supervised", {}).items():
                for metric_name, value in metrics.items():
                    filename = json_file.stem
                    
                    # All files begin with "iqa_", so remove that prefix
                    if filename.startswith("iqa_"):
                        filename = filename[4:]
                    
                    # All files end with "_metrics_results", so remove that suffix
                    if filename.endswith("_metrics_results"):
                        filename = filename[:-18]
                    
                    records.append({
                        "filename": filename,
                        "model": model_name,
                        "metric": metric_name,
                        "value": value
                    })

    return pd.DataFrame(records)

def plot_metrics(df: pd.DataFrame) -> None:
    """
    Creates scatter plots for each metric (r2, mae, mse), with different colors per model.

    Args:
        df (pd.DataFrame): DataFrame with 'filename', 'model', 'metric', 'value'
    """
    metrics = df['metric'].unique()
    for metric in metrics:
        metric_df = df[df['metric'] == metric]
        plt.figure()
        for model in metric_df['model'].unique():
            model_df = metric_df[metric_df['model'] == model]
            plt.scatter(model_df['filename'], model_df['value'], label=model)
        plt.title(f'{metric.upper()} Values Across Files')
        plt.xlabel('JSON File')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    directory_path = "results/results_20250418_120940"  # replace with your path
    df = collect_metrics_from_jsons(directory_path)
    plot_metrics(df)

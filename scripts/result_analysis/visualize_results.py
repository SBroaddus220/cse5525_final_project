import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_results(directory: Path) -> pd.DataFrame:
    """Load and aggregate model results from the compiled_results directory.

    Args:
        directory (Path): Path to the compiled_results directory.

    Returns:
        pd.DataFrame: Aggregated DataFrame with columns 
                      ['metric', 'model_type', 'model_name', 'mse', 'mae', 'r2'].
    """
    records = []
    
    # Loop through each metric directory
    for metric_dir in directory.iterdir():
        if metric_dir.is_dir():
            result_file = metric_dir / 'results.json'
            if result_file.exists():
                with open(result_file, 'r') as f:
                    results = json.load(f)

                for model_type in ['unsupervised', 'supervised']:
                    if model_type in results:
                        for model_name, metrics in results[model_type].items():
                            cleaned_metric = metric_dir.name
                            if cleaned_metric.startswith("iqa_"):
                                cleaned_metric = cleaned_metric[len("iqa_"):]
                            if cleaned_metric.endswith("_metrics"):
                                cleaned_metric = cleaned_metric[:-len("_metrics")]

                            records.append({
                                'metric': cleaned_metric,
                                'model_type': model_type,
                                'model_name': model_name,
                                'mse': metrics.get('mse', None),
                                'mae': metrics.get('mae', None),
                                'r2': metrics.get('r2', None)
                            })
    return pd.DataFrame(records)


def plot_metric(
    df: pd.DataFrame, metric_name: str, output_path: Path, color_by_model: bool = True
) -> None:
    """Plot a single performance metric (R², MAE, or MSE) as a scatterplot grouped by metric category.

    Args:
        df (pd.DataFrame): DataFrame of results.
        metric_name (str): One of 'r2', 'mae', or 'mse'.
        output_path (Path): Path to save the plot.
        color_by_model (bool): Whether to use separate colors per model or by model_type.
    """
    plt.figure(figsize=(16, 8))
    plot_data = df.copy()
    plot_data["marker"] = plot_data["model_type"].map({"supervised": "o", "unsupervised": "X"})
    plot_data["model_label"] = plot_data["model_type"].str[0].str.upper() + "_" + plot_data["model_name"]

    # Sort metrics for consistency
    metric_order = sorted(plot_data["metric"].unique())

    # Plot using seaborn.scatterplot with hue as model_label or model_type
    hue = "model_label" if color_by_model else "model_type"
    sns.scatterplot(
        data=plot_data,
        x="metric",
        y=metric_name,
        hue=hue,
        style="model_type",
        markers={"supervised": "o", "unsupervised": "X"},
        s=100,
        palette="tab10",
        hue_order=sorted(plot_data[hue].unique()),
        style_order=["supervised", "unsupervised"]
    )

    plt.title(f"{metric_name.upper()} Values Across Metrics")
    plt.xlabel("Metric")
    plt.ylabel(metric_name.upper())
    plt.xticks(rotation=60, ha="right")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_all_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save all scatterplots for each performance metric.

    Args:
        df (pd.DataFrame): Aggregated results DataFrame.
        output_dir (Path): Output directory to save plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["r2", "mae", "mse"]:
        plot_metric(df, metric_name=metric, output_path=output_dir / f"{metric}_scatter_by_model.png", color_by_model=True)
        plot_metric(df, metric_name=metric, output_path=output_dir / f"{metric}_scatter_by_type.png", color_by_model=False)


def main() -> None:
    """Main function to parse arguments and run the aggregation and visualization."""
    parser = argparse.ArgumentParser(description="Aggregate and visualize model evaluation results.")
    parser.add_argument(
        'compiled_results_dir',
        type=Path,
        help="Path to the compiled_results directory."
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path("./plots"),
        help="Directory where plots will be saved."
    )
    args = parser.parse_args()

    df = load_results(args.compiled_results_dir)
    plot_all_metrics(df, args.output_dir)


if __name__ == "__main__":
    main()

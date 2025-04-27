import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_timestamp(directory: Path) -> Optional[datetime]:
    """Parse timestamp from a directory name of format 'results_%Y%m%d_%H%M%S'."""
    try:
        return datetime.strptime(directory.name, "results_%Y%m%d_%H%M%S")
    except ValueError:
        return None


def find_valid_results_dirs(base_dir: Path) -> Dict[str, Dict[datetime, Path]]:
    """
    Recursively find valid results directories with the expected structure.

    Args:
        base_dir: Path to the root directory to search.

    Returns:
        A dictionary mapping each metric name to a dictionary of timestamps and their corresponding paths.
    """
    results_map: Dict[str, Dict[datetime, Path]] = {}

    for dir_path in base_dir.rglob("results_*"):
        timestamp = parse_timestamp(dir_path)
        if timestamp is None:
            continue

        for metric_dir in dir_path.iterdir():
            if not metric_dir.is_dir():
                continue

            results_file = metric_dir / "results.json"
            if results_file.exists():
                metric_name = metric_dir.name
                results_map.setdefault(metric_name, {})
                results_map[metric_name][timestamp] = metric_dir

    return results_map


def collect_latest_model_metrics(
    ts_to_dir: Dict[datetime, Path]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Collect the latest model metrics for all models ever seen across timestamps.

    Args:
        ts_to_dir: Mapping of timestamps to directories for a given metric.

    Returns:
        Aggregated model metrics dictionary.
    """
    latest_metrics: Dict[str, Dict[str, Dict[str, float]]] = {"supervised": {}, "unsupervised": {}}
    model_latest_ts: Dict[Tuple[str, str], datetime] = {}

    # Iterate all timestamps in order
    for ts in sorted(ts_to_dir):
        path = ts_to_dir[ts]
        results_file = path / "results.json"
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to read {results_file}: {e}")
            continue

        for category in ["supervised", "unsupervised"]:
            models = data.get(category, {})
            for model_name, metrics in models.items():
                key = (category, model_name)
                if key not in model_latest_ts or ts > model_latest_ts[key]:
                    model_latest_ts[key] = ts
                    latest_metrics[category][model_name] = metrics

    return latest_metrics


def collect_latest_files(
    ts_to_dir: Dict[datetime, Path],
    extensions: Tuple[str, ...] = (".csv", ".png", ".npy")
) -> Dict[str, Tuple[datetime, Path]]:
    """
    Collect the latest file for each model shap file across all timestamps.

    Args:
        ts_to_dir: Mapping of timestamps to directories for a given metric.
        extensions: Tuple of file extensions to track.

    Returns:
        Dictionary mapping filename to the latest (timestamp, path).
    """
    latest_files: Dict[str, Tuple[datetime, Path]] = {}

    for ts, dir_path in ts_to_dir.items():
        for file_path in dir_path.iterdir():
            if file_path.suffix.lower() not in extensions:
                continue
            fname = file_path.name
            if fname not in latest_files or ts > latest_files[fname][0]:
                latest_files[fname] = (ts, file_path)

    return latest_files


def copy_latest_results_and_files(
    metric: str, ts_dirs: Dict[datetime, Path], output_root: Path
) -> None:
    """
    Copy the latest results.json and all latest SHAP files for a given metric.

    Args:
        metric: The metric name.
        ts_dirs: Mapping of timestamps to directories for this metric.
        output_root: Root path where aggregated results should be written.
    """
    output_dir = output_root / metric
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate latest model metrics
    latest_metrics = collect_latest_model_metrics(ts_dirs)
    out_json_path = output_dir / "results.json"
    with open(out_json_path, "w") as f:
        json.dump(latest_metrics, f, indent=4)
    logging.info(f"Wrote aggregated results.json for metric '{metric}'")

    # Copy latest SHAP files
    latest_files = collect_latest_files(ts_dirs)
    for fname, (_, src_path) in latest_files.items():
        dst_path = output_dir / fname
        shutil.copy2(src_path, dst_path)
        logging.info(f"Copied latest {fname} to {dst_path}")


def aggregate_latest_results(base_dir: Path, output_dir: Path) -> None:
    """
    Aggregate latest results.json and all SHAP files per metric into a new structure.

    Args:
        base_dir: Path to search for results directories.
        output_dir: Path to store aggregated results.
    """
    logging.info(f"Searching for results in {base_dir}")
    results = find_valid_results_dirs(base_dir)

    for metric, ts_dirs in results.items():
        logging.info(f"Processing metric: {metric}")
        copy_latest_results_and_files(metric, ts_dirs, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate latest metric results and SHAP files.")
    parser.add_argument("base_dir", type=Path, help="Base directory containing results.")
    parser.add_argument("output_dir", type=Path, help="Directory to write aggregated results.")
    args = parser.parse_args()

    aggregate_latest_results(args.base_dir, args.output_dir)

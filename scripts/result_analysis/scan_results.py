import json
from pathlib import Path
from typing import Dict, List, Set
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def collect_present_models(results_json: Path) -> Dict[str, Set[str]]:
    """
    Extract model names from a results.json file.

    Args:
        results_json: Path to results.json.

    Returns:
        A dictionary with 'supervised' and 'unsupervised' keys and sets of model names.
    """
    with open(results_json, "r") as f:
        data = json.load(f)

    return {
        "supervised": set(data.get("supervised", {}).keys()),
        "unsupervised": set(data.get("unsupervised", {}).keys()),
    }


def extract_shap_file_model_prefix(filename: str) -> str:
    """
    Extract model name prefix from a SHAP file name of the form '<model>_shap.<ext>'.

    Args:
        filename: File name to process.

    Returns:
        Extracted model name or empty string if invalid.
    """
    if "_shap." in filename:
        return filename.split("_shap.")[0]
    return ""


def check_compiled_results(compiled_dir: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Check for missing models or SHAP files in compiled results.

    Args:
        compiled_dir: Root directory of compiled results.

    Returns:
        A dictionary detailing missing models and files per metric.
    """
    all_supervised: Set[str] = set()
    all_unsupervised: Set[str] = set()
    shap_models_with_files: Set[str] = set()
    shap_file_suffixes = {".csv", ".png", ".npy"}

    # First pass: collect all models and all models that have at least one SHAP file
    for metric_dir in compiled_dir.iterdir():
        if not metric_dir.is_dir():
            continue
        json_file = metric_dir / "results.json"
        if not json_file.exists():
            logging.warning(f"No results.json in {metric_dir}")
            continue

        models = collect_present_models(json_file)
        all_supervised.update(models["supervised"])
        all_unsupervised.update(models["unsupervised"])

        for file_path in metric_dir.iterdir():
            if file_path.suffix.lower() in shap_file_suffixes:
                model = extract_shap_file_model_prefix(file_path.name)
                if model:
                    shap_models_with_files.add(model)

    # Generate expected SHAP file names
    expected_shap_files: Set[str] = {
        f"{model}_shap{ext}" for model in shap_models_with_files for ext in shap_file_suffixes
    }

    # Second pass: identify missing models and files per metric
    missing_report: Dict[str, Dict[str, List[str]]] = {}

    for metric_dir in compiled_dir.iterdir():
        if not metric_dir.is_dir():
            continue
        metric_name = metric_dir.name
        json_file = metric_dir / "results.json"
        if not json_file.exists():
            continue

        models = collect_present_models(json_file)
        files_present = {f.name for f in metric_dir.iterdir() if f.is_file()}

        missing_supervised = sorted(list(all_supervised - models["supervised"]))
        missing_unsupervised = sorted(list(all_unsupervised - models["unsupervised"]))
        missing_files = sorted(list(expected_shap_files - files_present))

        if missing_supervised or missing_unsupervised or missing_files:
            missing_report[metric_name] = {
                "missing_supervised_models": missing_supervised,
                "missing_unsupervised_models": missing_unsupervised,
                "missing_files": missing_files,
            }

    return missing_report


def write_missing_report(report: Dict[str, Dict[str, List[str]]], output_path: Path) -> None:
    """
    Write the missing report to a JSON file.

    Args:
        report: The dictionary of missing items.
        output_path: The path to write the JSON file to.
    """
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
    logging.info(f"Wrote missing models/files report to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check compiled results for missing models and SHAP files.")
    parser.add_argument("compiled_dir", type=Path, help="Directory with compiled results.")
    parser.add_argument("output_json", type=Path, help="Output JSON file path for missing report.")
    args = parser.parse_args()

    report = check_compiled_results(args.compiled_dir)
    write_missing_report(report, args.output_json)

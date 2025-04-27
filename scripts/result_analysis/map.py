import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

import pyiqa
import torch  # Required by pyiqa

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize(name: str) -> str:
    """Normalize name by replacing + and - with _, collapsing underscores, and lowercasing."""
    name = name.replace('+', '_').replace('-', '_')
    name = re.sub(r'_+', '_', name)
    return name.lower()

def extract_core_from_dir(dir_path: Path) -> str:
    """Strip prefix/suffix from directory name and normalize."""
    name = dir_path.name
    if name.startswith("iqa_"):
        name = name[len("iqa_"):]
    if name.endswith("_metrics"):
        name = name[:-len("_metrics")]
    return normalize(name)

def get_metric_mapping(directory: Path, metric_names: List[str]) -> Dict[str, str]:
    """Recursively search directory and match subdirectories to known metrics."""
    mapping = {}
    norm_metrics = {normalize(name): name for name in metric_names}
    
    for subdir in directory.rglob('*'):
        if subdir.is_dir():
            core_name = extract_core_from_dir(subdir)
            matched_metric = norm_metrics.get(core_name)
            if matched_metric:
                mapping[str(subdir)] = matched_metric
            else:
                logger.warning(f"No match found for directory: {subdir}")
    return mapping

def main():
    parser = argparse.ArgumentParser(description="Match pyiqa metrics to subdirectories.")
    parser.add_argument("directory", type=Path, help="Directory to search recursively.")
    parser.add_argument("--output", type=Path, default="matched_metrics.json", help="Output JSON file.")
    args = parser.parse_args()

    # Step 1: Get all available metrics
    metric_list = pyiqa.list_models()
    logger.info(f"Loaded {len(metric_list)} IQA metrics from pyiqa.")

    # Step 2: Match and collect
    mapping = get_metric_mapping(args.directory, metric_list)

    # Step 3: Output JSON
    with args.output.open('w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4)
    
    logger.info(f"Mapping saved to {args.output}")

if __name__ == "__main__":
    main()

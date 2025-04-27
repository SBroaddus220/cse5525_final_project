#!/usr/bin/env python3
"""Aggregate and visualise SHAP key-phrase importances for IQA metrics.

The script walks the *compiled_results/* tree that contains one sub-directory
per metric (e.g. *iqa_arniqa_clive_metrics/*).  Every metric folder may hold
one or more ``*_shap.csv`` files – one for each model that produced SHAP
analysis.  A CSV looks like::

    keyphrase,mean_abs_shap
    abandoned,0.001567873102830481
    alien,0.00137877887465989
    alien planet,3.342807580140753e-05
    android cyborg,0.0004799780617945857

For every CSV row we **adjust the value’s sign** so that “positive = better”.
That is, if the underlying IQA metric is *lower-better* we multiply the SHAP
values by -1.  Metric polarity is discovered with *pyiqa*.

Three kinds of aggregations are produced:

1.  **Per-metric (within one metric, across all models)**
2.  **Cross-metric (all metrics together, averaging over every CSV seen)**
3.  **Per-model (one model name across all metrics it appeared in)**

Each aggregation is saved as a *sorted* CSV (descending by adjusted mean) and
visualised as a horizontal bar chart (top-*N* key-phrases).  Per-metric files
are written **inside** the corresponding metric directory, cross-metric output
goes to *<output_dir>/global*, and per-model output to *<output_dir>/per_model*.

The file/figure names include *“adjusted”* to emphasise that values were
polarity-corrected.

Run::

    python collect_shap.py compiled_results/ --output_dir shap_plots --top_n 20
"""
from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch  # noqa: F401 – required by pyiqa
import pyiqa

# --------------------------------------------------------------------------- #
# SUPPRESS noisy FutureWarnings                                                #
# (place this right after the standard imports section)                        #
# --------------------------------------------------------------------------- #
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

###############################################################################
# ------------------------------- logging setup ----------------------------- #
###############################################################################
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

###############################################################################
# ----------------------------- helper utilities ---------------------------- #
###############################################################################


def normalize(name: str) -> str:
    """Normalise a string (replace +/- with _, collapse _, lower-case)."""
    name = name.replace("+", "_").replace("-", "_")
    name = re.sub(r"_+", "_", name)
    return name.lower()


def strip_metric_dir_name(metric_dir: Path) -> str:
    """Remove ``iqa_`` prefix and ``_metrics`` suffix, then *normalize*."""
    stem = metric_dir.name
    if stem.startswith("iqa_"):
        stem = stem[len("iqa_") :]
    if stem.endswith("_metrics"):
        stem = stem[: -len("_metrics")]
    return normalize(stem)


_LOWER_BETTER_CACHE: Dict[str, bool] = {}
# --------------------------------------------------------------------------- #
# EXTRA mapping file support + smarter dash/underscore fallback               #
# (update *before* calling metric_lower_better)                               #
# --------------------------------------------------------------------------- #
from json import loads as _json_loads

# --------------------------------------------------------------------------- #
# REBUILD alias mapping so keys match *strip_metric_dir_name()* output        #
# --------------------------------------------------------------------------- #
try:
    _RAW_ALIAS: Dict[str, str] = _json_loads(Path("matched_metrics.json").read_text())
except FileNotFoundError:  # pragma: no cover
    _RAW_ALIAS = {}

_METRIC_ALIAS: Dict[str, str] = {}

for path_str, canonical in _RAW_ALIAS.items():
    # key = directory core name after stripping prefix/suffix + normalise
    core_key = strip_metric_dir_name(Path(path_str))
    _METRIC_ALIAS[core_key] = canonical
    # also allow a 'stripped trailing _' version (handles “clipiqa_”)
    _METRIC_ALIAS[core_key.rstrip("_")] = canonical
    # and the canonical *itself* in normalised form
    _METRIC_ALIAS[normalize(canonical)] = canonical


# --------------------------------------------------------------------------- #
# ENHANCED metric_lower_better() – add more fall-back variants                #
# --------------------------------------------------------------------------- #
def metric_lower_better(metric_name: str) -> bool:
    """
    Determine if *metric_name* is lower-better.

    Tries (in order):
    1. exact              4. remove trailing '_'         7. alias mapping
    2. swap _ ↔ '-'       5. replace trailing '_' → '+'  8. assume higher-better
    3. strip double '__'  6. plus sign directly
    """
    if metric_name in _LOWER_BETTER_CACHE:
        return _LOWER_BETTER_CACHE[metric_name]

    tried: List[str] = []

    def _probe(name: str) -> bool | None:
        if not name or name in tried:
            return None
        tried.append(name)
        try:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return bool(pyiqa.create_metric(name, device=dev).lower_better)
        except Exception:
            return None

    variants = [
        metric_name,
        metric_name.replace("_", "-"),
        metric_name.replace("__", "_"),
        metric_name.rstrip("_"),
        metric_name.rstrip("_") + "+",
        metric_name.replace("_", "+"),
    ]

    for cand in variants:
        res = _probe(cand)
        if res is not None:
            _LOWER_BETTER_CACHE[metric_name] = res
            return res

    # alias mapping (from json)
    alias = _METRIC_ALIAS.get(metric_name)
    if alias:
        res = _probe(alias)
        if res is not None:
            _LOWER_BETTER_CACHE[metric_name] = res
            return res

    logger.warning(
        "Could not create IQA metric %s (tried %s). Assuming higher-better.",
        metric_name,
        tried + ([alias] if alias else []),
    )
    _LOWER_BETTER_CACHE[metric_name] = False
    return False



###############################################################################
# ----------------------- core data-collection logic ------------------------ #
###############################################################################


def read_shap_csv(csv_path: Path, adjust_sign: bool) -> pd.Series:
    """Return Series keyed by *keyphrase*; invert only if data are signed."""
    df = pd.read_csv(csv_path)
    value_col = "mean_abs_shap" if "mean_abs_shap" in df.columns else df.columns[1]
    series = df.set_index("keyphrase")[value_col]

    # Only flip if values are *signed* – abs metrics remain positive
    if adjust_sign and not value_col.endswith("abs_shap"):
        series = -series
    return series


def gather_metric_data(metric_dir: Path) -> Tuple[str, Dict[str, pd.Series]]:
    """Return metric_name and a mapping: *model_name → adjusted SHAP Series*."""
    metric_core = strip_metric_dir_name(metric_dir)    
    lower_better_flag = metric_lower_better(metric_core)
    adjust = lower_better_flag  # multiply by -1 if lower-better

    model_shap: Dict[str, pd.Series] = {}

    for csv_file in metric_dir.glob("*_shap.csv"):
        # Model name is file stem without _shap
        model = csv_file.stem.replace("_shap", "")
        series = read_shap_csv(csv_file, adjust_sign=adjust)
        model_shap[model] = series
        logger.debug("Loaded %s (%d key-phrases, adjust=%s)", csv_file, len(series), adjust)

    if not model_shap:  # pragma: no cover
        logger.warning("No CSV SHAP files found in %s", metric_dir)
    return metric_core, model_shap


###############################################################################
# ----------------------------- aggregation math ---------------------------- #
###############################################################################


def average_over_series(series_list: List[pd.Series]) -> pd.Series:
    """Compute element-wise mean over *series_list* (missing filled per index)."""
    if not series_list:
        return pd.Series(dtype=float)
    combined = pd.concat(series_list, axis=1)  # columns = different models/metrics
    return combined.mean(axis=1, skipna=True)  # average ignoring NaNs

# --------------------------------------------------------------------------- #
# NORMALISED (scaled) aggregation helper                                      #
# --------------------------------------------------------------------------- #
def scaled_average_over_series(series_list: List[pd.Series]) -> pd.Series:
    """Normalise each Series to its own max(|value|) before averaging."""
    if not series_list:
        return pd.Series(dtype=float)
    scaled = []
    for s in series_list:
        denom = s.abs().max()
        scaled.append(s / denom if denom and not pd.isna(denom) else s)
    return average_over_series(scaled)

###############################################################################
# ------------------------------- visual helper ----------------------------- #
###############################################################################


def save_barplot(series: pd.Series, title: str, out_png: Path, top_n: int) -> None:
    """Plot *top_n* entries of *series* as a horizontal bar chart."""
    series = series.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, 0.4 * top_n + 2))
    sns.barplot(
        y=series.index,
        x=series.values,
        orient="h",
        palette="viridis_r",
    )
    plt.title(title)
    plt.xlabel("Adjusted Mean SHAP (positive = key-phrase improves metric)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    logger.info("Saved plot → %s", out_png)


###############################################################################
# ------------------------------- main worker ------------------------------ #
###############################################################################


def process_tree(compiled_root: Path, output_root: Path, top_n: int) -> None:
    """Walk the directory tree, aggregate & visualise adjusted SHAP data."""
    output_root.mkdir(parents=True, exist_ok=True)
    global_dir = output_root / "global"
    per_model_dir = output_root / "per_model"
    global_dir.mkdir(exist_ok=True)
    per_model_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------------------- #
    # 1) load every metric → {metric_name: {model_name: Series}}            #
    # --------------------------------------------------------------------- #
    metric_to_models: Dict[str, Dict[str, pd.Series]] = {}
    for metric_dir in compiled_root.iterdir():
        if not metric_dir.is_dir():
            continue
        metric_name, model_shap = gather_metric_data(metric_dir)
        if model_shap:
            metric_to_models[metric_name] = model_shap

            # ---------------------------------------------------------- #
            # per-metric aggregation (average across models)            #
            # ---------------------------------------------------------- #
            mean_series = average_over_series(list(model_shap.values())).sort_values(ascending=False)
            csv_path = metric_dir / f"{metric_name}_adjusted_per_metric_avg.csv"
            img_path = metric_dir / f"{metric_name}_adjusted_per_metric_top{top_n}.png"
            mean_series.to_csv(csv_path, header=["adjusted_mean_shap"])
            logger.info("Saved per-metric CSV → %s", csv_path)
            save_barplot(
                mean_series,
                title=f"{metric_name}: top {top_n} key-phrases (avg over models, adjusted)",
                out_png=img_path,
                top_n=top_n,
            )

    if not metric_to_models:
        logger.error("No metrics with SHAP CSVs were found under %s", compiled_root)
        return

    # --------------------------------------------------------------------- #
    # 2) cross-metric global average (all models + metrics mixed)           #
    # --------------------------------------------------------------------- #
    all_series: List[pd.Series] = [
        series for models in metric_to_models.values() for series in models.values()
    ]
    global_avg = average_over_series(all_series).sort_values(ascending=False)
    global_csv = global_dir / "global_keyphrase_adjusted_avg.csv"
    global_img = global_dir / f"global_keyphrase_adjusted_top{top_n}.png"
    global_avg.to_csv(global_csv, header=["adjusted_mean_shap"])
    logger.info("Saved global CSV → %s", global_csv)
    save_barplot(
        global_avg,
        title=f"Global top {top_n} key-phrases (avg over *all* metrics & models, adjusted)",
        out_png=global_img,
        top_n=top_n,
    )

    # --------------------------------------------------------------------- #
    # 3) per-model averages across *all* metrics                            #
    # --------------------------------------------------------------------- #
    # transpose metric_to_models into model → list[Series]
    model_to_series: Dict[str, List[pd.Series]] = defaultdict(list)
    for model_dict in metric_to_models.values():
        for mdl, series in model_dict.items():
            model_to_series[mdl].append(series)

    for mdl, series_list in model_to_series.items():
        mdl_avg = average_over_series(series_list).sort_values(ascending=False)
        mdl_csv = per_model_dir / f"{mdl}_adjusted_keyphrase_avg.csv"
        mdl_img = per_model_dir / f"{mdl}_adjusted_top{top_n}.png"
        mdl_avg.to_csv(mdl_csv, header=["adjusted_mean_shap"])
        logger.info("Saved per-model CSV → %s", mdl_csv)
        save_barplot(
            mdl_avg,
            title=f"{mdl}: top {top_n} key-phrases (avg over metrics, adjusted)",
            out_png=mdl_img,
            top_n=top_n,
        )
        
    # ---------- GLOBAL & PER-MODEL *scaled* (normalised per series) -------- #
    global_scaled = scaled_average_over_series(all_series).sort_values(ascending=False)
    g_csv = global_dir / "global_keyphrase_adjusted_avg_scaled.csv"
    g_img = global_dir / f"global_keyphrase_adjusted_scaled_top{top_n}.png"
    global_scaled.to_csv(g_csv, header=["adjusted_scaled_mean_shap"])
    save_barplot(
        global_scaled,
        title=f"Global top {top_n} key-phrases (scaled per series, adjusted)",
        out_png=g_img,
        top_n=top_n,
    )

    for mdl, series_list in model_to_series.items():
        mdl_scaled = scaled_average_over_series(series_list).sort_values(ascending=False)
        m_csv = per_model_dir / f"{mdl}_adjusted_keyphrase_avg_scaled.csv"
        m_img = per_model_dir / f"{mdl}_adjusted_scaled_top{top_n}.png"
        mdl_scaled.to_csv(m_csv, header=["adjusted_scaled_mean_shap"])
        save_barplot(
            mdl_scaled,
            title=f"{mdl}: top {top_n} key-phrases (scaled per series, adjusted)",
            out_png=m_img,
            top_n=top_n,
        )


###############################################################################
# ---------------------------------- CLI ----------------------------------- #
###############################################################################


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Aggregate SHAP key-phrase scores across IQA metrics/models "
        "and visualise adjusted importances (positive = better)."
    )
    parser.add_argument(
        "compiled_results_dir",
        type=Path,
        help="Path to *compiled_results/* directory that holds metric folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("shap_summary_plots"),
        help="Root directory where aggregate CSV/PNG files will be written.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top key-phrases to display in each bar chart.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_tree(args.compiled_results_dir, args.output_dir, args.top_n)


if __name__ == "__main__":
    main()

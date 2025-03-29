# -*- coding: utf-8 -*-

"""
Utilities for the application.
"""

# **** IMPORTS ****
import os
import re
import shap
import json
import joblib
import inspect
import logging
import numpy as np
import pandas as pd
import importlib.util
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Type, List, Any, Dict, Set
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

# **** LOGGING ****
logger = logging.getLogger(__name__)

# **** FUNCTIONS ****
def sanitize_sqlite_identifier(name: str) -> str:
    """Replaces problematic characters in SQLite identifiers with underscores."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def sanitize_classname(name: str) -> str:
    """Sanitizes a string to be a valid Python class name."""
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return name[0].upper() + name[1:] if name else "Unnamed"

def get_image_uuid(file_path: Path) -> str:
    """
    Returns the UUID of the image file.
    The UUID is assumed to be the file's stem (filename without extension).
    Args:
        file_path (Path): The path to the image file.
    Returns:
        str: The UUID of the image (file stem).
    """
    return file_path.stem


def load_image_from_uuid(search_directory: Path, image_uuid: str) -> Optional[Path]:
    """
    Searches through the given directory (recursively) for the first file matching the UUID (stem).
    Args:
        search_directory (Path): The base directory to search in.
        image_uuid (str): The UUID (filename stem) to match.
    Returns:
        Optional[Path]: The first path found with a matching stem, or None if none found.
    """
    logger.debug(f"Searching for image with UUID: {image_uuid} in {search_directory}")
    for candidate in search_directory.rglob("*"):
        if candidate.is_file() and candidate.stem == image_uuid:
            return candidate
    return None


def discover_classes_in_file(file_path: Path, base_class: Type) -> List:
    """
    Discovers any subclasses of a given base class within a single .py file.

    Args:
        file_path (Path): Path to the Python file containing potential class definitions.
        base_class (Type): The base class to search for subclasses.

    Returns:
        List: Instantiated subclasses of the specified base class.
    """
    discovered_classes = []

    if file_path.suffix != ".py":
        return discovered_classes

    module_name = file_path.stem

    # Load the module from the file
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Inspect classes defined in the module
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, base_class) and obj is not base_class:
                logger.info(f"Discovered {obj.__name__}")
                discovered_classes.append(obj)

    return discovered_classes

def load_json_documents(directory: Path) -> List[dict]:
    """
    Recursively loads all JSON documents in a directory and its subdirectories,
    returning them as a list of dictionaries.

    Args:
        directory (Path): Path to the directory containing JSON files.

    Returns:
        List[dict]: List of dictionaries loaded from JSON files.
    """
    documents = []

    for file_path in directory.rglob("*.json"):
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    documents.append(data)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from {file_path}")

    return documents

def combine_dictionaries(dicts: List[dict]) -> dict:
    """
    Combines a list of dictionaries into a single dictionary.

    Args:
        dicts (List[dict]): List of dictionaries to combine.

    Returns:
        dict: A single merged dictionary containing all key-value pairs.
    """
    combined = {}

    for d in dicts:
        combined.update(d)

    return combined

def compute_shap_explanations(texts: List[str], metrics: List[List[float]]) -> Dict[str, Any]:
    """
    Computes SHAP explanations for a set of texts and their corresponding metrics.
    
    Args:
        texts (List[str]): A list of text strings.
        metrics (List[List[float]]): A list of float lists representing numeric metrics.
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - "model": The trained RandomForestRegressor model.
            - "feature_names": The feature names from the TF-IDF vectorizer.
            - "shap_values": The SHAP values for each metric.
            - "feature_importances": A list of pandas Series objects containing mean absolute SHAP values 
              for each metric, indexed by feature names.
    
    Raises:
        ValueError: If the lengths of texts and metrics do not match.
    """
    
    # Validate inputs
    if len(texts) != len(metrics):
        raise ValueError("The number of texts must match the number of metric rows.")
    
    logging.info("Step 1: Vectorizing text data using TF-IDF.")
    vectorizer = TfidfVectorizer()
    X_sparse = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert the sparse matrix to a dense numpy array
    X = X_sparse.toarray().astype(float)
    Y = np.array(metrics)
    
    logging.info("Step 2: Training a multi-output RandomForestRegressor.")
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, Y)
    
    logging.info("Step 3: Computing SHAP values using TreeExplainer.")
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X)
    
    # Convert the list of arrays returned by shap_values into a single 3D array
    shap_values_3d = np.array(shap_values_raw)
    
    logging.info("Step 4: Computing mean absolute SHAP values for each metric.")
    feature_importances = []
    
    # Each metric is the 3rd dimension
    for metric_idx in range(shap_values_3d.shape[-1]):
        shap_values_for_metric = shap_values_3d[:, :, metric_idx]
        mean_abs_shap = np.abs(shap_values_for_metric).mean(axis=0)
        importance_series = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
        feature_importances.append(importance_series)
    
    # Package results into a dictionary
    results = {
        "model": model,
        "feature_names": feature_names,
        "shap_values": shap_values_3d,
        "feature_importances": feature_importances
    }
    
    return results

def save_shap_results(results: Dict[str, Any], output_dir: str, metric_names: Optional[List[str]] = None) -> None:
    """
    Saves SHAP explanation results to the specified output directory.

    Args:
        results (Dict[str, Any]): The dictionary returned by `compute_shap_explanations`.
        output_dir (str): Path to the directory where files will be saved.
        metric_names (Optional[List[str]]): Optional list of metric names to name feature importance files.

    Returns:
        None
    """
    logging.info(f"Saving SHAP results to directory: {output_dir}")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the trained model
    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(results["model"], model_path)
    logging.info(f"Model saved to {model_path}")

    # Save the feature names
    feature_names_path = os.path.join(output_dir, "feature_names.csv")
    pd.Series(results["feature_names"]).to_csv(feature_names_path, index=False)
    logging.info(f"Feature names saved to {feature_names_path}")

    # Save the SHAP values (3D array)
    shap_values_path = os.path.join(output_dir, "shap_values.npy")
    np.save(shap_values_path, results["shap_values"])
    logging.info(f"SHAP values saved to {shap_values_path}")

    # Save the feature importances per metric (sorted by importance)
    for idx, fi in enumerate(results["feature_importances"]):
        sorted_fi = fi.sort_values(ascending=False)
        if metric_names and idx < len(metric_names):
            filename = f"feature_importance_{metric_names[idx]}.csv"
        else:
            filename = f"feature_importance_metric_{idx + 1}.csv"
        fi_path = os.path.join(output_dir, filename)
        sorted_fi.to_csv(fi_path)
        logging.info(f"Sorted feature importance for metric {idx + 1} saved to {fi_path}")



def create_shap_importance_visuals(csv_dir: str, top_n: int = 10) -> None:
    """
    Creates bar chart visualizations for the most important words from SHAP feature importance files.

    Args:
        csv_dir (str): Directory containing the feature importance CSV files.
        top_n (int, optional): Number of top words to display. Defaults to 10.

    Returns:
        None
    """
    logging.info(f"Creating SHAP feature importance visualizations from directory: {csv_dir}")

    # Gather all CSV files that match the naming convention
    csv_files = [f for f in os.listdir(csv_dir)
                 if f.startswith("feature_importance_") and f.endswith(".csv")]
    
    if not csv_files:
        logging.warning("No feature importance CSV files found.")
        return

    for csv_file in csv_files:
        file_path = os.path.join(csv_dir, csv_file)
        logging.info(f"Reading feature importance data from: {file_path}")

        # Load the CSV as a Series (index=feature, values=importance).
        # Squeeze has been removed in newer pandas versions, so we'll do:
        df = pd.read_csv(file_path, index_col=0)
        # We assume a single column of feature importance, so let's grab the first column as a Series.
        fi_series = df.iloc[:, 0]

        # Sort and take the top N
        fi_series = fi_series.sort_values(ascending=False).head(top_n)

        # Create a bar chart
        plt.figure()
        fi_series.plot(kind="barh")
        plt.gca().invert_yaxis()  # So the highest value is on top
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Features in {csv_file}")
        plt.tight_layout()

        # Save the figure
        output_png = os.path.splitext(file_path)[0] + ".png"
        plt.savefig(output_png)
        plt.close()
        logging.info(f"Saved visualization to {output_png}")


def create_shap_heatmap(csv_dir: str, top_n: int = 20) -> None:
    """
    Creates a heatmap of top feature importances across all metrics.

    Args:
        csv_dir (str): Directory containing the feature importance CSV files.
        top_n (int): Number of top words to include across all metrics.

    Returns:
        None
    """
    logging.info("Creating SHAP heatmap across all metrics.")

    csv_files = [f for f in os.listdir(csv_dir)
                 if f.startswith("feature_importance_") and f.endswith(".csv")]

    if not csv_files:
        logging.warning("No feature importance CSV files found.")
        return

    # Aggregate importances into a DataFrame
    importance_frames = []
    metric_names = []

    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(csv_dir, csv_file), index_col=0)
        series = df.iloc[:, 0]
        importance_frames.append(series)
        metric_names.append(os.path.splitext(csv_file)[0])  # use filename as metric label

    all_df = pd.concat(importance_frames, axis=1)
    all_df.columns = metric_names
    all_df.fillna(0, inplace=True)

    # Select top_n words by total importance across all metrics
    top_features = all_df.sum(axis=1).sort_values(ascending=False).head(top_n).index
    filtered_df = all_df.loc[top_features]

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(filtered_df, annot=False, cmap="viridis")
    plt.title(f"Top {top_n} Word Importances Across Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Features")
    plt.tight_layout()

    output_path = os.path.join(csv_dir, "shap_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Heatmap saved to {output_path}")


def compute_average_importance_from_csvs(csv_dir: str, output_csv: Optional[str] = None) -> pd.Series:
    """
    Computes the average feature importance across multiple CSV files in the specified directory.

    Each CSV file is assumed to have exactly two columns (no headers):
        - Column 0: feature name
        - Column 1: feature importance

    Example CSV content:
        ,0
        by,0.007241219747425433
        marble,0.004113825845780783

    In practice, some files might have an empty string for the first row's feature name.
    This function handles that gracefully, but you should ensure each file
    has strictly two columns and no extra trailing commas.

    Steps:
    1) Load each CSV into a DataFrame with columns: ['feature', 'importance'].
    2) Set 'feature' as the index, rename 'importance' to the filename.
    3) Concatenate across columns, fill missing values with 0.
    4) Compute the average across all columns.
    5) Optionally save as a CSV.

    Args:
        csv_dir (str): Directory containing feature importance CSV files.
        output_csv (Optional[str]): If provided, saves the final average Series to the given filepath.

    Returns:
        pd.Series: A Series of average importances, indexed by feature name.
    """
    logging.info(f"Computing average feature importance from CSV files in {csv_dir}.")
    
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    if not csv_files:
        logging.warning("No CSV files found in the directory.")
        return pd.Series(dtype=float)

    dfs = []
    for csv_file in csv_files:
        file_path = os.path.join(csv_dir, csv_file)
        logging.info(f"Loading CSV file: {file_path}")

        try:
            # Explicitly declare two columns, no header
            # Note: We use usecols to avoid parsing extra columns (if trailing commas exist).
            df = pd.read_csv(
                file_path,
                header=None,
                names=["feature", "importance"],
                usecols=[0, 1]  # ensures exactly two columns
            )
        except Exception as e:
            logging.error(f"Failed to read {csv_file}: {e}")
            continue

        if df.empty:
            logging.warning(f"File {csv_file} is empty or invalid.")
            continue

        # Set "feature" as the index
        df.set_index("feature", inplace=True)

        # We'll store the importance values under a column named after the file
        df.columns = [csv_file]

        # Ensure numeric
        df[csv_file] = pd.to_numeric(df[csv_file], errors="coerce").fillna(0)

        # df now has shape (n_features_in_this_file, 1)
        dfs.append(df)

    if not dfs:
        logging.warning("No valid CSV files to process after reading.")
        return pd.Series(dtype=float)

    # Concatenate across columns => shape (sum_of_features, number_of_csvs)
    combined_df = pd.concat(dfs, axis=1)

    # Fill missing features with 0
    combined_df.fillna(0, inplace=True)

    # Compute the average across all CSV columns
    avg_series = combined_df.mean(axis=1).sort_values(ascending=False)

    # Optionally save to CSV
    if output_csv:
        avg_series.to_csv(output_csv, header=["average_importance"])
        logging.info(f"Saved average feature importance to {output_csv}")

    return avg_series





















def plot_average_importance_heatmap(
    avg_importance: pd.Series,
    top_n: int = 20,
    output_path: str = "average_shap_importance_heatmap.png"
) -> None:
    """
    Generates a heatmap from a Series of average SHAP feature importances.

    Args:
        avg_importance (pd.Series): A Series of feature importances, indexed by feature name.
        top_n (int): Number of top features to display.
        output_path (str): File path for saving the heatmap image.

    Returns:
        None
    """
    logging.info(f"Generating heatmap for top {top_n} average SHAP features.")
    
    # Select the top N features
    top_avg_importance = avg_importance.head(top_n)
    
    # Convert to a DataFrame for plotting
    df = pd.DataFrame(top_avg_importance, columns=["Average SHAP Importance"])

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.T, annot=False, cmap="rocket", cbar_kws={"label": "Importance"})
    plt.title(f"Top {top_n} Features by Average SHAP Importance")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()
    logging.info(f"Average SHAP importance heatmap saved to {output_path}")

































def filter_common_metrics(data: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Filters out metrics that are not present in every UUID entry in the dataset.

    Args:
        data (Dict[str, Dict[str, float]]): Dictionary mapping UUIDs to dictionaries of metric names and values.

    Returns:
        Dict[str, Dict[str, float]]: Filtered dictionary where each UUID only retains metrics present in all UUIDs.
    """
    # Step 1: Get the intersection of all metric keys across UUIDs
    common_keys: Set[str] = set(next(iter(data.values())).keys())  # Start with first UUID's keys
    for uuid, metrics in data.items():
        common_keys &= set(metrics.keys())  # Keep only keys present in all entries

    # Step 2: Filter each UUID's metric dictionary to only keep the common keys
    filtered_data: Dict[str, Dict[str, float]] = {}
    for uuid, metrics in data.items():
        filtered_data[uuid] = {k: v for k, v in metrics.items() if k in common_keys}

    return filtered_data

# ****
if __name__ == "__main__":
    raise RuntimeError("This module is not meant to be run directly.")

# -*- coding: utf-8 -*-

"""
Model training pipeline for prompt quality prediction.
"""

# ---------------------------------
# Imports
# ---------------------------------
import re
import json
import logging
import logging.config
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from datetime import datetime
import spacy
from keybert import KeyBERT
from pathlib import Path
import sqlite3
import joblib
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import torch
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import QuantileTransformer

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# **** LOCAL IMPORTS ****
from cse5525_final_project.util import (
    load_json_documents,
    combine_dictionaries,
)
from cse5525_final_project.config import (
    LOGGER_CONFIG,
    DB_PATH,
    IMAGES_DIR,
    DATA_DIR,
    RESULTS_DIR
)

# **** LOGGING SETUP ****
logging.config.dictConfig(LOGGER_CONFIG)
logging.getLogger().setLevel(logging.INFO)
# Silence SHAP / explain
for name in ("explain", "shap", "shap.explain"):
    log = logging.getLogger(name)
    log.setLevel(logging.WARNING)               # drop INFO
    log.propagate = False                       # <-- critical
logger = logging.getLogger(__name__)


###############################################################################
#                              DATA LOADING                                   #
###############################################################################
def load_metric_with_prompts(
    db_path: Path,
    metric_table_name: str,
    value_col: str = "value"
) -> pd.DataFrame:
    """
    Load a single metric table and combine with prompts from JSON files.

    Args:
        db_path (Path): Path to the SQLite database.
        metric_table_name (str): Name of the table with UUIDs and metric values.
        value_col (str): Name of the column containing the metric values (default: 'value').

    Returns:
        pd.DataFrame: DataFrame with columns 'uuid', 'prompt', and the metric.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    # Load JSON prompt data
    json_docs = load_json_documents(IMAGES_DIR)
    if not json_docs:
        raise FileNotFoundError(f"No JSON files found in {IMAGES_DIR}")
    combined_dict = combine_dictionaries(json_docs)

    # Load the metric table
    query = f"SELECT uuid, {value_col} FROM {metric_table_name}"
    with sqlite3.connect(db_path) as conn:
        df_metric = pd.read_sql_query(query, conn)

    # Add prompts based on uuid
    df_metric["prompt"] = df_metric["uuid"].apply(
        lambda u: combined_dict.get(f"{u}.png", {}).get("p", None)
    )

    # Optional: drop rows with missing prompts
    df_metric.dropna(subset=["prompt"], inplace=True)

    return df_metric

###############################################################################
#                           FEATURE ENGINEERING                               #
###############################################################################
# Load spaCy model
nlp = spacy.load("en_core_web_sm")
def count_modifiers_and_entities(prompt: str) -> Dict[str, Any]:
    doc = nlp(prompt)
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    named_entities = [ent.text for ent in doc.ents]

    return {
        'num_adjectives': len(adjectives),
        'num_named_entities': len(named_entities),
        # 'adjectives': adjectives,
        # 'named_entities': named_entities,
    }

def calculate_complexity(prompt: str) -> Dict[str, Any]:
    words = re.findall(r'\b\w+\b', prompt)
    word_count = len(words)
    unique_word_count = len(set(words))
    comma_count = prompt.count(',')

    complexity_score = (
        0.3 * word_count +
        0.5 * unique_word_count +
        0.2 * comma_count
    )

    return {
        'word_count': word_count,
        'unique_word_count': unique_word_count,
        'comma_count': comma_count,
        'complexity_score': round(complexity_score, 2)
    }

def add_prompt_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds prompt-derived features to the given DataFrame.
    Assumes a 'prompt' column exists.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with additional columns for prompt features.
    """
    logging.info("Extracting prompt features...")

    # Extract complexity features
    complexity_features = df["prompt"].apply(calculate_complexity).apply(pd.Series)

    # Extract modifier/entity features
    modifier_features = df["prompt"].apply(count_modifiers_and_entities).apply(pd.Series)

    # Combine into original dataframe
    df = pd.concat([df, complexity_features, modifier_features], axis=1)
    return df

_CACHE_FILE = Path("prompts_keywords.feather")
def attach_keywords(
    df: pd.DataFrame,
    text_col: str = "prompt",
    ngram_range: Tuple[int, int] = (1, 2),
    top_n: int = 10,
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.Series:
    """Return a Series of cached / newly‑extracted KeyBERT keywords.

    If new prompts are encountered, they are extracted once, appended to the
    on‑disk cache, and the cache is persisted.

    Args:
        df: DataFrame that contains a ``text_col`` column.
        text_col: Name of the column with the raw prompt text.
        ngram_range: ``keyphrase_ngram_range`` passed to KeyBERT.
        top_n: ``top_n`` phrases returned per prompt.
        model_name: Sentence‑transformer model used by KeyBERT.

    Returns:
        pd.Series: Space‑joined keywords aligned with ``df.index``.
    """
    # ── 1 load existing cache ──────────────────────────────────────────────
    if _CACHE_FILE.exists():
        cache_df = pd.read_feather(_CACHE_FILE)
        cache: dict[str, str] = dict(zip(cache_df["prompt"], cache_df["keywords"]))
    else:
        cache = {}

    # ── 2 process unseen prompts ───────────────────────────────────────────
    unseen: List[str] = [p for p in df[text_col].tolist() if p not in cache]
    if unseen:
        logger.info("KeyBERT: extracting %d unseen prompt(s)…", len(unseen))
        kw_model = KeyBERT(model_name)
        for prompt in unseen:
            phrases = kw_model.extract_keywords(
                prompt,
                keyphrase_ngram_range=ngram_range,
                stop_words="english",
                top_n=top_n,
            )
            cache[prompt] = " ".join([kw for kw, _ in phrases])

        # persist expanded cache
        pd.DataFrame({"prompt": list(cache.keys()),
                      "keywords": list(cache.values())}
        ).to_feather(_CACHE_FILE)
        logger.info("KeyBERT cache now holds %d prompt(s).", len(cache))

    # ── 3 return Series aligned to df ──────────────────────────────────────
    return pd.Series([cache[p] for p in df[text_col]], index=df.index, name="keywords")



###############################################################################
#                             PREPROCESSING                                   #
###############################################################################
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to remove duplicates from.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    initial_shape = df.shape
    df = df.drop_duplicates()
    final_shape = df.shape
    logger.info(f"Removed {initial_shape[0] - final_shape[0]} duplicates")
    return df

def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Detect outliers using the IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (series < lower_bound) | (series > upper_bound)

def detect_spike_at_cap(series: pd.Series, cap_value: float, threshold_ratio: float = 0.01) -> bool:
    """
    Detects if capping caused a spike at the upper bound.
    """
    total = len(series)
    spike_count = (series == cap_value).sum()
    spike_ratio = spike_count / total
    return spike_ratio > threshold_ratio

def choose_imputation_strategy(series: pd.Series, outlier_mask: pd.Series) -> Tuple[str, Optional[float]]:
    """
    Choose best imputation strategy based on skewness and outlier %.
    """
    skewness = series.skew()
    outlier_ratio = outlier_mask.sum() / len(series)

    logger.info(f"{series.name} - Skew: {skewness:.2f}, Outlier ratio: {outlier_ratio:.2%}")

    if outlier_ratio < 0.01:
        return "none", None
    elif outlier_ratio < 0.05:
        return "cap", None
    elif abs(skewness) > 1:
        return "median", series.median()
    else:
        return "mean", series.mean()

def handle_outliers_and_impute(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Systematically handle outliers and correct capping-induced spikes.
    """
    df_result = df.copy()

    for col in columns:
        if df_result[col].dtype.kind not in 'biufc':
            continue

        series = df_result[col]
        outlier_mask = detect_outliers_iqr(series)
        strategy, value = choose_imputation_strategy(series, outlier_mask)

        if strategy == "cap":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            capped = series.clip(lower=lower_bound, upper=upper_bound)

            if detect_spike_at_cap(capped, upper_bound):
                # Spike detected, switch to imputation instead
                value = series.median() if abs(series.skew()) > 1 else series.mean()
                df_result.loc[outlier_mask, col] = value
                logger.info(f"{col}: Spike at cap detected, switched to {('median' if abs(series.skew()) > 1 else 'mean')} imputation")
            else:
                df_result[col] = capped
                logger.info(f"{col}: Capped to IQR bounds")
        elif strategy in {"mean", "median"}:
            df_result.loc[outlier_mask, col] = value
            logger.info(f"{col}: Imputed {outlier_mask.sum()} outliers with {strategy} = {value:.2f}")
        elif strategy == "none":
            logger.info(f"{col}: No action taken")
        else:
            logger.warning(f"{col}: Unknown strategy {strategy}")

    return df_result

def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Logs and drops constant columns from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to inspect and modify.

    Returns:
        pd.DataFrame: A copy of the DataFrame with constant columns removed.
    """
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    logger.info(f"Found {len(constant_cols)} constant column(s).")
    if constant_cols:
        logger.info(f"Dropping constant columns: {constant_cols}")
    return df.drop(columns=constant_cols)

def scale_continuous_columns(
    df: pd.DataFrame, blacklist: Optional[List[str]] = None
) -> pd.DataFrame:
    """Scales continuous numerical columns in the DataFrame using StandardScaler,
    excluding any columns specified in the blacklist.

    Args:
        df (pd.DataFrame): The input DataFrame with numerical features.
        blacklist (Optional[List[str]]): List of column names to exclude from scaling (e.g., encoded categorical codes).

    Returns:
        pd.DataFrame: A copy of the DataFrame with scaled continuous columns.
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Identify binary columns (likely already normalized or categorical)
    binary_cols = [col for col in numeric_cols if df[col].nunique() == 2]

    # Define continuous columns as numeric, non-binary, and not blacklisted
    if blacklist is None:
        blacklist = []
    continuous_cols = [
        col for col in numeric_cols
        if col not in binary_cols and col not in blacklist
    ]

    # Apply scaling
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[continuous_cols] = scaler.fit_transform(df[continuous_cols])
    logger.info(f"Scaled continuous columns: {continuous_cols}")

    return df_scaled

def drop_sparse_binary_columns(df: pd.DataFrame, sparse_threshold: float = 0.01) -> pd.DataFrame:
    """
    Drop binary columns with low or high presence of 1s.

    Args:
        df (pd.DataFrame): Input DataFrame.
        sparse_threshold (float): Threshold to consider a binary column sparse.

    Returns:
        pd.DataFrame: Modified DataFrame with sparse binary columns dropped.
    """
    cols_to_drop = []
    for col in df.columns:
        if df[col].nunique() == 2:
            p = df[col].mean()
            if p < sparse_threshold or p > 1 - sparse_threshold:
                cols_to_drop.append(col)

    logger.info(f"Dropping sparse binary columns: {cols_to_drop}")
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df

def drop_leaky_columns(df: pd.DataFrame, target: pd.Series, correlation_threshold: float = 0.98) -> pd.DataFrame:
    """
    Drop columns that are highly correlated with the target (excluding the target itself).

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (pd.Series): Target variable.
        correlation_threshold (float): Correlation threshold for leakage.

    Returns:
        pd.DataFrame: Modified DataFrame with leaky columns dropped.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corrwith(target).abs()
    leaky_cols = correlations[(correlations > correlation_threshold) & (correlations.index != target.name)].index.tolist()

    logger.info(f"Dropping leaky columns: {leaky_cols}")
    df.drop(columns=leaky_cols, inplace=True, errors='ignore')
    return df

def collapse_rare_one_hot_columns(df: pd.DataFrame, sparse_threshold: float = 0.01) -> pd.DataFrame:
    """
    Collapse rare one-hot columns into a single 'category_other' column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        sparse_threshold (float): Minimum frequency to be considered common.

    Returns:
        pd.DataFrame: Modified DataFrame with rare one-hot columns collapsed.
    """
    from collections import defaultdict

    ohe_candidates = [
        col for col in df.columns if df[col].nunique() == 2 and df[col].max() == 1 and df[col].min() == 0
    ]

    # Group by prefix
    groups = defaultdict(list)
    for col in ohe_candidates:
        parts = col.split("_")
        if len(parts) >= 2:
            prefix = "_".join(parts[:-1])
            groups[prefix].append(col)

    for group_cols in groups.values():
        col_sums = df[group_cols].sum()
        rare_cols = col_sums[col_sums < sparse_threshold * len(df)].index.tolist()

        if rare_cols:
            existing = [col for col in rare_cols if col in df.columns]
            if existing:
                logger.info(f"Collapsing rare one-hot columns into 'category_other': {existing}")
                df["category_other"] = df.get("category_other", 0) + df[existing].sum(axis=1).clip(upper=1)
                df.drop(columns=existing, inplace=True)

    return df

def apply_quantile_transform_to_skewed(
    df: pd.DataFrame,
    skew_threshold: float = 1.0
) -> pd.DataFrame:
    """
    Apply QuantileTransformer to skewed numerical columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        skew_threshold (float): Absolute skewness above which transformation is applied.

    Returns:
        pd.DataFrame: Transformed dataframe with quantile normalization applied to skewed columns.
    """
    df_transformed = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    transformed_cols = []

    for col in numeric_cols:
        skew = df[col].skew()
        logger.debug(f"Skewness for column '{col}': {skew:.2f}")
        if abs(skew) > skew_threshold:
            qt = QuantileTransformer(output_distribution="normal", random_state=42)
            df_transformed[col] = qt.fit_transform(df[[col]])
            transformed_cols.append(col)
            logger.info(f"Applied QuantileTransformer to skewed column: '{col}' (skewness = {skew:.2f})")

    if not transformed_cols:
        logger.info("No columns exceeded skew threshold. No quantile transformation applied.")

    return df_transformed


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_duplicates(df)
    # TODO: Handle missing values like outliers
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        raise ValueError("DataFrame contains missing values. Please handle them before proceeding.")
    df = handle_outliers_and_impute(df, df.columns.tolist())
    df = drop_constant_columns(df)
    df = scale_continuous_columns(df)
    df = drop_sparse_binary_columns(df, sparse_threshold=0.01)
    df = drop_leaky_columns(df, df["value"], correlation_threshold=0.98)
    df = collapse_rare_one_hot_columns(df, sparse_threshold=0.01)
    df = apply_quantile_transform_to_skewed(df, skew_threshold=1.0)
    return df

###############################################################################
#                             BERT EMBEDDER                                   #
###############################################################################
class BertCLSEmbedder:
    """Lightweight util to obtain 768‑D [CLS] embeddings from a BERT model.

    This class is *not* an sklearn transformer to avoid heavy pickling
    overhead; instead, embedding is done inside a FunctionTransformer wrapper
    in the pipeline.

    Attributes
    ----------
    model_name : str
        HuggingFace checkpoint to load.
    max_len : int
        Maximum token length (padding / truncation).
    device : str
        `'cuda'` if available else `'cpu'`.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_len: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_len = max_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None

    # --------------------------- internals -------------------------------- #

    def _lazy_load(self) -> None:
        if self._tokenizer is None or self._model is None:
            logger.info("Loading BERT (%s) on %s", self.model_name, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

    # --------------------------- public API ------------------------------- #

    def __call__(self, prompts: pd.Series) -> np.ndarray:
        """Vectorise a batch of prompts into CLS embeddings."""
        self._lazy_load()
        assert self._tokenizer and self._model
        batch_size = 32
        all_vecs: List[np.ndarray] = []

        for i in tqdm(
            range(0, len(prompts), batch_size), desc="Embedding prompts"
        ):
            batch = prompts.iloc[i : i + batch_size].tolist()
            enc = self._tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                out = self._model(**enc).last_hidden_state[:, 0, :].cpu().numpy()
            all_vecs.append(out)

        return np.vstack(all_vecs)
    
###############################################################################
#                             BERT EMBEDDER                                   #
###############################################################################
class BertCLSEmbedder:
    """Lightweight util to obtain 768‑D [CLS] embeddings from a BERT model.

    This class is *not* an sklearn transformer to avoid heavy pickling
    overhead; instead, embedding is done inside a FunctionTransformer wrapper
    in the pipeline.

    Attributes
    ----------
    model_name : str
        HuggingFace checkpoint to load.
    max_len : int
        Maximum token length (padding / truncation).
    device : str
        `'cuda'` if available else `'cpu'`.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_len: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_len = max_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None

    # --------------------------- internals -------------------------------- #

    def _lazy_load(self) -> None:
        if self._tokenizer is None or self._model is None:
            logger.info("Loading BERT (%s) on %s", self.model_name, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

    # --------------------------- public API ------------------------------- #

    def __call__(self, prompts: pd.Series) -> np.ndarray:
        """Vectorise a batch of prompts into CLS embeddings."""
        self._lazy_load()
        assert self._tokenizer and self._model
        batch_size = 32
        all_vecs: List[np.ndarray] = []

        for i in tqdm(
            range(0, len(prompts), batch_size), desc="Embedding prompts"
        ):
            batch = prompts.iloc[i : i + batch_size].tolist()
            enc = self._tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                out = self._model(**enc).last_hidden_state[:, 0, :].cpu().numpy()
            all_vecs.append(out)

        return np.vstack(all_vecs)


###############################################################################
#                          BASE PIPELINE RUNNER                               #
###############################################################################
class BasePipelineRunner:
    """
    Base class for pipeline model runners. Subclasses set self.param_grid
    and implement _build_pipeline to return a Pipeline object.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.param_grid: Dict[str, Any] = {}
        self._best_estimator = None

    def _build_pipeline(self) -> Pipeline:
        raise NotImplementedError("Subclasses must implement _build_pipeline.")

    def fit(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Fits a GridSearchCV using the pipeline and self.param_grid,
        then evaluates on the test set.
        """
        from sklearn.model_selection import GridSearchCV, KFold
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        self.logger.info("Fitting pipeline with GridSearchCV.")
        pipeline = self._build_pipeline()

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=self.param_grid,
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        self._best_estimator = grid_search.best_estimator_
        self.logger.info("Best Params: %s", grid_search.best_params_)

        y_pred = self._best_estimator.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            # "best_estimator": self._best_estimator,
            # "y_true": y_test,
            # "y_pred": y_pred,
            "mse": mse,
            "mae": mae,
            "r2": r2
        }
        
###############################################################################
#                     ANN (MLP) PIPELINE RUNNER                               #
###############################################################################
class ANNPipelineRunner(BasePipelineRunner):
    """
    Pipeline runner for an ANN (MLPRegressor) with no preprocessing step.
    Assumes input DataFrame is already preprocessed.
    """

    def __init__(self):
        super().__init__()
        self.param_grid = {
            "model__hidden_layer_sizes": [(64, 64)],
            "model__activation": ["relu"],
            "model__solver": ["adam"],
            "model__alpha": [0.0001, 0.001, 0.00001],
            "model__learning_rate": ["constant", "adaptive"]
        }

    def _build_pipeline(self) -> Pipeline:
        steps = [
            ("model", MLPRegressor(random_state=42, max_iter=1000))
        ]
        return Pipeline(steps=steps)

###############################################################################
#                  ADDITIONAL REGRESSION PIPELINE RUNNERS                     #
###############################################################################
class RandomForestPipelineRunner(BasePipelineRunner):
    """Pipeline runner for a RandomForestRegressor on preprocessed data."""

    def __init__(self) -> None:
        super().__init__()
        # Hyper‑parameter grid to explore
        self.param_grid = {
            # "model__n_estimators": [200, 500],
            "model__n_estimators": [200],
            # "model__max_depth": [None, 30, 60],
            "model__max_depth": [None],
            # "model__min_samples_split": [2, 4],
            "model__min_samples_split": [2],
        }

    def _build_pipeline(self) -> Pipeline:
        from sklearn.ensemble import RandomForestRegressor

        steps = [
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
        ]
        return Pipeline(steps=steps)


class GradientBoostingPipelineRunner(BasePipelineRunner):
    """Pipeline runner for a GradientBoostingRegressor on preprocessed data."""

    def __init__(self) -> None:
        super().__init__()
        self.param_grid = {
            # "model__n_estimators": [300, 600],
            "model__n_estimators": [300],
            # "model__learning_rate": [0.05, 0.1],
            "model__learning_rate": [0.05],
            # "model__max_depth": [2, 3],
            "model__max_depth": [2],
        }

    def _build_pipeline(self) -> Pipeline:
        from sklearn.ensemble import GradientBoostingRegressor

        steps = [
            ("model", GradientBoostingRegressor(random_state=42))
        ]
        return Pipeline(steps=steps)


class SVRPipelineRunner(BasePipelineRunner):
    """Pipeline runner for an SVR with feature scaling."""

    def __init__(self) -> None:
        super().__init__()
        self.param_grid = {
            "model__C": [1.0, 10.0],
            # "model__gamma": ["scale", "auto"],
            "model__gamma": ["scale"],
            # "model__kernel": ["rbf", "poly"],
            "model__kernel": ["rbf"],
        }

    def _build_pipeline(self) -> Pipeline:
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        steps = [
            # Scale is important for SVR convergence
            # ("scaler", StandardScaler()),
            ("model", SVR())
        ]
        return Pipeline(steps=steps)


###############################################################################
#                GENERIC SUPERVISED MODEL RUNNER FUNCTION                     #
###############################################################################
def run_supervised_models(
    df: pd.DataFrame,
    runner_classes: Optional[List[type[BasePipelineRunner]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Execute multiple pipeline runners on the same feature matrix.

    Args:
        df (pd.DataFrame): Input dataframe (already pre‑processed).
        runner_classes (Optional[List[type[BasePipelineRunner]]]): List of
            runner subclasses to execute. Defaults to a sensible set.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping from model name to evaluation metrics.
    """
    import logging
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    logger = logging.getLogger(__name__)
    logger.info("Running supervised pipelines for multiple models.")

    # --------------------------- data prep -------------------------------- #
    y = df["value"]

    # Embed prompts once to reuse across models
    logger.info("Embedding prompts via BERT (shared across models).")
    embedder = BertCLSEmbedder()
    X_text = embedder(df["prompt"])

    # Scale structured numeric columns once
    structured_cols = [
        "word_count",
        "unique_word_count",
        "comma_count",
        "complexity_score",
        "num_adjectives",
        "num_named_entities",
    ]
    logger.info("Scaling structured numeric features.")
    scaler = StandardScaler()
    X_struct = scaler.fit_transform(df[structured_cols])

    # Concatenate into final feature matrix
    X = np.hstack([X_text, X_struct])

    # Consistent split for fair comparison
    logger.info("Performing single train/test split (shared).")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------- define runners ----------------------------- #
    if runner_classes is None:
        runner_classes = [
            ANNPipelineRunner,
            RandomForestPipelineRunner,
            GradientBoostingPipelineRunner,
            SVRPipelineRunner,
        ]

    results: Dict[str, Dict[str, Any]] = {}

    # ------------------------- run each model ----------------------------- #
    for runner_cls in runner_classes:
        model_name = runner_cls.__name__.replace("PipelineRunner", "")
        logger.info("=== Fitting %s model ===", model_name)
        runner = runner_cls()
        results[model_name] = runner.fit(X_train, X_test, y_train, y_test)
        logger.info(
            "%s results — MSE: %.4f | MAE: %.4f | R²: %.4f",
            model_name,
            results[model_name]["mse"],
            results[model_name]["mae"],
            results[model_name]["r2"],
        )

    return results


###############################################################################
#  REVISED  KeyBERT_TFIDFPipelineRunner  – uses the cached attach_keywords    #
###############################################################################
class KeyBERT_TFIDFPipelineRunner(BasePipelineRunner):
    """Pipeline runner using cached KeyBERT tokens + TF‑IDF vectorization."""

    def __init__(self) -> None:
        super().__init__()
        self.param_grid = {
            "vectorizer__max_features": [300],
            "model__alpha": [1e-4, 1e-3],
            "model__hidden_layer_sizes": [(64, 64)],
            "model__activation": ["relu"],
            "model__solver": ["adam"],
            "model__learning_rate": ["constant", "adaptive"],
        }

    # ▼ this now loads from the shared cache instead of full extraction
    def _extract_keywords(self, prompts: pd.Series) -> List[str]:
        """Return cached keywords aligned with *prompts* index."""
        df_prompts = prompts.to_frame(name="prompt")
        return attach_keywords(df_prompts).tolist()

    def _build_pipeline(self) -> Pipeline:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neural_network import MLPRegressor

        return Pipeline(
            steps=[
                ("vectorizer", TfidfVectorizer()),
                ("model", MLPRegressor(max_iter=1000, random_state=42)),
            ]
        )


###############################################################################
#            CROSS‑VALIDATED  KeyBERT→TFIDF→MLP  WITH SHAP                    #
###############################################################################
def run_unsupervised_keybert_model(
    df: pd.DataFrame,
    output_dir: Path,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """5‑fold CV evaluation of the KeyBERT‑TFIDF‑MLP pipeline + SHAP dump."""
    from sklearn.model_selection import KFold
    import numpy as np

    y = df["value"].reset_index(drop=True)
    keywords = attach_keywords(df)  # cached extraction
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    metrics: Dict[str, List[float]] = {"mse": [], "mae": [], "r2": []}
    last_runner: Optional[KeyBERT_TFIDFPipelineRunner] = None

    for fold, (tr_idx, te_idx) in enumerate(kf.split(keywords)):
        X_train = keywords.iloc[tr_idx].tolist()
        X_test = keywords.iloc[te_idx].tolist()
        y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]

        runner = KeyBERT_TFIDFPipelineRunner()
        res = runner.fit(X_train, X_test, y_train, y_test)
        for k in metrics:
            metrics[k].append(res[k])
        last_runner = runner

    # ── aggregate ──────────────────────────────────────────────────────────
    agg = {k: float(np.mean(v)) for k, v in metrics.items()}

    # ── SHAP on full fit ───────────────────────────────────────────────────
    assert last_runner is not None
    pipeline = last_runner._best_estimator           # vectorizer + model
    vec = pipeline.named_steps["vectorizer"]
    X_full = vec.transform(keywords.tolist())
    record_shap_tokens(
        model=pipeline.named_steps["model"],
        X=X_full,
        feature_names=vec.get_feature_names_out().tolist(),
        save_stem=output_dir / "KeyBERT_TFIDF_MLP_shap",
    )

    return agg


###############################################################################
#       CROSS‑VALIDATED UNSUPERVISED MODELS  +  TOKEN‑LEVEL SHAP              #
###############################################################################
def run_unsupervised_models(
    df: pd.DataFrame,
    output_dir: Path,
    runner_classes: Optional[List[type[BasePipelineRunner]]] = None,
    n_splits: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """5‑fold CV on TF‑IDF keyword features for multiple regressors + SHAP."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import KFold
    import numpy as np

    y = df["value"].reset_index(drop=True)
    keywords = attach_keywords(df)
    tfidf = TfidfVectorizer(max_features=300)
    X_all = tfidf.fit_transform(keywords.tolist())

    if runner_classes is None:
        runner_classes = [
            RandomForestPipelineRunner,
            GradientBoostingPipelineRunner,
            SVRPipelineRunner,
        ]

    results: Dict[str, Dict[str, Any]] = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for runner_cls in runner_classes:
        name = runner_cls.__name__.replace("PipelineRunner", "")
        logger.info("=== %s : %d‑fold CV ===", name, n_splits)

        fold_metrics: Dict[str, List[float]] = {"mse": [], "mae": [], "r2": []}
        last_runner: Optional[BasePipelineRunner] = None

        for tr_idx, te_idx in kf.split(X_all):
            X_tr, X_te = X_all[tr_idx], X_all[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            runner = runner_cls()
            res = runner.fit(X_tr, X_te, y_tr, y_te)
            for k in fold_metrics:
                fold_metrics[k].append(res[k])
            last_runner = runner

        # ── aggregate & store ───────────────────────────────────────────────
        results[name] = {k: float(np.mean(v)) for k, v in fold_metrics.items()}

        # ── SHAP on model trained on full data ──────────────────────────────
        assert last_runner is not None
        full_runner = runner_cls()
        _ = full_runner.fit(X_all, X_all, y, y)          # fit on all data
        record_shap_tokens(
            model=full_runner._best_estimator,
            X=X_all,
            feature_names=tfidf.get_feature_names_out().tolist(),
            save_stem=output_dir / f"{name}_shap",
        )

    # ── KeyBERT‑TFIDF‑MLP run (handled separately) ─────────────────────────
    results["KeyBERT"] = run_unsupervised_keybert_model(
        df=df, output_dir=output_dir, n_splits=n_splits
    )
    return results




###############################################################################
#                       SHAP UTILITIES FOR TOKENS                             #
###############################################################################
def record_shap_tokens(
    model,
    X,
    feature_names: List[str],
    save_stem: Path,
    max_background: int = 100,
    max_display: int = 20,
) -> None:
    """Compute and save SHAP values + beeswarm plot for token features.

    Args:
        model: Fitted regression model (no vectorizer).
        X: Feature matrix used for prediction (dense or sparse).
        feature_names: List of token / feature names (length = X.shape[1]).
        save_stem:   Path stem like ``…/my_metric/KeyBERT`` (no suffix!).
        max_background:  Background sample size for KernelExplainer.
        max_display:     Top‑k tokens to display in the beeswarm plot.
    """
    import numpy as np
    import shap
    import matplotlib.pyplot as plt

    # ── ensure dense ndarray ───────────────────────────────────────────────
    if not isinstance(X, np.ndarray):
        X_dense = X.toarray()
    else:
        X_dense = X

    # ── pick best explainer ────────────────────────────────────────────────
    if hasattr(model, "estimators_"):              # forests / boosting
        explainer = shap.TreeExplainer(model)
    else:                                          # MLP, SVR, etc.
        bg_idx = np.random.choice(
            X_dense.shape[0], min(max_background, X_dense.shape[0]), replace=False
        )
        explainer = shap.KernelExplainer(model.predict, X_dense[bg_idx])

    # ── compute SHAP values ────────────────────────────────────────────────
    shap_values = explainer.shap_values(X_dense, nsamples=max_background)

    # ── persist raw values ─────────────────────────────────────────────────
    np.save(save_stem.with_suffix(".npy"), shap_values)

    # ── beeswarm summary plot ──────────────────────────────────────────────
    shap.summary_plot(
        shap_values,
        X_dense,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(save_stem.with_suffix(".png"))
    plt.close()




###############################################################################
#                             MAIN FUNCTION                                   #
###############################################################################
def main():
    metric_table_names = [
        'iqa_arniqa_metrics',
        'iqa_arniqa_clive_metrics',
        'iqa_arniqa_csiq_metrics',
        'iqa_arniqa_flive_metrics',
        'iqa_arniqa_kadid_metrics',
        'iqa_arniqa_live_metrics',
        'iqa_arniqa_spaq_metrics',
        'iqa_arniqa_tid_metrics',
        'iqa_brisque_metrics',
        'iqa_brisque_matlab_metrics',
        'iqa_clipiqa_metrics',
        'iqa_clipiqa__metrics',
        'iqa_clipiqa__rn50_512_metrics',
        'iqa_clipiqa__vitL14_512_metrics',
        'iqa_cnniqa_metrics',
        'iqa_dbcnn_metrics',
        'iqa_entropy_metrics',
        'iqa_hyperiqa_metrics',
        'iqa_ilniqe_metrics',
        'iqa_laion_aes_metrics',
        'iqa_liqe_metrics',
        'iqa_liqe_mix_metrics',
        'iqa_maniqa_metrics',
        'iqa_maniqa_kadid_metrics',
        'iqa_maniqa_pipal_metrics',
        'iqa_musiq_metrics',
        'iqa_musiq_ava_metrics',
        'iqa_musiq_paq2piq_metrics',
        'iqa_musiq_spaq_metrics',
        'iqa_nima_metrics',
        'iqa_nima_koniq_metrics',
        'iqa_nima_spaq_metrics',
        'iqa_nima_vgg16_ava_metrics',
        'iqa_niqe_metrics',
        'iqa_niqe_matlab_metrics',
        'iqa_nrqm_metrics',
        'iqa_paq2piq_metrics',
        'iqa_pi_metrics',
        'iqa_piqe_metrics',
        'iqa_topiq_iaa_metrics',
        'iqa_topiq_iaa_res50_metrics',
        'iqa_topiq_nr_metrics',
        # 'iqa_topiq_nr_face_metrics',
        'iqa_topiq_nr_flive_metrics',
        'iqa_topiq_nr_spaq_metrics',
        'iqa_tres_metrics',
        'iqa_tres_flive_metrics',
        'iqa_unique_metrics',
        'iqa_uranker_metrics',
        'iqa_wadiqam_nr_metrics',
    ]

    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for metric_table_name in metric_table_names:
        try:
            logger.info("Processing table: %s", metric_table_name)
            df = load_metric_with_prompts(DB_PATH, metric_table_name)
            df.drop(columns=["uuid"], inplace=True, errors="ignore")
            df = add_prompt_features(df)
            df = preprocess_data(df)

            results_dir = (
                RESULTS_DIR / f"results_{current_timestamp}" / metric_table_name
            )
            results_dir.mkdir(parents=True, exist_ok=True)

            unsupervised_results = run_unsupervised_models(
                df.copy(), output_dir=results_dir
            )

            with open(results_dir / "results.json", "w") as fh:
                json.dump({"unsupervised": unsupervised_results}, fh, indent=4)

        except Exception as exc:
            logger.error("Error processing %s: %s", metric_table_name, exc)
            continue
        

# ****
if __name__ == "__main__":
    main()
    
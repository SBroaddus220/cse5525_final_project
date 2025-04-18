# -*- coding: utf-8 -*-

"""
Model training pipeline for prompt quality prediction.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import torch
from sklearn.model_selection import ParameterGrid
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# --------------------------------------------------------------------------- #
#                                 HELPER UTILITIES                            #
# --------------------------------------------------------------------------- #
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

def cap_zscore_outliers(df: pd.DataFrame, columns: list, threshold: float = 3.0) -> pd.DataFrame:
    """
    Caps outliers in the specified columns of a DataFrame based on Z-scores.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to cap.
        threshold (float): Z-score threshold for capping (default: 3.0).

    Returns:
        pd.DataFrame: A copy of the DataFrame with capped values.
    """
    df_capped = df.copy()

    for col in columns:
        if df_capped[col].dtype.kind in 'biufc':  # Ensure it's numeric
            col_zscores = stats.zscore(df_capped[col], nan_policy='omit')
            mean = df_capped[col].mean()
            std = df_capped[col].std()

            upper_cap = mean + threshold * std
            lower_cap = mean - threshold * std

            num_upper = np.sum(df_capped[col] > upper_cap)
            num_lower = np.sum(df_capped[col] < lower_cap)

            df_capped[col] = np.where(df_capped[col] > upper_cap, upper_cap, df_capped[col])
            df_capped[col] = np.where(df_capped[col] < lower_cap, lower_cap, df_capped[col])

            logger.info(
                f"{col}: Capped outliers to mean ± {threshold} * std "
                f"(lower: {lower_cap:.2f}, upper: {upper_cap:.2f}) | "
                f"{num_lower} lower, {num_upper} upper values capped"
            )
    return df_capped

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

    return df_scaled

def get_highly_correlated_pairs(
    df: pd.DataFrame, threshold: float = 0.95
) -> List[Tuple[str, str, float]]:
    """Finds all pairs of columns in a DataFrame that are highly correlated.

    Args:
        df (pd.DataFrame): The input DataFrame with numerical features only.
        threshold (float): Correlation threshold above which pairs are considered highly correlated.

    Returns:
        List[Tuple[str, str, float]]: List of tuples with each containing two column names and their correlation.
    """
    logger.info("Finding column pairs with correlation greater than %.2f", threshold)

    # Compute correlation matrix
    corr_matrix = df.corr()
    logger.debug("Correlation matrix computed.")

    # Store highly correlated pairs
    correlated_pairs = []

    # Iterate only over upper triangle to avoid duplicate pairs and self-correlation
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                correlated_pairs.append((col1, col2, corr_value))

    logger.info("Found %d highly correlated column pairs.", len(correlated_pairs))
    return correlated_pairs

def compute_correlations(
    df: pd.DataFrame, target_col: str = "next_inspection_grade_c_or_below"
) -> pd.Series:
    """
    Computes the absolute Pearson correlation between each feature and the target variable.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame containing numerical features and the target column.
        target_col (str): Name of the target variable column.

    Returns:
        pd.Series: Series of absolute correlations, indexed by feature names.
    """
    # Drop the target column to isolate features
    feature_cols = df.drop(columns=[target_col])
    
    # Compute absolute Pearson correlations with the target
    correlation_series = feature_cols.corrwith(df[target_col]).abs()

    logger.info("Computed correlations for %d features", len(correlation_series))

    return correlation_series

def plot_top_correlations(
    correlation_series: pd.Series, top_k: int = 20
) -> None:
    """
    Plots the top-k most correlated features with the target variable.

    Args:
        correlation_series (pd.Series): Series of feature correlations with the target variable.
        top_k (int): Number of top correlated features to display in the plot.

    Returns:
        None
    """
    # Sort and select the top_k features
    top_correlations = correlation_series.abs().sort_values(ascending=False).head(top_k)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=top_correlations.index, y=top_correlations.values, palette="crest", width=0.6)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3, fontsize=10)

    plt.title(f"Top {top_k} Most Correlated Features with Target", fontsize=14, weight='bold')
    plt.xlabel("Feature", fontsize=12)
    plt.ylabel("Absolute Correlation", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=10)
    plt.ylim(0, 0.65)  # scale from zero and leave a little headroom
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# --------------------------------------------------------------------------- #
#                        TEXT → FIXED‑SIZE EMBEDDING                          #
# --------------------------------------------------------------------------- #


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
    
def perform_shap_analysis(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """SHAP on numeric features with XGBRegressor (robust to 1‑feature case)."""
    X = data.drop(columns=[target_col]).select_dtypes(
        include=["number", "bool", "category"]
    )
    y = data[target_col]

    if X.shape[1] == 0:
        raise ValueError("No numeric features left for SHAP analysis.")

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    feature_importances = np.abs(shap_values.values).mean(axis=0)
    feature_importances = np.atleast_1d(feature_importances)   # <-- key line

    importance_df = (
        pd.DataFrame(
            {"feature": X.columns, "importance": feature_importances}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df




def plot_top_k_shap_features_from_df(
    importance_df: pd.DataFrame,
    top_k: int = 10
) -> None:
    """
    Plots a bar chart of the top-k most important features based on SHAP values.

    Args:
        importance_df (pd.DataFrame): DataFrame containing columns ['feature', 'importance'].
        top_k (int): Number of top features to plot.

    Returns:
        None
    """
    logger.info(f"Generating SHAP feature importance plot for top {top_k} features.")

    # Get top_k by descending importance
    top_features = importance_df.nlargest(top_k, 'importance')

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x="importance",
        y="feature",
        data=top_features,
        palette="flare"
    )

    # Add padding to x-axis so value labels don’t overflow
    x_max = top_features["importance"].max()
    plt.xlim(0, x_max * 1.1)

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)

    plt.xlabel("Mean Absolute SHAP Value", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"Top {top_k} SHAP Feature Importances", fontsize=14, weight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    logger.info("Feature importance plot displayed.")


# --------------------------------------------------------------------------- #
#                            TRAINING CONFIG DATACLASS                        #
# --------------------------------------------------------------------------- #


@dataclass
class TrainerConfig:
    """Experiment configuration."""

    target_col: str
    test_size: float = 0.2
    random_state: int = 42
    # search space for MLPRegressor
    param_grid: Dict[str, Any] = field(
        default_factory=lambda: {
            "regressor__hidden_layer_sizes": [(256,), (512, 256)],
            "regressor__alpha": [1e-4, 1e-3],
            "regressor__learning_rate_init": [1e-3, 5e-4],
        }
    )
    model_dir: pathlib.Path = pathlib.Path("saved_models")
    metrics_file: pathlib.Path = pathlib.Path("model_metrics.json")


# --------------------------------------------------------------------------- #
#                                  TRAINER                                    #
# --------------------------------------------------------------------------- #


class PromptQualityTrainer:
    """End‑to‑end pipeline: preprocessing ➔ training ➔ evaluation ➔ SHAP."""

    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self.best_estimator_: Optional[Pipeline] = None
        self.metrics_: Dict[str, float] = {}
        # single BertCLSEmbedder instance reused inside pipeline
        self._embedder = BertCLSEmbedder()

    # ------------------------------------------------------------------ #
    #                          data preparation                          #
    # ------------------------------------------------------------------ #

    def _prepare(self, df: pd.DataFrame, prompt_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply existing preprocessing helpers, returning X, y."""
        logger.info("Running shared preprocessing utilities")

        df = remove_duplicates(df)
        df = drop_constant_columns(df)

        # numeric feature engineering (exclude prompt + target columns)
        numeric_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in (prompt_col, self.cfg.target_col)
        ]
        df = cap_zscore_outliers(df, numeric_cols)
        df = scale_continuous_columns(df, blacklist=[prompt_col, self.cfg.target_col])

        # optional diagnostic: log highly correlated pairs
        corr_pairs = get_highly_correlated_pairs(df[numeric_cols])
        if corr_pairs:
            logger.info("Highly correlated feature pairs (>|0.95|): %s", corr_pairs)

        X = df.drop(columns=[self.cfg.target_col])
        y = df[self.cfg.target_col]
        return X, y

    # ------------------------------------------------------------------ #
    #                         pipeline definition                        #
    # ------------------------------------------------------------------ #

    def _build_pipeline(self, prompt_col: str, numeric_cols: List[str]) -> Pipeline:
        """ColumnTransformer → MLPRegressor wrapped in GridSearchCV."""
        from sklearn.preprocessing import FunctionTransformer  # local import

        text_branch = Pipeline(
            steps=[("embed", FunctionTransformer(self._embedder, validate=False))]
        )

        preprocessor = ColumnTransformer(
            [
                ("text", text_branch, prompt_col),
                ("numeric", "passthrough", numeric_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", MLPRegressor(
                    hidden_layer_sizes=(256,),      # still tuned via grid
                    max_iter=300,
                    random_state=self.cfg.random_state,
                    early_stopping=False            # <- disable for tiny folds
                )),
            ]
        )
        return pipeline

    # ------------------------------------------------------------------ #
    #                              training                              #
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame, prompt_col: str = "prompt") -> None:
        """Train model using grid search CV and store best estimator + metrics."""
        X, y = self._prepare(df, prompt_col)
        numeric_cols = [c for c in X.columns if c != prompt_col]

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
        )

        pipeline = self._build_pipeline(prompt_col, numeric_cols)
        # choose a fold count that never exceeds the training‑set size
        n_train = len(y_train)
        n_splits = min(3, n_train) if n_train > 1 else 2          # fallback to 2

        cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.cfg.random_state,
        )

        search = GridSearchCV(
            pipeline,
            param_grid=self.cfg.param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )
        n_cfgs = len(list(ParameterGrid(self.cfg.param_grid)))
        logger.info("Starting grid search over %d hyper‑parameter sets", n_cfgs)

        search = GridSearchCV(
            pipeline,
            param_grid=self.cfg.param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )
        search.fit(X_train, y_train)
        self.best_estimator_ = search.best_estimator_
        logger.info("Best params: %s", search.best_params_)

        preds = self.best_estimator_.predict(X_val)
        self.metrics_ = {
            "mse": mean_squared_error(y_val, preds),
            "mae": mean_absolute_error(y_val, preds),
            "r2": r2_score(y_val, preds),
        }
        logger.info("Validation metrics: %s", self.metrics_)

        # Optional SHAP diagnostics using your existing helper
        shap_df = pd.concat([X_val, y_val], axis=1)
        shap_imp = perform_shap_analysis(shap_df, self.cfg.target_col)  # re‑use helper
        logger.info("Top‑5 SHAP features:\n%s", shap_imp.head())

    # ------------------------------------------------------------------ #
    #                           persistence                              #
    # ------------------------------------------------------------------ #

    def save(self, name: str) -> None:
        """Persist trained pipeline + metrics."""
        if self.best_estimator_ is None:
            raise RuntimeError("Fit must be called before save().")

        self.cfg.model_dir.mkdir(exist_ok=True, parents=True)
        model_path = self.cfg.model_dir / f"{name}.joblib"
        joblib.dump(self.best_estimator_, model_path)
        logger.info("Saved model ➜ %s", model_path)

        # append / create metrics JSON
        metrics = {}
        if self.cfg.metrics_file.exists():
            metrics = json.loads(self.cfg.metrics_file.read_text())
        metrics[name] = self.metrics_
        self.cfg.metrics_file.write_text(json.dumps(metrics, indent=2))
        logger.info("Logged metrics ➜ %s", self.cfg.metrics_file)

    # ------------------------------------------------------------------ #
    #                        post‑hoc interpretability                    #
    # ------------------------------------------------------------------ #

    def compute_shap(self, X: pd.DataFrame, nsamples: int = 100) -> shap.Explanation:
        """Compute Kernel SHAP values for an arbitrary feature batch."""
        if self.best_estimator_ is None:
            raise RuntimeError("Model not trained yet.")
        # KernelExplainer can wrap any regressor
        bg = X.sample(nsamples, random_state=self.cfg.random_state)
        explainer = shap.KernelExplainer(self.best_estimator_.predict, bg)
        return explainer.shap_values(X)

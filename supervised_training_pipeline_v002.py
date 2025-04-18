import logging
import re
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from transformers import BertTokenizer, BertModel
import shap

logger = logging.getLogger(__name__)

# ****
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


###############################################################################
#                           PREPROCESSING LOGIC                               #
###############################################################################
def preprocess_dataframe(
    df: pd.DataFrame,
    columns_to_cap: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Perform minor preprocessing, including dropping duplicates, capping outliers,
    dropping constant columns, and scaling numerical columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns_to_cap (Optional[List[str]]): Columns to apply outlier capping.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    # Reference existing utility functions without re-defining if unchanged
    logger.info("Starting preprocessing.")
    df = remove_duplicates(df)
    if columns_to_cap is not None:
        df = cap_zscore_outliers(df, columns=columns_to_cap)
    df = drop_constant_columns(df)
    df = scale_continuous_columns(df)
    logger.info("Preprocessing complete.")
    return df


###############################################################################
#                      BERT EMBEDDING TRANSFORMER                             #
###############################################################################
class BertPromptEmbedder(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that converts text prompts into
    BERT [CLS] embeddings.
    """

    def __init__(self, pretrained_model_name: str = "bert-base-uncased"):
        """
        Initializes the BertPromptEmbedder with a tokenizer and model.

        Args:
            pretrained_model_name (str): Name of the pretrained BERT model.
        """
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
        self.model = BertModel.from_pretrained(self.pretrained_model_name)
        self.model.eval()
        self.model.requires_grad_(False)

    def fit(self, X: pd.DataFrame, y=None):
        """No fitting required for the embedding model."""
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms the input dataframe of prompts into BERT [CLS] embeddings.

        Args:
            X (pd.DataFrame): DataFrame that must contain a column 'prompt'.

        Returns:
            np.ndarray: Array of shape (N, 768) containing the BERT [CLS] embeddings.
        """
        logger.info("Generating BERT [CLS] embeddings for prompts.")
        embeddings = []
        for prompt in X["prompt"]:
            tokens = self.tokenizer.encode_plus(
                prompt,
                add_special_tokens=True,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64
            )
            with torch.no_grad():
                output = self.model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"]
                )
            cls_vector = output.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_vector)
        return np.array(embeddings)


###############################################################################
#                           FEATURE FUSION STEP                               #
###############################################################################
class FeatureFusion(BaseEstimator, TransformerMixin):
    """
    Concatenates BERT embeddings (or any vector) with additional numeric features.
    """

    def __init__(self, numeric_feature_cols: Optional[List[str]] = None):
        """
        Initializes the FeatureFusion object.

        Args:
            numeric_feature_cols (Optional[List[str]]): Columns containing numeric features.
        """
        self.numeric_feature_cols = numeric_feature_cols if numeric_feature_cols else []

    def fit(self, X, y=None):
        """No fitting required for feature fusion."""
        return self

    def transform(self, X) -> np.ndarray:
        """
        Concatenates the previously computed BERT embeddings with the numeric features.

        Args:
            X (pd.DataFrame or np.ndarray): If np.ndarray, it is assumed to be BERT embeddings.
                                            If pd.DataFrame, you can extract numeric columns.

        Returns:
            np.ndarray: Concatenated array of shape (N, embeddings_dim + numeric_feature_dim).
        """
        if isinstance(X, pd.DataFrame):
            # If X is a DataFrame, only numeric columns are needed here
            numeric_data = X[self.numeric_feature_cols].values
            return numeric_data
        # If X is already an array of BERT embeddings, just return it
        return X


###############################################################################
#                     DATAFRAME-TO-EMB + NUMERIC PIPELINE                     #
###############################################################################
class PromptDataTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn Transformer that orchestrates:
      1) Getting BERT embeddings from 'prompt'
      2) Extracting numeric columns
      3) Concatenating them
    """

    def __init__(self, numeric_feature_cols: Optional[List[str]] = None):
        """
        Initialize with the numeric columns to extract.

        Args:
            numeric_feature_cols (Optional[List[str]]): Numeric columns to fuse with BERT embeddings.
        """
        self.bert_embedder = BertPromptEmbedder()
        self.feature_fusion = FeatureFusion(numeric_feature_cols=numeric_feature_cols)
        self.numeric_feature_cols = numeric_feature_cols if numeric_feature_cols else []
        self._embeddings = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the BERT embedder to the data (no op).

        Args:
            X (pd.DataFrame): DataFrame with at least the column 'prompt'.
            y: Ignored.

        Returns:
            PromptDataTransformer: Self.
        """
        self.bert_embedder.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate BERT embeddings and fuse them with numeric features.

        Args:
            X (pd.DataFrame): DataFrame containing 'prompt' and numeric columns.

        Returns:
            np.ndarray: Fused feature array of shape (N, 768 + len(numeric_feature_cols)).
        """
        # Step A: BERT embeddings
        emb = self.bert_embedder.transform(X)

        # Step B: Numeric features
        numeric_data = X[self.numeric_feature_cols].values

        # Step C: Concatenate
        fused = np.concatenate([emb, numeric_data], axis=1)
        return fused


###############################################################################
#                     REGRESSION PIPELINE RUNNER                              #
###############################################################################
class RegressionBasePipelineRunner:
    """
    Base class for running a regression pipeline with scikit-learn's GridSearchCV.
    """

    def __init__(self):
        """
        Initializes a regression pipeline runner with an empty param_grid.
        """
        self.logger = logging.getLogger(__name__)
        self.param_grid: Dict[str, Any] = {}
        self._best_estimator = None

    def _build_pipeline(self) -> Pipeline:
        """
        Builds and returns an uninitialized Pipeline.
        Must be overridden in subclasses.
        """
        raise NotImplementedError("Subclasses must implement _build_pipeline.")

    def fit(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Fits a GridSearchCV using the pipeline and param_grid, then evaluates on the test set.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.
            y_train (pd.Series): Training targets.
            y_test (pd.Series): Testing targets.

        Returns:
            Dict[str, Any]: Dictionary containing the best estimator, predictions, and performance metrics.
        """
        self.logger.info("Fitting regression pipeline with GridSearchCV.")
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
            "best_estimator": self._best_estimator,
            "y_true": y_test,
            "y_pred": y_pred,
            "mse": mse,
            "mae": mae,
            "r2": r2,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts using the best estimator found by fit.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            np.ndarray: Predictions.
        """
        if self._best_estimator is None:
            raise ValueError("You must fit the model before predicting.")
        return self._best_estimator.predict(X)

    def get_best_estimator(self) -> Any:
        """
        Returns the best estimator after fitting.

        Returns:
            Any: The best fitted estimator.
        """
        return self._best_estimator


class BERTMLPRegressorPipelineRunner(RegressionBasePipelineRunner):
    """
    A pipeline runner for a BERT + MLPRegressor architecture.
    """

    def __init__(self, numeric_feature_cols: Optional[List[str]] = None):
        """
        Generates a param grid for searching MLP hyperparams and initializes
        the numeric_feature_cols to fuse with BERT embeddings.

        Args:
            numeric_feature_cols (Optional[List[str]]): List of numeric columns.
        """
        super().__init__()
        self.numeric_feature_cols = numeric_feature_cols if numeric_feature_cols else []
        self.param_grid = {
            "model__hidden_layer_sizes": [(64,), (64, 64)],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate": ["constant", "adaptive"]
        }

    def _build_pipeline(self) -> Pipeline:
        """
        Builds the regression pipeline:
          1) PromptDataTransformer - BERT embeddings + numeric features
          2) MLPRegressor
        """
        steps = [
            ("data_transform", PromptDataTransformer(numeric_feature_cols=self.numeric_feature_cols)),
            ("model", MLPRegressor(random_state=42, max_iter=1000))
        ]
        return Pipeline(steps=steps)


###############################################################################
#               SHAP-BASED INTERPRETATION FOR PROMPT MODEL                    #
###############################################################################
def merge_wordpiece_tokens(tokens: List[str]) -> List[str]:
    """
    Merges WordPiece tokens into more readable forms (e.g., 'com' + '##puter' -> 'computer').

    Args:
        tokens (List[str]): List of tokens from a BERT tokenizer.

    Returns:
        List[str]: List of merged tokens.
    """
    merged_tokens = []
    current_token = ""
    for tok in tokens:
        if tok.startswith("##"):
            current_token += tok[2:]
        else:
            if current_token:
                merged_tokens.append(current_token)
            current_token = tok
    if current_token:
        merged_tokens.append(current_token)
    return merged_tokens


def run_shap_interpretation(
    model_pipeline: Pipeline,
    df: pd.DataFrame,
    numeric_feature_cols: List[str],
    explainer_samples: int = 100
) -> Dict[str, Any]:
    """
    Runs a SHAP analysis on a pipeline that first transforms data with BERT
    and then feeds it into a regression model. Merges WordPiece tokens for readability.

    Args:
        model_pipeline (Pipeline): The fitted pipeline (contains PromptDataTransformer and a regressor).
        df (pd.DataFrame): The dataframe containing 'prompt' and numeric features.
        numeric_feature_cols (List[str]): Names of the numeric features.
        explainer_samples (int): Number of samples to use for SHAP.

    Returns:
        Dict[str, Any]: A dictionary with shap_values, expected_value, etc.
    """
    logger.info("Beginning SHAP interpretation for the prompt model.")
    if explainer_samples < len(df):
        sample_df = df.sample(explainer_samples, random_state=42)
    else:
        sample_df = df

    # Transform data to get the final features that go into the regressor
    final_features = model_pipeline["data_transform"].transform(sample_df)
    regressor = model_pipeline["model"]

    logger.info("Initializing DeepExplainer / KernelExplainer.")
    # If MLP is used, you can usually use KernelExplainer or similar
    explainer = shap.Explainer(regressor, final_features)
    shap_values = explainer(final_features)

    # Merge WordPiece tokens for each row (example approach)
    # Note that in a real scenario, you'd need to capture the tokens from the BERT transform step
    merged_token_info = []
    for idx, row in sample_df.iterrows():
        tokens = model_pipeline["data_transform"].bert_embedder.tokenizer.tokenize(row["prompt"])
        merged_tokens = merge_wordpiece_tokens(tokens)
        merged_token_info.append(merged_tokens)

    result = {
        "shap_values": shap_values.values,
        "base_values": shap_values.base_values,
        "merged_tokens": merged_token_info
    }
    logger.info("SHAP interpretation complete.")
    return result


###############################################################################
#                           MODEL SAVING UTILITY                              #
###############################################################################
def save_regression_model(best_estimator: Any, filename: str) -> None:
    """
    Saves the best performing model (pipeline) to disk using joblib.

    Args:
        best_estimator (Any): The scikit-learn pipeline or model to save.
        filename (str): Path to the output file.
    """
    from joblib import dump
    logger.info(f"Saving best estimator to {filename}.")
    dump(best_estimator, filename)


###############################################################################
#                   MODEL TRAINING ORCHESTRATION FUNCTION                     #
###############################################################################
def train_and_evaluate_prompt_model(
    data: pd.DataFrame,
    target_col: str,
    numeric_feature_cols: Optional[List[str]] = None,
    columns_to_cap: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Orchestrates the entire pipeline:
      - Minor preprocessing
      - Train/test split
      - Fit regression pipeline
      - Evaluate model
      - Return results

    Args:
        data (pd.DataFrame): The input dataframe with columns 'prompt', numeric features, and a target.
        target_col (str): The name of the target column.
        numeric_feature_cols (Optional[List[str]]): List of numeric feature columns.
        columns_to_cap (Optional[List[str]]): Columns to cap outliers in.
        test_size (float): Test set fraction.
        random_state (int): Seed for reproducibility.

    Returns:
        Dict[str, Any]: Dictionary with pipeline results (best_estimator, metrics, etc.).
    """
    logger.info("Starting model training and evaluation for prompt-based regression.")
    df = preprocess_dataframe(data, columns_to_cap=columns_to_cap)

    # Split
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # Train
    runner = BERTMLPRegressorPipelineRunner(numeric_feature_cols=numeric_feature_cols)
    results = runner.fit(X_train, X_test, y_train, y_test)
    logger.info(
        f"Training complete | MSE: {results['mse']:.4f}, MAE: {results['mae']:.4f}, R2: {results['r2']:.4f}"
    )

    return results
"""
example_prompt_regression.py
============================
End‑to‑end demo that exercises:

* preprocess_dataframe            – minor cleansing / scaling
* train_and_evaluate_prompt_model – model training + CV search
* run_shap_interpretation         – post‑hoc explanation
* save_regression_model           – persistence

Replace the stub utility functions with your production versions.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 0.  Infrastructure ---------------------------------------------------
# ---------------------------------------------------------------------
# Assume the entire block you pasted lives in a module named `prompt_pipeline`.
# If it sits in the same file, just comment the import and use the symbols directly.
# ---------------------------------------------------------------------
from supervised_training_pipeline_v002 import (
    preprocess_dataframe,
    train_and_evaluate_prompt_model,
    run_shap_interpretation,
    save_regression_model,
    remove_duplicates,
    cap_zscore_outliers,
    drop_constant_columns,
    scale_continuous_columns,
)
# ---------------------------------------------------------------------
# 1.  Logging setup
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------
# 2.  Fake (but plausible) dataset
# ---------------------------------------------------------------------
data = pd.DataFrame(
    {
        "prompt": [
            # ───────── original five ─────────
            "A photo‑realistic portrait of a golden retriever wearing sunglasses",
            "Low‑poly voxel spaceship in deep space, 8‑bit colour palette",
            "Surrealistic painting of a clock melting over desert dunes",
            "An elegant minimalist product photo of a wristwatch on black satin",
            "Macro shot of dew on a spider web at dawn, shallow depth of field",
            # ───────── seven new samples ─────
            "Watercolour landscape of snow‑capped mountains reflected in a lake",
            "Cyber‑punk cityscape at night with neon signs and rainy streets",
            "Isometric pixel‑art coffee shop interior, warm morning light",
            "High‑contrast black‑and‑white portrait of an elderly sailor",
            "Still‑life oil painting of a fruit bowl with dramatic chiaroscuro",
            "Photoreal render of a chrome robot hand holding a red rose",
            "Fantasy illustration of a dragon circling a crystal tower at sunset",
        ],
        "complexity_score": [
            0.56, 0.32, 0.73, 0.41, 0.67,   # original
            0.45, 0.61, 0.38, 0.52, 0.47, 0.59, 0.66  # new
        ],
        "adjective_count": [
            3, 1, 2, 2, 2,   # original
            3, 3, 2, 2, 2, 2, 3   # new
        ],
        "brisque": [
            22.1, 48.3, 31.7, 18.9, 25.4,   # original
            29.8, 34.6, 40.2, 19.5, 23.7, 27.1, 32.4  # new (dummy values)
        ],
    }
)
NUMERIC_FEATURES = ["complexity_score", "adjective_count"]
TARGET_COLUMN = "brisque"

# ---------------------------------------------------------------------
# 3.  Training + evaluation
# ---------------------------------------------------------------------
results = train_and_evaluate_prompt_model(
    data=data,
    target_col=TARGET_COLUMN,
    numeric_feature_cols=NUMERIC_FEATURES,
    columns_to_cap=NUMERIC_FEATURES,   # cap the engineered scores just for illustration
    test_size=0.4,                     # use 60 / 40 split so we get a couple test samples
)

logging.info(
    "=== Metrics on hold‑out set ===\n"
    "MSE = %.3f | MAE = %.3f | R² = %.3f",
    results["mse"],
    results["mae"],
    results["r2"],
)

# ---------------------------------------------------------------------
# 4.  SHAP interpretation
# ---------------------------------------------------------------------
shap_out = run_shap_interpretation(
    model_pipeline=results["best_estimator"],
    df=data,
    numeric_feature_cols=NUMERIC_FEATURES,
    explainer_samples=min(100, len(data)),
)

logging.info("SHAP values shape: %s", shap_out["shap_values"].shape)
logging.info("First row merged tokens: %s", shap_out["merged_tokens"][0][:10])

# (Optional) Visualize – uncomment if running in a notebook or GUI environment
# import shap
# shap.plots.beeswarm(shap_out["shap_values"])

# ---------------------------------------------------------------------
# 5.  Persist the pipeline
# ---------------------------------------------------------------------
model_path = Path("bert_mlp_prompt_pipeline.joblib")
save_regression_model(results["best_estimator"], model_path)
logging.info("Model saved to %s", model_path.resolve())

# ---------------------------------------------------------------------
# 6.  Reload and inference example
# ---------------------------------------------------------------------
from joblib import load

reloaded_pipeline = load(model_path)
new_prompts = pd.DataFrame(
    {
        "prompt": ["Hyper‑realistic render of a red sports car speeding through rain"],
        "complexity_score": [0.62],
        "adjective_count": [2],
    }
)
prediction = reloaded_pipeline.predict(new_prompts)
print(f"\nPredicted quality score: {prediction[0]:.2f}")

"""
run_prompt_quality_with_viz.py

Demonstration of the full workflow:
  • training & saving a BERT‑MLP regressor
  • correlation analysis  ➜  saves top_correlations.png
  • SHAP feature analysis ➜  saves top_shap.png
"""

from __future__ import annotations
import logging
import matplotlib.pyplot as plt
import pandas as pd

# ——— infrastructure built earlier ——————————————————————————————— #
from supervised_training_pipeline import (
    TrainerConfig, 
    PromptQualityTrainer,
    remove_duplicates,
    drop_constant_columns,
    cap_zscore_outliers,
    scale_continuous_columns,
    perform_shap_analysis,
    compute_correlations,
    plot_top_correlations,
    plot_top_k_shap_features_from_df,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# --------------------------------------------------------------------- #
# 1.  Assemble/Load your dataframe (replace this dummy example)         #
# --------------------------------------------------------------------- #
df = pd.DataFrame(
    {
        "prompt": [
            "a serene mountain lake at sunrise",
            "dark cyberpunk alley with neon signs",
            "portrait of a golden retriever wearing glasses",
        ],
        "complexity_score": [2.7, 4.3, 3.1],
        "adjective_count": [2, 2, 1],
        "BRISQUE": [28.4, 35.1, 22.9],  # ← target quality metric
    }
)

# --------------------------------------------------------------------- #
# 2.  Train + save model                                               #
# --------------------------------------------------------------------- #
cfg = TrainerConfig(target_col="BRISQUE", test_size=0.25, random_state=123)
trainer = PromptQualityTrainer(cfg)
trainer.fit(df, prompt_col="prompt")
trainer.save("brisque_mlp")
logger.info("Metrics: %s", trainer.metrics_)

# --------------------------------------------------------------------- #
# 3.  Correlation plot (numeric features ↔ target)                      #
# --------------------------------------------------------------------- #
#   – reuse your helper utilities so the preprocessing is identical –
numeric_cols = [c for c in df.columns if c not in ("prompt", cfg.target_col)]
df_numeric = df[["BRISQUE", *numeric_cols]].copy()

df_numeric = remove_duplicates(df_numeric)
df_numeric = drop_constant_columns(df_numeric)
df_numeric = cap_zscore_outliers(df_numeric, numeric_cols)
df_numeric = scale_continuous_columns(df_numeric, blacklist=[cfg.target_col])

corr_series = compute_correlations(df_numeric, target_col=cfg.target_col)
plot_top_correlations(corr_series, top_k=10)
plt.gcf().savefig("top_correlations.png", dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved correlation figure → top_correlations.png")

# --------------------------------------------------------------------- #
# 4.  SHAP importance plot (numeric only for quick demo)                #
#     You can include BERT embeddings too, but that yields 768 features #
# --------------------------------------------------------------------- #
importance_df = perform_shap_analysis(df_numeric, target_col=cfg.target_col)
plot_top_k_shap_features_from_df(importance_df, top_k=10)
plt.gcf().savefig("top_shap.png", dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved SHAP figure → top_shap.png")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# 1. Example Text + Metrics
# ---------------------------
texts = [
    "Bright red flower in a sunny garden",
    "Dark indoor room with black couch",
    "Bright sunny day at the beach",
    "Night sky with shining stars",
    "Dark forest path at sunset"
]

# Two float metrics per image/text pair
metrics = [
    [0.9,  0.7],
    [0.1,  0.2],
    [0.8,  0.85],
    [0.3,  0.4],
    [0.2,  0.6]
]

# ---------------------------
# 2. TF-IDF Vectorization
# ---------------------------
vectorizer = TfidfVectorizer()
X_sparse = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

# Convert to dense float array
X = X_sparse.toarray().astype(float)
Y = np.array(metrics)

# ---------------------------
# 3. Train Multi-output Model
# ---------------------------
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, Y)

# ---------------------------
# 4. SHAP Explanation
# ---------------------------
# SHAP Explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)  # returns shape (n_samples, n_features, n_outputs)
shap_values = np.array(shap_values)

# Split per metric
shap_values_metric_1 = shap_values[:, :, 0]
shap_values_metric_2 = shap_values[:, :, 1]

# ---------------------------
# 5. Compute Feature Importances
# ---------------------------
# Compute mean abs SHAP values
mean_abs_shap_1 = np.abs(shap_values_metric_1).mean(axis=0)
mean_abs_shap_2 = np.abs(shap_values_metric_2).mean(axis=0)

# Map to feature names
feature_importance_1 = pd.Series(mean_abs_shap_1, index=feature_names).sort_values(ascending=False)
feature_importance_2 = pd.Series(mean_abs_shap_2, index=feature_names).sort_values(ascending=False)

# ---------------------------
# 6. Plot Top Words per Metric
# ---------------------------
def plot_top_words(feature_importance, metric_name, top_n=5):
    top_features = feature_importance.head(top_n)
    plt.figure(figsize=(6, 4))
    plt.barh(top_features.index[::-1], top_features[::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top Word Features for {metric_name}")
    plt.tight_layout()
    plt.show()

plot_top_words(feature_importance_1, "Metric 1")
plot_top_words(feature_importance_2, "Metric 2")

# ---------------------------
# 7. Optional: SHAP for One Document
# ---------------------------
sample_idx = 0
doc_shap_1 = pd.DataFrame({
    "word": feature_names,
    "shap_value": shap_values_metric_1[sample_idx]
}).sort_values("shap_value", ascending=False)

print(f"\nTop SHAP Words for Document {sample_idx} (Metric 1):")
print(doc_shap_1.head(10))

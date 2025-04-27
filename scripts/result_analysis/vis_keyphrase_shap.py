# visualize_keyphrase_shap.py

"""
Visualize keyphrases with SHAP values using Word Cloud, Waterfall Plot, and Treemap.
"""

from openai import OpenAI
from dotenv import load_dotenv
import os

# Load your API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from typing import List
import re

def clean_label(label: str, max_words: int = 3) -> str:
    """
    Clean and standardize the cluster label.
    - Limit to max words
    - Title-case it
    - Remove unnecessary punctuation
    """
    label = label.strip()
    label = re.sub(r"[\"'’.,;:!?()\-]+", "", label)
    words = label.split()
    return " ".join(words[:max_words]).title()


import pandas as pd
import matplotlib.pyplot as plt
import squarify  # for treemap
from wordcloud import WordCloud
import numpy as np

# === add_to_visualize_keyphrase_shap.py ======================================
# Put these imports near the top of your existing script
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

try:                     # optional dimensionality-reduction upgrade
    import umap          # pip install umap-learn
except ImportError:      # fall back to PCA if UMAP not available
    umap = None

import plotly.graph_objects as go   # pip install plotly

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV with columns 'keyphrase' and 'adjusted_scaled_mean_shap'.
    
    Args:
        csv_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    df = pd.read_csv(csv_path)
    return df

def plot_wordcloud(df: pd.DataFrame) -> None:
    """
    Create a high-resolution word cloud weighted by SHAP value,
    using all keyphrases.

    Args:
        df (pd.DataFrame): DataFrame with keyphrases and SHAP values.
    """
    shap_dict = dict(zip(df['keyphrase'], df['adjusted_scaled_mean_shap']))

    wordcloud = WordCloud(
        width=3000,          # Higher resolution
        height=1500,
        background_color='white',
        max_words=len(shap_dict),  # Include all records
        prefer_horizontal=0.9,     # Prefer horizontal words for better readability
        collocations=False         # Treat each keyphrase separately even if repeated words
    ).generate_from_frequencies(shap_dict)

    plt.figure(figsize=(18, 9))  # Bigger figure for display
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_waterfall(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Create a waterfall-style bar plot of top keyphrases.
    
    Args:
        df (pd.DataFrame): DataFrame with keyphrases and SHAP values.
        top_n (int): Number of top keyphrases to display.
    """
    top_df = df.head(top_n)
    
    cumulative = np.cumsum(top_df['adjusted_scaled_mean_shap'])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(top_df['keyphrase'], top_df['adjusted_scaled_mean_shap'], color='skyblue')
    ax.plot(top_df['keyphrase'], cumulative, color='red', marker='o', label='Cumulative SHAP')
    
    ax.set_ylabel('Adjusted Scaled Mean SHAP')
    ax.set_title(f"Waterfall Plot of Top {top_n} Keyphrases", fontsize=18)
    ax.set_xticklabels(top_df['keyphrase'], rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_treemap(df: pd.DataFrame, top_n: int = 100) -> None:
    """
    Create a treemap of keyphrases.
    
    Args:
        df (pd.DataFrame): DataFrame with keyphrases and SHAP values.
        top_n (int): Number of top keyphrases to include.
    """
    top_df = df.head(top_n)
    
    fig = plt.figure(figsize=(12, 8))
    squarify.plot(
        sizes=top_df['adjusted_scaled_mean_shap'],
        label=top_df['keyphrase'],
        alpha=0.8,
        pad=True
    )
    plt.axis('off')
    plt.title(f"Treemap of Top {top_n} Keyphrases (by SHAP)", fontsize=18)
    plt.tight_layout()
    plt.show()
    
# --------------------------------------------------------------------------- #
# NEW VISUALIZATION FUNCTIONS                                                 #
# --------------------------------------------------------------------------- #
def plot_radial_bar(df: pd.DataFrame, top_n: int = 30) -> None:
    """Radial (polar) bar chart of the top‐N keyphrases."""
    top_df = df.head(top_n)
    angles = np.linspace(0, 2 * np.pi, top_n, endpoint=False)
    values = top_df['adjusted_scaled_mean_shap'].values
    labels = top_df['keyphrase'].values

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    bars = ax.bar(angles, values, width=2 * np.pi / top_n, alpha=0.8)

    # put labels slightly outside bar tips
    for angle, height, label in zip(angles, values, labels):
        ax.text(angle, height + height * 0.05, label,
                ha='center', va='center', fontsize=8,
                rotation=np.rad2deg(angle), rotation_mode='anchor')

    ax.set_title(f"Radial Bar Chart of Top {top_n} Keyphrases", fontsize=18)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.tight_layout()
    plt.show()


def plot_bubble_chart(df: pd.DataFrame, top_n: int = 100) -> None:
    """Bubble scatter in 2-D space (random jitter) sized by SHAP."""
    rnd = np.random.RandomState(42)
    top_df = df.head(top_n).copy()
    top_df['x'] = rnd.rand(top_n)
    top_df['y'] = rnd.rand(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(top_df['x'], top_df['y'],
                         s=top_df['adjusted_scaled_mean_shap'] * 4000,
                         alpha=0.6, edgecolors='k')
    for _, row in top_df.iterrows():
        ax.text(row['x'], row['y'], row['keyphrase'],
                ha='center', va='center', fontsize=7)
    ax.set_title(f"Bubble Chart of Top {top_n} Keyphrases", fontsize=18)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_sankey(df: pd.DataFrame, top_n: int = 50) -> None:
    """
    Simple Sankey: one root node → phrase nodes.
    (Great for interactive display in Jupyter / browsers.)
    """
    top_df = df.head(top_n)
    phrases = top_df['keyphrase'].tolist()
    values = top_df['adjusted_scaled_mean_shap'].tolist()

    # Node 0 is the "Total" root
    node_labels = ['Total'] + phrases
    sources = [0] * top_n
    targets = list(range(1, top_n + 1))

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=node_labels, pad=15),
        link=dict(source=sources, target=targets, value=values)
    )])
    fig.update_layout(title_text=f"Sankey Diagram of Top {top_n} Keyphrases",
                      font_size=10)
    fig.show()


def plot_heatstrip(df: pd.DataFrame) -> None:
    """Colored list (heat‐strip)."""
    fig, ax = plt.subplots(figsize=(6, len(df) * 0.15))
    sns.heatmap(df[['adjusted_scaled_mean_shap']].T,
                cmap="Reds", cbar=True, ax=ax,
                yticklabels=['SHAP Importance'],
                xticklabels=df['keyphrase'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                       fontsize=7)
    plt.title("Heatstrip of Keyphrases (All Records)", fontsize=18)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_violin_swarm(df: pd.DataFrame) -> None:
    """Violin + swarm: phrase length vs SHAP."""
    df = df.copy()
    df['length'] = df['keyphrase'].str.split().apply(len)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(x='length', y='adjusted_scaled_mean_shap',
                   data=df, inner=None, ax=ax, color='lightgray')
    sns.swarmplot(x='length', y='adjusted_scaled_mean_shap',
                  data=df, size=3, ax=ax)
    ax.set_xlabel("Keyphrase Length (words)")
    ax.set_ylabel("SHAP Value")
    ax.set_title("Keyphrase Length vs SHAP Importance", fontsize=18)
    plt.tight_layout()
    plt.show()


def plot_cumulative_contribution(df: pd.DataFrame) -> None:
    """Plot a refined cumulative SHAP contribution curve with aesthetic improvements."""
    df = df.copy()
    df['cum_pct'] = df['adjusted_scaled_mean_shap'].cumsum() / df['adjusted_scaled_mean_shap'].sum() * 100

    sns.set_theme(style="whitegrid")  # Elegant background with gridlines
    fig, ax = plt.subplots(figsize=(10, 6))

    # Line and marker styling
    ax.plot(
        range(1, len(df) + 1),
        df['cum_pct'],
        marker='o',
        markersize=4,
        linewidth=2,
        color='#1f77b4',
        label='Cumulative SHAP'
    )

    # Threshold line
    ax.axhline(80, color='red', linestyle='--', linewidth=1.5)
    ax.text(len(df) * 0.95, 82, '80% Threshold', color='red', ha='right', fontsize=11)

    # Labels and titles
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Cumulative % of Total SHAP", fontsize=12)
    ax.set_title("Cumulative SHAP Contribution Curve", fontsize=16, weight='bold')

    # Remove top and right spines
    sns.despine()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_embedding_scatter(df: pd.DataFrame, use_umap: bool = True) -> None:
    """
    Embed keyphrases (TF-IDF) to 2-D (UMAP or PCA), show scatter sized by SHAP.
    """
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df['keyphrase'])

    if use_umap and umap is not None:
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(X)
        title = "UMAP Embedding of Keyphrases"
    else:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X.toarray())
        title = "PCA Embedding of Keyphrases"

    fig, ax = plt.subplots(figsize=(10, 8))
    sizes = df['adjusted_scaled_mean_shap'] * 3000
    scatter = ax.scatter(coords[:, 0], coords[:, 1], s=sizes, alpha=0.6,
                         edgecolors='k')

    # Annotate a few top phrases for clarity
    for idx in df.head(20).index:
        ax.text(coords[idx, 0], coords[idx, 1], df.loc[idx, 'keyphrase'],
                fontsize=8)

    ax.set_title(title, fontsize=18)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    
from sklearn.cluster import DBSCAN
from collections import Counter

from adjustText import adjust_text
def plot_clustered_umap(df: pd.DataFrame, coords: np.ndarray, cluster_labels: dict) -> None:
    """
    Improved UMAP cluster visualization with smarter label placement and lower overlap.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Bubble plot using SHAP values
    scatter = ax.scatter(coords[:, 0], coords[:, 1],
                         c=df['cluster'],
                         cmap='tab20',
                         s=df['adjusted_scaled_mean_shap'] * 3000,
                         alpha=0.7,
                         edgecolors='k', linewidth=0.5)

    # Create labels with better placement logic
    texts = []
    for cluster_id, label in cluster_labels.items():
        cluster_points = coords[df['cluster'] == cluster_id]
        if cluster_points.shape[0] == 0:
            continue

        # Weighted center: emphasize higher SHAP
        shap_vals = df.loc[df['cluster'] == cluster_id, 'adjusted_scaled_mean_shap'].values
        weighted_center = np.average(cluster_points, axis=0, weights=shap_vals)

        texts.append(ax.text(weighted_center[0], weighted_center[1], label,
                             fontsize=10, weight='bold',
                             bbox=dict(facecolor='white', edgecolor='black',
                                       boxstyle='round,pad=0.25', alpha=0.85)))

    # Improve positioning
    adjust_text(
        texts,
        expand_points=(1.2, 1.4),
        force_points=0.2,
        force_text=0.2,
        only_move={'points': 'y', 'text': 'xy'},
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
        ax=ax
    )

    ax.set_title("UMAP Embedding with Cluster Labels", fontsize=18)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.show()
def gpt_label_cluster(phrases: List[str], model="gpt-4o") -> str:
    """
    Ask GPT-4o to provide a clean, conceptual label for a cluster of keyphrases.
    """
    examples = """
Examples of good cluster labels:
- Artist Names
- Futuristic Settings
- Digital Art Styles
- Color Palette Types
- Surreal Aesthetic
- Science Fiction Themes
- Fantasy Creatures
"""

    prompt = (
        "These keyphrases are from an image generation dataset, representing artistic concepts.\n"
        + examples +
        "\nNow here is a new group of keyphrases:\n\n"
        + "\n".join(f"- {phrase}" for phrase in phrases)
        + "\n\nWhat is a short, general label (2–3 words) that best describes this group?\n"
          "Avoid abstract or vague terms (e.g., 'Inspired', 'Dreamlike'). "
          "Avoid color names unless the entire group is centered around color. "
          "Prefer conceptual categories or concrete visual themes (e.g., 'Science Fiction Scenes', 'Mechanical Components'). "
          "Only return the label."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert at summarizing visual themes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=30
    )

    label = response.choices[0].message.content.strip()
    return clean_label(label)

def llm_label_clusters(df: pd.DataFrame, cluster_col: str = 'cluster', max_per_cluster: int = 12) -> dict:
    """
    Generate cluster labels using LLM for each group.
    Returns: dict of {cluster_id: label}
    """
    cluster_labels = {}
    for cluster_id in sorted(set(df[cluster_col])):
        if cluster_id == -1:
            continue

        cluster_df = df[df[cluster_col] == cluster_id]
        if cluster_df.empty:
            cluster_labels[cluster_id] = "Unknown"
            continue

        phrases = cluster_df['keyphrase'].head(max_per_cluster).tolist()
        label = gpt_label_cluster(phrases)
        cluster_labels[cluster_id] = label

    return cluster_labels


def main():
    csv_path = "./shap_summary_plots/global/global_keyphrase_adjusted_avg_scaled.csv"  # 🔥 Change this to your CSV path
    df = load_data(csv_path)

    # Classic visuals
    # plot_wordcloud(df)
    # plot_waterfall(df)
    # plot_treemap(df)

    # ✨ NEW fancy visuals
    # plot_radial_bar(df)
    # plot_bubble_chart(df)
    # plot_sankey(df)
    # plot_heatstrip(df)
    # plot_violin_swarm(df)
    # plot_cumulative_contribution(df)
    # plot_embedding_scatter(df)        # set use_umap=False if UMAP not installed
    # === Step 1: Embed keyphrases using TF-IDF
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df["keyphrase"])

    # === Step 2: UMAP projection to 2D
    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(X)
    
    # === Step 3: Clustering
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(coords)
    df["cluster"] = clustering.labels_

    # === Step 4: Label each cluster using GPT-4o
    cluster_labels = llm_label_clusters(df)

    # === Step 5: Visualize
    plot_clustered_umap(df, coords, cluster_labels)

if __name__ == "__main__":
    main()

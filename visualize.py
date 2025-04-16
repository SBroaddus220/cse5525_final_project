import sqlite3
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data_from_db(db_path: str) -> pd.DataFrame:
    """Load feature data from the SQLite database into a Pandas DataFrame.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM features", conn)
    conn.close()

    # Convert JSON strings back to lists
    df['adjectives'] = df['adjectives'].apply(json.loads)
    df['named_entities'] = df['named_entities'].apply(json.loads)
    return df


def save_plot(fig, filename: str) -> None:
    """Save a matplotlib figure to the visualizations folder.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        filename (str): The filename (without path) to save the figure as.
    """
    os.makedirs("visualizations", exist_ok=True)
    fig.savefig(os.path.join("visualizations", filename), bbox_inches="tight")
    plt.close(fig)


def generate_visualizations(df: pd.DataFrame) -> None:
    """Generate and save useful visualizations from the extracted feature dataset.

    Args:
        df (pd.DataFrame): DataFrame containing extracted features.
    """
    # 1. Histogram of complexity scores
    fig, ax = plt.subplots()
    sns.histplot(df['complexity_score'], bins=20, kde=True, ax=ax)
    ax.set_title("Distribution of Complexity Scores")
    ax.set_xlabel("Complexity Score")
    save_plot(fig, "complexity_score_distribution.png")

    # 2. Bar plot of most common art styles
    fig, ax = plt.subplots()
    df['art_style'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Art Style Frequency")
    ax.set_ylabel("Count")
    save_plot(fig, "art_style_frequency.png")

    # 3. Bar plot of most common lighting types
    fig, ax = plt.subplots()
    df['lighting'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Lighting Type Frequency")
    ax.set_ylabel("Count")
    save_plot(fig, "lighting_type_frequency.png")

    # 4. Number of adjectives vs complexity score
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['num_adjectives'], y=df['complexity_score'], ax=ax)
    ax.set_title("Adjective Count vs Complexity Score")
    ax.set_xlabel("Number of Adjectives")
    ax.set_ylabel("Complexity Score")
    save_plot(fig, "adjective_vs_complexity.png")

    # 5. Number of named entities vs complexity score
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['num_named_entities'], y=df['complexity_score'], ax=ax)
    ax.set_title("Named Entity Count vs Complexity Score")
    ax.set_xlabel("Number of Named Entities")
    ax.set_ylabel("Complexity Score")
    save_plot(fig, "entities_vs_complexity.png")

    # 6. Pie chart of known_template values
    fig, ax = plt.subplots()
    df['known_template'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel('')
    ax.set_title("Known Template Distribution")
    save_plot(fig, "known_template_pie.png")

    # 7. Countplot of explicitness
    fig, ax = plt.subplots()
    sns.countplot(x='explicitness', data=df, ax=ax)
    ax.set_title("Explicitness Classification")
    save_plot(fig, "explicitness_count.png")

    # 8. Heatmap of correlation matrix between numeric fields
    numeric_cols = ['word_count', 'unique_word_count', 'comma_count', 'complexity_score',
                    'num_adjectives', 'num_named_entities']
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix of Numeric Features")
    save_plot(fig, "correlation_matrix.png")


if __name__ == "__main__":
    db_path = "./extracted_features.db"
    df = load_data_from_db(db_path)
    generate_visualizations(df)

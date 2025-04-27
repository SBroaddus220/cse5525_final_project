import pandas as pd
import argparse
from pathlib import Path

def generate_tabular_table(df: pd.DataFrame, num_rows: int = None) -> str:
    """
    Generate a LaTeX table* environment using tabular with the style matching standard ACL tables.
    
    Args:
        df (pd.DataFrame): DataFrame with 'keyphrase' and 'adjusted_scaled_mean_shap'.
        num_rows (int, optional): Number of rows to include (default: all).
    
    Returns:
        str: LaTeX code.
    """
    if num_rows is not None:
        df = df.head(num_rows)

    lines = []
    lines.append("\\begin{table*}")
    lines.append("  \\centering")
    lines.append("  \\begin{tabular}{ll}")
    lines.append("    \\hline")
    lines.append("    \\textbf{Keyphrase} & \\textbf{Adjusted Scaled Mean SHAP} \\\\")
    lines.append("    \\hline")

    for _, row in df.iterrows():
        key = row['keyphrase'].replace('_', '\\_')
        value = f"{row['adjusted_scaled_mean_shap']:.4f}"
        lines.append(f"    \\verb|{key}| & {value} \\\\")

    lines.append("    \\hline")
    lines.append("  \\end{tabular}%")
    lines.append("  \\caption{Top Keyphrases Ranked by Adjusted Scaled Mean SHAP}")
    lines.append("\\end{table*}")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table* from CSV.")
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file.")
    parser.add_argument("--num_rows", type=int, default=None, help="Number of rows to include.")
    parser.add_argument("--output", type=Path, help="Path to save the LaTeX output.", default=None)

    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    latex_table = generate_tabular_table(df, num_rows=args.num_rows)

    if args.output:
        args.output.write_text(latex_table, encoding="utf-8")
        print(f"Wrote LaTeX table to {args.output}")
    else:
        print(latex_table)

if __name__ == "__main__":
    main()

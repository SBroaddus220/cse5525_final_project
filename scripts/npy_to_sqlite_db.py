import numpy as np
import sqlite3
from pathlib import Path
from typing import Union

def npy_to_sqlite(npy_path: Union[str, Path], sqlite_path: Union[str, Path], table_name: str = "data") -> None:
    """
    Convert a .npy file (containing a 2D array or structured array) into a SQLite database table.

    Args:
        npy_path (str or Path): Path to the .npy file.
        sqlite_path (str or Path): Path to the SQLite .db file to create or update.
        table_name (str): Name of the table to insert data into.
    """
    npy_path = Path(npy_path)
    sqlite_path = Path(sqlite_path)

    # Load the NumPy array
    data = np.load(npy_path, allow_pickle=True)

    # Connect to SQLite database
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # If it's a structured array (e.g., dtype=[('col1', float), ('col2', int)]):
    if data.dtype.names:
        columns = data.dtype.names
        col_defs = ', '.join(f"{col} TEXT" for col in columns)
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})")
        for row in data:
            cursor.execute(f"INSERT INTO {table_name} VALUES ({','.join(['?']*len(columns))})", tuple(row))
    else:
        # Assume it's a regular 2D array
        num_cols = data.shape[1] if data.ndim > 1 else 1
        col_defs = ', '.join(f"col{i} REAL" for i in range(num_cols))
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})")

        if data.ndim == 1:
            data = data[:, np.newaxis]

        for row in data:
            cursor.execute(f"INSERT INTO {table_name} VALUES ({','.join(['?']*num_cols)})", tuple(row))

    conn.commit()
    conn.close()
    print(f"Saved {data.shape[0]} rows to {sqlite_path} in table '{table_name}'.")

# Example usage
npy_to_sqlite("GradientBoosting_shap.npy", "gbshap.db", table_name="npy_data")

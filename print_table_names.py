import sqlite3

def get_table_names(db_path: str) -> None:
    """
    Prints all table names from the SQLite database in Python list format.

    Args:
        db_path (str): Path to the SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    print("metric_table_names = [")
    for table in tables:
        print(f"    '{table}',")
    print("]")

    conn.close()

# Example usage
get_table_names("data/database.sqlite3")

import sqlite3

def main():
    # Path to your SQLite database file
    db_path = "data/database.sqlite3"

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query:
    # 1. Counts rows by metric_name (count_instances)
    # 2. Computes average duration in seconds (avg_duration_seconds)
    # 3. Groups by metric_name
    # 4. Sorts by avg_duration_seconds DESC
    query = """
        SELECT
            metric_name,
            COUNT(*) AS count_instances,
            AVG((julianday(end_time) - julianday(start_time)) * 86400) AS avg_duration_seconds
        FROM metric_computation_times
        GROUP BY metric_name
        ORDER BY avg_duration_seconds DESC
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    # Define column widths
    col1_width = 30
    col2_width = 8
    col3_width = 15

    # Prepare headers with fixed width formatting
    header = (
        f"{'METRIC NAME':<{col1_width}}"
        f"{'COUNT':>{col2_width}}"
        f"{'AVERAGE (s)':>{col3_width}}"
    )
    print(header)
    print("-" * (col1_width + col2_width + col3_width))

    # Print rows with fixed width formatting
    for metric_name, count_instances, avg_duration in rows:
        print(
            f"{metric_name:<{col1_width}}"
            f"{count_instances:>{col2_width}}"
            f"{avg_duration:>{col3_width}.2f}"
        )

    conn.close()

if __name__ == "__main__":
    main()

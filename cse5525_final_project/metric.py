# -*- coding: utf-8 -*-

"""
Framework for metrics in the application.
"""

# **** IMPORTS ****
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# **** LOCAL IMPORTS ****
from cse5525_final_project.sqlite_table import SQLITETable

# **** LOGGING ****
logger = logging.getLogger(__name__)

# **** ABSTRACT CLASSES ****
class Metric:
    """Abstract base class for some metric to analyze some data."""
    
    class MetricTable(SQLITETable):
        """Inner classes for managing the metric in the SQLITE table."""
        table_name: str = ""
        required_columns: set = set()
        
        @classmethod
        def create_table(cls, conn: sqlite3.Connection):
            raise NotImplementedError("Subclasses must implement create_table()")

        @classmethod
        def entry_exists(cls, conn: sqlite3.Connection, uuid: str) -> bool:
            row = conn.execute(
                f"SELECT 1 FROM {cls.table_name} WHERE uuid = ?",
                (uuid,)
            ).fetchone()
            return row is not None


    @classmethod
    def compute_and_store(cls, conn: sqlite3.Connection, file_path: Path) -> None:
        """
        Abstract method to compute and store the metric.
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement compute_and_store()")
    

# **** CLASSES ****
class MetricComputationTimes(SQLITETable):
    """
    A SQLITE table to store start and end times of each metric computation for each image UUID.

    Attributes:
        table_name (str): The name of the table.
        required_columns (set): Required column names for this table.
    """

    table_name: str = "metric_computation_times"
    required_columns: set = {"uuid", "metric_name", "start_time", "end_time"}

    @classmethod
    def create_table(cls, conn: sqlite3.Connection) -> None:
        """
        Create the table if it doesn't already exist.

        Args:
            conn (sqlite3.Connection): The active sqlite connection.
        """
        # Create table, storing times as TEXT in ISO 8601 format
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {cls.table_name} (
                uuid TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                PRIMARY KEY (uuid, metric_name, start_time, end_time)
            )
            """
        )
        conn.commit()

    @classmethod
    def insert_record(cls, conn: sqlite3.Connection, data: Dict[str, Any]) -> Optional[int]:
        """
        Insert a single record into the table if it doesn't already exist.
        Returns the newly inserted record ID, or None if no insert performed.

        Args:
            conn (sqlite3.Connection): The active sqlite connection.
            data (Dict[str, Any]): The record data to be inserted.

        Returns:
            Optional[int]: The last inserted row ID, or None if already existed.
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        sql = f"INSERT OR IGNORE INTO {cls.table_name} ({columns}) VALUES ({placeholders})"
        conn.execute(sql, list(data.values()))
        conn.commit()

        row = conn.execute("SELECT last_insert_rowid()").fetchone()
        if row and row[0]:
            return int(row[0])
        return None

    
# ****
if __name__ == "__main__":
    raise RuntimeError("This module is not meant to be run directly.")

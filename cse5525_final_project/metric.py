# -*- coding: utf-8 -*-

"""
Framework for metrics in the application.
"""

# **** IMPORTS ****
import sqlite3
import logging
from pathlib import Path

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
    
# ****
if __name__ == "__main__":
    raise RuntimeError("This module is not meant to be run directly.")

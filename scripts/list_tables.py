# -*- coding: utf-8 -*-

"""
Prompts the user for sqlite3 database path and lists all tables and their records.
"""

# **** IMPORTS ****
import sqlite3

# **** FUNCTIONS ****
def list_tables_and_records(db_path: str):
    """
    Lists all tables and their records in the given SQLite database.

    Args:
        db_path (str): Path to the SQLite database file.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query to get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if tables:
            print("Tables and their records in the database:")
            for table in tables:
                table_name = table[0]
                print(f"\nTable: {table_name}")
                
                # Query to get all records from the table
                cursor.execute(f"SELECT * FROM {table_name}")
                records = cursor.fetchall()
                
                if records:
                    for record in records:
                        print(record)
                else:
                    print("(No records found)")
        else:
            print("No tables found in the database.")
        
        conn.close()
    except sqlite3.Error as e:
        print(f"Error accessing database: {e}")

# **** MAIN FUNCTION ****
def main():
    db_path = input("Enter the path to the SQLite database: ")
    list_tables_and_records(db_path)

# ****
if __name__ == "__main__":
    main()

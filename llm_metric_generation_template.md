<instructions>
In my application, I have a set of images which for each I generate some metrics. These metrics are stored in a sqlite database to enable persistance with sufficient ACID compliance. Each metric has a class where they define the table that can store the metric and the logic to compute and store the metric. 

I want you to generate a new class for the following metric(s): 
# Add metric(s) here

Please ONLY provide the implementation for the metric. DO NOT TALK outside of code blocks. Please add type hints and docstrings in Google's format. Please add comments for any large step in any algorithm.
</instructions>

Provided below is some existing logic to reference for creating the logic. Please reimplement this in the class(es). If more efficient, please add another method to separate the computation logic.
<metric_calculation_logic>
# Remove if not needed
</metric_calculation_logic>

Provided below is the abstract class for every metric.
<base_class>
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
</base_class>
<example>
```python
class Dimension(Metric):
    """Represents the dimensions for a file."""

    class Table(Metric.MetricTable):
        table_name = "dimension_metrics"
        required_columns = {"uuid", "width", "height"}

        @classmethod
        def create_table(cls, conn: sqlite3.Connection):
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {cls.table_name} (
                    uuid TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL
                );
                """
            )
            conn.commit()

    @classmethod
    def compute_and_store(cls, conn: sqlite3.Connection, file_path: Path) -> None:
        cls.Table: Metric.MetricTable
        cls.Table.create_table(conn)

        if not cls.Table.verify_table(conn):
            raise Exception("Required columns missing in dimension_metrics table.")

        uuid = get_image_uuid(file_path)

        if cls.Table.entry_exists(conn, uuid):
            logger.debug("Metric already exists for UUID: %s", uuid)
            return

        try:
            with Image.open(file_path) as img:
                width, height = img.size
        except Exception as e:
            logging.error("Error opening file %s: %s", file_path, e)
            raise

        cls.Table.insert_record(conn, {
            "uuid": uuid,
            "width": width,
            "height": height
        })

        logging.debug("Stored dimensions for %s: %dx%d", uuid, width, height)
```
</example>

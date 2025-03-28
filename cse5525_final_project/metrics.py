# -*- coding: utf-8 -*-

"""
Logic regarding metrics for application data.
"""

# **** IMPORTS ****
import sqlite3
import logging
import numpy as np
from PIL import Image
from pathlib import Path

# **** LOCAL IMPORTS ****
from cse5525_final_project.metric import Metric
from cse5525_final_project.util import get_image_uuid

# **** LOGGING ****
logger = logging.getLogger(__name__)

# **** CLASSES ****
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


class ColorHistogram(Metric):
    """Represents the color histogram for a file."""

    class Table(Metric.MetricTable):
        table_name = "color_histogram_metrics"
        required_columns = {"uuid"} | {f"{channel}_{i}" for channel in ("r", "g", "b") for i in range(256)}

        @classmethod
        def create_table(cls, conn: sqlite3.Connection) -> None:
            """
            Creates the color histogram table with columns for each RGB channel intensity (0-255).
            """
            columns = ",\n".join(
                [f"{channel}_{i} INTEGER NOT NULL" for channel in ("r", "g", "b") for i in range(256)]
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {cls.table_name} (
                    uuid TEXT NOT NULL,
                    {columns}
                );
                """
            )
            conn.commit()

    @classmethod
    def compute_and_store(cls, conn: sqlite3.Connection, file_path: Path) -> None:
        """
        Computes the RGB color histogram for the image and stores it in the database.

        Args:
            conn (sqlite3.Connection): The SQLite connection object.
            file_path (Path): The path to the image file.
        """
        cls.Table: Metric.MetricTable
        cls.Table.create_table(conn)

        if not cls.Table.verify_table(conn):
            raise Exception("Required columns missing in color_histogram_metrics table.")

        uuid = get_image_uuid(file_path)

        if cls.Table.entry_exists(conn, uuid):
            logger.debug("Metric already exists for UUID: %s", uuid)
            return

        try:
            # Compute color histograms
            histogram = cls.get_color_histogram(file_path)
        except Exception as e:
            logging.error("Error computing histogram for %s: %s", file_path, e)
            raise

        # Flatten histogram dictionary into column-value pairs
        record = {"uuid": uuid}
        for channel in ("r", "g", "b"):
            for i in range(256):
                record[f"{channel}_{i}"] = histogram[channel][i]

        # Insert the record into the database
        cls.Table.insert_record(conn, record)

        logging.debug("Stored color histogram for %s", uuid)

    @staticmethod
    def get_color_histogram(file_path: Path):
        """
        Returns a dictionary containing the histogram for each color channel.
        The histogram is given as a list of counts for intensity values [0..255].
        
        :param file_path: Path to the PNG image.
        :return: Dictionary with keys 'r', 'g', and 'b', each mapping to a list of 256 counts.
        """
        with Image.open(file_path) as img:
            # Ensure RGB
            img = img.convert('RGB')
            # Get histograms for each channel
            r_hist = img.getchannel(0).histogram()
            g_hist = img.getchannel(1).histogram()
            b_hist = img.getchannel(2).histogram()
            
        return {
            'r': r_hist,
            'g': g_hist,
            'b': b_hist
        }


class AverageBrightness(Metric):
    """Represents the average brightness for a file."""

    class Table(Metric.MetricTable):
        table_name = "average_brightness_metrics"
        required_columns = {"uuid", "average_brightness"}

        @classmethod
        def create_table(cls, conn: sqlite3.Connection):
            """
            Creates the table for storing average brightness if it does not exist.
            
            Args:
                conn (sqlite3.Connection): The SQLite connection object.
            """
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {cls.table_name} (
                    uuid TEXT NOT NULL,
                    average_brightness REAL NOT NULL
                );
                """
            )
            conn.commit()

    @staticmethod
    def calculate_average_brightness(file_path: Path) -> float:
        """
        Computes the average brightness of an image.

        Args:
            file_path (Path): Path to the image file.

        Returns:
            float: Average brightness value (0 = black, 255 = white).
        """
        with Image.open(file_path) as img:
            # Convert image to grayscale
            gray_img = img.convert('L')
            # Convert to NumPy array for numerical processing
            gray_data = np.array(gray_img, dtype=np.float32)

        # Calculate and return the mean pixel intensity
        return float(np.mean(gray_data))

    @classmethod
    def compute_and_store(cls, conn: sqlite3.Connection, file_path: Path) -> None:
        """
        Computes the average brightness and stores the result in the database.

        Args:
            conn (sqlite3.Connection): The SQLite connection object.
            file_path (Path): Path to the image file.
        """
        cls.Table: Metric.MetricTable
        cls.Table.create_table(conn)

        if not cls.Table.verify_table(conn):
            raise Exception("Required columns missing in average_brightness_metrics table.")

        uuid = get_image_uuid(file_path)

        if cls.Table.entry_exists(conn, uuid):
            logging.debug("Average brightness metric already exists for UUID: %s", uuid)
            return

        try:
            brightness = cls.calculate_average_brightness(file_path)
        except Exception as e:
            logging.error("Error calculating average brightness for %s: %s", file_path, e)
            raise

        cls.Table.insert_record(conn, {
            "uuid": uuid,
            "average_brightness": brightness
        })

        logging.debug("Stored average brightness for %s: %.2f", uuid, brightness)


class Contrast(Metric):
    """Represents the contrast metric for a given image."""

    class Table(Metric.MetricTable):
        table_name = "contrast_metrics"
        required_columns = {"uuid", "contrast"}

        @classmethod
        def create_table(cls, conn: sqlite3.Connection):
            """
            Creates the contrast_metrics table in the SQLite database if it does not already exist.
            
            Args:
                conn (sqlite3.Connection): SQLite connection object.
            """
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {cls.table_name} (
                    uuid TEXT NOT NULL,
                    contrast REAL NOT NULL
                );
                """
            )
            conn.commit()

    @staticmethod
    def _calculate_contrast(file_path: Path) -> float:
        """
        Calculates the contrast of an image using the standard deviation
        of grayscale pixel values (RMS contrast).
        
        Args:
            file_path (Path): Path to the image file.
        
        Returns:
            float: RMS contrast of the image.
        """
        with Image.open(file_path) as img:
            # Convert image to grayscale
            gray_img = img.convert('L')
            # Convert to numpy array for pixel value processing
            gray_data = np.array(gray_img, dtype=np.float32)

        # Compute standard deviation of intensity as the contrast measure
        return float(np.std(gray_data))

    @classmethod
    def compute_and_store(cls, conn: sqlite3.Connection, file_path: Path) -> None:
        """
        Computes the contrast of the image and stores it in the database.
        
        Args:
            conn (sqlite3.Connection): SQLite connection object.
            file_path (Path): Path to the image file.
        """
        cls.Table: Metric.MetricTable
        cls.Table.create_table(conn)

        if not cls.Table.verify_table(conn):
            raise Exception("Required columns missing in contrast_metrics table.")

        uuid = get_image_uuid(file_path)

        if cls.Table.entry_exists(conn, uuid):
            logger.debug("Contrast metric already exists for UUID: %s", uuid)
            return

        try:
            contrast = cls._calculate_contrast(file_path)
        except Exception as e:
            logging.error("Error calculating contrast for %s: %s", file_path, e)
            raise

        cls.Table.insert_record(conn, {
            "uuid": uuid,
            "contrast": contrast
        })

        logging.debug("Stored contrast for %s: %.4f", uuid, contrast)


class SaturationScore(Metric):
    """Computes and stores the average saturation score of an image."""

    class Table(Metric.MetricTable):
        table_name = "saturation_metrics"
        required_columns = {"uuid", "saturation"}

        @classmethod
        def create_table(cls, conn: sqlite3.Connection):
            """Creates the saturation_metrics table if it doesn't exist."""
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {cls.table_name} (
                    uuid TEXT NOT NULL,
                    saturation REAL NOT NULL
                );
                """
            )
            conn.commit()

        @classmethod
        def insert_record(cls, conn: sqlite3.Connection, record: dict):
            """Inserts a new record into the saturation_metrics table."""
            conn.execute(
                f"""
                INSERT INTO {cls.table_name} (uuid, saturation)
                VALUES (?, ?);
                """,
                (record["uuid"], record["saturation"])
            )
            conn.commit()

    @staticmethod
    def compute_saturation_score(file_path: Path) -> float:
        """
        Computes the average saturation score by converting the image to HSV.

        Args:
            file_path (Path): Path to the image file.

        Returns:
            float: Average saturation score in the range [0, 1].
        """
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            rgb_data = np.array(img, dtype=np.float32) / 255.0  # normalize to [0, 1]

        # Split channels
        r = rgb_data[:, :, 0]
        g = rgb_data[:, :, 1]
        b = rgb_data[:, :, 2]

        # Compute max and min per pixel
        cmax = np.max(rgb_data, axis=2)
        cmin = np.min(rgb_data, axis=2)
        delta = cmax - cmin

        # Compute saturation channel
        saturation = np.zeros_like(cmax)
        mask = cmax != 0
        saturation[mask] = delta[mask] / cmax[mask]

        # Return the mean saturation
        return float(np.mean(saturation))

    @classmethod
    def compute_and_store(cls, conn: sqlite3.Connection, file_path: Path) -> None:
        """
        Computes and stores the saturation score in the database.

        Args:
            conn (sqlite3.Connection): Connection to the SQLite database.
            file_path (Path): Path to the image file.
        """
        cls.Table.create_table(conn)

        if not cls.Table.verify_table(conn):
            raise Exception("Required columns missing in saturation_metrics table.")

        uuid = get_image_uuid(file_path)

        if cls.Table.entry_exists(conn, uuid):
            logging.debug("Saturation metric already exists for UUID: %s", uuid)
            return

        try:
            saturation_score = cls.compute_saturation_score(file_path)
        except Exception as e:
            logging.error("Failed to compute saturation for %s: %s", file_path, e)
            raise

        cls.Table.insert_record(conn, {
            "uuid": uuid,
            "saturation": saturation_score
        })

        logging.debug("Stored saturation score for %s: %.4f", uuid, saturation_score)


# ****
if __name__ == "__main__":
    raise RuntimeError("This module is not meant to be run directly.")

# -*- coding: utf-8 -*-

"""
This module holds variables used by other modules.
The data here can be edited, just be careful.
"""

# **** IMPORTS ****
import logging
from pathlib import Path

# **** LOGGING ****
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------
# CONFIGURATION VARIABLES
# ------------------------------------------------------------------------------------------

# **** CONSTANTS ****
IMAGE_METRIC_COMPUTATION_TIME_LOGGING_THRESHOLD = 5  # Seconds

# **** PATHS ****
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR / "cse5525_final_project"
DATA_DIR = BASE_DIR / "data"
METRICS_FILE_PATH = PROJECT_DIR / "metrics.py"  # Path to the metrics file
DB_PATH = DATA_DIR / "database.sqlite3"
IMAGES_DIR = DATA_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"

# **** LOGGING CONFIGURATION ****
# Logging configurations
LOG_FILE_PATH = BASE_DIR / "program_log.txt"

LOGGER_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # Doesn't disable other loggers that might be active
    "formatters": {
        "default": {
            "format": "[%(levelname)s][%(funcName)s] | %(asctime)s | %(message)s",
        },
        "simple": {  # Used for console logging
            "format": "[%(levelname)s][%(funcName)s] | %(message)s",
        },
    },
    "handlers": {
        "logfile": {
            "class": "logging.FileHandler",  # Basic file handler
            "formatter": "default",
            "level": "WARNING",
            "filename": LOG_FILE_PATH.as_posix(),
            "mode": "a",
            "encoding": "utf-8",
        },
        "console": {
            "class": "logging.StreamHandler",  # Basic stream handler
            "formatter": "simple",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {  # Simple program, so root logger uses all handlers
        "level": "DEBUG",
        "handlers": [
            "logfile",
            "console",
        ]
    }
}


# ****
if __name__ == "__main__":
    raise Exception("This file is not meant to run on its own.")

# -*- coding: utf-8 -*-

"""
Utilities for the application.
"""

# **** IMPORTS ****
import re
import json
import inspect
import logging
import importlib.util
from pathlib import Path
from typing import Optional, Type, List

# **** LOGGING ****
logger = logging.getLogger(__name__)

# **** FUNCTIONS ****
def sanitize_sqlite_identifier(name: str) -> str:
    """Replaces problematic characters in SQLite identifiers with underscores."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def sanitize_classname(name: str) -> str:
    """Sanitizes a string to be a valid Python class name."""
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return name[0].upper() + name[1:] if name else "Unnamed"

def get_image_uuid(file_path: Path) -> str:
    """
    Returns the UUID of the image file.
    The UUID is assumed to be the file's stem (filename without extension).
    Args:
        file_path (Path): The path to the image file.
    Returns:
        str: The UUID of the image (file stem).
    """
    return file_path.stem


def load_image_from_uuid(search_directory: Path, image_uuid: str) -> Optional[Path]:
    """
    Searches through the given directory (recursively) for the first file matching the UUID (stem).
    Args:
        search_directory (Path): The base directory to search in.
        image_uuid (str): The UUID (filename stem) to match.
    Returns:
        Optional[Path]: The first path found with a matching stem, or None if none found.
    """
    logger.debug(f"Searching for image with UUID: {image_uuid} in {search_directory}")
    for candidate in search_directory.rglob("*"):
        if candidate.is_file() and candidate.stem == image_uuid:
            return candidate
    return None


def discover_classes_in_file(file_path: Path, base_class: Type) -> List:
    """
    Discovers any subclasses of a given base class within a single .py file.

    Args:
        file_path (Path): Path to the Python file containing potential class definitions.
        base_class (Type): The base class to search for subclasses.

    Returns:
        List: Instantiated subclasses of the specified base class.
    """
    discovered_classes = []

    if file_path.suffix != ".py":
        return discovered_classes

    module_name = file_path.stem

    # Load the module from the file
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Inspect classes defined in the module
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, base_class) and obj is not base_class:
                logger.info(f"Discovered {obj.__name__}")
                discovered_classes.append(obj)

    return discovered_classes

def load_json_documents(directory: Path) -> List[dict]:
    """
    Recursively loads all JSON documents in a directory and its subdirectories,
    returning them as a list of dictionaries.

    Args:
        directory (Path): Path to the directory containing JSON files.

    Returns:
        List[dict]: List of dictionaries loaded from JSON files.
    """
    documents = []

    for file_path in directory.rglob("*.json"):
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    documents.append(data)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from {file_path}")

    return documents

def combine_dictionaries(dicts: List[dict]) -> dict:
    """
    Combines a list of dictionaries into a single dictionary.

    Args:
        dicts (List[dict]): List of dictionaries to combine.

    Returns:
        dict: A single merged dictionary containing all key-value pairs.
    """
    combined = {}

    for d in dicts:
        combined.update(d)

    return combined

# ****
if __name__ == "__main__":
    raise RuntimeError("This module is not meant to be run directly.")

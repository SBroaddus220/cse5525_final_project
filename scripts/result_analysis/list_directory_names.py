import argparse
import logging
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def list_directories_recursively(base_path: Path) -> List[Path]:
    """Recursively list all directories starting from the base path.

    Args:
        base_path (Path): The root directory to start searching from.

    Returns:
        List[Path]: A list of directory paths found recursively.
    """
    directories = []
    for path in base_path.rglob('*'):
        if path.is_dir():
            directories.append(path)
    return directories


def save_directories_to_file(directories: List[Path], output_file: Path) -> None:
    """Save the list of directories to a text file.

    Args:
        directories (List[Path]): List of directory paths.
        output_file (Path): Path to the output text file.
    """
    with output_file.open('w', encoding='utf-8') as f:
        for directory in directories:
            f.write(str(directory) + '\n')


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Recursively list directories and save to a file.')
    parser.add_argument('directory', type=str, help='Base directory to start listing from.')
    parser.add_argument('output', type=str, help='Output text file path.')
    return parser.parse_args()


def main() -> None:
    """Main function to run the script."""
    args = parse_args()
    base_path = Path(args.directory)
    output_file = Path(args.output)

    if not base_path.exists() or not base_path.is_dir():
        logging.error(f"The specified path {base_path} does not exist or is not a directory.")
        return

    logging.info(f"Listing directories under {base_path}")
    directories = list_directories_recursively(base_path)

    logging.info(f"Found {len(directories)} directories. Saving to {output_file}")
    save_directories_to_file(directories, output_file)
    logging.info("Finished saving directories.")


if __name__ == "__main__":
    main()

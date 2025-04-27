import os

def remove_png_files(directory: str) -> None:
    """
    Recursively removes all .png files in the provided directory.

    Args:
        directory (str): Path to the directory to search.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    dir_to_clean = input("Enter the path to the directory: ").strip()
    if os.path.isdir(dir_to_clean):
        remove_png_files(dir_to_clean)
    else:
        print("Invalid directory.")

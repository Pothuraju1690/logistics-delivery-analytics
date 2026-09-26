import os
import pandas as pd

def check_file_exists(file_path):
    """Check whether the given file exists."""
    return os.path.exists(file_path)

def load_data(file_path):
    """Load CSV data into a pandas DataFrame."""
    if not check_file_exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)
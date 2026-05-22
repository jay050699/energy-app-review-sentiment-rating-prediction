import pandas as pd
from pathlib import Path


def load_dataset(file_path):
    """
    Load a CSV dataset from the given file path.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    return pd.read_csv(path)


def load_processed_reviews():
    """
    Load the cleaned energy app review dataset.
    """

    return load_dataset("data/processed/energy_app_reviews_cleaned.csv")


def load_raw_reviews():
    """
    Load the raw energy app review dataset.
    """

    return load_dataset("data/raw/energy_app_reviews_raw.csv")
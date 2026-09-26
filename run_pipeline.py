from src.data_loader import load_data
from src.data_cleaner import clean_data, save_cleaned_data


RAW_FILE = "data/raw/logistics_data.csv"
CLEANED_FILE = "data/cleaned/logistics_cleaned.csv"


def run_pipeline():
    """Execute complete data pipeline."""

    print("Loading raw data...")
    df = load_data(RAW_FILE)

    print("Checking data quality...")
    print("Cleaning data...")

    df = clean_data(df)

    print("Applying business rules...")

    print("Saving cleaned data...")
    save_cleaned_data(df, CLEANED_FILE)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
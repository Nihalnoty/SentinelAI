from pathlib import Path

import pandas as pd


# Project root: SentinelAI/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "arab_potash copy"
    / "CFG_42_EDS_Data_clean.xlsx"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "centrifuge_clean.parquet"


def load_raw_data() -> pd.DataFrame:
    """Load the raw centrifuge dataset from Excel."""
    df = pd.read_excel(RAW_FILE)

    # Remove completely empty columns.
    df = df.dropna(axis=1, how="all")

    # Convert timestamp to datetime.
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort chronologically.
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def validate_data(df: pd.DataFrame) -> None:
    """Run basic data-quality checks."""
    required_columns = [
        "Date",
        "Torque",
        "Vibration",
        "Bearing_Temp_SolidEnd",
        "Bearing_Temp_LiquidEnd",
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if df[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if df["Date"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Timestamps are not sorted correctly.")


def save_processed_data(df: pd.DataFrame) -> None:
    """Save the cleaned dataset in Parquet format."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_FILE, index=False)


def main() -> None:
    print("Loading SentinelAI dataset...")

    df = load_raw_data()

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    validate_data(df)

    save_processed_data(df)

    print(f"Processed dataset saved to: {PROCESSED_FILE}")
    print("Data validation: PASSED")


if __name__ == "__main__":
    main()
    
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest


# Project root: SentinelAI/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "centrifuge_clean.parquet"
OUTPUT_DIR = PROJECT_ROOT / "reports"
OUTPUT_PATH = OUTPUT_DIR / "anomaly_results.csv"

TIMESTAMP_COLUMN = "Date"

SENSOR_COLUMNS = [
    "Torque",
    "Vibration",
    "Bearing_Temp_SolidEnd",
    "Bearing_Temp_LiquidEnd",
]


def main():
    print("Loading processed centrifuge dataset...")

    df = pd.read_parquet(DATA_PATH)
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])

    # Validate the expected columns.
    required_columns = [TIMESTAMP_COLUMN] + SENSOR_COLUMNS
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Sort chronologically and remove rows with invalid sensor readings.
    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    df = df.dropna(subset=SENSOR_COLUMNS).copy()

    if df.empty:
        raise ValueError("No valid sensor observations remain.")

    print(f"Rows available: {len(df):,}")
    print(f"Time range: {df[TIMESTAMP_COLUMN].min()} to "
          f"{df[TIMESTAMP_COLUMN].max()}")

    # Baseline model: identify unusual combinations of sensor readings.
    # contamination=0.01 requests approximately 1% of observations
    # to be flagged as anomalies; this is a modelling assumption,
    # not a measured equipment-failure rate.
    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )

    print("Fitting Isolation Forest...")
    predictions = model.fit_predict(df[SENSOR_COLUMNS])

    # sklearn returns -1 for anomalies and 1 for inliers.
    df["anomaly"] = predictions == -1

    # Higher scores indicate more unusual observations.
    df["anomaly_score"] = -model.score_samples(df[SENSOR_COLUMNS])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    anomaly_count = int(df["anomaly"].sum())
    anomaly_percentage = 100 * anomaly_count / len(df)

    print("\nBaseline model complete.")
    print(f"Observations analysed: {len(df):,}")
    print(f"Observations flagged: {anomaly_count:,}")
    print(f"Flagged percentage: {anomaly_percentage:.2f}%")
    print(f"Results saved to: {OUTPUT_PATH}")

    print("\nTen most unusual observations:")
    print(
        df.nlargest(10, "anomaly_score")[
            [TIMESTAMP_COLUMN] + SENSOR_COLUMNS + ["anomaly_score"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
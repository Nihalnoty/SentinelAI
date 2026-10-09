from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "centrifuge_model_data.parquet"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

ALL_SENSORS = [
    "Torque",
    "Vibration",
    "Bearing_Temp_SolidEnd",
    "Bearing_Temp_LiquidEnd",
]

THREE_SENSORS = [
    "Torque",
    "Vibration",
    "Bearing_Temp_SolidEnd",
]


def run_model(df, sensors, model_name):
    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )

    predictions = model.fit_predict(df[sensors])

    results = pd.DataFrame({
        "Date": df["Date"],
        "anomaly": predictions == -1,
        "anomaly_score": -model.score_samples(df[sensors]),
    })

    results["model"] = model_name

    return results


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Model A: Four sensors, using the original readings.
    model_a = run_model(
        df,
        ALL_SENSORS,
        "four_sensors",
    )

    # Model B: Three sensors, excluding liquid-end temperature.
    model_b = run_model(
        df,
        THREE_SENSORS,
        "three_sensors",
    )

    model_a.to_csv(
        REPORTS_DIR / "anomalies_four_sensors.csv",
        index=False,
    )

    model_b.to_csv(
        REPORTS_DIR / "anomalies_three_sensors.csv",
        index=False,
    )

    flagged_a = set(
        model_a.loc[model_a["anomaly"], "Date"]
    )

    flagged_b = set(
        model_b.loc[model_b["anomaly"], "Date"]
    )

    common = flagged_a & flagged_b
    union = flagged_a | flagged_b

    overlap = (
        len(common) / len(union) * 100
        if union
        else 0
    )

    print("\nSENSOR MODEL COMPARISON")
    print("=" * 50)

    print(f"Total observations: {len(df):,}")
    print(f"Four-sensor anomalies: {len(flagged_a):,}")
    print(f"Three-sensor anomalies: {len(flagged_b):,}")
    print(f"Shared anomaly timestamps: {len(common):,}")
    print(f"Jaccard overlap: {overlap:.2f}%")

    print("\nReports saved:")
    print("reports/anomalies_four_sensors.csv")
    print("reports/anomalies_three_sensors.csv")


if __name__ == "__main__":
    main()
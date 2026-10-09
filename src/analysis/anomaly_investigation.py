from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANOMALY_PATH = PROJECT_ROOT / "reports" / "anomalies_three_sensors.csv"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "centrifuge_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "anomaly_episodes.csv"


SENSOR_COLUMNS = [
    "Torque",
    "Vibration",
    "Bearing_Temp_SolidEnd",
    "Bearing_Temp_LiquidEnd",
]


def main():
    if not ANOMALY_PATH.exists():
        raise FileNotFoundError(f"Missing anomaly report: {ANOMALY_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing processed dataset: {DATA_PATH}")

    # Load predictions from the selected three-sensor baseline.
    predictions = pd.read_csv(ANOMALY_PATH, parse_dates=["Date"])
    predictions["anomaly"] = (
        predictions["anomaly"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    if predictions["anomaly"].isna().any():
        raise ValueError("Unexpected values found in the anomaly column.")

    # Load original sensor measurements.
    sensor_data = pd.read_parquet(DATA_PATH)
    sensor_data["Date"] = pd.to_datetime(sensor_data["Date"])

    # Attach sensor readings to every prediction timestamp.
    df = predictions.merge(
        sensor_data[["Date", *SENSOR_COLUMNS]],
        on="Date",
        how="left",
        validate="one_to_one",
    )

    if df[SENSOR_COLUMNS].isna().all(axis=1).any():
        raise ValueError("Some anomaly timestamps have no matching sensor readings.")

    df = df.sort_values("Date").reset_index(drop=True)

    # Detect gaps so separated anomaly periods cannot become one episode.
    time_gap = df["Date"].diff().ne(pd.Timedelta(minutes=1))
    status_change = df["anomaly"].ne(df["anomaly"].shift())

    df["episode_group"] = (time_gap | status_change).cumsum()

    anomalies = df[df["anomaly"]].copy()

    episodes = (
        anomalies.groupby("episode_group")
        .agg(
            start=("Date", "min"),
            end=("Date", "max"),
            flagged_minutes=("Date", "size"),
            mean_anomaly_score=("anomaly_score", "mean"),
            max_anomaly_score=("anomaly_score", "max"),
            mean_torque=("Torque", "mean"),
            min_torque=("Torque", "min"),
            max_torque=("Torque", "max"),
            mean_vibration=("Vibration", "mean"),
            max_vibration=("Vibration", "max"),
            mean_solid_temp=("Bearing_Temp_SolidEnd", "mean"),
            max_solid_temp=("Bearing_Temp_SolidEnd", "max"),
            mean_liquid_temp=("Bearing_Temp_LiquidEnd", "mean"),
        )
        .reset_index(drop=True)
    )

    # Elapsed time differs from the number of flagged observations.
    episodes["elapsed_minutes"] = (
        (episodes["end"] - episodes["start"]).dt.total_seconds() / 60
    ).astype(int) + 1

    episodes.insert(
        0,
        "incident_id",
        [f"SNT-{i:04d}" for i in range(1, len(episodes) + 1)],
    )

    episodes["status"] = "requires_investigation"
    episodes["interpretation"] = "anomaly_detected_not_confirmed_failure"

    # Put the most persistent episodes first.
    episodes = episodes.sort_values(
        ["flagged_minutes", "max_anomaly_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(OUTPUT_PATH, index=False)

    print("\nSENTINELAI — INCIDENT EPISODE GENERATION")
    print(f"Observations processed: {len(df):,}")
    print(f"Flagged observations: {len(anomalies):,}")
    print(f"Incident episodes generated: {len(episodes):,}")
    print(f"Total sensor timestamps unmatched: {df[SENSOR_COLUMNS].isna().all(axis=1).sum():,}")

    print("\nTen longest episodes:")
    print(episodes.head(10).to_string(index=False))

    print(f"\nSaved incident episodes to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "centrifuge_clean.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "reports"

SENSORS = [
    "Torque",
    "Vibration",
    "Bearing_Temp_SolidEnd",
    "Bearing_Temp_LiquidEnd",
]


def main():
    df = pd.read_parquet(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    print("DATA QUALITY AUDIT")
    print("=" * 65)
    print("Rows:", len(df))
    print("Date range:", df["Date"].min(), "to", df["Date"].max())

    # 1. Count zero-valued observations for each sensor.
    print("\n1. ZERO-VALUED READINGS")

    zero_counts = (df[SENSORS] == 0).sum()

    print(zero_counts.to_string())

    # 2. Find the largest absolute minute-to-minute changes.
    print("\n2. LARGEST ABSOLUTE CHANGES")

    change_records = []

    for sensor in SENSORS:
        change = df[sensor].diff().abs()

        largest_indices = change.nlargest(10).index

        for idx in largest_indices:
            change_records.append(
                {
                    "sensor": sensor,
                    "timestamp": df.loc[idx, "Date"],
                    "previous_value": df.loc[idx - 1, sensor]
                    if idx > 0 else None,
                    "current_value": df.loc[idx, sensor],
                    "absolute_change": change.loc[idx],
                }
            )

    changes_df = pd.DataFrame(change_records)
    changes_df = changes_df.sort_values(
        "absolute_change",
        ascending=False,
    )

    print(changes_df.head(20).round(4).to_string(index=False))

    # 3. Save audit outputs for later inspection.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zero_counts.rename("zero_count").to_csv(
        OUTPUT_DIR / "sensor_zero_counts.csv"
    )

    changes_df.to_csv(
        OUTPUT_DIR / "sensor_large_changes.csv",
        index=False,
    )

    print("\nSaved reports:")
    print(OUTPUT_DIR / "sensor_zero_counts.csv")
    print(OUTPUT_DIR / "sensor_large_changes.csv")


if __name__ == "__main__":
    main()
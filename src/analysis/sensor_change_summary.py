from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "centrifuge_clean.parquet"

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

    print("Dataset rows:", len(df))

    for month in ["2022-12", "2023-02", "2023-07", "2023-08"]:
        period = df[df["Date"].dt.to_period("M") == month]

        print(f"\n{'=' * 65}")
        print(f"MONTH: {month}")
        print(f"Observations: {len(period)}")

        print("\nSensor statistics:")
        print(period[SENSORS].describe().round(3).to_string())

        print("\nLargest absolute minute-to-minute changes:")
        for sensor in SENSORS:
            changes = period[sensor].diff().abs()
            largest_positions = changes.nlargest(3).index

            result = period.loc[largest_positions, ["Date", sensor]].copy()
            result["absolute_change"] = changes.loc[largest_positions]

            print(f"\n{sensor}:")
            print(result.to_string(index=False))


if __name__ == "__main__":
    main()
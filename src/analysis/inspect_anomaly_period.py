from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "reports" / "anomaly_results.csv"


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    start = pd.Timestamp("2022-12-25 20:00:00")
    end = pd.Timestamp("2022-12-29 06:00:00")

    period = df.loc[
        df["Date"].between(start, end)
    ].copy()

    if period.empty:
        print("No observations found for the selected period.")
        return

    sensors = [
        "Torque",
        "Vibration",
        "Bearing_Temp_SolidEnd",
        "Bearing_Temp_LiquidEnd",
    ]

    print("Inspection period:", start, "to", end)
    print("Rows:", len(period))
    print("Flagged observations:", int(period["anomaly"].astype(bool).sum()))

    print("\nSensor statistics:")
    print(
        period[sensors]
        .describe()
        .round(3)
        .to_string()
    )

    print("\nHourly summary:")
    hourly = (
        period.set_index("Date")
        .resample("1h")
        .agg(
            {
                "Torque": "mean",
                "Vibration": "mean",
                "Bearing_Temp_SolidEnd": "mean",
                "Bearing_Temp_LiquidEnd": "mean",
                "anomaly": "sum",
            }
        )
    )

    print(hourly.to_string())

    output_path = PROJECT_ROOT / "reports" / "december_anomaly_inspection.csv"
    hourly.to_csv(output_path)

    print("\nHourly inspection saved to:", output_path)


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "centrifuge_model_data.parquet"
)

FOUR_SENSOR_PATH = (
    PROJECT_ROOT
    / "reports"
    / "anomalies_four_sensors.csv"
)

THREE_SENSOR_PATH = (
    PROJECT_ROOT
    / "reports"
    / "anomalies_three_sensors.csv"
)


def inspect_period(df, four, three, start, end, title):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    period = df[
        df["Date"].between(start, end)
    ].copy()

    period = period.merge(
        four[["Date", "anomaly"]].rename(
            columns={"anomaly": "four_sensor_anomaly"}
        ),
        on="Date",
        how="left",
        validate="one_to_one",
    )

    period = period.merge(
        three[["Date", "anomaly"]].rename(
            columns={"anomaly": "three_sensor_anomaly"}
        ),
        on="Date",
        how="left",
        validate="one_to_one",
    )

    print(f"\n{title}")
    print("=" * 90)

    print(
        period[
            [
                "Date",
                "Torque",
                "Vibration",
                "Bearing_Temp_SolidEnd",
                "Bearing_Temp_LiquidEnd",
                "liquid_temp_zero",
                "four_sensor_anomaly",
                "three_sensor_anomaly",
            ]
        ].to_string(index=False)
    )


def main():
    df = pd.read_parquet(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    four = pd.read_csv(
        FOUR_SENSOR_PATH,
        parse_dates=["Date"],
    )

    three = pd.read_csv(
        THREE_SENSOR_PATH,
        parse_dates=["Date"],
    )

    inspect_period(
        df,
        four,
        three,
        "2023-07-17 16:43:00",
        "2023-07-17 17:40:00",
        "PERIOD 1: JULY 17 SENSOR DISAGREEMENT",
    )

    inspect_period(
        df,
        four,
        three,
        "2023-08-13 13:35:00",
        "2023-08-13 14:15:00",
        "PERIOD 2: AUGUST 13 SENSOR DISAGREEMENT",
    )


if __name__ == "__main__":
    main()
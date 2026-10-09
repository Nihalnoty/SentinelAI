from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "centrifuge_clean.parquet"
)

START = pd.Timestamp("2022-11-03 04:50:00")
END = pd.Timestamp("2022-11-30 14:29:00")

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

    # Inspect 30 minutes before and after the zero-reading period.
    before_start = START - pd.Timedelta(minutes=30)
    after_end = END + pd.Timedelta(minutes=30)

    period = df[df["Date"].between(before_start, after_end)].copy()

    period["phase"] = "outside_selected_period"
    period.loc[period["Date"].between(before_start, START), "phase"] = "before"
    period.loc[period["Date"].between(START, END), "phase"] = "zero_period"
    period.loc[period["Date"].between(END, after_end), "phase"] = "after"

    print("Selected zero-reading period:")
    print(START, "to", END)

    print("\nSensor statistics by phase:")
    print(
        period.groupby("phase")[SENSORS]
        .agg(["count", "mean", "min", "max"])
        .round(3)
        .to_string()
    )

    print("\nFirst 5 observations before the period:")
    print(
        df[df["Date"] < START]
        .tail(5)[["Date"] + SENSORS]
        .to_string(index=False)
    )

    print("\nFirst 5 observations inside the period:")
    print(
        df[df["Date"].between(START, END)]
        .head(5)[["Date"] + SENSORS]
        .to_string(index=False)
    )

    print("\nLast 5 observations inside the period:")
    print(
        df[df["Date"].between(START, END)]
        .tail(5)[["Date"] + SENSORS]
        .to_string(index=False)
    )

    print("\nFirst 5 observations after the period:")
    print(
        df[df["Date"] > END]
        .head(5)[["Date"] + SENSORS]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
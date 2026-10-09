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


def main():
    df = pd.read_parquet(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    sensor = "Bearing_Temp_LiquidEnd"

    # Identify observations where liquid-end temperature is exactly zero.
    df["is_zero"] = df[sensor].eq(0)

    # Assign a new group whenever the zero/non-zero status changes.
    df["group"] = df["is_zero"].ne(df["is_zero"].shift()).cumsum()

    # Summarize consecutive runs of zero readings.
    zero_runs = (
        df[df["is_zero"]]
        .groupby("group")
        .agg(
            start=("Date", "min"),
            end=("Date", "max"),
            observations=("Date", "size"),
            mean_torque=("Torque", "mean"),
            mean_vibration=("Vibration", "mean"),
            mean_solid_temp=("Bearing_Temp_SolidEnd", "mean"),
        )
        .reset_index(drop=True)
        .sort_values("observations", ascending=False)
    )

    print("LIQUID-END TEMPERATURE ZERO INVESTIGATION")
    print("=" * 60)

    print("Total observations:", len(df))
    print("Zero-valued observations:", int(df["is_zero"].sum()))
    print("Consecutive zero-reading runs:", len(zero_runs))

    print("\nTen longest zero-reading runs:")
    print(zero_runs.head(10).to_string(index=False))

    # Count zero readings by month.
    monthly = (
        df.groupby(df["Date"].dt.to_period("M"))["is_zero"]
        .agg(observations="size", zero_count="sum")
    )

    monthly["zero_percentage"] = (
        100 * monthly["zero_count"] / monthly["observations"]
    ).round(2)

    print("\nMonthly zero-reading summary:")
    print(monthly.to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zero_runs.to_csv(
        OUTPUT_DIR / "liquid_temp_zero_runs.csv",
        index=False,
    )

    monthly.to_csv(
        OUTPUT_DIR / "liquid_temp_zero_monthly.csv"
    )

    print("\nReports saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
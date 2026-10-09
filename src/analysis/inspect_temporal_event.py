from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "centrifuge_model_data.parquet"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "temporal_anomaly_results.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "december_24_temporal_event.csv"
)


def main():
    data = pd.read_parquet(DATA_PATH)
    results = pd.read_csv(
        RESULTS_PATH,
        parse_dates=["Date"],
    )

    data["Date"] = pd.to_datetime(data["Date"])

    # Inspect 30 minutes before and after the observed event.
    start = pd.Timestamp("2022-12-24 17:59:00")
    end = pd.Timestamp("2022-12-24 19:06:00")

    period = data[
        data["Date"].between(start, end)
    ].copy()

    period = period.merge(
        results[["Date", "anomaly", "anomaly_score"]],
        on="Date",
        how="left",
        validate="one_to_one",
    )

    columns = [
        "Date",
        "Torque",
        "Vibration",
        "Bearing_Temp_SolidEnd",
        "Bearing_Temp_LiquidEnd",
        "anomaly",
        "anomaly_score",
    ]

    period.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nDECEMBER 24 TEMPORAL EVENT")
    print("=" * 100)

    print("Time range:", start, "to", end)
    print("Observations:", len(period))
    print("Flagged observations:", int(period["anomaly"].sum()))

    print("\nHourly summary:")

    hourly = (
        period.set_index("Date")
        .resample("10min")
        .agg({
            "Torque": "mean",
            "Vibration": "mean",
            "Bearing_Temp_SolidEnd": "mean",
            "Bearing_Temp_LiquidEnd": "mean",
            "anomaly": "sum",
        })
    )

    print(hourly.to_string())

    print("\nHighest-scoring observations in this period:")

    top = period.nlargest(15, "anomaly_score")

    print(
        top[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
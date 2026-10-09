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
    / "december_transition_detail.csv"
)


def main():
    data = pd.read_parquet(DATA_PATH)
    results = pd.read_csv(
        RESULTS_PATH,
        parse_dates=["Date"],
    )

    data["Date"] = pd.to_datetime(data["Date"])

    start = pd.Timestamp("2022-12-24 17:55:00")
    end = pd.Timestamp("2022-12-24 19:05:00")

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

    print("\nDECEMBER 24 TRANSITION DETAIL")
    print("=" * 90)

    print("Observations:", len(period))
    print("Anomalies:", int(period["anomaly"].sum()))

    print(
        period[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
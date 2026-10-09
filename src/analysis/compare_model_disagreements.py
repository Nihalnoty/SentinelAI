from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

FOUR_SENSOR_PATH = REPORTS_DIR / "anomalies_four_sensors.csv"
THREE_SENSOR_PATH = REPORTS_DIR / "anomalies_three_sensors.csv"

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "centrifuge_model_data.parquet"
)


def main():
    four = pd.read_csv(FOUR_SENSOR_PATH, parse_dates=["Date"])
    three = pd.read_csv(THREE_SENSOR_PATH, parse_dates=["Date"])
    data = pd.read_parquet(DATA_PATH)

    data["Date"] = pd.to_datetime(data["Date"])

    comparison = four[
        ["Date", "anomaly", "anomaly_score"]
    ].merge(
        three[["Date", "anomaly", "anomaly_score"]],
        on="Date",
        suffixes=("_four", "_three"),
        validate="one_to_one",
    )

    disagreements = comparison[
        comparison["anomaly_four"] != comparison["anomaly_three"]
    ].copy()

    disagreements = disagreements.merge(
        data[
            [
                "Date",
                "Torque",
                "Vibration",
                "Bearing_Temp_SolidEnd",
                "Bearing_Temp_LiquidEnd",
                "liquid_temp_zero",
                "torque_zero",
            ]
        ],
        on="Date",
        how="left",
        validate="one_to_one",
    )

    disagreements["month"] = (
        disagreements["Date"].dt.to_period("M").astype(str)
    )

    disagreements.to_csv(
        REPORTS_DIR / "model_disagreements.csv",
        index=False,
    )

    print("\nMODEL DISAGREEMENT INVESTIGATION")
    print("=" * 55)
    print("Disagreement timestamps:", len(disagreements))

    print("\nDisagreements by month:")
    print(
        disagreements.groupby("month").size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nLiquid-temperature zero readings among disagreements:")
    print(int(disagreements["liquid_temp_zero"].sum()))

    print("\nTorque-zero readings among disagreements:")
    print(int(disagreements["torque_zero"].sum()))

    print("\nFirst 15 disagreements:")
    print(
        disagreements[
            [
                "Date",
                "anomaly_four",
                "anomaly_three",
                "Torque",
                "Vibration",
                "Bearing_Temp_SolidEnd",
                "Bearing_Temp_LiquidEnd",
                "liquid_temp_zero",
            ]
        ].head(15).to_string(index=False)
    )

    print("\nSaved: reports/model_disagreements.csv")


if __name__ == "__main__":
    main()
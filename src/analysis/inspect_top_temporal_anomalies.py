from pathlib import Path

import pandas as pd


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "temporal_anomaly_results.csv"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "centrifuge_model_data.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "top_temporal_anomalies.csv"
)


# --------------------------------------------------
# MAIN INVESTIGATION
# --------------------------------------------------

def main():

    print("\nLoading temporal anomaly results...")

    results = pd.read_csv(
        RESULTS_PATH,
        parse_dates=["Date"],
    )

    print("Loading original sensor data...")

    data = pd.read_parquet(DATA_PATH)

    data["Date"] = pd.to_datetime(data["Date"])

    # Select the 20 highest-scoring observations.
    # A higher anomaly_score means a more unusual observation
    # according to the fitted Isolation Forest.
    top = results.nlargest(
        20,
        "anomaly_score",
    ).copy()

    # Select the original sensor measurements and quality flags.
    sensor_columns = [
        "Date",
        "Torque",
        "Vibration",
        "Bearing_Temp_SolidEnd",
        "Bearing_Temp_LiquidEnd",
        "liquid_temp_zero",
        "torque_zero",
    ]

    sensor_data = data[sensor_columns].copy()

    # Merge the sensor readings with the model results.
    # The suffix prevents duplicate liquid_temp_zero column names.
    top = top.merge(
        sensor_data,
        on="Date",
        how="left",
        validate="one_to_one",
        suffixes=("_model", "_source"),
    )

    # Identify the correct liquid-temperature quality flag.
    if "liquid_temp_zero_source" in top.columns:
        liquid_zero_column = "liquid_temp_zero_source"
    elif "liquid_temp_zero" in top.columns:
        liquid_zero_column = "liquid_temp_zero"
    else:
        raise KeyError(
            "Could not find the liquid-temperature quality flag."
        )

    # Select the columns for inspection.
    display_columns = [
        "Date",
        "anomaly",
        "anomaly_score",
        "Torque",
        "Vibration",
        "Bearing_Temp_SolidEnd",
        "Bearing_Temp_LiquidEnd",
        liquid_zero_column,
        "torque_zero",
    ]

    # Rename the quality flag for a consistent output format.
    top = top.rename(
        columns={
            liquid_zero_column: "liquid_temp_zero"
        }
    )

    display_columns = [
        "Date",
        "anomaly",
        "anomaly_score",
        "Torque",
        "Vibration",
        "Bearing_Temp_SolidEnd",
        "Bearing_Temp_LiquidEnd",
        "liquid_temp_zero",
        "torque_zero",
    ]

    # Display the results.
    print("\nTOP 20 TEMPORAL ANOMALIES")
    print("=" * 100)

    print(
        top[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # Save the investigation results.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    top[display_columns].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nINVESTIGATION COMPLETE")
    print("=" * 50)
    print("Observations inspected:", len(top))
    print("Liquid-temperature zero readings:",
          int(top["liquid_temp_zero"].sum()))
    print("Torque-zero readings:",
          int(top["torque_zero"].sum()))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
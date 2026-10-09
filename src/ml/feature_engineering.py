from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "centrifuge_model_data.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "centrifuge_features.parquet"
)

SENSOR_COLUMNS = [
    "Torque",
    "Vibration",
    "Bearing_Temp_SolidEnd",
    "Bearing_Temp_LiquidEnd",
]


def main():
    df = pd.read_parquet(INPUT_PATH)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Confirm that the dataset contains one-minute intervals.
    time_gaps = df["Date"].diff()
    irregular_intervals = (
        time_gaps.notna()
        & (time_gaps != pd.Timedelta(minutes=1))
    )

    print("Irregular time intervals:", int(irregular_intervals.sum()))

    # Use the original readings for the first three sensors.
    # Use the masked liquid-temperature column for modelling.
    modelling_columns = [
        "Torque",
        "Vibration",
        "Bearing_Temp_SolidEnd",
    ]

    modelling_columns.append("Bearing_Temp_LiquidEnd_model")

    feature_names = []

    for column in modelling_columns:
        # Short-term rolling behaviour over 5 minutes.
        mean_5 = f"{column}_mean_5m"
        std_5 = f"{column}_std_5m"

        df[mean_5] = df[column].rolling(
            window=5,
            min_periods=5,
        ).mean()

        df[std_5] = df[column].rolling(
            window=5,
            min_periods=5,
        ).std()

        # Longer-term behaviour over 30 minutes.
        mean_30 = f"{column}_mean_30m"
        std_30 = f"{column}_std_30m"

        df[mean_30] = df[column].rolling(
            window=30,
            min_periods=30,
        ).mean()

        df[std_30] = df[column].rolling(
            window=30,
            min_periods=30,
        ).std()

        # Change relative to the previous minute.
        change = f"{column}_change_1m"
        df[change] = df[column].diff()

        feature_names.extend([
            mean_5,
            std_5,
            mean_30,
            std_30,
            change,
        ])

    # Record missing feature values instead of inventing sensor readings.
    df["liquid_temp_model_missing"] = (
        df["Bearing_Temp_LiquidEnd_model"].isna().astype("int8")
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print("\nFEATURE ENGINEERING COMPLETE")
    print("=" * 50)
    print("Rows:", len(df))
    print("New temporal features:", len(feature_names))
    print("Total columns:", len(df.columns))
    print("Liquid-temperature missing flags:",
          int(df["liquid_temp_model_missing"].sum()))
    print("Output:", OUTPUT_PATH)

    print("\nFirst five generated features:")
    print(feature_names[:5])


if __name__ == "__main__":
    main()
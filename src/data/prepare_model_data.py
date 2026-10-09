from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "centrifuge_clean.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_PATH = OUTPUT_DIR / "centrifuge_model_data.parquet"


def main():
    df = pd.read_parquet(INPUT_PATH)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Preserve explicit indicators of suspicious zero readings.
    df["liquid_temp_zero"] = (
        df["Bearing_Temp_LiquidEnd"] == 0
    ).astype("int8")

    df["torque_zero"] = (
        df["Torque"] == 0
    ).astype("int8")

    # Create a separate modelling column.
    # The original sensor column remains unchanged.
    df["Bearing_Temp_LiquidEnd_model"] = (
        df["Bearing_Temp_LiquidEnd"]
        .mask(df["liquid_temp_zero"] == 1)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUTPUT_PATH, index=False)

    print("MODEL DATA PREPARATION COMPLETE")
    print("=" * 55)

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    print(
        "Liquid temperature zero flags:",
        int(df["liquid_temp_zero"].sum()),
    )

    print(
        "Torque zero flags:",
        int(df["torque_zero"].sum()),
    )

    print(
        "Missing values in modelling temperature:",
        int(df["Bearing_Temp_LiquidEnd_model"].isna().sum()),
    )

    print("\nOriginal sensor data preserved.")
    print("Output saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
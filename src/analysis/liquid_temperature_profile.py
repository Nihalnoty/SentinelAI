from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "centrifuge_clean.parquet"
)

SENSOR = "Bearing_Temp_LiquidEnd"


def main():
    df = pd.read_parquet(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    temperature = df[SENSOR]

    categories = {
        "Exactly zero": temperature.eq(0),
        "Above zero and below 1": temperature.gt(0) & temperature.lt(1),
        "At least 1 and below 10": temperature.ge(1) & temperature.lt(10),
        "At least 10 and below 30": temperature.ge(10) & temperature.lt(30),
        "At least 30 and below 60": temperature.ge(30) & temperature.lt(60),
        "At least 60": temperature.ge(60),
        "Missing": temperature.isna(),
    }

    print("LIQUID-END TEMPERATURE PROFILE")
    print("=" * 60)

    total = len(df)

    for label, mask in categories.items():
        count = int(mask.sum())
        percentage = 100 * count / total

        print(
            f"{label:<30} "
            f"{count:>8,} observations "
            f"({percentage:>6.2f}%)"
        )

    print("\nTemperature statistics:")
    print(temperature.describe().round(3).to_string())

    print("\nMonthly mean and median:")
    monthly = (
        df.set_index("Date")[SENSOR]
        .resample("MS")
        .agg(["count", "mean", "median", "min", "max"])
        .round(3)
    )

    print(monthly.to_string())

    output_path = PROJECT_ROOT / "reports" / "liquid_temperature_profile.csv"
    monthly.to_csv(output_path)

    print("\nMonthly report saved to:", output_path)


if __name__ == "__main__":
    main()
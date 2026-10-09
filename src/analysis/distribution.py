from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "centrifuge_clean.parquet"
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


SENSORS = [
    "Torque",
    "Vibration",
    "Bearing_Temp_SolidEnd",
    "Bearing_Temp_LiquidEnd",
]


def load_data():
    return pd.read_parquet(DATA_FILE)


def analyze_distributions(df):
    print("\nDISTRIBUTION ANALYSIS")

    for sensor in SENSORS:
        series = df[sensor]

        print(f"\n--- {sensor} ---")
        print(f"Minimum: {series.min():.3f}")
        print(f"25th percentile: {series.quantile(0.25):.3f}")
        print(f"Median: {series.median():.3f}")
        print(f"75th percentile: {series.quantile(0.75):.3f}")
        print(f"Maximum: {series.max():.3f}")


def plot_distributions(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for sensor in SENSORS:
        plt.figure(figsize=(10, 5))

        plt.hist(df[sensor], bins=100)

        plt.title(f"{sensor} Distribution")
        plt.xlabel(sensor)
        plt.ylabel("Frequency")

        plt.tight_layout()

        output_file = (
            FIGURES_DIR
            / f"{sensor.lower()}_distribution.png"
        )

        plt.savefig(output_file)
        plt.close()

        print(f"Saved: {output_file}")


def main():
    print("Loading SentinelAI processed dataset...")

    df = load_data()

    analyze_distributions(df)
    plot_distributions(df)

    print("\nDistribution analysis completed successfully.")


if __name__ == "__main__":
    main()
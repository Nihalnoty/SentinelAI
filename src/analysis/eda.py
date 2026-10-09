from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Project root: SentinelAI/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "centrifuge_clean.parquet"
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def load_data():
    """Load the processed SentinelAI dataset."""
    return pd.read_parquet(DATA_FILE)


def basic_analysis(df):
    """Print basic statistics and sensor correlations."""

    sensors = [
        "Torque",
        "Vibration",
        "Bearing_Temp_SolidEnd",
        "Bearing_Temp_LiquidEnd",
    ]

    print("\nDATASET SHAPE")
    print(df.shape)

    print("\nSENSOR STATISTICS")
    print(df[sensors].describe().T)

    print("\nCORRELATION MATRIX")
    print(df[sensors].corr())


def plot_sensor_trends(df):
    """Create time-series plots for each sensor."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sensors = [
        "Torque",
        "Vibration",
        "Bearing_Temp_SolidEnd",
        "Bearing_Temp_LiquidEnd",
    ]

    for sensor in sensors:
        plt.figure(figsize=(12, 5))

        plt.plot(df["Date"], df[sensor])

        plt.title(f"{sensor} Over Time")
        plt.xlabel("Time")
        plt.ylabel(sensor)

        plt.tight_layout()

        output_file = FIGURES_DIR / f"{sensor.lower()}_trend.png"
        plt.savefig(output_file)
        plt.close()

        print(f"Saved: {output_file}")


def main():
    print("Loading SentinelAI processed dataset...")

    df = load_data()

    basic_analysis(df)

    plot_sensor_trends(df)

    print("\nEDA Step 1 completed successfully.")


if __name__ == "__main__":
    main()
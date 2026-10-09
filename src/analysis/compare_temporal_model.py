from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_flags(filename, column_name):
    path = REPORTS_DIR / filename

    df = pd.read_csv(path, parse_dates=["Date"])

    return df[["Date", "anomaly"]].rename(
        columns={"anomaly": column_name}
    )


def compare_models(df, first, second):
    a = set(df.loc[df[first], "Date"])
    b = set(df.loc[df[second], "Date"])

    shared = len(a & b)
    union = len(a | b)

    overlap = shared / union * 100 if union else 0

    print(f"\n{first} vs {second}")
    print(f"  First model anomalies: {len(a):,}")
    print(f"  Second model anomalies: {len(b):,}")
    print(f"  Shared timestamps: {shared:,}")
    print(f"  Jaccard overlap: {overlap:.2f}%")


def main():
    four = load_flags(
        "anomalies_four_sensors.csv",
        "four_sensor",
    )

    three = load_flags(
        "anomalies_three_sensors.csv",
        "three_sensor",
    )

    temporal = load_flags(
        "temporal_anomaly_results.csv",
        "temporal",
    )

    comparison = (
        four.merge(three, on="Date", validate="one_to_one")
        .merge(temporal, on="Date", validate="one_to_one")
    )

    print("\nTHREE-MODEL COMPARISON")
    print("=" * 50)

    compare_models(comparison, "four_sensor", "three_sensor")
    compare_models(comparison, "four_sensor", "temporal")
    compare_models(comparison, "three_sensor", "temporal")

    quality_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "centrifuge_model_data.parquet"
    )

    quality = pd.read_parquet(
        quality_path,
        columns=["Date", "liquid_temp_zero"],
    )

    temporal_quality = temporal.merge(
        quality,
        on="Date",
        validate="one_to_one",
    )

    flagged = temporal_quality["temporal"]
    zero = temporal_quality["liquid_temp_zero"].eq(1)

    print("\nTEMPORAL MODEL AND SENSOR QUALITY")
    print("=" * 50)
    print("Anomalies during liquid-temperature zero readings:",
          int((flagged & zero).sum()))
    print("Anomalies during nonzero readings:",
          int((flagged & ~zero).sum()))

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
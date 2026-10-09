from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "reports" / "anomaly_results.csv"


def main():
    df = pd.read_csv(RESULTS_PATH, parse_dates=["Date"])

    df["anomaly"] = (
        df["anomaly"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    if df["anomaly"].isna().any():
        raise ValueError("Unexpected values found in the anomaly column.")

    df["month"] = df["Date"].dt.to_period("M")

    summary = (
        df.groupby("month")
        .agg(
            observations=("anomaly", "size"),
            flagged=("anomaly", "sum"),
        )
    )

    summary["flagged_percentage"] = (
        100 * summary["flagged"] / summary["observations"]
    ).round(2)

    print("\nMonthly anomaly summary:")
    print(summary.to_string())

    output_path = PROJECT_ROOT / "reports" / "anomaly_monthly_summary.csv"
    summary.to_csv(output_path)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
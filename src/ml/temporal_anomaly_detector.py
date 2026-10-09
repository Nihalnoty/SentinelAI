from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "centrifuge_features.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "temporal_anomaly_results.csv"
)

SENSORS = [
    "Torque",
    "Vibration",
    "Bearing_Temp_SolidEnd",
]


def main():
    df = pd.read_parquet(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Use temporal features from the three selected sensors.
    feature_columns = []

    for sensor in SENSORS:
        feature_columns.extend([
            f"{sensor}_mean_5m",
            f"{sensor}_std_5m",
            f"{sensor}_mean_30m",
            f"{sensor}_std_30m",
            f"{sensor}_change_1m",
        ])

    X = df[feature_columns].copy()

    # Replace missing feature values using each feature's median.
    # This fills feature gaps, not the original sensor measurements.
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        IsolationForest(
            n_estimators=100,
            contamination=0.01,
            random_state=42,
            n_jobs=-1,
        ),
    )

    predictions = model.fit_predict(X)

    # Access the fitted Isolation Forest within the pipeline.
    forest = model.named_steps["isolationforest"]

    X_imputed = model.named_steps["simpleimputer"].transform(X)

    results = pd.DataFrame({
        "Date": df["Date"],
        "anomaly": predictions == -1,
        "anomaly_score": -forest.score_samples(X_imputed),
    })

    results["liquid_temp_zero"] = df["liquid_temp_zero"].astype("int8")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\nTEMPORAL ANOMALY DETECTION COMPLETE")
    print("=" * 55)
    print("Observations:", len(results))
    print("Features used:", len(feature_columns))
    print("Anomalies detected:", int(results["anomaly"].sum()))
    print(
        "Anomaly percentage:",
        round(results["anomaly"].mean() * 100, 2),
        "%",
    )
    print(
        "Anomalies during liquid-temperature zero readings:",
        int(
            (
                results["anomaly"]
                & (results["liquid_temp_zero"] == 1)
            ).sum()
        ),
    )
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
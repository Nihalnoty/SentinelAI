from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INCIDENTS_PATH = PROJECT_ROOT / "reports" / "anomaly_episodes.csv"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "centrifuge_clean.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports" / "incident_evidence"

SENSOR_COLUMNS = [
    "Torque",
    "Vibration",
    "Bearing_Temp_SolidEnd",
    "Bearing_Temp_LiquidEnd",
]

WINDOW_MINUTES = 60
CHANGE_WINDOW_MINUTES = 60


def load_incident(incident_id):
    if not INCIDENTS_PATH.exists():
        raise FileNotFoundError(f"Incident report not found: {INCIDENTS_PATH}")

    incidents = pd.read_csv(INCIDENTS_PATH, parse_dates=["start", "end"])
    matches = incidents.loc[incidents["incident_id"] == incident_id]

    if matches.empty:
        raise ValueError(f"Incident ID not found: {incident_id}")

    incident = matches.iloc[0]

    if incident["end"] < incident["start"]:
        raise ValueError("Incident end time is earlier than its start time.")

    return incident


def load_sensor_data(start, end):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    data = pd.read_parquet(DATA_PATH)

    required_columns = ["Date", *SENSOR_COLUMNS]
    missing_columns = [c for c in required_columns if c not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing dataset columns: {missing_columns}")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    for column in SENSOR_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.sort_values("Date")

    if data["Date"].duplicated().any():
        raise ValueError("Duplicate timestamps found in sensor dataset.")

    window_start = start - pd.Timedelta(minutes=WINDOW_MINUTES)
    window_end = end + pd.Timedelta(minutes=WINDOW_MINUTES)

    return data.loc[data["Date"].between(window_start, window_end)].copy()


def summarize_window(data, start, end):
    window = data.loc[data["Date"].between(start, end)]

    result = {
        "start": str(start),
        "end": str(end),
        "observations": int(len(window)),
        "expected_minute_observations": int(
            (end - start).total_seconds() / 60
        ) + 1,
        "sensors": {},
    }

    for column in SENSOR_COLUMNS:
        values = window[column].dropna()

        result["sensors"][column] = {
            "valid_readings": int(len(values)),
            "missing_readings": int(len(window) - len(values)),
            "mean": round(float(values.mean()), 6) if len(values) else None,
            "minimum": round(float(values.min()), 6) if len(values) else None,
            "maximum": round(float(values.max()), 6) if len(values) else None,
            "standard_deviation": (
                round(float(values.std()), 6)
                if len(values) > 1
                else (0.0 if len(values) == 1 else None)
            ),
        }

    return result


def compare_windows(before, during, after):
    comparison = {}

    for column in SENSOR_COLUMNS:
        before_mean = before["sensors"][column]["mean"]
        during_mean = during["sensors"][column]["mean"]
        after_mean = after["sensors"][column]["mean"]

        def percent_change(reference, current):
            if reference is None or current is None:
                return None

            if np.isclose(reference, 0.0):
                return None

            return round((current - reference) / abs(reference) * 100, 2)

        comparison[column] = {
            "before_mean": before_mean,
            "during_mean": during_mean,
            "after_mean": after_mean,
            "during_vs_before_percent": percent_change(
                before_mean, during_mean
            ),
            "after_vs_during_percent": percent_change(
                during_mean, after_mean
            ),
        }

    return comparison


def analyze_changes_within_incident(data, start, end):
    """Summarize sensor behaviour in consecutive 60-minute blocks."""
    incident_data = data.loc[data["Date"].between(start, end)].copy()

    if incident_data.empty:
        return []

    elapsed_minutes = (
        (incident_data["Date"] - start).dt.total_seconds() // 60
    ).astype(int)

    incident_data["incident_block"] = (
        elapsed_minutes // CHANGE_WINDOW_MINUTES
    ).astype(int)

    changes = []

    for block_id, block in incident_data.groupby("incident_block"):
        block_start = block["Date"].min()
        block_end = block["Date"].max()

        block_result = {
            "block_start": str(block_start),
            "block_end": str(block_end),
            "observations": int(len(block)),
            "sensors": {},
        }

        for column in SENSOR_COLUMNS:
            values = block[column].dropna()

            block_result["sensors"][column] = {
                "mean": round(float(values.mean()), 6) if len(values) else None,
                "minimum": round(float(values.min()), 6) if len(values) else None,
                "maximum": round(float(values.max()), 6) if len(values) else None,
                "valid_readings": int(len(values)),
            }

        changes.append(block_result)

    return changes


def build_findings(incident, before, during, after, comparison, blocks):
    findings = []

    findings.append(
        f"{incident['incident_id']} was flagged from "
        f"{incident['start']} to {incident['end']}."
    )

    findings.append(
        f"The incident contains {int(incident['flagged_minutes'])} "
        "flagged observations."
    )

    for column, values in comparison.items():
        findings.append(
            f"{column}: before mean={values['before_mean']}; "
            f"during mean={values['during_mean']}; "
            f"after mean={values['after_mean']}; "
            f"during-vs-before change="
            f"{values['during_vs_before_percent']}%."
        )

    findings.append(
        f"The incident was divided into {len(blocks)} consecutive "
        f"{CHANGE_WINDOW_MINUTES}-minute analysis blocks."
    )

    if before["observations"] < before["expected_minute_observations"]:
        findings.append(
            "The pre-incident window has fewer observations than expected."
        )

    if after["observations"] < after["expected_minute_observations"]:
        findings.append(
            "The post-incident window has fewer observations than expected."
        )

    if during["observations"] < during["expected_minute_observations"]:
        findings.append(
            "The incident window has fewer observations than expected."
        )

    findings.append(
        "Sensor changes and recovery comparisons are descriptive evidence. "
        "They do not independently confirm equipment failure or its cause."
    )

    return findings


def investigate(incident_id):
    incident = load_incident(incident_id)

    start = incident["start"]
    end = incident["end"]

    data = load_sensor_data(start, end)

    before = summarize_window(
        data,
        start - pd.Timedelta(minutes=WINDOW_MINUTES),
        start - pd.Timedelta(minutes=1),
    )

    during = summarize_window(data, start, end)

    after = summarize_window(
        data,
        end + pd.Timedelta(minutes=1),
        end + pd.Timedelta(minutes=WINDOW_MINUTES),
    )

    comparison = compare_windows(before, during, after)

    blocks = analyze_changes_within_incident(data, start, end)

    findings = build_findings(
        incident, before, during, after, comparison, blocks
    )

    report = {
        "incident": {
            "incident_id": str(incident["incident_id"]),
            "start": str(start),
            "end": str(end),
            "flagged_minutes": int(incident["flagged_minutes"]),
            "mean_anomaly_score": float(incident["mean_anomaly_score"]),
            "max_anomaly_score": float(incident["max_anomaly_score"]),
            "status": str(incident["status"]),
            "interpretation": str(incident["interpretation"]),
        },
        "analysis_windows": {
            "before": before,
            "during": during,
            "after": after,
        },
        "sensor_comparison": comparison,
        "within_incident_blocks": blocks,
        "findings": findings,
        "conclusion": {
            "confirmed_failure": False,
            "reason": (
                "Available anomaly predictions and sensor measurements "
                "do not independently confirm an equipment failure."
            ),
            "requires_human_review": True,
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / f"{incident_id}_evidence.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    print("\nSENTINELAI — INCIDENT EVIDENCE REPORT")
    print("=" * 55)
    print(f"Incident: {incident_id}")
    print(f"Start: {start}")
    print(f"End: {end}")
    print(f"Flagged observations: {int(incident['flagged_minutes'])}")
    print(f"Within-incident blocks: {len(blocks)}")

    print("\nSensor comparison: incident vs pre-incident mean")

    for column, values in comparison.items():
        print(
            f"{column}: before={values['before_mean']}, "
            f"during={values['during_mean']}, "
            f"after={values['after_mean']}, "
            f"change={values['during_vs_before_percent']}%"
        )

    print("\nFirst three within-incident blocks:")

    for block in blocks[:3]:
        print(
            f"{block['block_start']} to {block['block_end']} "
            f"({block['observations']} observations)"
        )

    print("\nFindings:")

    for finding in findings:
        print(f"- {finding}")

    print("\nConfirmed failure: NO")
    print("Human review required: YES")
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze sensor evidence for a SentinelAI incident."
    )
    parser.add_argument(
        "--incident",
        default="SNT-0015",
        help="Incident ID to investigate.",
    )
    args = parser.parse_args()

    investigate(args.incident)

if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f"\nERROR: {error}")
        raise SystemExit(1)
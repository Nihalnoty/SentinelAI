from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "reports" / "model_disagreements.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "disagreement_episodes.csv"


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Start a new episode when consecutive disagreement timestamps
    # are separated by more than one minute.
    time_gap = df["Date"].diff()

    df["episode_id"] = (
        time_gap.ne(pd.Timedelta(minutes=1))
    ).cumsum()

    episodes = (
        df.groupby("episode_id")
        .agg(
            start_time=("Date", "min"),
            end_time=("Date", "max"),
            disagreement_minutes=("Date", "size"),
            four_sensor_anomalies=("anomaly_four", "sum"),
            three_sensor_anomalies=("anomaly_three", "sum"),
            liquid_temp_zero_count=("liquid_temp_zero", "sum"),
            torque_zero_count=("torque_zero", "sum"),
        )
        .reset_index(drop=True)
    )

    episodes["duration_minutes"] = (
        (episodes["end_time"] - episodes["start_time"])
        .dt.total_seconds()
        .div(60)
        .add(1)
        .astype(int)
    )

    episodes = episodes.sort_values(
        "disagreement_minutes",
        ascending=False,
    )

    episodes.to_csv(OUTPUT_PATH, index=False)

    print("\nDISAGREEMENT EPISODE INVESTIGATION")
    print("=" * 60)
    print("Total disagreement timestamps:", len(df))
    print("Total disagreement episodes:", len(episodes))

    print("\nTop 15 episodes:")
    print(
        episodes.head(15).to_string(index=False)
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
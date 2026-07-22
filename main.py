import argparse
import pandas as pd
import os
from model.metrics.srcml.pipeline import main as run_pipeline


def main(path: str, sample: int | None = None):
    df = pd.read_csv(path)
    if sample is not None:
        df = df.sample(n=sample, random_state=42)
    print(f"Dataset size: {len(df)} rows")
    reports = run_pipeline(df)
    for combo, report in reports.items():
        print(f"\n=== {combo} ===")
        for k, v in report.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of rows to sample for a quick run (omit for full data)",
    )
    args = parser.parse_args()

    print("Training embedding model: SRCML")
    path = os.getenv("DATA_CSV", "data/output.csv")
    main(path=path, sample=args.sample)

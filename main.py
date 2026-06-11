import pandas as pd
import os
from model.metrics.srcml.embedding import main as run_embedding


def main(path: str):
    df = pd.read_csv(path).sample(n=100)
    reports = run_embedding(df, ["code", "code_xml", "code_ast_xml"])
    for rep, report in reports.items():
        print(f"\n=== {rep} ===")
        print(report)


if __name__ == "__main__":
    print("Training embedding model: SRCML")
    path = os.getenv("DATA_CSV", "data/output.csv")
    main(path=path)

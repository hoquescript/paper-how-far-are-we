import pandas as pd
import os
from model.embedding import kfold_embedding_model


def main(path: str):
    df = pd.read_csv(path)
    report = kfold_embedding_model(df, ["code", "ast", "combined"])
    print(report)


if __name__ == "__main__":
    path = os.getenv("DATA_CSV", "data/sample/python.csv")
    main(path=path)

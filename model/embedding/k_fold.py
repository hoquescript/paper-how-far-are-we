import numpy as np
import pandas as pd
from datetime import datetime


from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from utils.ast.ast_generator import generate_ast_sequence
from utils.snapshot import save_snapshot
from .partials.helper import set_seed, safe_str
from .partials.embedder import CodeEmbedder


def get_report(y_true, y_pred, y_score=None):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="binary"),
        "average_f1_score": f1_score(y_true, y_pred, average="macro"),
        "roc/auc": roc_auc_score(y_true, y_score),
    }


def train(
    df: pd.DataFrame,
    representations: list[str] = ["code"],
    seed: int = 42,
):
    set_seed(seed)
    embedder = CodeEmbedder()
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    # Parsing AST
    df["ast"] = [
        generate_ast_sequence(safe_str(row.code).lower(), safe_str(row.language))
        for row in df.itertuples(index=False)
    ]

    reports = {}

    for rep in representations:
        print("_" * 80)
        print(f'Representation: "{rep}"')
        print("_" * 80)

        entities = (
            df["code"].str.lower() + "\n" + embedder.separator_token + "\n" + df["ast"]
            if rep == "combined"
            else df[rep]
        ).to_list()

        X = embedder.embed_texts(entities, batch_size=128)
        y = df["label"].to_numpy()

        # Collecting scores for each fold, We will aggregate it later
        y_tests = []
        y_preds = []
        y_scores = []
        for i, (train_idx, validation_idx) in enumerate(kfold.split(X)):
            print(
                f"Started: fold {i + 1} of 10 at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            X_train, X_test, y_train, y_test = (
                X[train_idx],
                X[validation_idx],
                y[train_idx],
                y[validation_idx],
            )

            pipe = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)),
                ]
            )
            pipe.fit(X_train, y_train)
            save_snapshot(f"{rep}_{i}", pipe)

            y_pred = pipe.predict(X_test)
            y_score = pipe.decision_function(X_test)
            y_tests.append(y_test)
            y_preds.append(y_pred)
            y_scores.append(y_score)

            print(get_report(y_test, y_pred, y_score))
            print(
                f"Completed: fold {i + 1} of 10 at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print("-" * 20)

        reports[rep] = get_report(
            np.concatenate(y_tests), np.concatenate(y_preds), np.concatenate(y_scores)
        )

    return reports

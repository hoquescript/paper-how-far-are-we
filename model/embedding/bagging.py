import numpy as np
import pandas as pd
from datetime import datetime


from sklearn.model_selection import KFold, train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
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

        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.8, random_state=seed
        )

        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)),
            ]
        )

        bag_model = BaggingClassifier(
            pipe,
            n_estimators=100,
            max_samples=0.8,
            oob_score=True,
            random_state=seed,
            n_jobs=10,
        )
        bag_model.fit(X_train, y_train)
        y_pred = bag_model.predict(X_test)
        y_score = bag_model.decision_function(X_test)

        print(get_report(y_test, y_pred, y_score))
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 20)

        reports[rep] = get_report(y_test, y_pred, y_score)

    return reports

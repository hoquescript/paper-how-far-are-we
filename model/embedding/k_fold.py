import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split, cross_val_score

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

from scripts.utils.ast.ast_generator import generate_ast_sequence
from .utils.helper import set_seed, safe_str
from .utils.embedder import CodeEmbedder


def get_report(y_true, y_pred, y_score=None):
    f1_h = f1_score(y_true, y_pred, pos_label=1)
    f1_a = f1_score(y_true, y_pred, pos_label=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "custom_f1_score": (f1_h + f1_a) / 2.0,
        "roc/auc": roc_auc_score(y_true, y_score),
    }


def train(
    df: pd.DataFrame,
    representations: list[str] = ["code"],
    seed: int = 42,
):
    set_seed(seed)
    embedder = CodeEmbedder()

    # Parsing AST
    df["ast"] = [
        generate_ast_sequence(safe_str(row.code).lower(), safe_str(row.language))
        for row in df.itertuples(index=False)
    ]

    reports = {}

    for rep in representations:
        entities = (
            df["code"].str.lower() + "\n" + embedder.separator_token + "\n" + df["ast"]
            if rep == "combined"
            else df[rep]
        ).to_list()

        X = embedder.embed_texts(entities, batch_size=128)
        y = df["label"]

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.20, random_state=seed, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.125,
            random_state=seed,
            stratify=y_train_val,
        )
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)),
            ]
        )

        X_fit = np.vstack([X_train, X_val])
        y_fit = np.concatenate([y_train, y_val])
        pipe.fit(X_fit, y_fit)

        y_pred = pipe.predict(X_test)
        y_score = pipe.decision_function(X_test)
        reports[rep] = get_report(y_test, y_pred, y_score)

    return reports

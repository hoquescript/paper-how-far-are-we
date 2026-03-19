import os
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import (
    PredefinedSplit,
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from scipy.stats import loguniform

from scripts.utils.ast.ast_generator import generate_ast_sequence


# -----------------------------
# 0) Utils
# -----------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_str(x):
    return "" if x is None else str(x)


def ast_preorder_types(code: str, language: str) -> str:
    normalized_language = "cpp" if language == "c" else language
    return generate_ast_sequence(code, normalized_language)


# -----------------------------
# 2) Representations
# -----------------------------
DEFAULT_SEP_TOKEN = "</s>"


def resolve_separator_token(tokenizer) -> str:
    return tokenizer.sep_token or tokenizer.eos_token or DEFAULT_SEP_TOKEN


def make_representations(code: str, language: str, sep_token: str) -> Dict[str, str]:
    code_only = code
    ast_only = ast_preorder_types(code, language)
    combined = code_only + "\n" + sep_token + "\n" + ast_only
    return {"code": code_only, "ast": ast_only, "combined": combined}


# -----------------------------
# 3) CodeT5 embedding model wrapper
# -----------------------------
@dataclass
class CodeEmbedder:
    model_name: str = "Salesforce/codet5p-110m-embedding"  # [web:2]
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.separator_token = resolve_separator_token(self.tokenizer)
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Extract normalized sequence embeddings from the CodeT5+ checkpoint.
        Returns array shape (n, embed_dim).
        """
        all_vecs = []
        n_batches = (len(texts) + batch_size - 1) // batch_size

        print(
            f"Embedding {len(texts)} samples on {self.device} "
            f"in {n_batches} batches (batch_size={batch_size})",
            flush=True,
        )

        for i in range(0, len(texts), batch_size):
            print(
                f"Embedding batch {i // batch_size + 1}/{n_batches}",
                flush=True,
            )
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            embeddings = self.model(**enc)
            all_vecs.append(embeddings.detach().cpu().numpy())

        return np.vstack(all_vecs)


# -----------------------------
# 4) ML training with RandomizedSearchCV (like Section III-E steps) [file:1]
# -----------------------------
def average_f1(y_true, y_pred) -> float:
    """
    The paper reports an "Average F1-score" computed by averaging F1 for Human-positive
    and AI-positive (macro over the two single-class F1s). [file:1]
    With binary labels 1=Human, 0=AI:
    - F1_Human is standard f1_score(pos_label=1)
    - F1_AI is f1_score(pos_label=0)
    """
    f1_h = f1_score(y_true, y_pred, pos_label=1)
    f1_a = f1_score(y_true, y_pred, pos_label=0)
    return (f1_h + f1_a) / 2.0


def compute_paper_metrics(y_true, y_pred) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "tpr": tpr,
        "tnr": tnr,
        "avg_f1_custom": average_f1(y_true, y_pred),
    }


def train_and_eval_classifier(
    X_train, y_train, X_val, y_val, X_test, y_test, seed: int = 42
):
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=False)),
        ]
    )
    param_dist = [
        {
            "clf__kernel": ["linear"],
            "clf__C": loguniform(1e-3, 1e3),
        },
        {
            "clf__kernel": ["rbf"],
            "clf__C": loguniform(1e-3, 1e3),
            "clf__gamma": ["scale", "auto"],
        },
    ]

    X_search = np.vstack([X_train, X_val])
    y_search = np.concatenate([y_train, y_val])
    validation_fold = np.concatenate(
        [np.full(len(y_train), -1, dtype=int), np.zeros(len(y_val), dtype=int)]
    )

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=25,
        scoring="f1_macro",  # you can also optimize for custom Average-F1 with a scorer
        cv=PredefinedSplit(validation_fold),
        random_state=seed,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_search, y_search)
    best = search.best_estimator_

    y_pred = best.predict(X_test)
    metrics = compute_paper_metrics(y_test, y_pred)
    report = {
        "best_params": search.best_params_,
        "accuracy": metrics["accuracy"],
        "tpr": metrics["tpr"],
        "tnr": metrics["tnr"],
        "avg_f1_custom": metrics["avg_f1_custom"],
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "classification_report": classification_report(y_test, y_pred, digits=4),
    }
    return best, report


def train_and_eval_svm_only(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
):
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel=kernel, C=C, gamma=gamma, probability=False)),
        ]
    )

    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])
    pipe.fit(X_fit, y_fit)

    y_pred = pipe.predict(X_test)
    metrics = compute_paper_metrics(y_test, y_pred)
    report = {
        "best_params": {
            "clf__kernel": kernel,
            "clf__C": C,
            "clf__gamma": gamma,
        },
        "accuracy": metrics["accuracy"],
        "tpr": metrics["tpr"],
        "tnr": metrics["tnr"],
        "avg_f1_custom": metrics["avg_f1_custom"],
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "classification_report": classification_report(y_test, y_pred, digits=4),
    }
    return pipe, report


# -----------------------------
# 5) End-to-end driver
# -----------------------------
def main(
    df: pd.DataFrame,
    language_col: str = "language",
    code_col: str = "code",
    label_col: str = "label",
    representation: str = "ast",  # "code" | "ast" | "combined"
    model_kind: str = "svm",
    seed: int = 42,
):
    """
    df must have:
      - language: "python"/"java"/"c"
      - code: code snippet string
      - label: 1=Human, 0=AI
    """
    set_seed(seed)

    embedder = CodeEmbedder()
    texts = []
    labels = []

    for _, row in df.iterrows():
        lang = safe_str(row[language_col]).lower()
        code = safe_str(row[code_col])
        y = int(row[label_col])

        reps = make_representations(code, lang, embedder.separator_token)
        texts.append(reps[representation])
        labels.append(y)

    default_batch_size = 16 if embedder.device == "cuda" else 4
    batch_size = int(os.environ.get("EMBED_BATCH_SIZE", str(default_batch_size)))
    X = embedder.embed_texts(texts, batch_size=batch_size)
    y = np.array(labels)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.10, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=1 / 9,
        random_state=seed,
        stratify=y_train_val,
    )

    # best_model, report = train_and_eval_classifier(
    #     X_train, y_train, X_val, y_val, X_test, y_test, seed=seed
    # )
    best_model, report = train_and_eval_svm_only(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    return report


if __name__ == "__main__":
    # Example input format:
    # data.csv columns: language, code, label
    # label: 1=Human, 0=AI
    path = os.environ.get("DATA_CSV", "data.csv")
    if not os.path.exists(path):
        raise SystemExit(
            "Create a data.csv with columns: language, code, label (1=Human, 0=AI). "
            "Then run: DATA_CSV=data.csv python section3f_codet5_embeddings.py"
        )

    df = pd.read_csv(path)

    for rep in ["code", "ast", "combined"]:
        print("\n==============================")
        print(f"Representation = {rep}")
        print("==============================")
        report = main(df, representation=rep, model_kind="svm")
        print("Best params:", report["best_params"])
        print("Accuracy:", report["accuracy"])
        print("TPR:", report["tpr"])
        print("TNR:", report["tnr"])
        print("Average-F1 (custom):", report["avg_f1_custom"])
        print(report["classification_report"])

import os
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModel

from tree_sitter_languages import get_parser

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from scipy.stats import loguniform, randint


# -----------------------------
# 0) Utils
# -----------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_str(x):
    return "" if x is None else str(x)


# -----------------------------
# 1) AST serialization (AST Only)
#    This is a simple preorder traversal of node types.
#    The paper references a specific serialization from [42],
#    but this is a good starting point to reproduce the pipeline. [file:1]
# -----------------------------
def ast_preorder_types(code: str, language: str) -> str:
    """
    Returns a whitespace-separated sequence of node types from the Tree-sitter AST.
    language: "python" (works), "java", "c" (should work if parser available).
    """
    parser = get_parser(language)
    tree = parser.parse(code.encode("utf8"))
    root = tree.root_node

    tokens = []

    def walk(node):
        tokens.append(node.type)
        for child in node.children:
            walk(child)

    walk(root)
    return " ".join(tokens)


# -----------------------------
# 2) Representations
# -----------------------------
SEP_TOKEN = "<CODESPLIT_ASTSEP>"  # "special separator token" idea [file:1]


def make_representations(code: str, language: str) -> Dict[str, str]:
    code_only = code
    ast_only = ast_preorder_types(code, language)
    combined = code_only + "\n" + SEP_TOKEN + "\n" + ast_only
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Mean-pool last_hidden_state with attention mask.
        Returns array shape (n, hidden_dim).
        """
        all_vecs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            out = self.model(**enc)
            # out.last_hidden_state: (B, T, H)
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)

            summed = (last * mask).sum(dim=1)  # (B, H)
            counts = mask.sum(dim=1).clamp(min=1)  # (B, 1)
            mean_pooled = summed / counts

            all_vecs.append(mean_pooled.detach().cpu().numpy())

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


def train_and_eval_classifier(X_train, y_train, X_test, y_test, seed: int = 42):
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

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=25,
        scoring="f1_macro",  # you can also optimize for custom Average-F1 with a scorer
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    best = search.best_estimator_

    y_pred = best.predict(X_test)
    report = {
        "best_params": search.best_params_,
        "accuracy": accuracy_score(y_test, y_pred),
        "avg_f1_custom": average_f1(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "classification_report": classification_report(y_test, y_pred, digits=4),
    }
    return best, report


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

    texts = []
    labels = []

    for _, row in df.iterrows():
        lang = safe_str(row[language_col]).lower()
        code = safe_str(row[code_col])
        y = int(row[label_col])

        reps = make_representations(code, lang)
        texts.append(reps[representation])
        labels.append(y)

    embedder = CodeEmbedder()
    X = embedder.embed_texts(texts, batch_size=16)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=seed, stratify=y
    )

    best_model, report = train_and_eval_classifier(
        X_train, y_train, X_test, y_test, seed=seed
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
        print("Average-F1 (custom):", report["avg_f1_custom"])
        print(report["classification_report"])

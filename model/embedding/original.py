import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

from sklearn.model_selection import (
    train_test_split,
)

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


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_str(x):
    return "" if x is None else str(x)


DEFAULT_SEP_TOKEN = "</s>"


def resolve_separator_token(tokenizer) -> str:
    return tokenizer.sep_token or tokenizer.eos_token or DEFAULT_SEP_TOKEN


# -----------------------------
# CodeT5 embedding model wrapper
# -----------------------------
@dataclass
class CodeEmbedder:
    model_name: str = "Salesforce/codet5p-110m-embedding"
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.separator_token = resolve_separator_token(self.tokenizer)
        # Newer transformers T5Stack expects config.is_decoder; the hub's
        # CodeT5pEmbeddingConfig does not define it (encoder-only checkpoint).
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        if not hasattr(config, "is_decoder"):
            config.is_decoder = False
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True, config=config
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


def train_and_eval_svm_only(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
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
    return get_report(y_test, y_pred, y_score)


def main(
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

        reports[rep] = train_and_eval_svm_only(
            X_train, y_train, X_val, y_val, X_test, y_test
        )

    return reports

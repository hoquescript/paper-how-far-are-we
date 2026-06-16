import argparse
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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

from .embedding import CodeEmbedder, safe_str, set_seed, ts


def log(msg: str):
    print(f"[{ts()}] {msg}", flush=True)


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


def load_or_embed(embedder, texts: list, path: str, name: str) -> np.ndarray:
    """Load embeddings from disk if they exist, otherwise embed and save."""
    if os.path.exists(path):
        log(f"Loading cached {name} embeddings from {path} ...")
        vecs = pd.read_csv(path).values
        log(f"Loaded {name} embeddings  shape={vecs.shape}")
        return vecs

    log(f"Embedding {name} ({len(texts)} samples)...")
    vecs = embedder.embed_texts(texts, batch_size=64)
    cols = [f"{name}_{i}" for i in range(vecs.shape[1])]
    pd.DataFrame(vecs, columns=cols).to_csv(path, index=False)
    log(f"Saved {name} embeddings → {path}  shape={vecs.shape}")
    return vecs


def needs_embedding(output_dir: str) -> bool:
    """Return True if any embedding file is missing and GPU is required."""
    for split in ("train", "test"):
        for rep in ("code", "ast", "xml"):
            if not os.path.exists(os.path.join(output_dir, f"{split}_{rep}.csv")):
                return True
    return False


def main(df: pd.DataFrame, output_dir: str = "embeddings", seed: int = 42):
    set_seed(seed)
    split_dir = os.path.join(output_dir, "split")
    emb_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    log(f"Dataset size: {len(df)} rows")

    # ── Step 1: Train / Test split ────────────────────────────────────────
    log("=== Step 1/5: Train/Test Split ===")
    train_path = os.path.join(split_dir, "train.csv")
    test_path = os.path.join(split_dir, "test.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        log("Split files already exist — loading from disk")
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    else:
        df_train, df_test = train_test_split(
            df, test_size=0.20, random_state=seed, stratify=df["label"]
        )
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        log(f"Saved train ({len(df_train)} rows) → {train_path}")
        log(f"Saved test  ({len(df_test)} rows)  → {test_path}")

    log(f"Train: {len(df_train)} | Test: {len(df_test)}")

    # ── Step 2: Generate AST texts (needed before embedder init check) ────
    log("=== Step 2/5: Generating AST Sequences ===")
    from utils.ast.ast_generator import generate_ast_sequence

    train_ast = [
        generate_ast_sequence(safe_str(row.code), safe_str(row.language))
        for row in df_train.itertuples(index=False)
    ]
    test_ast = [
        generate_ast_sequence(safe_str(row.code), safe_str(row.language))
        for row in df_test.itertuples(index=False)
    ]
    log("AST sequences ready")

    # ── Step 3: Embed each representation (GPU only if files missing) ─────
    log("=== Step 3/5: Embeddings ===")
    embedder = CodeEmbedder() if needs_embedding(emb_dir) else None
    if embedder is None:
        log("All embedding files found — skipping GPU, loading from disk")

    splits = {
        "train": (df_train, train_ast),
        "test": (df_test, test_ast),
    }
    vecs = {"train": {}, "test": {}}

    for split_name, (df_split, ast_texts) in splits.items():
        texts_map = {
            "code": df_split["code"].apply(safe_str).to_list(),
            "ast": ast_texts,
            "xml": df_split["xml"].apply(safe_str).to_list(),
        }
        for rep, texts in texts_map.items():
            path = os.path.join(emb_dir, f"{split_name}_{rep}.csv")
            vecs[split_name][rep] = load_or_embed(embedder, texts, path, f"{split_name}/{rep}")

    # ── Step 4: Merge embeddings ──────────────────────────────────────────
    log("=== Step 4/5: Merging Embeddings ===")
    X_train = np.concatenate(
        [vecs["train"]["code"], vecs["train"]["ast"], vecs["train"]["xml"]], axis=1
    )
    X_test = np.concatenate(
        [vecs["test"]["code"], vecs["test"]["ast"], vecs["test"]["xml"]], axis=1
    )
    y_train = df_train["label"].values
    y_test = df_test["label"].values
    log(f"Train matrix: {X_train.shape} | Test matrix: {X_test.shape}")

    # ── Step 5: Train SVM ─────────────────────────────────────────────────
    log("=== Step 5/5: Training SVM ===")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)),
    ])
    pipe.fit(X_train, y_train)
    log("SVM training complete")

    y_pred = pipe.predict(X_test)
    y_score = pipe.decision_function(X_test)
    report = get_report(y_test, y_pred, y_score)

    log("=== Results ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    log("Done.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="embeddings")
    args = parser.parse_args()

    path = os.getenv("DATA_CSV", "data/hmcorp_xml.csv")
    df = pd.read_csv(path)
    if args.sample is not None:
        df = df.sample(n=args.sample, random_state=42)

    main(df, output_dir=args.output_dir)

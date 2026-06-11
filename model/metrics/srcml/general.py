import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from .embedding import CodeEmbedder, safe_str, set_seed


SCG_COLS = [
    "num_nodes",
    "num_edges",
    "num_self_loops",
    "graph_density",
    "num_cycles",
    "sum_degree",
    "avg_degree",
    "max_degree",
    "min_degree",
    "median_degree",
    "avg_in_degree",
    "avg_out_degree",
]


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


def main(df: pd.DataFrame, seed: int = 42):
    set_seed(seed)

    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # ── Embeddings ────────────────────────────────────────────────────────
    embedder = CodeEmbedder()
    codes = df["code"].apply(safe_str).to_list()
    embeddings = embedder.embed_texts(codes, batch_size=64)
    print(f"Embeddings shape: {embeddings.shape}")

    # ── SCG features ──────────────────────────────────────────────────────
    scg_values = df[SCG_COLS].values.astype(float)
    if np.isnan(scg_values).any():
        print("Warning: nulls found in SCG features — filling with 0")
        scg_values = np.nan_to_num(scg_values, nan=0.0)

    # ── Split ─────────────────────────────────────────────────────────────
    y = df["label"].values
    emb_train, emb_test, scg_train, scg_test, y_train, y_test = train_test_split(
        embeddings, scg_values, y, test_size=0.30, random_state=seed, stratify=y
    )
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")

    # ── Scale SCG (fit on train only) ─────────────────────────────────────
    scaler = StandardScaler()
    scg_train_scaled = scaler.fit_transform(scg_train)
    scg_test_scaled = scaler.transform(scg_test)

    # ── Concatenate: 256 (embedding) + 12 (SCG) = 268-dim ────────────────
    X_train = np.concatenate([emb_train, scg_train_scaled], axis=1)
    X_test = np.concatenate([emb_test, scg_test_scaled], axis=1)
    print(f"Train matrix: {X_train.shape} | Test matrix: {X_test.shape}")

    # ── Train ─────────────────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=100, random_state=seed, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]
    report = get_report(y_test, y_pred, y_score)

    print("\n=== Results ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # ── Feature importance breakdown ──────────────────────────────────────
    importances = clf.feature_importances_
    emb_imp = importances[:256].sum()
    scg_imp = importances[256:].sum()
    print(f"\nEmbedding importance : {emb_imp:.4f}")
    print(f"SCG importance       : {scg_imp:.4f}")
    print(f"SCG contribution     : {scg_imp / (emb_imp + scg_imp) * 100:.1f}%")
    for col, imp in zip(SCG_COLS, importances[256:]):
        print(f"  {col:<20} {imp:.4f}")

    return report


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    path = os.getenv("DATA_CSV", "data/hmcorp_srcml.csv")
    df = pd.read_csv(path)
    if args.sample is not None:
        df = df.sample(n=args.sample, random_state=42)
    print(f"Dataset size: {len(df)} rows")

    main(df)

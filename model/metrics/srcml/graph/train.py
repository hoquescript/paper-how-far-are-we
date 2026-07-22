import argparse
import os
import numpy as np
import pandas as pd
from lxml import etree
import networkx as nx

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

from embedding import CodeEmbedder, safe_str, set_seed


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


# ─────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────


def parse_xml_to_graph(xml_string):
    try:
        root = etree.fromstring(xml_string.encode())
    except etree.XMLSyntaxError:
        G = nx.DiGraph()
        G.add_node(0, tag="root", text="")
        return G

    G = nx.DiGraph()
    node_counter = [0]

    def assign_ids(element, parent_id=None):
        node_id = node_counter[0]
        node_counter[0] += 1
        tag = etree.QName(element.tag).localname
        text = (element.text or "").strip()
        G.add_node(node_id, tag=tag, text=text)
        if parent_id is not None:
            G.add_edge(parent_id, node_id, edge_type="hierarchical")
        for child in element:
            assign_ids(child, parent_id=node_id)
        return node_id

    assign_ids(root)

    # Sequential edges between siblings
    for node_id in list(G.nodes()):
        children = list(G.successors(node_id))
        for i in range(len(children) - 1):
            src, dst = children[i], children[i + 1]
            if not G.has_edge(src, dst):
                G.add_edge(src, dst, edge_type="sequential")

    # Self-loops where a node's tag matches its parent's tag
    for node_id in G.nodes():
        tag = G.nodes[node_id].get("tag", "")
        for pred in G.predecessors(node_id):
            if G.nodes[pred].get("tag", "") == tag:
                G.add_edge(node_id, node_id, edge_type="self_loop")
                break

    return G


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────


def extract_scg_features(G) -> dict:
    features = {}
    n = G.number_of_nodes()
    non_loop_edges = [(u, v) for u, v in G.edges() if u != v]
    self_loops = list(nx.selfloop_edges(G))

    features["num_nodes"] = n
    features["num_edges"] = len(non_loop_edges)
    features["num_self_loops"] = len(self_loops)
    features["graph_density"] = len(non_loop_edges) / (n * (n - 1)) if n > 1 else 0.0

    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))
    features["num_cycles"] = len(nx.cycle_basis(G_und))

    if n > 0:
        deg_seq = [G.in_degree(v) + G.out_degree(v) for v in G.nodes()]
        in_deg_seq = [G.in_degree(v) for v in G.nodes()]
        out_deg_seq = [G.out_degree(v) for v in G.nodes()]
        features["sum_degree"] = sum(deg_seq)
        features["avg_degree"] = sum(deg_seq) / n
        features["max_degree"] = max(deg_seq)
        features["min_degree"] = min(deg_seq)
        sorted_deg = sorted(deg_seq)
        mid = n // 2
        features["median_degree"] = (
            (sorted_deg[mid - 1] + sorted_deg[mid]) / 2 if n % 2 == 0 else sorted_deg[mid]
        )
        features["avg_in_degree"] = sum(in_deg_seq) / n
        features["avg_out_degree"] = sum(out_deg_seq) / n
    else:
        for f in ["sum_degree", "avg_degree", "max_degree", "min_degree",
                  "median_degree", "avg_in_degree", "avg_out_degree"]:
            features[f] = 0.0

    return features


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────


def get_report(y_true, y_pred, y_score=None) -> dict:
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


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────


def main(df: pd.DataFrame, seed: int = 42) -> dict:
    set_seed(seed)
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # ── Step 1: Extract SCG features from xml column ──────────────────────
    print("\n=== Step 1/4: Extracting SCG features ===")
    scg_rows = []
    skipped = 0
    for _, row in df.iterrows():
        xml = row.get("xml")
        if not isinstance(xml, str) or not xml.strip():
            skipped += 1
            scg_rows.append({f: 0.0 for f in SCG_COLS})
            continue
        try:
            scg_rows.append(extract_scg_features(parse_xml_to_graph(xml)))
        except Exception as e:
            print(f"  ERROR on id={row.get('id')}: {e}")
            skipped += 1
            scg_rows.append({f: 0.0 for f in SCG_COLS})

    if skipped:
        print(f"Skipped/zeroed {skipped} rows (missing or invalid XML)")

    scg_values = pd.DataFrame(scg_rows)[SCG_COLS].values.astype(float)
    if np.isnan(scg_values).any():
        print("Warning: NaNs in SCG features — filling with 0")
        scg_values = np.nan_to_num(scg_values, nan=0.0)

    # ── Step 2: Embed code ────────────────────────────────────────────────
    print("\n=== Step 2/4: Embedding code ===")
    embedder = CodeEmbedder()
    embeddings = embedder.embed_texts(df["code"].apply(safe_str).to_list(), batch_size=64)
    print(f"Embeddings shape: {embeddings.shape}")

    # ── Step 3: Train / test split ────────────────────────────────────────
    print("\n=== Step 3/4: Train/Test Split ===")
    y = df["label"].values
    emb_train, emb_test, scg_train, scg_test, y_train, y_test = train_test_split(
        embeddings, scg_values, y, test_size=0.20, random_state=seed, stratify=y
    )
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")

    scaler = StandardScaler()
    scg_train_scaled = scaler.fit_transform(scg_train)
    scg_test_scaled = scaler.transform(scg_test)

    X_train = np.concatenate([emb_train, scg_train_scaled], axis=1)
    X_test = np.concatenate([emb_test, scg_test_scaled], axis=1)
    print(f"Train matrix: {X_train.shape} | Test matrix: {X_test.shape}")

    # ── Step 4: Train RandomForest + evaluate ─────────────────────────────
    print("\n=== Step 4/4: Training RandomForest ===")
    clf = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)

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

    report["feature_importance"] = {
        "embedding": round(float(emb_imp), 4),
        "scg": round(float(scg_imp), 4),
        "scg_breakdown": {col: round(float(imp), 4) for col, imp in zip(SCG_COLS, importances[256:])},
    }

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCG + embedding pipeline")
    parser.add_argument("--input", "-i", default="data/hmcorp_xml_2.csv",
                        help="Input CSV with code, xml, label columns")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.sample is not None:
        df = df.sample(n=args.sample, random_state=42)
    print(f"Dataset size: {len(df)} rows")

    main(df)

from lxml import etree
import networkx as nx
import pandas as pd
import os
import re


# ─────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────


def parse_xml_to_graph(xml_string):
    """
    Takes a SrcML XML string and returns a NetworkX DiGraph
    with hierarchical, sequential, and self-loop edges.
    """
    try:
        root = etree.fromstring(xml_string.encode())
    except etree.XMLSyntaxError:
        G = nx.DiGraph()
        G.add_node(0, tag="root", text="")
        return G

    G = nx.DiGraph()
    node_counter = [0]

    # ── STEP A: Walk the XML tree, assign every element a unique node ID ──
    def assign_ids(element, parent_id=None):
        node_id = node_counter[0]
        node_counter[0] += 1

        tag = etree.QName(element.tag).localname  # strip namespace
        text = (element.text or "").strip()

        G.add_node(node_id, tag=tag, text=text)

        if parent_id is not None:
            G.add_edge(parent_id, node_id, edge_type="hierarchical")

        for child in element:
            assign_ids(child, parent_id=node_id)

        return node_id

    assign_ids(root)

    # ── STEP B: Sequential edges between siblings (document order) ──
    # Connects adjacent children of the same parent only,
    # avoids linking unrelated tokens across subtrees
    for node_id in list(G.nodes()):
        children = list(G.successors(node_id))
        for i in range(len(children) - 1):
            src = children[i]
            dst = children[i + 1]
            if not G.has_edge(src, dst):
                G.add_edge(src, dst, edge_type="sequential")

    # ── STEP C: Self-loops where a node's tag matches its parent's tag ──
    # Captures nested constructs of the same type (e.g. expr inside expr)
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


def extract_scg_features(G):
    """
    Computes all 12 SCG features from a NetworkX DiGraph.
    Returns a dict of feature name → value.
    """
    features = {}
    n = G.number_of_nodes()

    non_loop_edges = [(u, v) for u, v in G.edges() if u != v]
    self_loops = list(nx.selfloop_edges(G))

    # 1. Number of nodes
    features["num_nodes"] = n

    # 2. Number of edges (excluding self-loops)
    features["num_edges"] = len(non_loop_edges)

    # 3. Number of self-loops
    features["num_self_loops"] = len(self_loops)

    # 4. Graph density
    features["graph_density"] = len(non_loop_edges) / (n * (n - 1)) if n > 1 else 0.0

    # 5. Number of cycles
    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))
    features["num_cycles"] = len(nx.cycle_basis(G_und))

    # 6–12. Degree statistics
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
            (sorted_deg[mid - 1] + sorted_deg[mid]) / 2
            if n % 2 == 0
            else sorted_deg[mid]
        )

        features["avg_in_degree"] = sum(in_deg_seq) / n
        features["avg_out_degree"] = sum(out_deg_seq) / n
    else:
        for f in [
            "sum_degree",
            "avg_degree",
            "max_degree",
            "min_degree",
            "median_degree",
            "avg_in_degree",
            "avg_out_degree",
        ]:
            features[f] = 0.0

    return features


# ─────────────────────────────────────────────
# SINGLE FILE PROCESSING
# ─────────────────────────────────────────────


def process_xml_file(filepath):
    """
    Reads a single SrcML XML file, builds its graph,
    and returns the 12 SCG features as a dict.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        xml_string = f.read()

    G = parse_xml_to_graph(xml_string)
    return extract_scg_features(G)


# ─────────────────────────────────────────────
# FOLDER PROCESSING
# ─────────────────────────────────────────────


def process_folder(folder_path, output_csv="scg_features.csv"):
    """
    Processes all XML files in a folder.

    Expected filename format:
        gj{index}_human.xml  →  label = 1  (human-written)
        gj{index}_ai.xml     →  label = 0  (AI-generated)

    Any file not matching either pattern is skipped with a warning.

    Args:
        folder_path : path to folder containing XML files
        output_csv  : path to save the resulting CSV

    Returns:
        pandas DataFrame with columns:
            filename, index, label, + 12 SCG features
    """
    # Regex: captures the numeric index and the label token (gj-prefixed filenames)
    pattern = re.compile(r"^gj(\d+)_(human|ai)\.xml$", re.IGNORECASE)

    rows = []
    skipped = []

    xml_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".xml"))

    if not xml_files:
        print(f"No XML files found in: {folder_path}")
        return pd.DataFrame()

    print(f"Found {len(xml_files)} XML files in '{folder_path}'")

    for filename in xml_files:
        match = pattern.match(filename)
        if not match:
            skipped.append(filename)
            continue

        file_index = int(match.group(1))
        label_str = match.group(2).lower()
        label = 0 if label_str == "ai" else 1  # 0 = AI, 1 = Human

        filepath = os.path.join(folder_path, filename)

        try:
            features = process_xml_file(filepath)
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
            skipped.append(filename)
            continue

        row = {
            "filename": filename,
            "index": file_index,
            "label": label,  # 0 = AI, 1 = Human
            **features,
        }
        rows.append(row)

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s) (pattern mismatch or error):")
        for s in skipped:
            print(f"  {s}")

    if not rows:
        print("No valid files were processed.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    print(f"\nDone. Processed {len(df)} files.")
    print(f"  AI    (0): {(df['label'] == 0).sum()}")
    print(f"  Human (1): {(df['label'] == 1).sum()}")
    print(f"  Saved to : {output_csv}")

    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract SCG features from a folder of SrcML XML files."
    )
    parser.add_argument(
        "folder",
        help="Path to folder containing gj{index}_human.xml / gj{index}_ai.xml files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="scg_features.csv",
        help="Output CSV file path (default: scg_features.csv)",
    )
    args = parser.parse_args()

    df = process_folder(args.folder, output_csv=args.output)

    if not df.empty:
        print("\nSample output (first 5 rows):")
        print(df.head().to_string(index=False))

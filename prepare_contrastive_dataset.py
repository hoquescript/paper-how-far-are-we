"""Turn data/aidev_contrastive/*_paired.jsonl into train/valid/test files for
model/contrastive/model/run.py.

Problems in the raw paired files that this script fixes:

1. run.py needs an ``index`` key; the raw files only carry ``original_index``.
2. Every raw row is ``label=1`` (code=human, contrast=ChatGPT), so a classifier
   trained on them sees a single class. Each pair is emitted twice, once per
   direction, which also makes every split exactly 50/50.
3. A large share of pairs are degenerate: ``code`` and ``contrast`` are the same
   snippet once whitespace is normalised the way run.py normalises it. Flipping
   those would produce two identical inputs with opposite labels, so they are
   dropped.
4. The same (human, ai) pair appears many times over; duplicates are dropped.
5. There is no train/valid/test split. Snippets that appear in more than one
   pair are grouped into connected components and split component-wise, so a
   snippet never shows up in two different splits.
6. File names do not match content: javascript_paired.jsonl is mostly
   TypeScript. Rows are bucketed by their own ``language`` field, not by the
   file they came from, so each language gets its own directory and its own
   tree-sitter grammar.

Output goes to data/aidev_contrastive/<language>/{train,valid,test}.jsonl, a
path shape run.py's language inference understands.

    uv run python prepare_contrastive_dataset.py
    uv run python prepare_contrastive_dataset.py --languages python --seed 99
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from glob import glob

DATA_DIR = "data/aidev_contrastive"


def normalize(text: str) -> str:
    """Same normalisation run.py applies before tokenising."""
    return " ".join(str(text).split())


def read_pairs(data_dir):
    """Read every *_paired.jsonl, bucketed by each row's own language field.

    Returns {language: [(human_code, ai_code, original_index), ...]}.
    """
    buckets = defaultdict(list)
    malformed = 0
    for path in sorted(glob(os.path.join(data_dir, "*_paired.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not {"code", "contrast", "label", "language"} <= set(row):
                    malformed += 1
                    continue
                # label=1 means code is human-written, label=0 the model's.
                if int(row["label"]) == 1:
                    human, ai = row["code"], row["contrast"]
                else:
                    human, ai = row["contrast"], row["code"]
                buckets[str(row["language"]).lower()].append(
                    (human, ai, row.get("original_index"))
                )
    return buckets, malformed


def clean(pairs):
    """Drop empty and degenerate pairs, then de-duplicate on the normalised text."""
    kept = []
    seen = set()
    stats = Counter()
    for human, ai, original_index in pairs:
        human_norm, ai_norm = normalize(human), normalize(ai)
        if not human_norm or not ai_norm:
            stats["empty"] += 1
            continue
        if human_norm == ai_norm:
            stats["degenerate"] += 1
            continue
        key = (human_norm, ai_norm)
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        kept.append((human, ai, original_index, human_norm, ai_norm))
    return kept, stats


def group_pairs(pairs):
    """Group pairs that share a snippet, so they cannot be split apart.

    Union-find over normalised snippets; returns {component_id: [pair index]}.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for _, _, _, human_norm, ai_norm in pairs:
        union(human_norm, ai_norm)

    components = defaultdict(list)
    for i, (_, _, _, human_norm, _) in enumerate(pairs):
        components[find(human_norm)].append(i)
    return components


def split_components(components, train_ratio, valid_ratio, seed):
    """Assign whole components to train/valid/test, aiming at the given ratios."""
    ids = sorted(components)
    random.Random(seed).shuffle(ids)

    total = sum(len(components[cid]) for cid in ids)
    targets = {"train": train_ratio * total, "valid": valid_ratio * total}

    assignment = {}
    counts = Counter()
    for cid in ids:
        if counts["train"] < targets["train"]:
            name = "train"
        elif counts["valid"] < targets["valid"]:
            name = "valid"
        else:
            name = "test"
        assignment[cid] = name
        counts[name] += len(components[cid])
    return assignment


def write_split(path, pairs, indices, start_index, seed):
    """Write one split, emitting both label directions for every pair.

    Rows are shuffled before writing. run.py trains with a SequentialSampler and
    never reshuffles, so the order on disk is the order of every epoch; leaving
    the two directions of a pair adjacent would fill each batch with pairs of
    near-identical snippets.
    """
    rows = []
    for i in indices:
        human, ai, original_index, _, _ = pairs[i]
        rows.append({"code": human, "contrast": ai, "label": 1,
                     "original_index": original_index})
        rows.append({"code": ai, "contrast": human, "label": 0,
                     "original_index": original_index})
    random.Random(seed).shuffle(rows)

    index = start_index
    written = Counter()
    with open(path, "w") as f:
        for row in rows:
            row["index"] = index
            f.write(json.dumps(row) + "\n")
            written[row["label"]] += 1
            index += 1
    return index, written


def prepare(language, pairs, data_dir, train_ratio, valid_ratio, seed):
    cleaned, stats = clean(pairs)
    if not cleaned:
        print(f"[{language}] nothing left after cleaning")
        return None

    components = group_pairs(cleaned)
    assignment = split_components(components, train_ratio, valid_ratio, seed)

    by_split = defaultdict(list)
    for cid, indices in components.items():
        by_split[assignment[cid]].extend(indices)

    out_dir = os.path.join(data_dir, language)
    os.makedirs(out_dir, exist_ok=True)

    report = {
        "language": language,
        "raw_rows": len(pairs),
        "dropped_empty": stats["empty"],
        "dropped_degenerate": stats["degenerate"],
        "dropped_duplicate": stats["duplicate"],
        "kept_pairs": len(cleaned),
        "components": len(components),
        "seed": seed,
        "splits": {},
    }

    index = 0
    for name in ["train", "valid", "test"]:
        indices = sorted(by_split[name])
        path = os.path.join(out_dir, f"{name}.jsonl")
        index, written = write_split(path, cleaned, indices, index, seed)
        report["splits"][name] = {
            "path": path,
            "pairs": len(indices),
            "rows": sum(written.values()),
            "label_1_human": written[1],
            "label_0_ai": written[0],
        }

    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"[{language}]")
    print(f"  raw rows        : {len(pairs)}")
    print(
        f"  dropped         : {stats['degenerate']} degenerate, "
        f"{stats['duplicate']} duplicate, {stats['empty']} empty"
    )
    print(f"  kept pairs      : {len(cleaned)} in {len(components)} components")
    for name in ["train", "valid", "test"]:
        s = report["splits"][name]
        print(
            f"  {name:<15} : {s['rows']} rows from {s['pairs']} pairs "
            f"({s['label_1_human']} human / {s['label_0_ai']} ai) -> {s['path']}"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help=f"Directory holding the *_paired.jsonl files (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Languages to write (default: every language found in the data)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument(
        "--seed",
        type=int,
        default=99,
        help="Shuffle seed for the split (default: %(default)s, matching train.sh)",
    )
    args = parser.parse_args()

    if args.train_ratio + args.valid_ratio >= 1.0:
        parser.error("--train-ratio plus --valid-ratio must leave room for the test set")

    buckets, malformed = read_pairs(args.data_dir)
    if not buckets:
        parser.error(f"No *_paired.jsonl files found in {args.data_dir}")
    if malformed:
        print(f"Skipped {malformed} rows missing required keys")

    languages = args.languages or sorted(buckets)
    for language in languages:
        if language not in buckets:
            print(f"[{language}] skipped, not present in {args.data_dir}")
            continue
        prepare(
            language,
            buckets[language],
            args.data_dir,
            args.train_ratio,
            args.valid_ratio,
            args.seed,
        )


if __name__ == "__main__":
    main()

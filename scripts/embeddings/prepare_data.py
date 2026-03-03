"""
prepare_data.py — Convert raw JSONL to the CSV format expected by main.py

Input JSONL schema (sample_data.jsonl):
  {
    "index":    str,          # e.g. "gj258117"
    "code":     str,          # code snippet to classify
    "contrast": str,          # paired code snippet (optional, see --include-contrast)
    "label":    int,          # 1 = Human-written, 0 = AI-generated
    "language": str           # "java" | "python" | "cpp"
  }

Output CSV columns (consumed by main.py):
  language, code, label

Usage examples
--------------
# Basic: extract only the "code" field with its label
  python scripts/embeddings/prepare_data.py \
      --input  data/sample_data.jsonl \
      --output data/prepared_data.csv

# Also extract the "contrast" field with the INVERTED label
# (use when code=AI-generated / contrast=human-written pairs)
  python scripts/embeddings/prepare_data.py \
      --input  data/sample_data.jsonl \
      --output data/prepared_data.csv \
      --include-contrast

# Filter to a single language
  python scripts/embeddings/prepare_data.py \
      --input  data/sample_data.jsonl \
      --output data/prepared_data.csv \
      --language java

# Cap per-language volume (useful for AST+SVM runtime budget)
  python scripts/embeddings/prepare_data.py \
      --input  data/sample_data.jsonl \
      --output data/prepared_data.csv \
      --max-per-language 20000
"""

import argparse
import json
import os
import sys
from typing import Optional
import pandas as pd


SUPPORTED_LANGUAGES = {"python", "java", "cpp"}

# tree-sitter uses "cpp" but some datasets use "c++"
_LANG_ALIASES = {"c++": "cpp", "c_plus_plus": "cpp"}


def normalise_language(lang: str) -> str:
    lang = lang.strip().lower()
    return _LANG_ALIASES.get(lang, lang)


def infer_language_from_index(index: str) -> Optional[str]:
    idx = (index or "").strip().lower()
    if idx.startswith("gp"):
        return "python"
    if idx.startswith("gj"):
        return "java"
    if idx.startswith("gc"):
        return "cpp"
    return None


def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] line {lineno} skipped ({e})", file=sys.stderr)
    return records


def convert(
    records: list,
    include_contrast: bool = False,
    language_filter: Optional[str] = None,
) -> pd.DataFrame:
    rows = []

    for rec in records:
        code  = rec.get("code", "")
        label = rec.get("label")
        index = rec.get("index", "")
        raw_lang = rec.get("language")

        if raw_lang is None or str(raw_lang).strip() == "":
            lang = language_filter if language_filter else infer_language_from_index(str(index))
        else:
            lang = normalise_language(str(raw_lang))

        if language_filter and lang != language_filter:
            continue

        if lang not in SUPPORTED_LANGUAGES:
            print(f"[WARN] index={index}: unsupported language '{lang}', skipped.",
                  file=sys.stderr)
            continue

        if code is None or str(code).strip() == "":
            print(f"[WARN] index={index}: empty 'code' field, skipped.", file=sys.stderr)
            continue

        if label not in (0, 1):
            print(f"[WARN] index={index}: label={label!r} is not 0/1, skipped.",
                  file=sys.stderr)
            continue

        rows.append({
            "index":    index,
            "language": lang,
            "code":     str(code),
            "label":    int(label),
        })

        if include_contrast:
            contrast = rec.get("contrast", "")
            if contrast and str(contrast).strip():
                rows.append({
                    "index":    index + "_contrast",
                    "language": lang,
                    "code":     str(contrast),
                    "label":    1 - int(label),   # inverted label for the paired snippet
                })

    return pd.DataFrame(rows, columns=["index", "language", "code", "label"])


def print_summary(df: pd.DataFrame):
    print(f"\nTotal samples : {len(df)}")
    print(f"  Human  (1)  : {(df['label'] == 1).sum()}")
    print(f"  AI     (0)  : {(df['label'] == 0).sum()}")
    print("\nPer language:")
    for lang, grp in df.groupby("language"):
        h = (grp["label"] == 1).sum()
        a = (grp["label"] == 0).sum()
        print(f"  {lang:<8}  human={h}  ai={a}  total={len(grp)}")


def cap_per_language(df: pd.DataFrame, max_per_language: int, seed: int) -> pd.DataFrame:
    kept = []
    for lang, grp in df.groupby("language"):
        n = min(len(grp), max_per_language)
        kept.append(grp.sample(n=n, random_state=seed))
    return pd.concat(kept, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL dataset to CSV for Section III-F ML pipeline."
    )
    parser.add_argument(
        "--input", "-i",
        default="data/sample_data.jsonl",
        help="Path to input .jsonl file (default: data/sample_data.jsonl)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/prepared_data.csv",
        help="Path to output .csv file (default: data/prepared_data.csv)",
    )
    parser.add_argument(
        "--include-contrast",
        action="store_true",
        help=(
            "Also extract the 'contrast' field as a separate sample "
            "with the inverted label."
        ),
    )
    parser.add_argument(
        "--language",
        choices=list(SUPPORTED_LANGUAGES),
        default=None,
        help="Keep only samples of this language (default: all languages).",
    )
    parser.add_argument(
        "--max-per-language",
        type=int,
        default=None,
        help="Maximum number of samples per language after filtering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling when --max-per-language is set.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    print(f"Reading  : {args.input}")
    records = load_jsonl(args.input)
    print(f"  {len(records)} records loaded")

    df = convert(
        records,
        include_contrast=args.include_contrast,
        language_filter=args.language,
    )

    if df.empty:
        raise SystemExit("No valid samples after filtering. Check your input file.")

    if args.max_per_language is not None:
        if args.max_per_language <= 0:
            raise SystemExit("--max-per-language must be a positive integer.")
        df = cap_per_language(df, max_per_language=args.max_per_language, seed=args.seed)

    print_summary(df)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # Save without the index column; main.py only needs language, code, label
    df[["language", "code", "label"]].to_csv(args.output, index=False)
    print(f"\nSaved    : {args.output}")
    print(
        f"\nNext step:\n"
        f"  DATA_CSV={args.output} python scripts/embeddings/main.py"
    )


if __name__ == "__main__":
    main()

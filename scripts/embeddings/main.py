"""
AST + SVM detector (CodeT5+ embeddings).

Pipeline:
  1) Build AST-Only representation using paper repo traversal functions
  2) Generate embeddings with CodeT5+ 110M embedding model
  3) Train SVM with randomized hyperparameter search (80/10/10 split)
  4) Report detector accuracy_score and supporting metrics
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import List, Dict

AST_SAMPLE_TIMEOUT_SEC = int(os.environ.get("AST_TIMEOUT", "90"))
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "500"))

from transformers import AutoTokenizer, AutoModel
try:
    from tree_sitter_languages import get_parser as _get_parser  # pyright: ignore[reportMissingImports]
except ImportError:
    _get_parser = None

from sklearn.model_selection import train_test_split, RandomizedSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, recall_score
from scipy.stats import loguniform

# ── make project root importable ──────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── import the paper's actual traverse_ast implementations ────────────────────
from scripts.utils.ast.language.python_ast import traverse_ast as _py_traverse  # noqa: E402
from scripts.utils.ast.language.java_ast   import traverse_ast as _java_traverse  # noqa: E402
from scripts.utils.ast.language.cpp_ast    import traverse_ast as _cpp_traverse  # noqa: E402
from scripts.utils.ast.tree_sitter_loader import get_parser_for_language as _get_ts_parser  # noqa: E402

_TRAVERSE = {
    "python": _py_traverse,
    "java":   _java_traverse,
    "cpp":    _cpp_traverse,
}
_PARSER_CACHE: Dict[str, object] = {}


# ─────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_str(x) -> str:
    return "" if x is None else str(x)


def get_parser_for_language(language: str):
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]

    if _get_parser is not None:
        parser = _get_parser(language)
        _PARSER_CACHE[language] = parser
        return parser

    try:
        parser = _get_ts_parser(language)
    except RuntimeError as exc:
        raise RuntimeError(
            "Parser backend unavailable. Install 'tree_sitter_languages' or set "
            "'TS_LANGUAGE_SO_PATH' to a valid shared library."
        ) from exc

    _PARSER_CACHE[language] = parser
    return parser


# ─────────────────────────────────────────────
# AST serialization — uses the paper's repo traverse_ast functions
# (scripts/utils/ast/language/{python,java,cpp}_ast.py)
#
# Guo et al. [42] algorithm:
#   - non-leaf nodes  → "node_type::left" ... children ... "node_type::right"
#   - leaf nodes / identifiers / literals → actual source text
# ─────────────────────────────────────────────
def generate_ast_sequence(code: str, language: str) -> str:
    """
    Parse code and serialize AST using the paper's traverse_ast function.
    Uses tree_sitter_languages when available, otherwise falls back to
    build/my-languages.so.
    Returns a space-joined token sequence, or empty string on failure.
    """
    traverse = _TRAVERSE.get(language)
    if traverse is None:
        raise ValueError(f"Unsupported language: {language!r}. Use python/java/cpp.")

    parser = get_parser_for_language(language)
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    try:
        tokens = traverse(tree.root_node, code_bytes)
        return " ".join(tokens)
    except Exception as e:
        print(f"[AST ERROR] {e}")
        return ""


# separator token following Guo et al. [42]
_SEP = "<CODESPLIT>"


def make_representations(code: str, language: str) -> Dict[str, str]:
    ast_seq = generate_ast_sequence(code, language)
    return {
        "code":     code,
        "ast":      ast_seq,
        "combined": code + "\n" + _SEP + "\n" + ast_seq,
    }


# ─────────────────────────────────────────────
# CodeT5+ 110M embedding model (Section III-F)
# "Salesforce/codet5p-110m-embedding"
# ─────────────────────────────────────────────
@dataclass
class CodeEmbedder:
    model_name: str = "Salesforce/codet5p-110m-embedding"
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = 16, progress_every: int = 0) -> np.ndarray:
        all_vecs = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            if progress_every and (i // batch_size) % progress_every == 0 and i > 0:
                print(f"  Embed batches: {i // batch_size}/{n_batches}")
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            outputs = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            if isinstance(outputs, torch.Tensor):
                vecs = outputs
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                vecs = outputs.pooler_output
            else:
                vecs = outputs.last_hidden_state[:, 0, :]

            all_vecs.append(vecs.detach().cpu().numpy())
        return np.vstack(all_vecs)


# ─────────────────────────────────────────────
# Evaluation metrics (Section III-C)
# Positive label = Human (1), Negative = AI (0)
# All values as percentages to match paper tables
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    acc    = accuracy_score(y_true, y_pred) * 100
    tpr    = recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100
    tnr    = recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100
    f1_h   = f1_score(y_true, y_pred, pos_label=1, zero_division=0) * 100
    f1_ai  = f1_score(y_true, y_pred, pos_label=0, zero_division=0) * 100
    avg_f1 = (f1_h + f1_ai) / 2.0
    return {
        "accuracy": acc,
        "tpr":      tpr,
        "tnr":      tnr,
        "f1_human": f1_h,
        "f1_ai":    f1_ai,
        "avg_f1":   avg_f1,
    }


# ─────────────────────────────────────────────
# SVM classifier with random search
# ─────────────────────────────────────────────
def train_svm_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    seed: int = 42,
    n_iter: int = 25,
) -> Dict[str, object]:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=False)),
        ]
    )
    param_dist = {
        "clf__C": loguniform(1e-3, 1e3),
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", "auto"],
    }

    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    fold = [-1] * len(X_train) + [0] * len(X_val)
    ps = PredefinedSplit(test_fold=fold)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=ps,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_tv, y_tv)
    y_pred = search.best_estimator_.predict(X_test)
    return {
        "best_params": search.best_params_,
        **compute_metrics(y_test, y_pred),
    }


# ─────────────────────────────────────────────
# End-to-end driver — Section III-F
# ─────────────────────────────────────────────
def run_ast_svm(
    df: pd.DataFrame,
    language_col: str = "language",
    code_col:     str = "code",
    label_col:    str = "label",
    seed: int = 42,
) -> Dict[str, object]:
    """
    df columns:
      language : "python" / "java" / "cpp"
      code     : source code string
      label    : 1 = Human, 0 = AI

    Returns metrics for one AST + SVM model.
    """
    set_seed(seed)

    rows = list(df.iterrows())
    labels = [int(row[label_col]) for _, row in rows]
    n = len(rows)

    def ast_for_row(item):
        _, row = item
        lang = safe_str(row[language_col]).lower()
        code = safe_str(row[code_col])
        return make_representations(code, lang)["ast"]

    print(f"Building AST for {n} samples (timeout={AST_SAMPLE_TIMEOUT_SEC}s/sample) ...")
    texts = []
    with ThreadPoolExecutor(max_workers=1) as ex:
        futs = [ex.submit(ast_for_row, item) for item in rows]
        for i, fut in enumerate(futs):
            if (i + 1) % PROGRESS_EVERY == 0 or i == 0:
                print(f"  AST: {i + 1}/{n}")
            try:
                texts.append(fut.result(timeout=AST_SAMPLE_TIMEOUT_SEC))
            except (FuturesTimeoutError, Exception) as e:
                if (i + 1) % PROGRESS_EVERY == 0 or i == 0 or "timeout" in str(e).lower():
                    print(f"  [skip sample {i + 1}] {e}")
                texts.append("")

    print(f"Generating AST embeddings for {len(texts)} samples ...")
    embedder = CodeEmbedder()
    progress_every = max(1, (len(texts) + 15) // 16 // 10)
    X = embedder.embed_texts(texts, progress_every=progress_every)
    y = np.array(labels)

    # 80 : 10 : 10 split (Section III-C)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    n_iter = int(os.environ.get("SVM_N_ITER", "25"))
    return train_svm_classifier(
        X_train, y_train, X_val, y_val, X_test, y_test, seed=seed, n_iter=n_iter
    )


# ─────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    path = os.environ.get("DATA_CSV", "data.csv")
    if not os.path.exists(path):
        raise SystemExit(
            "Provide a CSV with columns: language, code, label (1=Human, 0=AI).\n"
            "Run prepare_data.py first, then:\n"
            "  DATA_CSV=data/prepared_data.csv python scripts/embeddings/main.py"
        )

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples from {path}")
    print(f"SVM random-search iterations: {os.environ.get('SVM_N_ITER', '25')}")

    metrics = run_ast_svm(df)
    print("\nAST + SVM results")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"TPR(Human): {metrics['tpr']:.2f}")
    print(f"TNR(AI): {metrics['tnr']:.2f}")
    print(f"AvgF1: {metrics['avg_f1']:.2f}")
    print(f"Best params: {metrics['best_params']}")

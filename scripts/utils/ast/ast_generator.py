"""
AST Generator Module

This module provides the main interface for generating AST sequence representations
from source code using Tree-Sitter parsers.

Used in:
- Section III-D: LLM-based approaches (zero-shot, in-context, fine-tuning)
- Section III-F: Machine learning with code embeddings

The generated AST sequences can be used as:
1. "AST Only" input for LLM prompting or embedding generation
2. Part of "Combined" (Code + AST) representation
"""

import os
import pandas as pd
from glob import glob
from tree_sitter_languages import get_parser

from scripts.utils.ast.language.python_ast import traverse_ast as F_python
from scripts.utils.ast.language.java_ast import traverse_ast as F_java
from scripts.utils.ast.language.cpp_ast import traverse_ast as F_cpp

providers = {
    "cpp": {"parser": get_parser("cpp"), "generator": F_cpp},
    "java": {"parser": get_parser("java"), "generator": F_java},
    "python": {"parser": get_parser("python"), "generator": F_python},
}


def generate_ast_sequence(code, lang):
    """
    Generate AST sequence representation for given source code.

    This function:
    1. Parses the source code using Tree-Sitter
    2. Traverses the AST using the Guo et al. approach
    3. Returns a space-separated string of tokens

    The output format uses:
    - "node_type::left" and "node_type::right" for non-leaf nodes
    - Actual text for leaf nodes and identifiers

    Args:
        code (str): Source code to parse
        lang (str): Programming language ('python', 'java', or 'cpp')

    Returns:
        str: Space-separated AST token sequence, or None if parsing fails

    Example:
        >>> code = "def foo(x): return x + 1"
        >>> ast_seq = generate_ast_sequence(code, 'python')
        >>> # Returns: "module::left function_definition::left def foo ..."
    """
    provider = providers[lang]
    parser = provider["parser"]
    generator = provider["generator"]

    code = str(code)
    code_bytes = code.encode("utf8")
    try:
        tree = parser.parse(code_bytes)
        ast_tokens = generator(tree.root_node, code_bytes)
        return " ".join(ast_tokens)
    except Exception as e:
        print(f"[ERROR] [GENERATE AST]: {e}")
        return None


def process_csv_files(input_dir, output_dir):
    """
    Batch process CSV files to generate AST sequences.

    This function:
    1. Reads CSV files containing 'code' column
    2. Generates AST sequence for each code snippet
    3. Saves results with columns: idx, code, ast, actual label

    The language is inferred from the filename pattern (e.g., *_python_*.csv)

    Args:
        input_dir (str): Directory containing input CSV files
        output_dir (str): Directory to save processed CSV files

    Note:
        Input CSV should have columns: code, actual label
        Output CSV will have columns: idx, code, ast, actual label
    """
    os.makedirs(output_dir, exist_ok=True)

    for csv_file in glob(input_dir + "/**/*.csv", recursive=True):
        print(f"Processing {csv_file}")
        data = pd.read_csv(csv_file)
        data["idx"] = data.index

        language = language_inference_from_path(csv_file)

        original_size = len(data)

        data["ast"] = data["code"].apply(
            lambda code: generate_ast_sequence(code, language)
        )

        data.dropna(subset=["ast"], inplace=True)

        number_removed = original_size - len(data)
        print(f"{csv_file} not parsed: {number_removed}/{original_size}")

        output_data = data[["idx", "code", "ast", "actual label"]]
        output_path = csv_file.replace(input_dir, output_dir)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_data.to_csv(output_path, index=False)


def language_inference_from_path(file_path):
    """
    Infer programming language from file path.

    Expects filename format: *_{dataset}_{model}_{language}_*.csv

    Args:
        file_path (str): Path to the CSV file

    Returns:
        str: Language name ('python', 'java', or 'cpp')
    """
    parts = file_path.split(os.sep)[-1].split("_")
    language = parts[2]
    return language.lower()


if __name__ == "__main__":
    # Example usage
    input_dir = "data"
    output_dir = "data_with_ast"
    process_csv_files(input_dir, output_dir)

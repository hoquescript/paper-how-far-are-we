"""
Code Feature Extraction Module

This module extracts static code features using Tree-Sitter for use in
Section III-E: Machine Learning Classifiers with Static Code Metrics.

Features extracted:
- keywords: Ratio of language keywords to total tokens
- if_else_while_operators: Ratio of operators in conditional statements to total tokens

These features are combined with other static metrics (from SciTools Understand)
for training machine learning classifiers.
"""

import os
import pandas as pd
from glob import glob
from tree_sitter import Language, Parser

from scripts.ast.language.python_ast import get_node_text
from scripts.ast.language.java_ast import analyze_java_code
from scripts.ast.language.cpp_ast import analyze_cpp_code


# Initialize Tree-Sitter parsers
CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')
cpp_parser = Parser()
cpp_parser.set_language(CPP_LANGUAGE)

JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
java_parser = Parser()
java_parser.set_language(JAVA_LANGUAGE)

PY_LANGUAGE = Language('build/my-languages.so', 'python')
python_parser = Parser()
python_parser.set_language(PY_LANGUAGE)

providers = {
    "cpp": {
        "parser": cpp_parser,
        "analyzer": analyze_cpp_code
    },
    "java": {
        "parser": java_parser,
        "analyzer": analyze_java_code
    },
    "python": {
        "parser": python_parser,
        "analyzer": analyze_python_code  # Defined below
    }
}


def analyze_python_code(tree, code):
    """
    Extract code features from Python code.
    
    Analyzes Python AST to extract:
    - Keywords ratio: count of language keywords / total tokens
    - Conditional operators ratio: count of operators in if/while / total tokens
    
    Args:
        tree: Tree-sitter parse tree
        code: Source code as string
    
    Returns:
        tuple: (keywords_ratio, conditional_operators_ratio)
    """
    total_tokens = 0
    unique_keywords = set()
    unique_operators = set()
    
    python_keywords = [
        'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
        'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
        'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
        'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
        'try', 'while', 'with', 'yield'
    ]
    
    python_operators = ['and', 'or', 'not', '<', '>', '==', '!=', '<=', '>=', 'is', 'is not', 'in', 'not in']

    def traverse(node, within_condition=False):
        nonlocal total_tokens
        
        if len(node.children) == 0:
            total_tokens += 1
            
        node_text = get_node_text(node, code).decode('utf8') if isinstance(get_node_text(node, code), bytes) else str(get_node_text(node, code))
        
        if node.type in python_keywords:
            unique_keywords.add(node.type)

        if within_condition and node_text in python_operators:
            unique_operators.add(node_text)

        new_within_condition = within_condition or node.type in ['if_statement', 'while_statement', 'for_statement']
        
        for child in node.children:
            traverse(child, new_within_condition)
            
    traverse(tree.root_node)
    
    keywords_ratio = len(unique_keywords) / total_tokens if total_tokens > 0 else 0
    operators_ratio = len(unique_operators) / total_tokens if total_tokens > 0 else 0
    
    return keywords_ratio, operators_ratio


def extract_features(code, lang):
    """
    Extract Tree-Sitter-based features from source code.
    
    Args:
        code (str): Source code
        lang (str): Programming language ('python', 'java', or 'cpp')
    
    Returns:
        tuple: (keywords_ratio, conditional_operators_ratio) or (None, None) if parsing fails
    """
    provider = providers[lang]
    parser = provider['parser']
    analyzer = provider['analyzer']

    try:
        tree = parser.parse(bytes(code, "utf8"))
        return analyzer(tree, code)
    except Exception as e:
        print(f"[ERROR] [EXTRACT_FEATURES]: {e}")
        return None, None


def process_csv_files(input_dir, output_dir):
    """
    Batch process CSV files to extract code features.
    
    This function:
    1. Reads CSV files with existing static metrics columns
    2. Extracts Tree-Sitter features (keywords, if_else_while_operators)
    3. Appends these features to the existing data
    4. Saves results
    
    Args:
        input_dir (str): Directory containing input CSV files with static metrics
        output_dir (str): Directory to save CSV files with additional features
    
    Note:
        Input CSV should already have columns from SciTools Understand:
        - SumCyclomatic, AvgCountLineCode, CountLineCodeDecl, CountDeclFunction,
          MaxNesting, CountLineBlank, etc.
        
        This function adds:
        - keywords
        - if_else_while_operators
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for csv_file in glob(input_dir + '/**/*.csv', recursive=True):
        print(f"Processing {csv_file}")
        data = pd.read_csv(csv_file)
        data['idx'] = data.index if 'idx' not in data.columns else data['idx']

        language = language_inference_from_path(csv_file)
        
        # Extract features for each code snippet
        for index, row in data.iterrows():
            keywords, conditional_operators = extract_features(row['code'], language)
            data.loc[index, 'keywords'] = keywords
            data.loc[index, 'if_else_while_operators'] = conditional_operators

        # Select output columns (assuming other metrics already exist)
        expected_columns = ['idx', 'code', 'actual label', 'SumCyclomatic', 'AvgCountLineCode', 
                          'CountLineCodeDecl', 'CountDeclFunction', 'MaxNesting', 'CountLineBlank',
                          'keywords', 'if_else_while_operators']
        
        # Only keep columns that exist in the dataframe
        output_columns = [col for col in expected_columns if col in data.columns]
        output_data = data[output_columns]

        output_path = csv_file.replace(input_dir, output_dir)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_data.to_csv(output_path, index=False)


def language_inference_from_path(file_path):
    """
    Infer programming language from file path.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        str: Language name ('python', 'java', or 'cpp')
    """
    filename = file_path.split(os.sep)[-1].lower()
    if 'python' in filename:
        return 'python'
    elif 'java' in filename:
        return 'java'
    elif 'cpp' in filename or 'c++' in filename:
        return 'cpp'
    else:
        # Default fallback - try to infer from file pattern
        parts = filename.split('_')
        if len(parts) > 2:
            return parts[2]
        return 'python'


if __name__ == "__main__":
    # Example usage
    input_dir = 'data_with_metrics'
    output_dir = 'data_with_all_features'
    process_csv_files(input_dir, output_dir)

# AST Processing Module

This module provides core AST processing functionality for the ICSE'25 paper: "An Empirical Study on Automatically Detecting AI-Generated Source Code: How Far Are We?"

## Overview

This module contains the essential Tree-Sitter-based code for implementing:
- **Section III-D**: LLM-based approaches (zero-shot, in-context learning, fine-tuning)
- **Section III-E**: Machine learning with static code metrics
- **Section III-F**: Machine learning with code embeddings

**Note**: Ablation study code (Section III-G) has been excluded for clarity.

## Files

### Core Modules

#### `ast_generator.py`
Main interface for generating AST sequence representations from source code.

**Purpose**: Generate "AST Only" representations for:
- LLM prompting (AST Only, Combined inputs)
- Code embedding generation (CodeT5+)

**Key Function**:
```python
generate_ast_sequence(code, lang) -> str
```

**Usage**:
```python
from ast_generator import generate_ast_sequence

code = "def foo(x): return x + 1"
ast_seq = generate_ast_sequence(code, 'python')
# Returns: "module::left function_definition::left def foo ..."
```

#### `feature_extractor.py`
Extracts static code features using Tree-Sitter for ML classification.

**Purpose**: Extract features for Section III-E (ML with static metrics):
- `keywords`: Ratio of language keywords to total tokens
- `if_else_while_operators`: Ratio of operators in conditionals to total tokens

**Key Function**:
```python
extract_features(code, lang) -> tuple[float, float]
```

**Usage**:
```python
from feature_extractor import extract_features

code = "if x > 0: return True"
keywords_ratio, ops_ratio = extract_features(code, 'python')
```

### Language-Specific Modules

#### `python_ast.py`
Python AST traversal using Tree-Sitter.

**Key Function**: `traverse_ast(node, code)` - Implements Guo et al. recursive traversal

#### `java_ast.py`
Java AST traversal and feature extraction.

**Key Functions**:
- `traverse_ast(node, code)` - AST traversal
- `analyze_java_code(tree, code)` - Feature extraction

#### `cpp_ast.py`
C++ AST traversal and feature extraction.

**Key Functions**:
- `traverse_ast(node, code)` - AST traversal
- `analyze_cpp_code(tree, code)` - Feature extraction

## AST Representation Format

Following **Guo et al. [42]** approach:
- Non-leaf nodes: `node_type::left` ... children ... `node_type::right`
- Leaf nodes and identifiers: actual text

**Example**:
```python
# Code
x = 5

# AST Sequence
module::left expression_statement::left assignment::left x = integer 5 assignment::right expression_statement::right module::right
```

## Dependencies

- `tree-sitter` - AST parsing
- `pandas` - Data processing

**Tree-Sitter Language Bindings**:
Place the compiled `my-languages.so` file in the `build/` directory.

## Usage Examples

### 1. Generate AST for LLM Prompting (Section III-D)

```python
from ast_generator import generate_ast_sequence

# For zero-shot/in-context/fine-tuning
code = "public int add(int a, int b) { return a + b; }"
ast_seq = generate_ast_sequence(code, 'java')

# Use in LLM prompt as "AST Only" or combine with code for "Combined"
prompt = f"AST: {ast_seq}"
```

### 2. Extract Features for ML (Section III-E)

```python
from feature_extractor import extract_features

code = "while (x < 10 && y > 0) { x++; }"
keywords, ops = extract_features(code, 'cpp')

# Combine with Understand metrics for ML training
features = {
    'SumCyclomatic': 5,
    'MaxNesting': 2,
    'keywords': keywords,
    'if_else_while_operators': ops
}
```

### 3. Generate AST for Embeddings (Section III-F)

```python
from ast_generator import generate_ast_sequence

# Generate AST sequence
ast_seq = generate_ast_sequence(code, 'python')

# Pass to CodeT5+ for embedding generation (external step)
# embedding = codet5_model.encode(ast_seq)
```

### 4. Batch Processing

```python
from ast_generator import process_csv_files

# Generate AST sequences for all code in CSV files
process_csv_files('input_data/', 'output_data_with_ast/')
```

```python
from feature_extractor import process_csv_files

# Extract features and add to existing metrics
process_csv_files('data_with_metrics/', 'data_with_all_features/')
```

## Paper Sections Mapping

| Paper Section | Module | Purpose |
|---------------|--------|---------|
| III-D (LLM approaches) | `ast_generator.py` + language modules | Generate AST sequences for prompting/fine-tuning |
| III-E (ML with metrics) | `feature_extractor.py` + language modules | Extract Tree-Sitter features (keywords, operators) |
| III-F (ML with embeddings) | `ast_generator.py` + language modules | Generate AST for CodeT5+ embedding input |

## What's NOT Included

This module excludes **Section III-G (Ablation Study)** components:
- `rename_variables()` - Not needed without ablation study
- `replace_function_names()` - Not needed without ablation study
- `remove_comments()` - Not needed without ablation study
- Code variant generation - Not needed without ablation study

## Notes

- **Embedding Generation**: CodeT5+ embedding generation is not included here. Use the HuggingFace `transformers` library separately.
- **Static Metrics**: SciTools Understand metrics (SumCyclomatic, etc.) must be extracted separately before using `feature_extractor.py`.
- **Tree-Sitter Setup**: Ensure `build/my-languages.so` contains compiled Python, Java, and C++ grammars.

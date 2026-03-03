"""
Java AST Traversal Module

This module provides functions to traverse Java AST nodes using Tree-Sitter
and generate sequential representations following the Guo et al. approach.

Used in:
- Section III-D: LLM-based approaches (AST Only, Combined)
- Section III-F: ML with code embeddings (AST Only, Combined)
"""

from tree_sitter import Language, Parser


def get_node_text(node, code):
    """
    Extract the text that a node corresponds to in the source code.
    
    Args:
        node: Tree-sitter AST node
        code: Source code as bytes
    
    Returns:
        bytes: The text corresponding to the node
    """
    start_byte = node.start_byte
    end_byte = node.end_byte
    return code[start_byte:end_byte]


def traverse_ast(node, code):
    """
    Map an AST node to a sequence of tokens using recursive traversal.
    
    This implements the Guo et al. approach where:
    - Non-leaf nodes are represented as "node_type::left" and "node_type::right"
    - Leaf nodes, identifiers, and literals are represented by their actual text
    
    Args:
        node: Tree-sitter AST node
        code: Source code as bytes
    
    Returns:
        list: Sequence of tokens representing the AST
    """
    seq = []
    name = node.type
    text = get_node_text(node, code).decode('utf8')

    # Include identifiers and literals directly
    if len(node.children) == 0 or node.type in ['identifier', 'string_literal', 'number_literal']:
        seq.append(text)
    else:
        seq.append(f"{name}::left")
        for child in node.children:
            seq.extend(traverse_ast(child, code))
        seq.append(f"{name}::right")

    return seq


def analyze_java_code(tree, code):
    """
    Extract code features from Java code for Section III-E.
    
    Analyzes Java AST to extract:
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
    
    java_keywords = [
        "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const",
        "continue", "default", "do", "double", "else", "enum", "exports", "extends", "final", "finally",
        "float", "for", "if", "implements", "import", "instanceof", "int", "interface", "long", "module",
        "native", "new", "package", "private", "protected", "public", "requires", "return", "short", "static",
        "strictfp", "super", "switch", "synchronized", "this", "throw", "throws", "transient", "try", "var",
        "void", "volatile", "while", "true", "false", "null"
    ]
    
    java_operators = ['&&', '||', '!', '<', '>', '<=', '>=', '==', '!=']

    def traverse(node, within_condition=False):
        nonlocal total_tokens
        
        # Increment token count for leaf nodes
        if len(node.children) == 0:
            total_tokens += 1
            
        node_text = get_node_text(node, code)
        
        # Add keywords
        if node.type in java_keywords:
            unique_keywords.add(node_text)

        # Check for operators within conditions
        if within_condition and node_text in java_operators:
            unique_operators.add(node_text)

        # Track if we're within a conditional statement
        new_within_condition = within_condition or node.type in ['if_statement', 'for_statement', 'while_statement']
        
        for child in node.children:
            traverse(child, new_within_condition)
            
    traverse(tree.root_node)
    
    keywords_ratio = len(unique_keywords) / total_tokens if total_tokens > 0 else 0
    operators_ratio = len(unique_operators) / total_tokens if total_tokens > 0 else 0
    
    return keywords_ratio, operators_ratio

"""
Python AST Traversal Module

This module provides functions to traverse Python AST nodes using Tree-Sitter
and generate sequential representations following the Guo et al. approach.

Used in:
- Section III-D: LLM-based approaches (AST Only, Combined)
- Section III-F: ML with code embeddings (AST Only, Combined)
"""


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
    - Leaf nodes and identifiers are represented by their actual text

    Args:
        node: Tree-sitter AST node
        code: Source code as bytes

    Returns:
        list: Sequence of tokens representing the AST

    Example:
        For code: x = 5
        Output: ['module::left', 'expression_statement::left', 'assignment::left',
                 'x', '=', 'integer', '5', 'assignment::right', ...]
    """
    seq = []
    name = node.type
    text = get_node_text(node, code).decode("utf8")

    # Leaf nodes and identifiers are included directly
    if len(node.children) == 0 or node.type == "identifier":
        seq.append(text)
    else:
        # Non-leaf nodes get left/right markers
        seq.append(f"{name}::left")
        for child in node.children:
            seq.extend(traverse_ast(child, code))
        seq.append(f"{name}::right")

    return seq

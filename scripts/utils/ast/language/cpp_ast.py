"""
C++ AST Traversal Module

This module provides functions to traverse C++ AST nodes using Tree-Sitter
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
    - Leaf nodes, identifiers, and literals are represented by their actual text

    Args:
        node: Tree-sitter AST node
        code: Source code as bytes

    Returns:
        list: Sequence of tokens representing the AST
    """
    seq = []
    name = node.type
    text = get_node_text(node, code).decode("utf8")

    # Include identifiers, literals, and preprocessor directives directly
    if len(node.children) == 0 or node.type in [
        "identifier",
        "string_literal",
        "number_literal",
        "char_literal",
        "preproc_include",
        "system_lib_string",
    ]:
        seq.append(text)
    else:
        seq.append(f"{name}::left")
        for child in node.children:
            seq.extend(traverse_ast(child, code))
        seq.append(f"{name}::right")

    return seq


def analyze_cpp_code(tree, code):
    """
    Extract code features from C++ code for Section III-E.

    Analyzes C++ AST to extract:
    - Keywords ratio: count of language keywords / total tokens
    - Conditional operators ratio: count of operators in if/while/for / total tokens

    Args:
        tree: Tree-sitter parse tree
        code: Source code as string

    Returns:
        tuple: (keywords_ratio, conditional_operators_ratio)
    """
    total_tokens = 0
    unique_keywords = set()
    unique_operators = set()

    cpp_keywords = [
        "alignas",
        "alignof",
        "and",
        "and_eq",
        "asm",
        "atomic_cancel",
        "atomic_commit",
        "atomic_noexcept",
        "auto",
        "bitand",
        "bitor",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "char8_t",
        "char16_t",
        "char32_t",
        "class",
        "compl",
        "concept",
        "const",
        "consteval",
        "constexpr",
        "constinit",
        "const_cast",
        "continue",
        "co_await",
        "co_return",
        "co_yield",
        "decltype",
        "default",
        "delete",
        "do",
        "double",
        "dynamic_cast",
        "else",
        "enum",
        "explicit",
        "export",
        "extern",
        "false",
        "float",
        "for",
        "friend",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "not",
        "not_eq",
        "nullptr",
        "operator",
        "or",
        "or_eq",
        "private",
        "protected",
        "public",
        "reflexpr",
        "register",
        "reinterpret_cast",
        "requires",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "synchronized",
        "template",
        "this",
        "thread_local",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
        "wchar_t",
        "while",
        "xor",
        "xor_eq",
    ]

    cpp_operators = ["&&", "||", "!", "<", ">", "<=", ">=", "==", "!=", "|", "&"]

    def traverse(node):
        nonlocal total_tokens

        # Increment token count for leaf nodes
        if len(node.children) == 0:
            total_tokens += 1

        if "primitive_type" in node.type:
            unique_keywords.add(get_node_text(node, code))

        if node.type in cpp_keywords:
            unique_keywords.add(node.type)

        # Handle conditional statements
        if node.type in ["if_statement", "while_statement", "for_statement"]:
            condition = None
            if node.type == "if_statement" or node.type == "while_statement":
                condition = node.child_by_field_name("condition")
            elif node.type == "for_statement":
                condition = node.child_by_field_name("condition")
            if condition:
                extract_operators(condition)

        for child in node.children:
            traverse(child)

    def extract_operators(node):
        """Extract operators from a condition node"""
        if node.type == "binary_expression" and any(
            op in code[node.start_byte : node.end_byte] for op in cpp_operators
        ):
            operator_text = code[node.start_byte : node.end_byte]
            for op in cpp_operators:
                if op in operator_text:
                    unique_operators.add(op)
        for child in node.children:
            extract_operators(child)

    traverse(tree.root_node)

    keywords_ratio = len(unique_keywords) / total_tokens if total_tokens > 0 else 0
    operators_ratio = len(unique_operators) / total_tokens if total_tokens > 0 else 0

    return keywords_ratio, operators_ratio

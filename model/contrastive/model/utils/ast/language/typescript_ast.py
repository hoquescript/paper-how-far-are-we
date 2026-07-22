"""
TypeScript AST Traversal Module

This module provides functions to traverse TypeScript (and TSX) AST nodes using Tree-Sitter
and generate sequential representations following the Guo et al. approach.

Supports:
- Regular TypeScript (.ts): functions, classes, interfaces, type aliases, generics
- TSX (.tsx): all of the above plus JSX elements, attributes, expressions, and fragments

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
    Map a TypeScript/TSX AST node to a sequence of tokens using recursive traversal.

    This implements the Guo et al. approach where:
    - Non-leaf nodes are represented as "node_type::left" and "node_type::right"
    - Leaf nodes, identifiers, literals, type annotations, and JSX text are
      represented by their actual text

    TSX-specific handling:
    - JSX element names (e.g. <MyComponent>) are treated as identifiers
    - JSX attribute names and string values are included as text
    - JSX expression containers `{...}` are traversed normally
    - JSX text content is included directly as a leaf

    Args:
        node: Tree-sitter AST node
        code: Source code as bytes

    Returns:
        list: Sequence of tokens representing the AST

    Example:
        For code: const x: number = 5;
        Output: ['program::left', 'lexical_declaration::left', 'const',
                 'variable_declarator::left', 'x', 'type_annotation::left',
                 ':', 'number', 'type_annotation::right', '=', '5',
                 'variable_declarator::right', 'lexical_declaration::right', ...]
    """
    seq = []
    name = node.type
    text = get_node_text(node, code).decode("utf8")

    # Node types whose text should always be inlined directly,
    # regardless of whether they have children.
    INLINE_TYPES = {
        # Core identifiers and literals
        "identifier",
        "property_identifier",
        "shorthand_property_identifier",
        "shorthand_property_identifier_pattern",
        "private_property_identifier",
        "string",
        "string_fragment",
        "number",
        "template_string",
        "regex",
        # TypeScript-specific
        "type_identifier",          # type alias names, generic params (e.g. T, Props)
        "predefined_type",          # built-in types: number, string, boolean, void, any, never
        # JSX-specific (TSX)
        "jsx_identifier",           # component names in JSX like <MyComp>
        "jsx_text",                 # raw text content between JSX tags
        "jsx_attribute_name",       # attribute names like className, onClick
    }

    if len(node.children) == 0 or node.type in INLINE_TYPES:
        seq.append(text)
    else:
        seq.append(f"{name}::left")
        for child in node.children:
            seq.extend(traverse_ast(child, code))
        seq.append(f"{name}::right")

    return seq


def analyze_typescript_code(tree, code):
    """
    Extract code features from TypeScript/TSX code for Section III-E.

    Analyzes the TypeScript AST to extract:
    - Keywords ratio: count of unique language keywords / total tokens
    - Conditional operators ratio: count of unique operators in if/while/for/ternary
      conditions / total tokens

    TSX-aware: JSX structure nodes (elements, attributes, fragments) are traversed
    normally; their leaf text is counted toward total_tokens.

    Args:
        tree: Tree-sitter parse tree
        code: Source code as bytes

    Returns:
        tuple: (keywords_ratio, conditional_operators_ratio)
    """
    total_tokens = 0
    unique_keywords = set()
    unique_operators = set()

    # Full TypeScript keyword set (ES2022 + TS-specific)
    ts_keywords = [
        # JavaScript reserved words
        "break", "case", "catch", "class", "const", "continue", "debugger",
        "default", "delete", "do", "else", "export", "extends", "false",
        "finally", "for", "function", "if", "import", "in", "instanceof",
        "let", "new", "null", "return", "static", "super", "switch", "this",
        "throw", "true", "try", "typeof", "var", "void", "while", "with",
        "yield", "async", "await", "of",
        # TypeScript-specific keywords
        "abstract", "as", "asserts", "declare", "enum", "from", "global",
        "implements", "interface", "infer", "is", "keyof", "module",
        "namespace", "never", "override", "private", "protected", "public",
        "readonly", "require", "satisfies", "type", "undefined", "unique",
        # JSX (used in .tsx files)
        "from",
    ]

    # Operators meaningful inside conditionals
    ts_operators = ["&&", "||", "!", "??", "<", ">", "<=", ">=", "==", "!=", "===", "!==", "|", "&"]

    # Node types that represent conditional contexts
    CONDITIONAL_NODE_TYPES = {
        "if_statement",
        "while_statement",
        "for_statement",
        "for_in_statement",   # for...of / for...in
        "ternary_expression", # condition ? a : b
        "do_statement",
    }

    def extract_operators(node):
        """Recursively extract conditional operators from a subtree."""
        node_text = get_node_text(node, code).decode("utf8")
        if node.type == "binary_expression" or node.type == "unary_expression":
            for op in ts_operators:
                if op in node_text:
                    unique_operators.add(op)
        for child in node.children:
            extract_operators(child)

    def traverse(node):
        nonlocal total_tokens

        # Count every leaf node as one token
        if len(node.children) == 0:
            total_tokens += 1

        node_text = get_node_text(node, code).decode("utf8").strip()

        # Detect keywords: tree-sitter surfaces TS keywords as nodes
        # whose type *is* the keyword string (e.g. node.type == "if")
        if node_text in ts_keywords and len(node.children) == 0:
            unique_keywords.add(node_text)

        # Also catch type annotations for predefined types (number, string, etc.)
        if node.type == "predefined_type":
            unique_keywords.add(node_text)

        # Scan condition sub-trees for operators
        if node.type in CONDITIONAL_NODE_TYPES:
            # For ternary, the whole expression is the condition context
            condition = None
            if node.type == "ternary_expression":
                condition = node  # operators can appear anywhere in it
            else:
                condition = node.child_by_field_name("condition")
            if condition:
                extract_operators(condition)

        for child in node.children:
            traverse(child)

    traverse(tree.root_node)

    keywords_ratio = len(unique_keywords) / total_tokens if total_tokens > 0 else 0
    operators_ratio = len(unique_operators) / total_tokens if total_tokens > 0 else 0

    return keywords_ratio, operators_ratio
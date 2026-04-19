"""
JavaScript AST Traversal Module

This module provides functions to traverse JavaScript AST nodes using Tree-Sitter
and generate sequential representations following the Guo et al. approach.

Supports:
- Regular JavaScript (.js): functions, classes, arrow functions, destructuring, etc.
- JSX-containing JavaScript (.jsx): all of the above plus JSX elements,
  attributes, expressions, and fragments (uses the tsx-equivalent JS grammar)

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
    Map a JavaScript/JSX AST node to a sequence of tokens using recursive traversal.

    This implements the Guo et al. approach where:
    - Non-leaf nodes are represented as "node_type::left" and "node_type::right"
    - Leaf nodes, identifiers, literals, and JSX text are represented by
      their actual text

    JSX-specific handling (for .jsx snippets):
    - JSX element names (e.g. <MyComponent>) are treated as identifiers
    - JSX attribute names and string values are included as text
    - JSX expression containers {...} are traversed normally
    - JSX text content is included directly as a leaf

    Args:
        node: Tree-sitter AST node
        code: Source code as bytes

    Returns:
        list: Sequence of tokens representing the AST

    Example:
        For code: const x = 5;
        Output: ['program::left', 'lexical_declaration::left', 'const',
                 'variable_declarator::left', 'x', '=', '5',
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
        # JSX-specific (for .jsx snippets)
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


def analyze_javascript_code(tree, code):
    """
    Extract code features from JavaScript/JSX code for Section III-E.

    Analyzes the JavaScript AST to extract:
    - Keywords ratio: count of unique language keywords / total tokens
    - Conditional operators ratio: count of unique operators in if/while/for/ternary
      conditions / total tokens

    JSX-aware: JSX structure nodes are traversed normally; their leaf text
    is counted toward total_tokens.

    Args:
        tree: Tree-sitter parse tree
        code: Source code as bytes

    Returns:
        tuple: (keywords_ratio, conditional_operators_ratio)
    """
    total_tokens = 0
    unique_keywords = set()
    unique_operators = set()

    # JavaScript keyword set (ES2022)
    js_keywords = [
        # Reserved words
        "break", "case", "catch", "class", "const", "continue", "debugger",
        "default", "delete", "do", "else", "export", "extends", "false",
        "finally", "for", "function", "if", "import", "in", "instanceof",
        "let", "new", "null", "return", "static", "super", "switch", "this",
        "throw", "true", "try", "typeof", "var", "void", "while", "with",
        "yield",
        # ES6+
        "async", "await", "of", "from", "as",
    ]

    # Operators meaningful inside conditionals
    js_operators = ["&&", "||", "!", "??", "<", ">", "<=", ">=", "==", "!=", "===", "!==", "|", "&"]

    # Node types that represent conditional contexts
    CONDITIONAL_NODE_TYPES = {
        "if_statement",
        "while_statement",
        "for_statement",
        "for_in_statement",     # for...of / for...in
        "ternary_expression",   # condition ? a : b
        "do_statement",
    }

    def extract_operators(node):
        """Recursively extract conditional operators from a subtree."""
        node_text = get_node_text(node, code).decode("utf8")
        if node.type in ("binary_expression", "unary_expression"):
            for op in js_operators:
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

        # tree-sitter surfaces JS keywords as nodes whose type *is*
        # the keyword string (e.g. node.type == "if") at leaf level
        if node_text in js_keywords and len(node.children) == 0:
            unique_keywords.add(node_text)

        # Scan condition sub-trees for operators
        if node.type in CONDITIONAL_NODE_TYPES:
            condition = None
            if node.type == "ternary_expression":
                condition = node
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
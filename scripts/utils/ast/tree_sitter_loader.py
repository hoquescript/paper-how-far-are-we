from __future__ import annotations

import os
from ctypes import c_void_p, cdll

from tree_sitter import Language, Parser

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_FALLBACK_SO_PATH = os.path.join(_PROJECT_ROOT, "build", "my-languages.so")

_TS_LIBS: dict[str, object] = {}
_LANGUAGE_CACHE: dict[tuple[str, str], Language] = {}
_PARSER_CACHE: dict[tuple[str, str], Parser] = {}


def load_ts_language(name: str, so_path: str | None = None) -> Language:
    resolved_so_path = os.path.abspath(
        so_path or os.environ.get("TS_LANGUAGE_SO_PATH", _FALLBACK_SO_PATH)
    )
    cache_key = (name, resolved_so_path)
    if cache_key in _LANGUAGE_CACHE:
        return _LANGUAGE_CACHE[cache_key]

    if not os.path.exists(resolved_so_path):
        raise RuntimeError(f"Tree-sitter library not found at '{resolved_so_path}'.")

    ts_lib = _TS_LIBS.get(resolved_so_path)
    if ts_lib is None:
        ts_lib = cdll.LoadLibrary(resolved_so_path)
        _TS_LIBS[resolved_so_path] = ts_lib

    fn = getattr(ts_lib, f"tree_sitter_{name}", None)
    if fn is None:
        raise RuntimeError(
            f"Language symbol 'tree_sitter_{name}' not found in '{resolved_so_path}'."
        )

    fn.restype = c_void_p
    language = Language(fn())
    _LANGUAGE_CACHE[cache_key] = language
    return language


def get_parser_for_language(name: str, so_path: str | None = None) -> Parser:
    resolved_so_path = os.path.abspath(
        so_path or os.environ.get("TS_LANGUAGE_SO_PATH", _FALLBACK_SO_PATH)
    )
    cache_key = (name, resolved_so_path)
    if cache_key in _PARSER_CACHE:
        return _PARSER_CACHE[cache_key]

    parser = Parser()
    parser.language = load_ts_language(name, resolved_so_path)
    _PARSER_CACHE[cache_key] = parser
    return parser

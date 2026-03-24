"""
gpucc/annotations.py — Annotation sentinel objects

These are dummy Python objects that make @kernel function definitions
syntactically valid Python. They have no runtime semantics — the actual
type information is extracted from the AST by the compiler.

Usage:
    from gpucc import kernel, float32, int32, N, M, K

    @kernel
    def vector_add(a: float32[N], b: float32[N], c: float32[N], n: int32):
        ...

Or use  from __future__ import annotations  to avoid needing them at all.
"""
from __future__ import annotations


class _TypeSentinel:
    """
    A subscriptable dummy type object for kernel annotations.
    e.g.  float32[N]  →  _TypeSentinel("float32")[N]  →  _TypeSentinel("float32")
    The actual type resolution happens in gpucc.types.parse_annotation() via AST.
    """
    def __init__(self, name: str):
        self._name = name

    def __class_getitem__(cls, item):
        return cls

    def __getitem__(self, item):
        return self

    def __repr__(self):
        return self._name


# Scalar type sentinels
float32 = _TypeSentinel("float32")
int32   = _TypeSentinel("int32")
int64   = _TypeSentinel("int64")

# Symbolic dimension sentinels (any value works; these are just name placeholders)
N = None
M = None
K = None

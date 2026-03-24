"""
gpucc/types.py — GPU type system

Defines the GPUType hierarchy used throughout the compiler:
  - ScalarType subclasses: Int32Type, Int64Type, Float32Type, BoolType
  - PointerType: 64-bit pointer to a ScalarType (global memory)
  - ArrayType: multi-dimensional array (function annotations only; lowered to PointerType in IR)

Each type knows its PTX register prefix (%r, %rd, %f, %p) and register type (.u32, etc.).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional, Tuple


# ── Base ──────────────────────────────────────────────────────────────────────

class GPUType:
    """Base class for all compiler types."""

    def ptx_reg_prefix(self) -> str:
        """PTX register prefix, e.g. '%f', '%r', '%rd', '%p'."""
        raise NotImplementedError

    def ptx_reg_type(self) -> str:
        """PTX register type declaration, e.g. '.f32', '.u32', '.u64', '.pred'."""
        raise NotImplementedError

    def byte_size(self) -> int:
        raise NotImplementedError

    def is_scalar(self) -> bool:
        return isinstance(self, ScalarType)

    def is_pointer(self) -> bool:
        return isinstance(self, PointerType)

    def is_float(self) -> bool:
        return isinstance(self, Float32Type)

    def is_int(self) -> bool:
        return isinstance(self, (Int32Type, Int64Type))

    def is_bool(self) -> bool:
        return isinstance(self, BoolType)


# ── Scalar types ─────────────────────────────────────────────────────────────

class ScalarType(GPUType):
    """Primitive scalar types (int32, int64, float32, bool)."""
    pass


@dataclass(frozen=True)
class Int32Type(ScalarType):
    def ptx_reg_prefix(self) -> str: return "%r"
    def ptx_reg_type(self) -> str:   return ".u32"
    def byte_size(self) -> int:      return 4
    def __repr__(self):              return "int32"


@dataclass(frozen=True)
class Int64Type(ScalarType):
    def ptx_reg_prefix(self) -> str: return "%rd"
    def ptx_reg_type(self) -> str:   return ".u64"
    def byte_size(self) -> int:      return 8
    def __repr__(self):              return "int64"


@dataclass(frozen=True)
class Float32Type(ScalarType):
    def ptx_reg_prefix(self) -> str: return "%f"
    def ptx_reg_type(self) -> str:   return ".f32"
    def byte_size(self) -> int:      return 4
    def __repr__(self):              return "float32"


@dataclass(frozen=True)
class BoolType(ScalarType):
    """Predicate type — maps to PTX .pred registers."""
    def ptx_reg_prefix(self) -> str: return "%p"
    def ptx_reg_type(self) -> str:   return ".pred"
    def byte_size(self) -> int:      return 1
    def __repr__(self):              return "bool"


# ── Pointer type ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PointerType(GPUType):
    """
    64-bit pointer to element_type in GPU global memory.
    Kernel array parameters are lowered to PointerType in the IR.
    """
    element_type: ScalarType

    def ptx_reg_prefix(self) -> str: return "%rd"
    def ptx_reg_type(self) -> str:   return ".u64"
    def byte_size(self) -> int:      return 8
    def __repr__(self):              return f"{self.element_type}*"


# ── Array type (annotation-only) ─────────────────────────────────────────────

@dataclass(frozen=True)
class ArrayType(GPUType):
    """
    Multi-dimensional array type used only in @kernel parameter annotations.
    e.g.  float32[N]    → ArrayType(Float32Type(), shape=(None,))
          float32[M, K] → ArrayType(Float32Type(), shape=(None, None))

    After parameter lowering in ast_to_ir.py, ArrayType is replaced by PointerType.
    This type never appears in the IR proper.
    """
    element_type: ScalarType
    shape: Tuple[Optional[int], ...]  # None = symbolic / unknown dimension

    def ptx_reg_prefix(self) -> str: raise TypeError("ArrayType has no PTX register")
    def ptx_reg_type(self) -> str:   raise TypeError("ArrayType has no PTX register")
    def byte_size(self) -> int:      raise TypeError("ArrayType has no fixed size")

    def ndim(self) -> int:
        return len(self.shape)

    def __repr__(self):
        dims = ", ".join("?" if d is None else str(d) for d in self.shape)
        return f"{self.element_type}[{dims}]"


# ── Singleton constants ───────────────────────────────────────────────────────

i32    = Int32Type()
i64    = Int64Type()
f32    = Float32Type()
bool_t = BoolType()


def pointer_to(t: ScalarType) -> PointerType:
    return PointerType(t)


# ── Annotation parser ─────────────────────────────────────────────────────────

_SCALAR_MAP: dict[str, ScalarType] = {
    "float32": f32,
    "int32":   i32,
    "int64":   i64,
    "bool":    bool_t,
}


def parse_annotation(node: ast.expr) -> GPUType:
    """
    Parse a Python AST annotation node into a GPUType.

    Handles:
      ast.Name         → scalar type  (e.g., 'int32' → Int32Type())
      ast.Subscript    → array type   (e.g., 'float32[N]' → ArrayType(f32, (None,)))
      ast.Attribute    → not supported
    """
    if isinstance(node, ast.Name):
        name = node.id
        if name not in _SCALAR_MAP:
            raise TypeError(
                f"Unknown type annotation '{name}'. "
                f"Supported: {list(_SCALAR_MAP)}"
            )
        return _SCALAR_MAP[name]

    if isinstance(node, ast.Subscript):
        base = parse_annotation(node.value)
        if not isinstance(base, ScalarType):
            raise TypeError(
                f"Array element type must be scalar, got {base!r}"
            )
        dims = _extract_dims(node.slice)
        return ArrayType(base, dims)

    raise TypeError(
        f"Unsupported annotation node: {ast.dump(node)}"
    )


def _extract_dims(slice_node: ast.expr) -> Tuple[Optional[int], ...]:
    """Convert a subscript slice into a tuple of dimension sizes (None = symbolic)."""
    if isinstance(slice_node, ast.Tuple):
        return tuple(_single_dim(elt) for elt in slice_node.elts)
    return (_single_dim(slice_node),)


def _single_dim(node: ast.expr) -> Optional[int]:
    """Return int if the node is an integer constant, else None (symbolic dim)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None

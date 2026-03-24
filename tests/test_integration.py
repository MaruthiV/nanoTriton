"""tests/test_integration.py — End-to-end pipeline tests (CPU only)

Tests the full pipeline: @kernel decorator → IR → PTX string.
Validates the PTX structure without requiring a GPU.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from gpucc import kernel, float32, int32, N, M, K


# ── vector_add end-to-end ─────────────────────────────────────────────────────

@kernel
def vector_add(a: float32[N], b: float32[N], c: float32[N], n: int32):
    tid = thread_id()
    if tid < n:
        c[tid] = a[tid] + b[tid]


def test_kernel_handle_name():
    assert vector_add.name == "vector_add"


def test_kernel_ir_not_none():
    assert vector_add.ir is not None
    assert vector_add.ir.name == "vector_add"


def test_kernel_ir_text():
    ir_text = vector_add.ir_text
    assert "function vector_add" in ir_text
    assert "tid" in ir_text.lower() or "TID" in ir_text.lower() or "tid" in ir_text


def test_kernel_ptx_is_string():
    ptx = vector_add.ptx
    assert isinstance(ptx, str)
    assert len(ptx) > 0


def test_kernel_ptx_valid_structure():
    ptx = vector_add.ptx
    assert ".version" in ptx
    assert ".target" in ptx
    assert ".visible .entry vector_add" in ptx
    assert "ret;" in ptx


def test_kernel_ptx_address_calc():
    """Verify the critical load/store address pattern is correct."""
    ptx = vector_add.ptx
    assert "mul.wide.u32" in ptx   # NOT mul.u64 on u32 register


def test_kernel_compile_opt_level_0():
    """opt_level=0 should skip all passes but still produce valid PTX."""
    ptx = vector_add.compile(opt_level=0)
    assert ".visible .entry vector_add" in ptx
    assert "ret;" in ptx


def test_kernel_compile_sm80():
    """Different SM target should be reflected in PTX."""
    ptx = vector_add.compile(sm_version="sm_80")
    assert ".target sm_80" in ptx


# ── scalar ops ────────────────────────────────────────────────────────────────

@kernel
def scalar_ops(a: float32[N], b: float32[N], c: float32[N], n: int32):
    tid = thread_id()
    val_a = a[tid]
    val_b = b[tid]
    c[tid] = val_a * val_b - val_a


def test_scalar_ops_ptx():
    ptx = scalar_ops.ptx
    assert "mul.f32" in ptx
    assert "sub.f32" in ptx


# ── block/grid intrinsics ─────────────────────────────────────────────────────

@kernel
def grid_kernel(a: float32[N], n: int32):
    row = block_id(0) * block_size(0) + thread_id(0)
    x = a[row]


def test_grid_kernel_ptx():
    ptx = grid_kernel.ptx
    assert "%ctaid.x" in ptx
    assert "%ntid.x" in ptx
    assert "%tid.x" in ptx

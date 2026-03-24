"""tests/test_codegen.py — PTX code generation tests (CPU only)"""
import ast
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from gpucc.frontend.ast_to_ir import ASTToIR
from gpucc.codegen.ptx_emit import PTXEmitter
from gpucc.codegen.regalloc import allocate


def _parse_and_emit(src: str, sm_version: str = "sm_75") -> str:
    tree = ast.parse(src)
    func_def = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    fn = ASTToIR().compile(func_def)
    emitter = PTXEmitter(sm_version=sm_version)
    return emitter.emit_function(fn)


# ── PTX structural tests ──────────────────────────────────────────────────────

def test_ptx_header():
    src = """
def k(n: int32):
    pass
"""
    ptx = _parse_and_emit(src)
    assert ".version 7.0" in ptx
    assert ".target sm_75" in ptx
    assert ".address_size 64" in ptx


def test_ptx_entry_name():
    src = """
def my_kernel(n: int32):
    pass
"""
    ptx = _parse_and_emit(src)
    assert ".visible .entry my_kernel(" in ptx


def test_ptx_param_declarations():
    src = """
def k(a: float32[N], n: int32):
    pass
"""
    ptx = _parse_and_emit(src)
    assert "param .u64 param_a" in ptx
    assert "param .u32 param_n" in ptx


def test_ptx_reg_declarations():
    src = """
def k(a: float32[N], n: int32):
    tid = thread_id()
    x = a[tid]
"""
    ptx = _parse_and_emit(src)
    assert ".reg .f32" in ptx or ".reg .u32" in ptx


def test_ptx_thread_intrinsic():
    src = """
def k(n: int32):
    tid = thread_id()
"""
    ptx = _parse_and_emit(src)
    assert "mov.u32" in ptx
    assert "%tid.x" in ptx


def test_ptx_load_global():
    src = """
def k(a: float32[N], n: int32):
    tid = thread_id()
    x = a[tid]
"""
    ptx = _parse_and_emit(src)
    assert "cvta.to.global.u64" in ptx
    assert "mul.wide.u32" in ptx
    assert "add.u64" in ptx
    assert "ld.global.f32" in ptx


def test_ptx_store_global():
    src = """
def k(c: float32[N], n: int32):
    tid = thread_id()
    c[tid] = 1.0
"""
    ptx = _parse_and_emit(src)
    assert "st.global.f32" in ptx


def test_ptx_float_add():
    src = """
def k(a: float32[N], b: float32[N], c: float32[N], n: int32):
    tid = thread_id()
    c[tid] = a[tid] + b[tid]
"""
    ptx = _parse_and_emit(src)
    assert "add.f32" in ptx


def test_ptx_conditional_branch():
    src = """
def k(a: float32[N], n: int32):
    tid = thread_id()
    if tid < n:
        x = a[tid]
"""
    ptx = _parse_and_emit(src)
    # Should have a predicated branch
    assert "setp." in ptx
    assert "@%" in ptx and "bra" in ptx


def test_ptx_ret():
    src = """
def k(n: int32):
    pass
"""
    ptx = _parse_and_emit(src)
    assert "ret;" in ptx


def test_ptx_sm_version_override():
    src = """
def k(n: int32):
    pass
"""
    ptx = _parse_and_emit(src, sm_version="sm_80")
    assert ".target sm_80" in ptx


def test_full_vector_add_ptx():
    src = """
def vector_add(a: float32[N], b: float32[N], c: float32[N], n: int32):
    tid = thread_id()
    if tid < n:
        c[tid] = a[tid] + b[tid]
"""
    ptx = _parse_and_emit(src)

    # Structural checks
    assert ".visible .entry vector_add(" in ptx
    assert "param .u64 param_a" in ptx
    assert "mov.u32" in ptx and "%tid.x" in ptx
    assert "ld.global.f32" in ptx
    assert "add.f32" in ptx
    assert "st.global.f32" in ptx
    assert "ret;" in ptx

    # Verify address calculation pattern for load
    assert "cvta.to.global.u64" in ptx
    assert "mul.wide.u32" in ptx   # 32-bit index → 64-bit offset (CRITICAL)
    assert "add.u64" in ptx        # final address

    # Ensure no type errors: mul.u64 on %r register is wrong
    # (mul.wide.u32 should be used instead)
    lines = ptx.split("\n")
    for line in lines:
        if "mul.u64" in line and "%r" in line:
            pytest.fail(
                f"Type error: mul.u64 used with u32 register (%r): {line!r}. "
                "Should use mul.wide.u32 instead."
            )

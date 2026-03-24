"""tests/test_frontend.py — AST → IR frontend unit tests (CPU only)"""
import ast
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from gpucc.frontend.ast_to_ir import ASTToIR
from gpucc.ir import Const, IRFunction, Op, VReg
from gpucc.types import Float32Type, Int32Type, PointerType, bool_t, f32, i32


def _parse_kernel(src: str) -> IRFunction:
    """Helper: parse a kernel source string → IRFunction."""
    tree = ast.parse(src)
    func_def = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    return ASTToIR().compile(func_def)


def test_vector_add_params():
    src = """
def vector_add(a: float32[N], b: float32[N], c: float32[N], n: int32):
    pass
"""
    fn = _parse_kernel(src)
    assert fn.name == "vector_add"
    assert len(fn.params) == 4
    names = [p[0] for p in fn.params]
    assert names == ["a", "b", "c", "n"]
    # ArrayType params should be lowered to PointerType
    assert isinstance(fn.params[0][1], PointerType)
    assert isinstance(fn.params[1][1], PointerType)
    assert isinstance(fn.params[3][1], Int32Type)


def test_thread_intrinsic():
    src = """
def k(n: int32):
    tid = thread_id()
"""
    fn = _parse_kernel(src)
    instrs = list(fn.all_instructions())
    # Should have a TID instruction
    tid_instrs = [i for i in instrs if i.op == Op.TID]
    assert len(tid_instrs) == 1
    assert tid_instrs[0].meta["dim"] == "x"
    assert isinstance(tid_instrs[0].dst.type, Int32Type)


def test_if_creates_blocks():
    src = """
def k(n: int32):
    tid = thread_id()
    if tid < n:
        x = 1
"""
    fn = _parse_kernel(src)
    fn.build_cfg()
    # Should have: entry, then, (possibly else), merge
    labels = [bb.label for bb in fn.blocks]
    assert any("then" in l for l in labels)
    assert any("merge" in l for l in labels)


def test_for_loop_creates_blocks():
    src = """
def k(n: int32):
    acc = 0
    for i in range(n):
        acc = acc + i
"""
    fn = _parse_kernel(src)
    labels = [bb.label for bb in fn.blocks]
    assert any("loop_header" in l for l in labels)
    assert any("loop_body" in l for l in labels)
    assert any("loop_latch" in l for l in labels)
    assert any("loop_exit" in l for l in labels)


def test_for_loop_constant_bound_unrollable():
    src = """
def k():
    for i in range(8):
        x = i + 1
"""
    fn = _parse_kernel(src)
    # The header CBRANCH should be tagged unrollable=True
    for bb in fn.blocks:
        for instr in bb.instructions:
            if instr.op == Op.CBRANCH and "unrollable" in instr.meta:
                assert instr.meta["unrollable"] is True
                return
    pytest.fail("No unrollable CBRANCH found for constant range loop")


def test_for_loop_param_bound_not_unrollable():
    src = """
def k(n: int32):
    for i in range(n):
        x = i + 1
"""
    fn = _parse_kernel(src)
    for bb in fn.blocks:
        for instr in bb.instructions:
            if instr.op == Op.CBRANCH and "unrollable" in instr.meta:
                assert instr.meta["unrollable"] is False
                return
    pytest.fail("No CBRANCH with unrollable=False found for param range loop")


def test_load_emitted_for_subscript():
    src = """
def k(a: float32[N], n: int32):
    tid = thread_id()
    x = a[tid]
"""
    fn = _parse_kernel(src)
    loads = [i for i in fn.all_instructions() if i.op == Op.LOAD]
    assert len(loads) >= 1
    assert isinstance(loads[0].type, Float32Type)
    assert loads[0].meta.get("space") == "global"


def test_store_emitted_for_subscript_assign():
    src = """
def k(c: float32[N], n: int32):
    tid = thread_id()
    c[tid] = 1.0
"""
    fn = _parse_kernel(src)
    stores = [i for i in fn.all_instructions() if i.op == Op.STORE]
    assert len(stores) >= 1
    assert stores[0].meta.get("space") == "global"


def test_binop_types():
    src = """
def k(n: int32):
    x = 1
    y = x + 2
    z = 1.0 + 2.0
"""
    fn = _parse_kernel(src)
    iadds = [i for i in fn.all_instructions() if i.op == Op.IADD]
    fadds = [i for i in fn.all_instructions() if i.op == Op.FADD]
    assert len(iadds) >= 1
    assert len(fadds) >= 1


def test_full_vector_add():
    src = """
def vector_add(a: float32[N], b: float32[N], c: float32[N], n: int32):
    tid = thread_id()
    if tid < n:
        c[tid] = a[tid] + b[tid]
"""
    fn = _parse_kernel(src)
    ops = {i.op for i in fn.all_instructions()}
    assert Op.TID in ops
    assert Op.LOAD in ops
    assert Op.STORE in ops
    assert Op.FADD in ops
    assert Op.LT in ops
    assert Op.CBRANCH in ops
    assert Op.RET in ops

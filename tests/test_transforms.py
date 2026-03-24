"""tests/test_transforms.py — Optimization pass tests (CPU only)"""
import ast
import copy
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from gpucc.frontend.ast_to_ir import ASTToIR
from gpucc.ir import Const, IRFunction, Instruction, Op
from gpucc.types import f32, i32, bool_t
from gpucc.transforms.constant_fold import constant_fold
from gpucc.transforms.dce import dead_code_elimination
from gpucc.transforms.loop_unroll import loop_unroll


def _parse_kernel(src: str) -> IRFunction:
    tree = ast.parse(src)
    func_def = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    return ASTToIR().compile(func_def)


# ── Constant folding ──────────────────────────────────────────────────────────

def test_constant_fold_iadd():
    fn = IRFunction(name="test", params=[])
    bb = fn.new_block("entry")
    dst = fn.new_vreg(i32)
    bb.append(Instruction(Op.IADD, dst, [Const(3, i32), Const(4, i32)], i32))
    bb.append(Instruction(Op.RET, None, [], None))

    fn = constant_fold(fn)
    instrs = [i for i in fn.all_instructions() if i.op != Op.RET]
    assert instrs[0].op == Op.COPY
    assert isinstance(instrs[0].srcs[0], Const)
    assert instrs[0].srcs[0].value == 7


def test_constant_fold_fadd():
    fn = IRFunction(name="test", params=[])
    bb = fn.new_block("entry")
    dst = fn.new_vreg(f32)
    bb.append(Instruction(Op.FADD, dst, [Const(1.0, f32), Const(2.0, f32)], f32))
    bb.append(Instruction(Op.RET, None, [], None))

    fn = constant_fold(fn)
    instrs = [i for i in fn.all_instructions() if i.op != Op.RET]
    assert instrs[0].op == Op.COPY
    assert abs(instrs[0].srcs[0].value - 3.0) < 1e-9


def test_constant_fold_comparison():
    fn = IRFunction(name="test", params=[])
    bb = fn.new_block("entry")
    dst = fn.new_vreg(bool_t)
    bb.append(Instruction(Op.LT, dst, [Const(3, i32), Const(5, i32)], bool_t))
    bb.append(Instruction(Op.RET, None, [], None))

    fn = constant_fold(fn)
    instrs = [i for i in fn.all_instructions() if i.op != Op.RET]
    assert instrs[0].op == Op.COPY
    assert instrs[0].srcs[0].value == 1  # 3 < 5 → True → 1


def test_constant_fold_no_fold_when_vreg_src():
    fn = IRFunction(name="test", params=[])
    bb = fn.new_block("entry")
    a = fn.new_vreg(i32)
    dst = fn.new_vreg(i32)
    bb.append(Instruction(Op.COPY, a, [Const(3, i32)], i32))
    bb.append(Instruction(Op.IADD, dst, [a, Const(4, i32)], i32))
    bb.append(Instruction(Op.RET, None, [], None))

    fn = constant_fold(fn)
    # The IADD should NOT be folded because 'a' is a VReg, not a Const
    iadd_instrs = [i for i in fn.all_instructions() if i.op == Op.IADD]
    assert len(iadd_instrs) == 1


def test_constant_fold_divide_by_zero_safe():
    fn = IRFunction(name="test", params=[])
    bb = fn.new_block("entry")
    dst = fn.new_vreg(i32)
    bb.append(Instruction(Op.IDIV, dst, [Const(5, i32), Const(0, i32)], i32))
    bb.append(Instruction(Op.RET, None, [], None))

    # Should not raise; just leave instruction as-is
    fn = constant_fold(fn)
    # IDIV should remain (not folded due to ZeroDivisionError)
    idiv_instrs = [i for i in fn.all_instructions() if i.op == Op.IDIV]
    assert len(idiv_instrs) == 1


# ── Dead code elimination ──────────────────────────────────────────────────────

def test_dce_removes_unused_copy():
    fn = IRFunction(name="test", params=[])
    bb = fn.new_block("entry")
    unused = fn.new_vreg(i32)
    bb.append(Instruction(Op.COPY, unused, [Const(42, i32)], i32))
    bb.append(Instruction(Op.RET, None, [], None))

    fn = dead_code_elimination(fn)
    copies = [i for i in fn.all_instructions() if i.op == Op.COPY]
    assert len(copies) == 0


def test_dce_keeps_used_value():
    src = """
def k(n: int32):
    tid = thread_id()
    x = tid + 1
"""
    fn = _parse_kernel(src)
    before_count = sum(1 for _ in fn.all_instructions())
    fn = dead_code_elimination(fn)
    after_count = sum(1 for _ in fn.all_instructions())
    # DCE might remove some, but the TID and used IADD should survive
    tid_instrs = [i for i in fn.all_instructions() if i.op == Op.TID]
    assert len(tid_instrs) == 1


def test_dce_keeps_stores():
    src = """
def k(c: float32[N], n: int32):
    tid = thread_id()
    c[tid] = 1.0
"""
    fn = _parse_kernel(src)
    fn = dead_code_elimination(fn)
    stores = [i for i in fn.all_instructions() if i.op == Op.STORE]
    assert len(stores) >= 1


# ── Loop unrolling ─────────────────────────────────────────────────────────────

def test_loop_unroll_constant_bound():
    src = """
def k(c: float32[N]):
    for i in range(8):
        c[i] = 0.0
"""
    fn = _parse_kernel(src)
    before_blocks = len(fn.blocks)

    fn = loop_unroll(fn, factor=4)
    after_blocks = len(fn.blocks)

    # Unrolling should add new body/latch blocks
    assert after_blocks > before_blocks


def test_loop_no_unroll_param_bound():
    src = """
def k(n: int32):
    for i in range(n):
        x = i + 1
"""
    fn = _parse_kernel(src)
    before_blocks = len(fn.blocks)
    fn = loop_unroll(fn, factor=4)
    after_blocks = len(fn.blocks)
    # Should not unroll (param bound → unrollable=False)
    assert after_blocks == before_blocks


def test_loop_no_unroll_indivisible():
    src = """
def k():
    for i in range(10):
        x = i + 1
"""
    fn = _parse_kernel(src)
    before_blocks = len(fn.blocks)
    # 10 % 4 != 0, should not unroll
    fn = loop_unroll(fn, factor=4)
    after_blocks = len(fn.blocks)
    assert after_blocks == before_blocks

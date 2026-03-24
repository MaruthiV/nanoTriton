"""
gpucc/transforms/constant_fold.py — Constant folding pass

Walks all instructions. If all source operands are Const, evaluates the
expression at compile time and replaces the instruction with a COPY of
the computed constant.

Example:
  %v5 = iadd int32 3, 4   →   %v5 = copy 7
  %v6 = fadd float32 1.0, 2.0  →  %v6 = copy 3.0
"""
from __future__ import annotations

import operator
from typing import Optional

from gpucc.ir import Const, Instruction, IRFunction, Op, Operand, VReg
from gpucc.types import BoolType, Float32Type, GPUType, Int32Type, bool_t, f32, i32


# Operations that can be constant-folded and their Python equivalents
_FOLD_TABLE: dict[str, callable] = {
    Op.FADD:  operator.add,
    Op.FSUB:  operator.sub,
    Op.FMUL:  operator.mul,
    Op.FDIV:  operator.truediv,
    Op.IADD:  operator.add,
    Op.ISUB:  operator.sub,
    Op.IMUL:  operator.mul,
    Op.IDIV:  operator.floordiv,
    Op.IMOD:  operator.mod,
    Op.LT:    operator.lt,
    Op.LE:    operator.le,
    Op.GT:    operator.gt,
    Op.GE:    operator.ge,
    Op.EQ:    operator.eq,
    Op.NE:    operator.ne,
    Op.AND:   operator.and_,
    Op.OR:    operator.or_,
    Op.CVTI2F: float,
    Op.CVTF2I: int,
}


def constant_fold(fn: IRFunction) -> IRFunction:
    """
    In-place constant folding on fn.  Returns fn (mutated).
    """
    for bb in fn.blocks:
        for i, instr in enumerate(bb.instructions):
            folded = _try_fold(instr)
            if folded is not None:
                bb.instructions[i] = folded
    return fn


def _try_fold(instr: Instruction) -> Optional[Instruction]:
    """
    Attempt to fold instr into a COPY of a constant.
    Returns the replacement Instruction, or None if not foldable.
    """
    op = instr.op
    if op not in _FOLD_TABLE:
        return None
    if instr.dst is None:
        return None
    # All sources must be constants
    if not all(isinstance(s, Const) for s in instr.srcs):
        return None

    fn = _FOLD_TABLE[op]
    vals = [s.value for s in instr.srcs]

    try:
        if op in (Op.CVTI2F,):
            result_val = float(vals[0])
            result_type = f32
        elif op in (Op.CVTF2I,):
            result_val = int(vals[0])
            result_type = i32
        elif op in (Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NE, Op.AND, Op.OR):
            result_val = int(bool(fn(*vals)))
            result_type = bool_t
        else:
            result_val = fn(*vals)
            result_type = instr.type or i32
            # Preserve float/int nature
            if isinstance(result_type, Float32Type):
                result_val = float(result_val)
            else:
                result_val = int(result_val)
    except (ZeroDivisionError, OverflowError, ValueError):
        return None

    folded_const = Const(result_val, result_type)
    return Instruction(
        op=Op.COPY,
        dst=instr.dst,
        srcs=[folded_const],
        type=result_type,
        meta={"folded_from": op},
    )

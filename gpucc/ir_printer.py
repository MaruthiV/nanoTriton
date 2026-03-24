"""
gpucc/ir_printer.py — IR pretty-printer

Dumps an IRFunction (or IRModule) to human-readable text.
Essential for debugging every stage of the compiler.

Example output:

  function vector_add(a: float32*, b: float32*, c: float32*, n: int32)
  ─────────────────────────────────────────────────────────────────────
  BB0_entry:
    %v0:tid    = tid[x]
    %v1:a      = param[0] a (float32*)
    %v2:b      = param[1] b (float32*)
    %v3:c      = param[2] c (float32*)
    %v4:n      = param[3] n (int32)
    %v5        = lt int32 %v0:tid, %v4:n
                 cbranch %v5 → BB2_then | BB3_merge
  BB2_then:
    %v6        = load.global float32 %v1:a[%v0:tid]
    %v7        = load.global float32 %v2:b[%v0:tid]
    %v8        = fadd float32 %v6, %v7
                 store.global float32 %v3:c[%v0:tid] = %v8
                 jump → BB3_merge
  BB3_merge:
                 ret
"""
from __future__ import annotations

from typing import Union

from gpucc.ir import (
    BasicBlock, Const, IRFunction, IRModule, Instruction, Op, Operand, VReg,
)
from gpucc.types import GPUType


def _fmt_operand(op: Operand) -> str:
    if isinstance(op, VReg):
        return repr(op)
    return repr(op)


def _fmt_type(t: GPUType) -> str:
    return repr(t)


def _fmt_instr(instr: Instruction) -> str:
    op = instr.op
    dst = instr.dst
    srcs = instr.srcs
    meta = instr.meta
    typ = instr.type

    dst_str = f"{repr(dst):<14}" if dst else " " * 14

    if op == Op.NOP:
        return f"{dst_str}  nop"

    if op == Op.RET:
        return f"{'':14}  ret"

    if op == Op.JUMP:
        return f"{'':14}  jump → {meta['label']}"

    if op == Op.CBRANCH:
        pred = _fmt_operand(srcs[0])
        return f"{'':14}  cbranch {pred} → {meta['true_label']} | {meta['false_label']}"

    if op == Op.PARAM:
        t = _fmt_type(typ) if typ else "?"
        return (f"{dst_str}  = param[{meta['param_index']}] "
                f"{meta['param_name']} ({t})")

    if op == Op.COPY:
        return f"{dst_str}  = copy {_fmt_operand(srcs[0])}"

    if op in (Op.TID, Op.CTAID, Op.NTID, Op.NCTAID):
        return f"{dst_str}  = {op}[{meta['dim']}]"

    if op == Op.LOAD:
        ptr = _fmt_operand(srcs[0])
        idx = _fmt_operand(srcs[1])
        space = meta.get("space", "global")
        t = _fmt_type(typ) if typ else "?"
        return f"{dst_str}  = load.{space} {t} {ptr}[{idx}]"

    if op == Op.STORE:
        ptr = _fmt_operand(srcs[0])
        idx = _fmt_operand(srcs[1])
        val = _fmt_operand(srcs[2])
        space = meta.get("space", "global")
        t = _fmt_type(typ) if typ else "?"
        return f"{'':14}  store.{space} {t} {ptr}[{idx}] = {val}"

    # Generic: dst = op type src0, src1, ...
    t = _fmt_type(typ) if typ else ""
    srcs_str = ", ".join(_fmt_operand(s) for s in srcs)
    type_str = f" {t}" if t else ""
    return f"{dst_str}  = {op}{type_str} {srcs_str}"


def print_function(fn: IRFunction, *, indent: int = 2) -> str:
    """Return a pretty-printed string representation of an IRFunction."""
    pad = " " * indent
    lines: list[str] = []

    # Header
    params_str = ", ".join(
        f"{name}: {_fmt_type(typ)}" for name, typ in fn.params
    )
    lines.append(f"function {fn.name}({params_str})")
    lines.append("─" * max(60, len(lines[0])))

    for bb in fn.blocks:
        lines.append(f"{bb.label}:")
        for instr in bb.instructions:
            lines.append(f"{pad}{_fmt_instr(instr)}")
        if not bb.instructions:
            lines.append(f"{pad}(empty)")
        lines.append("")

    return "\n".join(lines)


def print_module(mod: IRModule) -> str:
    """Return a pretty-printed string for all functions in an IRModule."""
    parts = [print_function(fn) for fn in mod.functions]
    return "\n\n".join(parts)


def dump(obj: Union[IRFunction, IRModule]) -> None:
    """Print to stdout. Convenience wrapper for debugging."""
    if isinstance(obj, IRFunction):
        print(print_function(obj))
    elif isinstance(obj, IRModule):
        print(print_module(obj))
    else:
        raise TypeError(f"Expected IRFunction or IRModule, got {type(obj)}")

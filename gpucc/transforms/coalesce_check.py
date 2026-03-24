"""
gpucc/transforms/coalesce_check.py — Memory coalescing analysis

Analyzes LOAD/STORE instructions and warns when access patterns
are likely to cause uncoalesced memory accesses.

A global memory access is coalesced when adjacent threads (warp lanes 0-31)
access adjacent memory addresses. On NVIDIA GPUs, a full warp coalesces into
a single 128-byte transaction if thread i accesses address[base + i].

This pass inspects the index operand of each LOAD/STORE:
  - COALESCED: index is threadIdx.x  (or threadIdx.x + constant)
  - STRIDED:   index is threadIdx.x * stride  (stride > 1 → uncoalesced)
  - UNKNOWN:   index is derived from block_id or loop variable (not analyzable)

This pass is ANALYSIS ONLY — it emits warnings to stderr, never mutates IR.
"""
from __future__ import annotations

import sys
from typing import Optional

from gpucc.ir import Const, IRFunction, Instruction, Op, Operand, VReg


def check_coalescing(fn: IRFunction) -> None:
    """
    Analyze memory access patterns in fn and print warnings for
    likely-uncoalesced accesses.  Does not modify the IR.
    """
    # Build a map from VReg id → defining instruction for quick lookup
    def_map: dict[int, Instruction] = {}
    for instr in fn.all_instructions():
        if instr.dst is not None:
            def_map[instr.dst.id] = instr

    for bb in fn.blocks:
        for instr in bb.instructions:
            if instr.op not in (Op.LOAD, Op.STORE):
                continue
            index_op = instr.srcs[1]
            kind, stride = _classify_index(index_op, def_map)

            if kind == "strided":
                _warn(fn.name, instr, stride)
            elif kind == "unknown":
                pass  # conservative: don't warn on unknowns


def _classify_index(
    index: Operand,
    def_map: dict[int, Instruction],
) -> tuple[str, Optional[int]]:
    """
    Classify the memory index access pattern.

    Returns (kind, stride) where kind is:
      "coalesced" — index is tid.x or tid.x + constant
      "strided"   — index is tid.x * k or similar (stride=k)
      "unknown"   — cannot determine statically
    """
    if isinstance(index, Const):
        return "unknown", None  # constant index — all threads same address (broadcast)

    if not isinstance(index, VReg):
        return "unknown", None

    defining = def_map.get(index.id)
    if defining is None:
        return "unknown", None

    # Direct tid.x use
    if defining.op == Op.TID and defining.meta.get("dim") == "x":
        return "coalesced", 1

    # tid.x + constant  (IADD with one TID source)
    if defining.op == Op.IADD:
        for src in defining.srcs:
            if isinstance(src, VReg):
                sub_def = def_map.get(src.id)
                if sub_def and sub_def.op == Op.TID and sub_def.meta.get("dim") == "x":
                    return "coalesced", 1

    # tid.x * constant  → strided
    if defining.op == Op.IMUL:
        for i, src in enumerate(defining.srcs):
            if isinstance(src, VReg):
                sub_def = def_map.get(src.id)
                if sub_def and sub_def.op == Op.TID and sub_def.meta.get("dim") == "x":
                    # Other operand is the stride
                    other = defining.srcs[1 - i]
                    if isinstance(other, Const):
                        return "strided", int(other.value)
                    return "strided", None  # unknown stride, still suspicious

    return "unknown", None


def _warn(fn_name: str, instr: Instruction, stride: Optional[int]) -> None:
    stride_str = f" (stride={stride})" if stride is not None else ""
    op_str = "LOAD" if instr.op == Op.LOAD else "STORE"
    print(
        f"[coalesce] WARNING in '{fn_name}': {op_str} may be uncoalesced{stride_str}. "
        f"Index: {instr.srcs[1]}",
        file=sys.stderr,
    )

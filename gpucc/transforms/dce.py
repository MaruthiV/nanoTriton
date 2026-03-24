"""
gpucc/transforms/dce.py — Dead code elimination pass

Removes instructions whose result (dst VReg) is never used anywhere
in the function, and whose operation has no observable side effects.

Side-effect-free ops (Op.is_pure) can be removed if their dst is unused.
Side-effecting ops (STORE, RET, CBRANCH, JUMP) are always kept.

Algorithm:
  1. Collect all VReg ids that appear as source operands → live_ids
  2. Walk instructions: if dst not in live_ids and op is pure → replace with NOP
  3. Remove NOPs from block instruction lists

This is a single-pass approximation (not iterative). It handles most
cases that arise from constant folding eliminating comparisons / copies.
"""
from __future__ import annotations

from gpucc.ir import IRFunction, Op, VReg


def dead_code_elimination(fn: IRFunction) -> IRFunction:
    """In-place DCE on fn. Returns fn (mutated)."""

    # ── Pass 1: collect all used VReg ids ─────────────────────────────────────
    live_ids: set[int] = set()
    for instr in fn.all_instructions():
        for src in instr.srcs:
            if isinstance(src, VReg):
                live_ids.add(src.id)

    # ── Pass 2: mark dead instructions as NOP ─────────────────────────────────
    for bb in fn.blocks:
        for i, instr in enumerate(bb.instructions):
            if instr.dst is None:
                continue  # no result → definitely not dead by value
            if not Op.is_pure(instr.op):
                continue  # side-effecting → keep
            if instr.dst.id not in live_ids:
                bb.instructions[i] = _make_nop()

    # ── Pass 3: remove NOPs from each block ───────────────────────────────────
    for bb in fn.blocks:
        bb.instructions = [i for i in bb.instructions if i.op != Op.NOP]

    return fn


def _make_nop():
    from gpucc.ir import Instruction
    return Instruction(op=Op.NOP, dst=None, srcs=[], type=None)

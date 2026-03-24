"""
gpucc/transforms/loop_unroll.py — Loop unrolling pass

Unrolls counted for-loops with statically-known bounds.

A loop is eligible for unrolling if:
  - Its header CBRANCH is tagged  meta["unrollable"] = True  (set by ast_to_ir)
  - The stop bound in the LT instruction is a Const (constant bound)
  - The trip count (stop - start) is divisible by the unroll factor K

When a loop is unrolled by factor K:
  - The body BasicBlock is duplicated K times
  - The loop counter increment in the latch is bumped by K per iteration
  - The header comparison uses the adjusted stride
  - A cleanup block handles the remainder (if trip_count % K != 0)

For the MVP, we only unroll loops where trip_count is fully divisible by K.
Loops with parameter bounds (unrollable=False) are left as-is.

Default unroll factor K = 4 (configurable via UNROLL_FACTOR).
"""
from __future__ import annotations

import copy
from typing import List, Optional, Tuple

from gpucc.ir import (
    BasicBlock, Const, IRFunction, Instruction, Op, VReg,
)
from gpucc.types import i32


UNROLL_FACTOR = 4  # default unroll factor


def loop_unroll(fn: IRFunction, factor: int = UNROLL_FACTOR) -> IRFunction:
    """
    Detect and unroll constant-bound loops in fn.
    Returns fn (mutated in-place).
    """
    loops = _find_unrollable_loops(fn)
    for loop_info in loops:
        _unroll_loop(fn, loop_info, factor)
    fn.build_cfg()
    return fn


# ── Loop detection ─────────────────────────────────────────────────────────────

def _find_unrollable_loops(fn: IRFunction) -> List[dict]:
    """
    Find loops eligible for unrolling.
    Returns a list of loop_info dicts:
      {
        "header": BasicBlock,   header block (has LT + CBRANCH)
        "body":   BasicBlock,   first body block
        "latch":  BasicBlock,   last block before back-edge to header
        "exit":   BasicBlock,   first block after the loop
        "counter": VReg,        loop counter vreg
        "start":  int,          loop start value (from Const)
        "stop":   int,          loop stop value (from Const)
        "step":   int,          loop step value (from Const or default 1)
      }
    """
    results = []
    for bb in fn.blocks:
        info = _check_loop_header(fn, bb)
        if info is not None:
            results.append(info)
    return results


def _check_loop_header(fn: IRFunction, bb: BasicBlock) -> Optional[dict]:
    """
    Check if bb is a loop header that can be unrolled.
    Pattern: last two instructions are  LT(counter, Const)  +  CBRANCH(unrollable=True)
    """
    instrs = bb.instructions
    if len(instrs) < 2:
        return None

    cbranch = instrs[-1]
    lt_instr = instrs[-2]

    if cbranch.op != Op.CBRANCH:
        return None
    if not cbranch.meta.get("unrollable", False):
        return None
    if lt_instr.op != Op.LT:
        return None

    counter_op = lt_instr.srcs[0]
    stop_op    = lt_instr.srcs[1]

    if not isinstance(counter_op, VReg):
        return None
    if not isinstance(stop_op, Const):
        return None

    stop_val = int(stop_op.value)

    # Find the start value from the COPY that initializes the counter
    start_val = _find_counter_init(fn, counter_op)
    if start_val is None:
        return None

    # Find body, latch, exit blocks
    body_label = cbranch.meta.get("true_label")
    exit_label = cbranch.meta.get("false_label")
    if not body_label or not exit_label:
        return None

    body_bb = fn.block_by_label(body_label)
    exit_bb = fn.block_by_label(exit_label)

    # Find latch: the block that jumps back to the header
    latch_bb = _find_latch(fn, bb.label)
    if latch_bb is None:
        return None

    # Find step from IADD in latch
    step_val = _find_step(latch_bb, counter_op)
    if step_val is None:
        step_val = 1

    return {
        "header":  bb,
        "body":    body_bb,
        "latch":   latch_bb,
        "exit":    exit_bb,
        "counter": counter_op,
        "start":   start_val,
        "stop":    stop_val,
        "step":    step_val,
    }


def _find_counter_init(fn: IRFunction, counter: VReg) -> Optional[int]:
    """Find the constant value used to initialize the loop counter."""
    for instr in fn.all_instructions():
        if instr.op == Op.COPY and instr.dst == counter:
            if instr.srcs and isinstance(instr.srcs[0], Const):
                return int(instr.srcs[0].value)
    return None


def _find_latch(fn: IRFunction, header_label: str) -> Optional[BasicBlock]:
    """Find the latch block (the one that jumps back to header_label)."""
    for bb in fn.blocks:
        term = bb.terminator()
        if term and term.op == Op.JUMP:
            if term.meta.get("label") == header_label:
                return bb
    return None


def _find_step(latch: BasicBlock, counter: VReg) -> Optional[int]:
    """Find the integer step increment for counter in the latch block."""
    for instr in latch.instructions:
        if instr.op == Op.IADD and instr.dst == counter:
            for src in instr.srcs:
                if isinstance(src, Const):
                    return int(src.value)
    return None


# ── Unrolling ──────────────────────────────────────────────────────────────────

def _unroll_loop(fn: IRFunction, info: dict, factor: int) -> None:
    """
    Unroll the loop described by info by the given factor.
    Only unrolls if (stop - start) % (step * factor) == 0.
    """
    start  = info["start"]
    stop   = info["stop"]
    step   = info["step"]
    trip   = (stop - start) // step  # total iterations

    if trip <= 0 or factor <= 1:
        return

    # Only unroll if trip count is divisible by factor
    if trip % factor != 0:
        return

    body_bb   = info["body"]
    latch_bb  = info["latch"]
    header_bb = info["header"]
    counter   = info["counter"]
    new_step  = step * factor

    # Duplicate the body block (factor - 1) additional times
    # Each copy has counter offset by step, step*2, ... step*(factor-1)
    prev_latch = latch_bb

    for k in range(1, factor):
        # Deep copy of body block
        new_body = _clone_block(fn, body_bb, suffix=f"_unroll{k}")
        # Shift counter references in the new body by k*step
        _shift_counter_uses(new_body, counter, k * step, fn)
        # Previous latch jumps into new body instead of back to header
        _redirect_jump(prev_latch, body_bb.label, new_body.label)
        # New body's latch: create a clone of the latch
        new_latch = _clone_block(fn, latch_bb, suffix=f"_unroll{k}_latch")
        _redirect_jump(new_body, latch_bb.label, new_latch.label)
        prev_latch = new_latch

    # Final latch jumps back to header (unchanged)
    # Update the step in the original latch to new_step
    _update_step(latch_bb, counter, new_step)
    # Update the step in all cloned latches to 0 (only original updates counter)
    # Actually all latches need no increment except the final one
    # Simpler: set step to 0 in all cloned latches, keep only original latch incrementing
    # Re-do: each cloned latch does not increment counter; only original latch does
    # (already updated to new_step above)
    # Set cloned latch increments to 0:
    for bb in fn.blocks:
        if "_unroll" in bb.label and "_latch" in bb.label:
            _update_step(bb, counter, 0)


def _clone_block(fn: IRFunction, template: BasicBlock, suffix: str = "") -> BasicBlock:
    """Deep-copy a BasicBlock, assign a new label, add to fn.blocks."""
    new_bb = BasicBlock(label=template.label + suffix)
    new_bb.instructions = copy.deepcopy(template.instructions)
    fn.blocks.append(new_bb)
    return new_bb


def _redirect_jump(bb: BasicBlock, old_label: str, new_label: str) -> None:
    """Change a JUMP or CBRANCH in bb from old_label to new_label."""
    for instr in bb.instructions:
        if instr.op == Op.JUMP and instr.meta.get("label") == old_label:
            instr.meta["label"] = new_label
        elif instr.op == Op.CBRANCH:
            if instr.meta.get("true_label") == old_label:
                instr.meta["true_label"] = new_label
            if instr.meta.get("false_label") == old_label:
                instr.meta["false_label"] = new_label


def _shift_counter_uses(bb: BasicBlock, counter: VReg, offset: int, fn: IRFunction) -> None:
    """
    In a cloned body block, add 'offset' to every use of the loop counter
    by replacing counter references with a new VReg initialized to counter + offset.
    This models the loop body executing at iteration k.
    """
    if offset == 0:
        return

    # Insert an IADD at the top of the block: tmp = counter + offset
    tmp = fn.new_vreg(i32, name=f"{counter.name}_off{offset}" if counter.name else None)
    add_instr = Instruction(
        op=Op.IADD, dst=tmp,
        srcs=[counter, Const(offset, i32)],
        type=i32,
    )
    bb.instructions.insert(0, add_instr)

    # Replace uses of counter with tmp (skip the first instruction we just added)
    for instr in bb.instructions[1:]:
        instr.replace_src(counter, tmp)


def _update_step(bb: BasicBlock, counter: VReg, new_step: int) -> None:
    """Update the IADD step for counter in bb to new_step."""
    for instr in bb.instructions:
        if instr.op == Op.IADD and instr.dst == counter:
            for j, src in enumerate(instr.srcs):
                if isinstance(src, Const):
                    instr.srcs[j] = Const(new_step, i32)
                    return

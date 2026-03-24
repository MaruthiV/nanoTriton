"""
gpucc/codegen/regalloc.py — Register allocator

MVP: trivial 1-to-1 mapping.
Each VReg gets its own dedicated PTX register (no spilling, no live-range analysis).
PTX is a virtual ISA — ptxas performs real register allocation when compiling PTX→SASS.

Returns:
    RegAlloc: dict-like mapping from VReg → PTX register name string (e.g., "f7", "r3", "rd2")
    (The '%' prefix is added by the PTX emitter, not here.)
"""
from __future__ import annotations

from typing import Dict

from gpucc.ir import IRFunction, VReg
from gpucc.types import BoolType, Float32Type, Int32Type, Int64Type, PointerType


# PTX register class → name prefix (without %)
_REG_PREFIX = {
    Float32Type: "f",
    Int32Type:   "r",
    Int64Type:   "rd",
    PointerType: "rd",
    BoolType:    "p",
}


def allocate(fn: IRFunction) -> "RegAlloc":
    """
    Assign each VReg in fn a unique PTX register name.

    Naming:  VReg(id=7, type=Float32Type()) → 'f7'
             VReg(id=3, type=Int32Type())   → 'r3'
             VReg(id=2, type=PointerType()) → 'rd2'
             VReg(id=0, type=BoolType())    → 'p0'
    """
    mapping: Dict[int, str] = {}  # VReg.id → register name

    for vreg in fn.all_vregs():
        prefix = _reg_prefix(vreg)
        mapping[vreg.id] = f"{prefix}{vreg.id}"

    return RegAlloc(mapping, fn)


def _reg_prefix(vreg: VReg) -> str:
    t = type(vreg.type)
    if t in _REG_PREFIX:
        return _REG_PREFIX[t]
    # Fallback: use 'r' for unknown integer-like types
    return "r"


class RegAlloc:
    """
    Mapping from VReg → PTX register name.
    Also tracks, per type class, how many registers were used
    (needed for PTX .reg declarations).
    """

    def __init__(self, mapping: Dict[int, str], fn: IRFunction):
        self._mapping = mapping
        # Count registers per PTX class
        self._counts: Dict[str, int] = {"f": 0, "r": 0, "rd": 0, "p": 0}
        for vreg in fn.all_vregs():
            prefix = _reg_prefix(vreg)
            # Track max id per prefix to determine declaration count
            # (IDs are not necessarily dense per prefix, so track max)
            cur = self._counts.get(prefix, 0)
            self._counts[prefix] = max(cur, vreg.id + 1)

    def get(self, vreg: VReg) -> str:
        """Return the PTX register name (without '%') for this VReg."""
        if vreg.id not in self._mapping:
            raise KeyError(f"No allocation for {vreg!r}")
        return self._mapping[vreg.id]

    def ptx_name(self, vreg: VReg) -> str:
        """Return the full PTX register reference (with '%')."""
        return f"%{self.get(vreg)}"

    def declaration_count(self, prefix: str) -> int:
        """
        How many registers of this class to declare in the PTX prologue.
        prefix: 'f', 'r', 'rd', 'p'
        """
        return self._counts.get(prefix, 0)

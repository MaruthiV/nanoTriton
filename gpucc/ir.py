"""
gpucc/ir.py — Intermediate Representation

Data structures for the compiler's IR:
  VReg        — virtual register (typed, uniquely numbered per function)
  Const       — compile-time constant literal
  Instruction — single IR operation (opcode + dst + srcs + metadata)
  BasicBlock  — straight-line sequence of instructions
  IRFunction  — a compiled @kernel function (params + list of basic blocks)
  IRModule    — container for all compiled functions

The IR is intentionally non-SSA: variables can be re-assigned (COPY is used
to model assignment from a loop update). Control flow is explicit via
JUMP / CBRANCH terminators at the end of each BasicBlock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple, Union

from gpucc.types import GPUType


# ── Op codes ──────────────────────────────────────────────────────────────────

class Op:
    # Float arithmetic
    FADD   = "fadd"
    FSUB   = "fsub"
    FMUL   = "fmul"
    FDIV   = "fdiv"
    FNEG   = "fneg"

    # Integer arithmetic
    IADD   = "iadd"
    ISUB   = "isub"
    IMUL   = "imul"
    IDIV   = "idiv"
    IMOD   = "imod"
    INEG   = "ineg"

    # Type conversions
    CVTI2F = "cvt.i2f"   # int → float
    CVTF2I = "cvt.f2i"   # float → int
    CVTI2I = "cvt.i2i"   # int widening / narrowing

    # Comparisons → BoolType result
    LT     = "lt"
    LE     = "le"
    GT     = "gt"
    GE     = "ge"
    EQ     = "eq"
    NE     = "ne"

    # Logical (on BoolType operands)
    AND    = "and"
    OR     = "or"
    NOT    = "not"

    # Memory
    LOAD   = "load"    # dst = mem[ptr + index]
    STORE  = "store"   # mem[ptr + index] = src  (no dst)

    # Thread / block intrinsics
    TID    = "tid"     # dst = threadIdx.{x,y,z}
    CTAID  = "ctaid"   # dst = blockIdx.{x,y,z}
    NTID   = "ntid"    # dst = blockDim.{x,y,z}
    NCTAID = "nctaid"  # dst = gridDim.{x,y,z}

    # Control flow
    JUMP    = "jump"    # unconditional branch  meta: {label}
    CBRANCH = "cbranch" # conditional branch    meta: {true_label, false_label}
    RET     = "ret"     # kernel return (void)

    # Data movement / misc
    PARAM  = "param"   # dst = kernel parameter  meta: {param_name, param_index}
    COPY   = "copy"    # dst = src  (assignment / loop variable update)
    NOP    = "nop"     # eliminated instruction placeholder

    # Side-effect-free ops (used by DCE)
    _PURE = frozenset({
        FADD, FSUB, FMUL, FDIV, FNEG,
        IADD, ISUB, IMUL, IDIV, IMOD, INEG,
        CVTI2F, CVTF2I, CVTI2I,
        LT, LE, GT, GE, EQ, NE,
        AND, OR, NOT,
        LOAD,
        TID, CTAID, NTID, NCTAID,
        PARAM, COPY,
    })

    @classmethod
    def is_pure(cls, op: str) -> bool:
        return op in cls._PURE


# ── Operands ──────────────────────────────────────────────────────────────────

@dataclass
class VReg:
    """
    Virtual register — an IR value slot.

    id   : unique integer within the parent IRFunction
    type : GPUType (determines PTX register class)
    name : optional debug hint (Python source variable name)
    """
    id:   int
    type: GPUType
    name: Optional[str] = None

    def __repr__(self) -> str:
        tag = f":{self.name}" if self.name else ""
        return f"%v{self.id}{tag}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, VReg) and self.id == other.id


@dataclass(frozen=True)
class Const:
    """Compile-time constant literal."""
    value: Union[int, float]
    type: GPUType

    def __repr__(self):
        return f"{self.value}"


Operand = Union[VReg, Const]


# ── Instruction ───────────────────────────────────────────────────────────────

@dataclass
class Instruction:
    """
    One IR instruction.

    op   : Op constant string
    dst  : VReg receiving the result, or None for STORE / JUMP / CBRANCH / RET / NOP
    srcs : operand list (VReg or Const)
    type : result type (same as dst.type when dst is not None)
    meta : pass-specific annotations

    Common meta keys:
      LOAD / STORE  : space = "global" | "shared"
      TID / CTAID / NTID / NCTAID : dim = "x" | "y" | "z"
      CBRANCH       : true_label, false_label
      JUMP          : label
      PARAM         : param_name, param_index
      FOR loop      : unrollable = True | False   (set by ast_to_ir on the header CBRANCH)
    """
    op:   str
    dst:  Optional[VReg]
    srcs: List[Operand]
    type: Optional[GPUType]
    meta: Dict = field(default_factory=dict)

    def uses(self) -> List[VReg]:
        """All VReg operands consumed by this instruction."""
        return [s for s in self.srcs if isinstance(s, VReg)]

    def defs(self) -> List[VReg]:
        """VReg defined by this instruction (empty list for instructions with no dst)."""
        return [self.dst] if self.dst is not None else []

    def replace_src(self, old: VReg, new: Operand) -> None:
        """Replace all occurrences of old VReg in srcs with new operand (in-place)."""
        self.srcs = [new if (isinstance(s, VReg) and s == old) else s for s in self.srcs]


# ── Basic Block ───────────────────────────────────────────────────────────────

@dataclass
class BasicBlock:
    """
    A straight-line sequence of instructions ending in JUMP, CBRANCH, or RET.

    label        : unique string label within the function (used by branch targets)
    instructions : ordered list of Instruction
    predecessors : labels of blocks that branch to this block (populated by build_cfg)
    successors   : labels of blocks this block branches to   (populated by build_cfg)
    """
    label:        str
    instructions: List[Instruction] = field(default_factory=list)
    predecessors: List[str]         = field(default_factory=list)
    successors:   List[str]         = field(default_factory=list)

    def append(self, instr: Instruction) -> Instruction:
        self.instructions.append(instr)
        return instr

    def terminator(self) -> Optional[Instruction]:
        """Return the last instruction if it is a control-flow terminator."""
        if self.instructions:
            last = self.instructions[-1]
            if last.op in (Op.JUMP, Op.CBRANCH, Op.RET):
                return last
        return None

    def is_terminated(self) -> bool:
        return self.terminator() is not None

    def __iter__(self) -> Iterator[Instruction]:
        return iter(self.instructions)

    def __len__(self) -> int:
        return len(self.instructions)


# ── IR Function ───────────────────────────────────────────────────────────────

@dataclass
class IRFunction:
    """
    Compiled representation of one @kernel function.

    name       : PTX entry point name (same as Python function name)
    params     : ordered list of (name, GPUType) after annotation lowering
                 (ArrayType params are already converted to PointerType here)
    blocks     : list of BasicBlock; blocks[0] is always the entry block
    vreg_count : counter for allocating new VRegs (monotonically increasing)
    """
    name:       str
    params:     List[Tuple[str, GPUType]]
    blocks:     List[BasicBlock]  = field(default_factory=list)
    vreg_count: int               = 0

    # ── Allocation helpers ────────────────────────────────────────────────────

    def new_vreg(self, typ: GPUType, name: Optional[str] = None) -> VReg:
        """Allocate a fresh virtual register."""
        v = VReg(id=self.vreg_count, type=typ, name=name)
        self.vreg_count += 1
        return v

    def new_block(self, label_hint: str = "") -> BasicBlock:
        """Create and register a new BasicBlock."""
        idx = len(self.blocks)
        label = f"BB{idx}_{label_hint}" if label_hint else f"BB{idx}"
        bb = BasicBlock(label=label)
        self.blocks.append(bb)
        return bb

    # ── Accessors ─────────────────────────────────────────────────────────────

    def entry_block(self) -> BasicBlock:
        return self.blocks[0]

    def all_instructions(self) -> Iterator[Instruction]:
        for bb in self.blocks:
            yield from bb.instructions

    def all_vregs(self) -> List[VReg]:
        """Return all VRegs defined in this function (in definition order)."""
        seen: set[int] = set()
        result: List[VReg] = []
        for instr in self.all_instructions():
            if instr.dst is not None and instr.dst.id not in seen:
                seen.add(instr.dst.id)
                result.append(instr.dst)
        return result

    def block_by_label(self, label: str) -> BasicBlock:
        for bb in self.blocks:
            if bb.label == label:
                return bb
        raise KeyError(f"No block with label '{label}' in function '{self.name}'")

    # ── CFG construction ──────────────────────────────────────────────────────

    def build_cfg(self) -> None:
        """
        Populate predecessor / successor lists on each BasicBlock by
        inspecting JUMP and CBRANCH terminators.
        """
        label_to_bb = {bb.label: bb for bb in self.blocks}
        # Clear existing edges
        for bb in self.blocks:
            bb.predecessors.clear()
            bb.successors.clear()
        # Re-populate
        for bb in self.blocks:
            term = bb.terminator()
            if term is None:
                continue
            if term.op == Op.JUMP:
                succ_label = term.meta["label"]
                bb.successors.append(succ_label)
                label_to_bb[succ_label].predecessors.append(bb.label)
            elif term.op == Op.CBRANCH:
                for key in ("true_label", "false_label"):
                    succ_label = term.meta[key]
                    bb.successors.append(succ_label)
                    label_to_bb[succ_label].predecessors.append(bb.label)


# ── IR Module ─────────────────────────────────────────────────────────────────

@dataclass
class IRModule:
    """Container for all compiled @kernel functions."""
    functions: List[IRFunction] = field(default_factory=list)

    def add(self, fn: IRFunction) -> None:
        self.functions.append(fn)

    def get(self, name: str) -> IRFunction:
        for fn in self.functions:
            if fn.name == name:
                return fn
        raise KeyError(f"No function '{name}' in module")

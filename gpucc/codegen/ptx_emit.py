"""
gpucc/codegen/ptx_emit.py — PTX code generator

Walks an IRFunction + RegAlloc and produces a PTX string.

Key design notes:
  - Target: sm_75 by default (T4, used on Colab free tier)
  - Register declarations use the RegAlloc counts
  - Each PTX instruction is annotated with a comment tracing it to the IR
  - The trickiest part is load/store address calculation:
      The index is a u32; the pointer is a u64; PTX requires:
        cvta.to.global.u64 %rdX, %rdPtr     ; generic → global space
        mul.wide.u32       %rdOff, %rIdx, 4  ; u32×u32 → u64 byte offset
        add.u64            %rdAddr, %rdX, %rdOff
        ld.global.f32      %fN, [%rdAddr]
      DO NOT use mul.u64 on a u32 register — that is a type error in PTX.
"""
from __future__ import annotations

from typing import List

from gpucc.codegen.regalloc import RegAlloc, allocate
from gpucc.ir import BasicBlock, Const, IRFunction, Instruction, Op, Operand, VReg
from gpucc.types import (
    BoolType, Float32Type, GPUType, Int32Type, Int64Type, PointerType,
)


# ── PTX type strings for instructions ─────────────────────────────────────────

def _ptx_arith_type(t: GPUType) -> str:
    if isinstance(t, Float32Type): return "f32"
    if isinstance(t, Int32Type):   return "u32"
    if isinstance(t, Int64Type):   return "u64"
    return "u32"

def _ptx_cmp_type(t: GPUType) -> str:
    if isinstance(t, Float32Type): return "f32"
    if isinstance(t, Int32Type):   return "s32"  # signed for comparisons
    if isinstance(t, Int64Type):   return "s64"
    return "s32"


# ── Main emitter ──────────────────────────────────────────────────────────────

class PTXEmitter:
    """
    Emit a complete PTX module (one .visible .entry per IRFunction).

    Usage:
        emitter = PTXEmitter(sm_version="sm_75")
        ptx_str = emitter.emit_function(ir_func)
    """

    def __init__(self, sm_version: str = "sm_75"):
        self.sm_version = sm_version
        self._lines: List[str] = []
        self._alloc: RegAlloc = None  # set during emit_function
        # Counter for anonymous temp registers used in address calc
        self._tmp_rd_id: int = 0
        self._fn: IRFunction = None

    def emit_function(self, fn: IRFunction) -> str:
        """Compile fn to a PTX string."""
        self._fn = fn
        self._alloc = allocate(fn)
        self._lines = []
        # Reserve temp register IDs beyond all VRegs
        self._tmp_rd_id = fn.vreg_count

        self._emit_file_header()
        self._emit_entry_signature(fn)
        self._emit_reg_declarations()
        self._emit_param_loads(fn)
        self._emit_blocks(fn)
        self._emit_line("}")
        return "\n".join(self._lines)

    # ── File header ───────────────────────────────────────────────────────────

    def _emit_file_header(self) -> None:
        self._emit_line(f".version 7.0")
        self._emit_line(f".target {self.sm_version}")
        self._emit_line(f".address_size 64")
        self._emit_line("")

    # ── Entry signature ───────────────────────────────────────────────────────

    def _emit_entry_signature(self, fn: IRFunction) -> None:
        self._emit_line(f".visible .entry {fn.name}(")
        for i, (pname, ptype) in enumerate(fn.params):
            comma = "," if i < len(fn.params) - 1 else ""
            if isinstance(ptype, PointerType):
                self._emit_line(f"    .param .u64 param_{pname}{comma}")
            elif isinstance(ptype, Int32Type):
                self._emit_line(f"    .param .u32 param_{pname}{comma}")
            elif isinstance(ptype, Int64Type):
                self._emit_line(f"    .param .u64 param_{pname}{comma}")
            elif isinstance(ptype, Float32Type):
                self._emit_line(f"    .param .f32 param_{pname}{comma}")
            else:
                self._emit_line(f"    .param .u32 param_{pname}{comma}  // unknown type")
        self._emit_line(") {")

    # ── Register declarations ─────────────────────────────────────────────────

    def _emit_reg_declarations(self) -> None:
        a = self._alloc
        # Extra rd registers needed for address calculation temporaries
        # We add headroom: 3 extra %rd regs per LOAD/STORE instruction
        n_mem_ops = sum(
            1 for instr in self._fn.all_instructions()
            if instr.op in (Op.LOAD, Op.STORE)
        )
        extra_rd = n_mem_ops * 3  # cvta result, offset, address

        counts = {
            "f":  a.declaration_count("f"),
            "r":  a.declaration_count("r"),
            "rd": a.declaration_count("rd") + extra_rd + 2,  # +2 for param loads
            "p":  a.declaration_count("p"),
        }

        if counts["f"]:
            self._emit_line(f"    .reg .f32 %f<{counts['f']}>;")
        if counts["r"]:
            self._emit_line(f"    .reg .u32 %r<{counts['r']}>;")
        if counts["rd"]:
            self._emit_line(f"    .reg .u64 %rd<{counts['rd']}>;")
        if counts["p"]:
            self._emit_line(f"    .reg .pred %p<{counts['p']}>;")
        self._emit_line("")

    # ── Parameter loads ───────────────────────────────────────────────────────

    def _emit_param_loads(self, fn: IRFunction) -> None:
        """
        Emit ld.param instructions for each kernel parameter.
        These replace the PARAM IR instructions; we handle them upfront
        so all parameters are loaded before any other instruction.
        """
        for pname, ptype in fn.params:
            # Find the VReg for this param from the PARAM instructions
            vreg = self._find_param_vreg(fn, pname)
            if vreg is None:
                continue
            reg = self._alloc.ptx_name(vreg)
            if isinstance(ptype, PointerType):
                self._emit_line(
                    f"    ld.param.u64 {reg}, [param_{pname}];  "
                    f"// load param '{pname}' (pointer)"
                )
            elif isinstance(ptype, Int32Type):
                self._emit_line(
                    f"    ld.param.u32 {reg}, [param_{pname}];  "
                    f"// load param '{pname}' (int32)"
                )
            elif isinstance(ptype, Int64Type):
                self._emit_line(
                    f"    ld.param.u64 {reg}, [param_{pname}];  "
                    f"// load param '{pname}' (int64)"
                )
            elif isinstance(ptype, Float32Type):
                self._emit_line(
                    f"    ld.param.f32 {reg}, [param_{pname}];  "
                    f"// load param '{pname}' (float32)"
                )
        self._emit_line("")

    def _find_param_vreg(self, fn: IRFunction, pname: str) -> VReg | None:
        for instr in fn.all_instructions():
            if instr.op == Op.PARAM and instr.meta.get("param_name") == pname:
                return instr.dst
        return None

    # ── Block and instruction emission ────────────────────────────────────────

    def _emit_blocks(self, fn: IRFunction) -> None:
        for i, bb in enumerate(fn.blocks):
            # Emit label for all blocks except the entry (BB0)
            if i > 0:
                self._emit_line(f"{bb.label}:")
            for instr in bb.instructions:
                self._emit_instr(instr)

    def _emit_instr(self, instr: Instruction) -> None:
        op = instr.op

        # PARAM instructions are handled in _emit_param_loads; skip here
        if op == Op.PARAM:
            return

        if op == Op.NOP:
            return

        if op == Op.RET:
            self._emit_line("    ret;")
            return

        if op == Op.JUMP:
            label = instr.meta["label"]
            self._emit_line(f"    bra {label};")
            return

        if op == Op.CBRANCH:
            pred = self._fmt_operand(instr.srcs[0])
            true_lbl  = instr.meta["true_label"]
            false_lbl = instr.meta["false_label"]
            self._emit_line(f"    @{pred} bra {true_lbl};")
            self._emit_line(f"    bra {false_lbl};")
            return

        if op == Op.COPY:
            dst = self._alloc.ptx_name(instr.dst)
            src = self._fmt_operand(instr.srcs[0])
            typ = _ptx_arith_type(instr.type)
            self._emit_line(
                f"    mov.{typ} {dst}, {src};  // {instr.dst}"
            )
            return

        if op in (Op.TID, Op.CTAID, Op.NTID, Op.NCTAID):
            self._emit_thread_intrinsic(instr)
            return

        if op == Op.LOAD:
            self._emit_load(instr)
            return

        if op == Op.STORE:
            self._emit_store(instr)
            return

        if op in (Op.FADD, Op.FSUB, Op.FMUL, Op.FDIV):
            self._emit_float_arith(instr)
            return

        if op in (Op.IADD, Op.ISUB, Op.IMUL, Op.IDIV, Op.IMOD):
            self._emit_int_arith(instr)
            return

        if op == Op.FNEG:
            dst = self._alloc.ptx_name(instr.dst)
            src = self._fmt_operand(instr.srcs[0])
            self._emit_line(f"    neg.f32 {dst}, {src};")
            return

        if op == Op.INEG:
            dst = self._alloc.ptx_name(instr.dst)
            src = self._fmt_operand(instr.srcs[0])
            self._emit_line(f"    neg.s32 {dst}, {src};")
            return

        if op in (Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NE):
            self._emit_compare(instr)
            return

        if op == Op.AND:
            dst = self._alloc.ptx_name(instr.dst)
            a = self._fmt_operand(instr.srcs[0])
            b = self._fmt_operand(instr.srcs[1])
            self._emit_line(f"    and.pred {dst}, {a}, {b};")
            return

        if op == Op.OR:
            dst = self._alloc.ptx_name(instr.dst)
            a = self._fmt_operand(instr.srcs[0])
            b = self._fmt_operand(instr.srcs[1])
            self._emit_line(f"    or.pred {dst}, {a}, {b};")
            return

        if op == Op.NOT:
            dst = self._alloc.ptx_name(instr.dst)
            src = self._fmt_operand(instr.srcs[0])
            self._emit_line(f"    not.pred {dst}, {src};")
            return

        if op == Op.CVTI2F:
            dst = self._alloc.ptx_name(instr.dst)
            src = self._fmt_operand(instr.srcs[0])
            self._emit_line(f"    cvt.rn.f32.s32 {dst}, {src};")
            return

        if op == Op.CVTF2I:
            dst = self._alloc.ptx_name(instr.dst)
            src = self._fmt_operand(instr.srcs[0])
            self._emit_line(f"    cvt.rzi.s32.f32 {dst}, {src};")
            return

        if op == Op.CVTI2I:
            dst = self._alloc.ptx_name(instr.dst)
            src = self._fmt_operand(instr.srcs[0])
            self._emit_line(f"    cvt.u64.u32 {dst}, {src};")
            return

        # Unknown op — emit a comment so the PTX is still parseable
        self._emit_line(f"    // UNIMPLEMENTED: {op}  {instr}")

    # ── Thread intrinsics ─────────────────────────────────────────────────────

    _OP_TO_PTX_SREG = {
        Op.TID:    "tid",
        Op.CTAID:  "ctaid",
        Op.NTID:   "ntid",
        Op.NCTAID: "nctaid",
    }

    def _emit_thread_intrinsic(self, instr: Instruction) -> None:
        dst = self._alloc.ptx_name(instr.dst)
        sreg = self._OP_TO_PTX_SREG[instr.op]
        dim  = instr.meta.get("dim", "x")
        self._emit_line(f"    mov.u32 {dst}, %{sreg}.{dim};  // {instr.dst}")

    # ── Float arithmetic ──────────────────────────────────────────────────────

    _FLOAT_OP = {
        Op.FADD: "add", Op.FSUB: "sub",
        Op.FMUL: "mul", Op.FDIV: "div.rn",
    }

    def _emit_float_arith(self, instr: Instruction) -> None:
        dst = self._alloc.ptx_name(instr.dst)
        a   = self._fmt_operand(instr.srcs[0])
        b   = self._fmt_operand(instr.srcs[1])
        ptx = self._FLOAT_OP[instr.op]
        self._emit_line(f"    {ptx}.f32 {dst}, {a}, {b};  // {instr.dst}")

    # ── Integer arithmetic ────────────────────────────────────────────────────

    _INT_OP = {
        Op.IADD: "add", Op.ISUB: "sub",
        Op.IMUL: "mul.lo", Op.IDIV: "div", Op.IMOD: "rem",
    }

    def _emit_int_arith(self, instr: Instruction) -> None:
        dst  = self._alloc.ptx_name(instr.dst)
        a    = self._fmt_operand(instr.srcs[0])
        b    = self._fmt_operand(instr.srcs[1])
        typ  = _ptx_arith_type(instr.type)
        ptx  = self._INT_OP[instr.op]
        # Use signed type for div/rem
        if instr.op in (Op.IDIV, Op.IMOD):
            typ = "s32"
        self._emit_line(f"    {ptx}.{typ} {dst}, {a}, {b};  // {instr.dst}")

    # ── Comparisons ───────────────────────────────────────────────────────────

    _CMP_PTX = {
        Op.LT: "lt", Op.LE: "le", Op.GT: "gt",
        Op.GE: "ge", Op.EQ: "eq", Op.NE: "ne",
    }

    def _emit_compare(self, instr: Instruction) -> None:
        dst  = self._alloc.ptx_name(instr.dst)
        a    = self._fmt_operand(instr.srcs[0])
        b    = self._fmt_operand(instr.srcs[1])
        cmp  = self._CMP_PTX[instr.op]
        # Determine operand type from source
        src_type = instr.srcs[0].type if instr.srcs else None
        typ  = _ptx_cmp_type(src_type) if src_type else "s32"
        self._emit_line(f"    setp.{cmp}.{typ} {dst}, {a}, {b};  // {instr.dst}")

    # ── Memory: load ──────────────────────────────────────────────────────────

    def _emit_load(self, instr: Instruction) -> None:
        """
        Emit a global memory load.

        IR:  dst = load.global f32  ptr_vreg, index_vreg

        PTX address calculation (3 instructions):
          1. cvta.to.global.u64  %rdGlobal, %rdPtr
             Convert generic pointer → global address space pointer
          2. mul.wide.u32        %rdOffset, %rIndex, elem_bytes
             Scale 32-bit index to 64-bit byte offset.
             IMPORTANT: mul.wide.u32 takes TWO u32 operands and produces u64.
             Do NOT use mul.u64 on a u32 source register.
          3. add.u64             %rdAddr, %rdGlobal, %rdOffset
             Final address = base + offset
          4. ld.global.f32       %fDst, [%rdAddr]
        """
        dst   = self._alloc.ptx_name(instr.dst)
        ptr   = self._fmt_operand(instr.srcs[0])
        index = self._fmt_operand(instr.srcs[1])
        typ   = instr.type
        elem_bytes = typ.byte_size() if typ else 4
        ptx_type   = _ptx_arith_type(typ) if typ else "f32"
        space = instr.meta.get("space", "global")

        # Allocate three temporary %rd registers for address calc
        rd_global = f"%rd{self._alloc_tmp_rd()}"
        rd_offset = f"%rd{self._alloc_tmp_rd()}"
        rd_addr   = f"%rd{self._alloc_tmp_rd()}"

        comment = f"// load {instr.dst} = {instr.srcs[0]}[{instr.srcs[1]}]"
        self._emit_line(f"    cvta.to.{space}.u64 {rd_global}, {ptr};  {comment}")
        self._emit_line(f"    mul.wide.u32 {rd_offset}, {index}, {elem_bytes};")
        self._emit_line(f"    add.u64 {rd_addr}, {rd_global}, {rd_offset};")
        self._emit_line(f"    ld.{space}.{ptx_type} {dst}, [{rd_addr}];")

    # ── Memory: store ─────────────────────────────────────────────────────────

    def _emit_store(self, instr: Instruction) -> None:
        """
        Emit a global memory store.

        IR:  store.global f32  ptr_vreg, index_vreg, value_vreg

        Same 3-step address calc as load, then:
          st.global.f32  [%rdAddr], %fVal
        """
        ptr   = self._fmt_operand(instr.srcs[0])
        index = self._fmt_operand(instr.srcs[1])
        val   = self._fmt_operand(instr.srcs[2])
        typ   = instr.type
        elem_bytes = typ.byte_size() if typ else 4
        ptx_type   = _ptx_arith_type(typ) if typ else "f32"
        space = instr.meta.get("space", "global")

        rd_global = f"%rd{self._alloc_tmp_rd()}"
        rd_offset = f"%rd{self._alloc_tmp_rd()}"
        rd_addr   = f"%rd{self._alloc_tmp_rd()}"

        comment = f"// store {instr.srcs[0]}[{instr.srcs[1]}] = {instr.srcs[2]}"
        self._emit_line(f"    cvta.to.{space}.u64 {rd_global}, {ptr};  {comment}")
        self._emit_line(f"    mul.wide.u32 {rd_offset}, {index}, {elem_bytes};")
        self._emit_line(f"    add.u64 {rd_addr}, {rd_global}, {rd_offset};")
        self._emit_line(f"    st.{space}.{ptx_type} [{rd_addr}], {val};")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _alloc_tmp_rd(self) -> int:
        """Allocate a temporary %rd register id (beyond all VReg ids)."""
        n = self._tmp_rd_id
        self._tmp_rd_id += 1
        return n

    def _fmt_operand(self, op: Operand) -> str:
        """Format an Operand as a PTX token."""
        if isinstance(op, VReg):
            return self._alloc.ptx_name(op)
        if isinstance(op, Const):
            if isinstance(op.type, Float32Type):
                # PTX requires float literals in hex or decimal with explicit type context
                # Using decimal notation here — ptxas accepts it
                return f"{op.value:g}"
            return str(int(op.value))
        raise TypeError(f"Unknown operand type: {type(op)}")

    def _emit_line(self, line: str) -> None:
        self._lines.append(line)

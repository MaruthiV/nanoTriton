"""
gpucc/frontend/ast_to_ir.py — Python AST → IR lowering

ASTToIR walks a Python ast.FunctionDef (from a @kernel-decorated function)
and emits IRFunction with basic blocks, virtual registers, and typed instructions.

Supported constructs:
  - Parameters with type annotations (float32[N], int32, etc.)
  - Assignments (x = expr)
  - Augmented assignments (acc += expr)
  - if / else
  - for i in range(n)  — both constant n and parameter-bound n
  - Binary ops: +, -, *, /
  - Unary ops: -
  - Comparisons: <, <=, >, >=, ==, !=
  - Boolean ops: and, or, not
  - Array subscript load:  a[i]  or  A[row, col]
  - Array subscript store: c[i] = val  or  C[row, col] = val
  - Intrinsic calls:
      thread_id()       → threadIdx.x
      thread_id(0/1/2)  → threadIdx.x/y/z
      block_id(0/1/2)   → blockIdx.x/y/z
      block_size(0/1/2) → blockDim.x/y/z
  - return (implicit or explicit void)
"""
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple, Union

from gpucc.ir import (
    BasicBlock, Const, IRFunction, Instruction, Op, Operand, VReg,
)
from gpucc.types import (
    ArrayType, BoolType, Float32Type, GPUType, Int32Type, Int64Type,
    PointerType, ScalarType, bool_t, f32, i32, i64, parse_annotation,
    pointer_to,
)


_DIM_MAP = {0: "x", 1: "y", 2: "z", "x": "x", "y": "y", "z": "z"}


def _inner_dim_symbol(annotation: ast.expr) -> Optional[str]:
    """
    Extract the inner (last) dimension symbol name from a 2D array annotation.

    float32[M, K]  →  "K"
    float32[K, N]  →  "N"
    float32[N]     →  None  (1D — no stride needed)
    float32[128]   →  None  (constant dim — no symbol)
    """
    if not isinstance(annotation, ast.Subscript):
        return None
    slice_node = annotation.slice
    # 2D: Subscript slice is a Tuple with ≥2 elements
    if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) >= 2:
        last = slice_node.elts[-1]
        if isinstance(last, ast.Name):
            return last.id  # e.g. "K", "N"
    return None


# ── Main visitor ──────────────────────────────────────────────────────────────

class ASTToIR:
    """
    Translates a Python ast.FunctionDef into an IRFunction.

    Usage:
        compiler = ASTToIR()
        ir_fn = compiler.compile(func_def_node)
    """

    def __init__(self):
        self._func: Optional[IRFunction] = None
        self._cur_block: Optional[BasicBlock] = None
        # scope stack: list of {varname: VReg}
        self._scope: List[Dict[str, VReg]] = []
        # stack of (header_label, exit_label) for for-loop break/continue
        self._loop_stack: List[Tuple[str, str]] = []
        # Maps array param name → stride param name derived from annotation symbols.
        # e.g. "A" → "k"  (from float32[M, K], inner dim K → param "k")
        #      "B" → "n"  (from float32[K, N], inner dim N → param "n")
        #      "C" → "n"  (from float32[M, N], inner dim N → param "n")
        self._array_strides: Dict[str, str] = {}

    # ── Public entry point ────────────────────────────────────────────────────

    def compile(self, node: ast.FunctionDef) -> IRFunction:
        """Compile a single @kernel function definition."""
        self._func = None
        self._cur_block = None
        self._scope = [{}]
        self._loop_stack = []
        self._array_strides = {}
        return self._visit_FunctionDef(node)

    # ── Function ─────────────────────────────────────────────────────────────

    def _visit_FunctionDef(self, node: ast.FunctionDef) -> IRFunction:
        params = self._lower_params(node.args)
        self._func = IRFunction(name=node.name, params=params)

        # Create entry block and switch to it
        entry = self._func.new_block("entry")
        self._set_block(entry)

        # Emit PARAM instructions and bind parameter names to VRegs
        for idx, (pname, ptype) in enumerate(params):
            vreg = self._func.new_vreg(ptype, name=pname)
            self._emit(Instruction(
                op=Op.PARAM, dst=vreg, srcs=[], type=ptype,
                meta={"param_name": pname, "param_index": idx},
            ))
            self._define(pname, vreg)

        # Lower body statements
        for stmt in node.body:
            self._visit_stmt(stmt)

        # Ensure entry block is terminated
        if not self._cur_block.is_terminated():
            self._emit(Instruction(op=Op.RET, dst=None, srcs=[], type=None))

        self._func.build_cfg()
        return self._func

    def _lower_params(self, args: ast.arguments) -> List[Tuple[str, GPUType]]:
        """
        Parse parameter annotations and lower ArrayType → PointerType.
        Also populates self._array_strides for 2D array params by reading the
        inner dimension symbol name directly from the annotation AST.

        Example: B: float32[K, N]
          - annotation AST: Subscript(Name("float32"), Tuple([Name("K"), Name("N")]))
          - inner dim symbol: "N" → lowercase "n" → stride param name "n"
          - self._array_strides["B"] = "n"
        """
        result: List[Tuple[str, GPUType]] = []
        for arg in args.args:
            name = arg.arg
            if arg.annotation is None:
                raise TypeError(
                    f"Kernel parameter '{name}' has no type annotation. "
                    "All @kernel parameters must be annotated."
                )
            raw_type = parse_annotation(arg.annotation)
            # Lower ArrayType to PointerType (arrays are passed as raw pointers)
            if isinstance(raw_type, ArrayType):
                lowered = pointer_to(raw_type.element_type)
                # Record the inner dimension's stride param name from annotation.
                inner_sym = _inner_dim_symbol(arg.annotation)
                if inner_sym is not None:
                    self._array_strides[name] = inner_sym.lower()
            else:
                lowered = raw_type
            result.append((name, lowered))
        return result

    # ── Statement dispatch ────────────────────────────────────────────────────

    def _visit_stmt(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign):
            self._visit_Assign(node)
        elif isinstance(node, ast.AugAssign):
            self._visit_AugAssign(node)
        elif isinstance(node, ast.If):
            self._visit_If(node)
        elif isinstance(node, ast.For):
            self._visit_For(node)
        elif isinstance(node, ast.Return):
            self._visit_Return(node)
        elif isinstance(node, ast.Expr):
            # Expression statement (e.g., a bare function call)
            self._visit_expr(node.value)
        elif isinstance(node, ast.Pass):
            pass  # no-op
        else:
            raise NotImplementedError(
                f"Unsupported statement: {type(node).__name__} at line {getattr(node, 'lineno', '?')}"
            )

    # ── Assignment ────────────────────────────────────────────────────────────

    def _visit_Assign(self, node: ast.Assign) -> None:
        val = self._visit_expr(node.value)

        for target in node.targets:
            if isinstance(target, ast.Name):
                # Simple scalar assignment: x = expr
                name = target.id
                existing = self._lookup_opt(name)
                if existing is not None:
                    # Overwrite via COPY
                    self._emit(Instruction(
                        op=Op.COPY, dst=existing, srcs=[val], type=existing.type,
                    ))
                else:
                    typ = val.type if isinstance(val, VReg) else val.type
                    vreg = self._func.new_vreg(typ, name=name)
                    self._emit(Instruction(
                        op=Op.COPY, dst=vreg, srcs=[val], type=typ,
                    ))
                    self._define(name, vreg)

            elif isinstance(target, ast.Subscript):
                # Array store: c[i] = val  or  C[row, col] = val
                self._emit_store(target, val)
            else:
                raise NotImplementedError(
                    f"Unsupported assignment target: {type(target).__name__}"
                )

    def _visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Lower  acc += expr  →  tmp = fadd acc, expr; copy acc = tmp"""
        right = self._visit_expr(node.value)

        if isinstance(node.target, ast.Name):
            name = node.target.id
            left = self._lookup(name)
            result = self._emit_binop(node.op, left, right)
            self._emit(Instruction(
                op=Op.COPY, dst=left, srcs=[result], type=left.type,
            ))
        elif isinstance(node.target, ast.Subscript):
            # a[i] += expr  →  tmp = load; tmp2 = op(tmp, expr); store
            load_val = self._emit_load(node.target)
            result = self._emit_binop(node.op, load_val, right)
            self._emit_store(node.target, result)
        else:
            raise NotImplementedError(
                f"Unsupported augmented assign target: {type(node.target).__name__}"
            )

    # ── If / else ─────────────────────────────────────────────────────────────

    def _visit_If(self, node: ast.If) -> None:
        pred = self._visit_expr(node.test)
        pred = self._coerce_to_bool(pred)

        fn = self._func
        then_bb   = fn.new_block("then")
        else_bb   = fn.new_block("else") if node.orelse else None
        merge_bb  = fn.new_block("merge")

        false_label = else_bb.label if else_bb else merge_bb.label

        self._emit(Instruction(
            op=Op.CBRANCH, dst=None, srcs=[pred], type=None,
            meta={"true_label": then_bb.label, "false_label": false_label},
        ))

        # Then branch
        self._set_block(then_bb)
        for stmt in node.body:
            self._visit_stmt(stmt)
        if not self._cur_block.is_terminated():
            self._emit(Instruction(op=Op.JUMP, dst=None, srcs=[], type=None,
                                   meta={"label": merge_bb.label}))

        # Else branch (optional)
        if else_bb is not None:
            self._set_block(else_bb)
            for stmt in node.orelse:
                self._visit_stmt(stmt)
            if not self._cur_block.is_terminated():
                self._emit(Instruction(op=Op.JUMP, dst=None, srcs=[], type=None,
                                       meta={"label": merge_bb.label}))

        self._set_block(merge_bb)

    # ── For loop ──────────────────────────────────────────────────────────────

    def _visit_For(self, node: ast.For) -> None:
        """
        Lower  for i in range(n)  to:
          - preheader: initialize counter
          - header: compare counter < bound; CBRANCH → body | exit
          - body: loop body
          - latch: counter += 1; JUMP → header
          - exit: continues

        The bound n may be a Const (constant) or a VReg (kernel parameter).
        Loops with Const bounds are tagged unrollable=True.
        """
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("Only simple for-loop targets supported (no tuple unpacking)")
        if not (isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"):
            raise NotImplementedError("Only for i in range(...) loops are supported")

        loop_var = node.target.id
        range_args = node.iter.args

        # Determine start, stop, step
        if len(range_args) == 1:
            start_op: Operand = Const(0, i32)
            stop_op  = self._visit_expr(range_args[0])
            step_op: Operand  = Const(1, i32)
        elif len(range_args) == 2:
            start_op = self._visit_expr(range_args[0])
            stop_op  = self._visit_expr(range_args[1])
            step_op  = Const(1, i32)
        elif len(range_args) == 3:
            start_op = self._visit_expr(range_args[0])
            stop_op  = self._visit_expr(range_args[1])
            step_op  = self._visit_expr(range_args[2])
        else:
            raise ValueError(f"range() takes 1–3 arguments, got {len(range_args)}")

        fn = self._func
        header_bb = fn.new_block("loop_header")
        body_bb   = fn.new_block("loop_body")
        latch_bb  = fn.new_block("loop_latch")
        exit_bb   = fn.new_block("loop_exit")

        # Is the bound a compile-time constant? → loop is unrollable
        is_const_bound = isinstance(stop_op, Const) and isinstance(start_op, Const)

        # ── Preheader: initialize counter ────────────────────────────────────
        counter = fn.new_vreg(i32, name=loop_var)
        self._emit(Instruction(op=Op.COPY, dst=counter, srcs=[start_op], type=i32))
        self._define(loop_var, counter)
        self._emit(Instruction(op=Op.JUMP, dst=None, srcs=[], type=None,
                               meta={"label": header_bb.label}))

        # ── Header: check counter < stop ─────────────────────────────────────
        self._set_block(header_bb)
        cond = fn.new_vreg(bool_t)
        self._emit(Instruction(
            op=Op.LT, dst=cond, srcs=[counter, stop_op], type=bool_t,
        ))
        self._emit(Instruction(
            op=Op.CBRANCH, dst=None, srcs=[cond], type=None,
            meta={
                "true_label":  body_bb.label,
                "false_label": exit_bb.label,
                "unrollable":  is_const_bound,
            },
        ))

        # ── Body ─────────────────────────────────────────────────────────────
        self._set_block(body_bb)
        self._loop_stack.append((header_bb.label, exit_bb.label))
        self._push_scope()
        for stmt in node.body:
            self._visit_stmt(stmt)
        self._pop_scope()
        self._loop_stack.pop()
        if not self._cur_block.is_terminated():
            self._emit(Instruction(op=Op.JUMP, dst=None, srcs=[], type=None,
                                   meta={"label": latch_bb.label}))

        # ── Latch: increment counter ──────────────────────────────────────────
        self._set_block(latch_bb)
        self._emit(Instruction(
            op=Op.IADD, dst=counter, srcs=[counter, step_op], type=i32,
        ))
        self._emit(Instruction(op=Op.JUMP, dst=None, srcs=[], type=None,
                               meta={"label": header_bb.label}))

        # ── Continue in exit block ────────────────────────────────────────────
        self._set_block(exit_bb)

    # ── Return ────────────────────────────────────────────────────────────────

    def _visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            raise NotImplementedError("Kernel functions must return void (no return value)")
        self._emit(Instruction(op=Op.RET, dst=None, srcs=[], type=None))

    # ── Expression dispatch ───────────────────────────────────────────────────

    def _visit_expr(self, node: ast.expr) -> Operand:
        if isinstance(node, ast.BinOp):
            return self._visit_BinOp(node)
        if isinstance(node, ast.UnaryOp):
            return self._visit_UnaryOp(node)
        if isinstance(node, ast.Compare):
            return self._visit_Compare(node)
        if isinstance(node, ast.BoolOp):
            return self._visit_BoolOp(node)
        if isinstance(node, ast.Call):
            return self._visit_Call(node)
        if isinstance(node, ast.Subscript):
            return self._emit_load(node)
        if isinstance(node, ast.Name):
            return self._visit_Name(node)
        if isinstance(node, ast.Constant):
            return self._visit_Constant(node)
        raise NotImplementedError(
            f"Unsupported expression: {type(node).__name__} at line {getattr(node, 'lineno', '?')}"
        )

    # ── Binary operations ─────────────────────────────────────────────────────

    def _visit_BinOp(self, node: ast.BinOp) -> Operand:
        left  = self._visit_expr(node.left)
        right = self._visit_expr(node.right)
        return self._emit_binop(node.op, left, right)

    def _emit_binop(self, ast_op: ast.operator, left: Operand, right: Operand) -> VReg:
        ltype = left.type if isinstance(left, (VReg, Const)) else None
        rtype = right.type if isinstance(right, (VReg, Const)) else None
        result_type = self._infer_binop_type(ltype, rtype)

        # Coerce operands to result type if needed
        left  = self._coerce(left, result_type)
        right = self._coerce(right, result_type)

        op_map_float = {
            ast.Add: Op.FADD, ast.Sub: Op.FSUB,
            ast.Mult: Op.FMUL, ast.Div: Op.FDIV,
        }
        op_map_int = {
            ast.Add: Op.IADD, ast.Sub: Op.ISUB,
            ast.Mult: Op.IMUL, ast.Div: Op.IDIV,
            ast.Mod: Op.IMOD,
        }

        if isinstance(result_type, Float32Type):
            ir_op = op_map_float.get(type(ast_op))
        else:
            ir_op = op_map_int.get(type(ast_op))

        if ir_op is None:
            raise NotImplementedError(
                f"Unsupported binary operator {type(ast_op).__name__} for type {result_type}"
            )

        dst = self._func.new_vreg(result_type)
        self._emit(Instruction(op=ir_op, dst=dst, srcs=[left, right], type=result_type))
        return dst

    def _infer_binop_type(self, ltype: Optional[GPUType], rtype: Optional[GPUType]) -> GPUType:
        """Float32 dominates int32 in arithmetic; otherwise use ltype."""
        if isinstance(ltype, Float32Type) or isinstance(rtype, Float32Type):
            return f32
        if isinstance(ltype, Int64Type) or isinstance(rtype, Int64Type):
            return i64
        return i32

    # ── Unary operations ──────────────────────────────────────────────────────

    def _visit_UnaryOp(self, node: ast.UnaryOp) -> Operand:
        operand = self._visit_expr(node.operand)
        if isinstance(node.op, ast.USub):
            typ = operand.type if isinstance(operand, (VReg, Const)) else i32
            ir_op = Op.FNEG if isinstance(typ, Float32Type) else Op.INEG
            dst = self._func.new_vreg(typ)
            self._emit(Instruction(op=ir_op, dst=dst, srcs=[operand], type=typ))
            return dst
        if isinstance(node.op, ast.Not):
            pred = self._coerce_to_bool(operand)
            dst = self._func.new_vreg(bool_t)
            self._emit(Instruction(op=Op.NOT, dst=dst, srcs=[pred], type=bool_t))
            return dst
        raise NotImplementedError(f"Unsupported unary op: {type(node.op).__name__}")

    # ── Comparisons ───────────────────────────────────────────────────────────

    _CMP_MAP = {
        ast.Lt: Op.LT, ast.LtE: Op.LE, ast.Gt: Op.GT,
        ast.GtE: Op.GE, ast.Eq: Op.EQ, ast.NotEq: Op.NE,
    }

    def _visit_Compare(self, node: ast.Compare) -> Operand:
        if len(node.ops) != 1:
            raise NotImplementedError("Chained comparisons (a < b < c) not supported")
        left   = self._visit_expr(node.left)
        right  = self._visit_expr(node.comparators[0])
        ir_op  = self._CMP_MAP.get(type(node.ops[0]))
        if ir_op is None:
            raise NotImplementedError(f"Unsupported comparison: {type(node.ops[0]).__name__}")

        # Coerce operands to same type
        cmp_type = self._infer_binop_type(
            left.type if isinstance(left, (VReg, Const)) else None,
            right.type if isinstance(right, (VReg, Const)) else None,
        )
        left  = self._coerce(left, cmp_type)
        right = self._coerce(right, cmp_type)

        dst = self._func.new_vreg(bool_t)
        self._emit(Instruction(op=ir_op, dst=dst, srcs=[left, right], type=bool_t))
        return dst

    # ── Boolean operations ────────────────────────────────────────────────────

    def _visit_BoolOp(self, node: ast.BoolOp) -> Operand:
        ir_op = Op.AND if isinstance(node.op, ast.And) else Op.OR
        result = self._coerce_to_bool(self._visit_expr(node.values[0]))
        for val_node in node.values[1:]:
            rhs = self._coerce_to_bool(self._visit_expr(val_node))
            dst = self._func.new_vreg(bool_t)
            self._emit(Instruction(op=ir_op, dst=dst, srcs=[result, rhs], type=bool_t))
            result = dst
        return result

    # ── Intrinsic calls ───────────────────────────────────────────────────────

    _INTRINSIC_MAP = {
        "thread_id": Op.TID,
        "block_id":  Op.CTAID,
        "block_size": Op.NTID,
        "grid_size": Op.NCTAID,
    }

    def _visit_Call(self, node: ast.Call) -> Operand:
        if not isinstance(node.func, ast.Name):
            raise NotImplementedError("Only simple function calls (intrinsics) are supported")
        fname = node.func.id

        if fname not in self._INTRINSIC_MAP:
            raise NotImplementedError(
                f"Unknown function '{fname}'. "
                "Supported intrinsics: thread_id, block_id, block_size, grid_size"
            )

        ir_op = self._INTRINSIC_MAP[fname]

        # Dimension argument (0/1/2 → x/y/z), default 0 → x
        if node.args:
            dim_node = node.args[0]
            if not isinstance(dim_node, ast.Constant):
                raise NotImplementedError("Intrinsic dimension must be a constant 0, 1, or 2")
            dim = _DIM_MAP.get(dim_node.value)
            if dim is None:
                raise ValueError(f"Invalid dimension {dim_node.value!r}, expected 0, 1, or 2")
        else:
            dim = "x"

        dst = self._func.new_vreg(i32)
        self._emit(Instruction(op=ir_op, dst=dst, srcs=[], type=i32, meta={"dim": dim}))
        return dst

    # ── Memory operations ─────────────────────────────────────────────────────

    def _emit_load(self, node: ast.Subscript) -> VReg:
        """Lower  a[i]  or  A[row, col]  to a LOAD instruction."""
        ptr, indices = self._subscript_ptr_and_indices(node)
        index = self._flatten_index(ptr, indices)

        ptr_type = ptr.type if isinstance(ptr, VReg) else None
        if not isinstance(ptr_type, PointerType):
            raise TypeError(f"Cannot index non-pointer type {ptr_type!r}")

        elem_type = ptr_type.element_type
        dst = self._func.new_vreg(elem_type)
        self._emit(Instruction(
            op=Op.LOAD, dst=dst, srcs=[ptr, index], type=elem_type,
            meta={"space": "global"},
        ))
        return dst

    def _emit_store(self, node: ast.Subscript, value: Operand) -> None:
        """Lower  c[i] = value  or  C[row, col] = value  to a STORE instruction."""
        ptr, indices = self._subscript_ptr_and_indices(node)
        index = self._flatten_index(ptr, indices)

        ptr_type = ptr.type if isinstance(ptr, VReg) else None
        if not isinstance(ptr_type, PointerType):
            raise TypeError(f"Cannot index non-pointer type {ptr_type!r}")

        elem_type = ptr_type.element_type
        value = self._coerce(value, elem_type)
        self._emit(Instruction(
            op=Op.STORE, dst=None, srcs=[ptr, index, value], type=elem_type,
            meta={"space": "global"},
        ))

    def _subscript_ptr_and_indices(
        self, node: ast.Subscript
    ) -> Tuple[Operand, List[Operand]]:
        """
        Extract the pointer VReg and list of index operands from a Subscript node.
        Handles both 1D (a[i]) and 2D (A[row, col]) access.
        """
        ptr = self._visit_expr(node.value)
        slice_node = node.slice
        if isinstance(slice_node, ast.Tuple):
            indices = [self._visit_expr(elt) for elt in slice_node.elts]
        else:
            indices = [self._visit_expr(slice_node)]
        return ptr, indices

    def _flatten_index(self, ptr: Operand, indices: List[Operand]) -> Operand:
        """
        For 1D: return index as-is (coerced to i32).
        For 2D: compute row * stride + col — requires stride info.

        NOTE: For the MVP we do not track static array shapes, so 2D access
        uses the caller-passed stride parameter (must be named '<array>_stride'
        or inferred). For now we just linearize 2D access naively as
        row * <stride_vreg> + col where stride_vreg must be in scope.

        TODO: pass array shapes through to enable proper 2D stride calculation.
        """
        if len(indices) == 1:
            return self._coerce(indices[0], i32)

        if len(indices) == 2:
            row = self._coerce(indices[0], i32)
            col = self._coerce(indices[1], i32)

            # Try to find a stride parameter in scope
            # Convention: for array X[M, K], stride param is named 'k' (inner dim)
            # For now, emit a stub multiply — the user must pass the stride as a param
            # named '{array_name}_stride' or we fall back to using the loop-bound variable.
            # We'll use a IMUL + IADD pattern.
            if not isinstance(ptr, VReg):
                raise TypeError("2D indexing requires a pointer VReg")

            # Look for a stride variable: prefer a param matching inner dimension name
            stride = self._find_stride_for(ptr)
            tmp = self._func.new_vreg(i32)
            self._emit(Instruction(op=Op.IMUL, dst=tmp, srcs=[row, stride], type=i32))
            flat = self._func.new_vreg(i32)
            self._emit(Instruction(op=Op.IADD, dst=flat, srcs=[tmp, col], type=i32))
            return flat

        raise NotImplementedError(f"Indexing with {len(indices)} dimensions not supported")

    def _find_stride_for(self, ptr: VReg) -> Operand:
        """
        Find the stride VReg for a 2D array access.

        Priority:
          1. Use self._array_strides[ptr.name] — populated from the annotation AST.
             e.g. B: float32[K, N] → inner dim "N" → stride param "n"
          2. Explicit <array>_stride parameter convention.
          3. Fallback candidates: k, n, cols, stride.
        """
        # 1. Annotation-derived stride (most accurate)
        if ptr.name and ptr.name in self._array_strides:
            stride_name = self._array_strides[ptr.name]
            vreg = self._lookup_opt(stride_name)
            if vreg is not None:
                return vreg

        # 2. Explicit <array>_stride parameter
        if ptr.name:
            vreg = self._lookup_opt(f"{ptr.name.lower()}_stride")
            if vreg is not None:
                return vreg

        # 3. Fallback candidates
        for name in ["k", "n", "cols", "stride"]:
            vreg = self._lookup_opt(name)
            if vreg is not None:
                return vreg

        raise TypeError(
            f"Cannot determine stride for 2D array access on '{ptr.name}'. "
            "Pass the inner dimension as a parameter (e.g. 'k' for [M,K], 'n' for [K,N])."
        )

    # ── Name / constant ───────────────────────────────────────────────────────

    def _visit_Name(self, node: ast.Name) -> Operand:
        return self._lookup(node.id)

    def _visit_Constant(self, node: ast.Constant) -> Const:
        val = node.value
        if isinstance(val, float):
            return Const(val, f32)
        if isinstance(val, int):
            return Const(val, i32)
        if isinstance(val, bool):
            return Const(int(val), bool_t)
        raise NotImplementedError(f"Unsupported constant type: {type(val).__name__}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _emit(self, instr: Instruction) -> Instruction:
        self._cur_block.append(instr)
        return instr

    def _set_block(self, bb: BasicBlock) -> None:
        self._cur_block = bb

    def _push_scope(self) -> None:
        self._scope.append({})

    def _pop_scope(self) -> None:
        self._scope.pop()

    def _define(self, name: str, vreg: VReg) -> None:
        """Bind name to vreg in the current (innermost) scope."""
        self._scope[-1][name] = vreg

    def _lookup(self, name: str) -> VReg:
        """Look up name through the scope stack (innermost first)."""
        for scope in reversed(self._scope):
            if name in scope:
                return scope[name]
        raise NameError(f"Undefined variable '{name}'")

    def _lookup_opt(self, name: str) -> Optional[VReg]:
        """Like _lookup but returns None if not found."""
        for scope in reversed(self._scope):
            if name in scope:
                return scope[name]
        return None

    def _coerce_to_bool(self, operand: Operand) -> VReg:
        """
        Ensure operand is a BoolType VReg.
        If it's already bool, return as-is.
        If it's int or float, emit  ne operand, 0.
        """
        if isinstance(operand, VReg) and isinstance(operand.type, BoolType):
            return operand
        # Emit   operand != 0
        zero: Const
        if isinstance(operand, VReg):
            if isinstance(operand.type, Float32Type):
                zero = Const(0.0, f32)
            else:
                zero = Const(0, i32)
        else:
            zero = Const(0, i32)
        dst = self._func.new_vreg(bool_t)
        self._emit(Instruction(op=Op.NE, dst=dst, srcs=[operand, zero], type=bool_t))
        return dst

    def _coerce(self, operand: Operand, target_type: GPUType) -> Operand:
        """
        Emit a type conversion if operand's type doesn't match target_type.
        For constants: just re-wrap with new type (no instruction needed).
        """
        src_type = operand.type

        if src_type == target_type:
            return operand

        # Constant coercion: no instruction, just re-wrap
        if isinstance(operand, Const):
            if isinstance(target_type, Float32Type):
                return Const(float(operand.value), target_type)
            if isinstance(target_type, (Int32Type, Int64Type)):
                return Const(int(operand.value), target_type)
            return operand

        # VReg coercion
        if isinstance(src_type, (Int32Type, Int64Type)) and isinstance(target_type, Float32Type):
            dst = self._func.new_vreg(target_type)
            self._emit(Instruction(op=Op.CVTI2F, dst=dst, srcs=[operand], type=target_type))
            return dst
        if isinstance(src_type, Float32Type) and isinstance(target_type, (Int32Type, Int64Type)):
            dst = self._func.new_vreg(target_type)
            self._emit(Instruction(op=Op.CVTF2I, dst=dst, srcs=[operand], type=target_type))
            return dst
        if isinstance(src_type, Int32Type) and isinstance(target_type, Int64Type):
            dst = self._func.new_vreg(target_type)
            self._emit(Instruction(op=Op.CVTI2I, dst=dst, srcs=[operand], type=target_type))
            return dst

        # Types already match or coercion not needed
        return operand

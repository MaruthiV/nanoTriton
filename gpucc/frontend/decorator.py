"""
gpucc/frontend/decorator.py — @kernel decorator

Usage:
    from gpucc import kernel, float32, int32

    @kernel
    def vector_add(a: float32[N], b: float32[N], c: float32[N], n: int32):
        tid = thread_id()
        if tid < n:
            c[tid] = a[tid] + b[tid]

    ptx_code = vector_add.ptx          # compile and return PTX string
    ir        = vector_add.ir          # inspect the IRFunction
    print(vector_add.ir_text)          # pretty-print the IR
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Optional

from gpucc.frontend.ast_to_ir import ASTToIR
from gpucc.ir import IRFunction


def kernel(fn) -> "KernelHandle":
    """
    Decorator that captures a Python function's source code, parses it,
    and compiles it to an IRFunction.  Does NOT execute the Python function.
    Returns a KernelHandle with .ptx, .ir, .ir_text properties.
    """
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    func_def = _extract_func_def(tree)
    compiler = ASTToIR()
    ir_func = compiler.compile(func_def)
    return KernelHandle(ir_func)


def _extract_func_def(tree: ast.Module) -> ast.FunctionDef:
    """Find the first FunctionDef in the parsed module."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError("No function definition found in kernel source")


class KernelHandle:
    """
    Wraps a compiled IRFunction and lazily produces PTX on demand.

    Attributes:
        name    : kernel entry point name
        ir      : IRFunction (the intermediate representation)
        ir_text : human-readable IR dump (for debugging)
        ptx     : compiled PTX string (cached after first call)
    """

    def __init__(self, ir_func: IRFunction):
        self._ir = ir_func
        self._ptx_cache: dict[tuple, str] = {}  # (opt_level, sm_version) → ptx

    @property
    def name(self) -> str:
        return self._ir.name

    @property
    def ir(self) -> IRFunction:
        return self._ir

    @property
    def ir_text(self) -> str:
        from gpucc.ir_printer import print_function
        return print_function(self._ir)

    def compile(self, opt_level: int = 2, sm_version: str = "sm_75") -> str:
        """
        Run optimization passes then PTX codegen.  Results are cached by (opt_level, sm_version).

        opt_level : 0 = no opts, 1 = fold+DCE, 2 = fold+DCE+unroll, 3 = all
        sm_version: PTX target architecture (default sm_75 = T4, Colab)
        """
        key = (opt_level, sm_version)
        if key not in self._ptx_cache:
            from gpucc.transforms import run_passes
            from gpucc.codegen.ptx_emit import PTXEmitter

            optimized = run_passes(self._ir, opt_level)
            emitter = PTXEmitter(sm_version=sm_version)
            self._ptx_cache[key] = emitter.emit_function(optimized)
        return self._ptx_cache[key]

    @property
    def ptx(self) -> str:
        """Compile with default settings and return PTX string."""
        return self.compile()

    def __repr__(self) -> str:
        return f"<KernelHandle '{self.name}'>"

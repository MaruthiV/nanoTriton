"""
gpucc — Python-to-GPU Kernel Compiler

A mini-Triton built from scratch: compile Python-like syntax to PTX.

Quick start:
    from gpucc import kernel
    from gpucc.types import f32, i32

    @kernel
    def vector_add(a: float32[N], b: float32[N], c: float32[N], n: int32):
        tid = thread_id()
        if tid < n:
            c[tid] = a[tid] + b[tid]

    print(vector_add.ir_text)   # inspect IR
    print(vector_add.ptx)       # compile → PTX string
"""
from gpucc.frontend.decorator import kernel, KernelHandle
from gpucc.annotations import float32, int32, int64, N, M, K
from gpucc.types import (
    GPUType, ScalarType, Float32Type, Int32Type, Int64Type, BoolType,
    PointerType, ArrayType,
    f32, i32, i64, bool_t, pointer_to, parse_annotation,
)

__all__ = [
    "kernel",
    "KernelHandle",
    # Annotation sentinels (for use in @kernel function signatures)
    "float32", "int32", "int64",
    "N", "M", "K",
    "GPUType",
    "ScalarType",
    "Float32Type",
    "Int32Type",
    "Int64Type",
    "BoolType",
    "PointerType",
    "ArrayType",
    "f32",
    "i32",
    "i64",
    "bool_t",
    "pointer_to",
    "parse_annotation",
]

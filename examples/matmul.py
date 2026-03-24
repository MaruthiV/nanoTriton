"""
examples/matmul.py — Matrix multiplication kernel

Each thread computes one element of C = A @ B.
A is [m, k], B is [k, n], C is [m, n].

The inner dimension 'k' must be passed as a parameter named 'k' so that
the compiler can find the stride for 2D array indexing.

Note: this is a naive matmul (no tiling, no shared memory).
Each thread does a full dot product over k. For actual performance,
you'd want a tiled implementation with shared memory (Phase 2).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpucc import kernel, float32, int32, N, M, K


@kernel
def matmul(A: float32[M, K], B: float32[K, N], C: float32[M, N],
           m: int32, k: int32, n: int32):
    row = block_id(0) * block_size(0) + thread_id(0)
    col = block_id(1) * block_size(1) + thread_id(1)
    acc = 0.0
    for ki in range(k):
        acc = acc + A[row, ki] * B[ki, col]
    C[row, col] = acc


if __name__ == "__main__":
    print("=" * 60)
    print("IR:")
    print("=" * 60)
    print(matmul.ir_text)

    print("=" * 60)
    print("PTX (opt_level=2):")
    print("=" * 60)
    print(matmul.compile(opt_level=2))

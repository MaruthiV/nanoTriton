"""
examples/vector_add.py — Vector addition kernel

Compile and inspect the vector_add kernel locally (no GPU needed).
To actually run it on a GPU, use notebooks/gpu_validation.ipynb on Colab.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpucc import kernel, float32, int32, N


@kernel
def vector_add(a: float32[N], b: float32[N], c: float32[N], n: int32):
    tid = thread_id()
    if tid < n:
        c[tid] = a[tid] + b[tid]


if __name__ == "__main__":
    print("=" * 60)
    print("IR:")
    print("=" * 60)
    print(vector_add.ir_text)

    print("=" * 60)
    print("PTX (opt_level=2):")
    print("=" * 60)
    print(vector_add.ptx)

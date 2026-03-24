"""
gpucc/runtime/cupy_runner.py — CuPy-based GPU kernel runner

Loads a compiled PTX string via CuPy's RawModule and wraps the kernel
launch interface.

Usage (on Colab with GPU):
    from gpucc import kernel
    from gpucc.runtime.cupy_runner import GPURunner
    import cupy as cp
    import numpy as np

    @kernel
    def vector_add(a: float32[N], b: float32[N], c: float32[N], n: int32):
        tid = thread_id()
        if tid < n:
            c[tid] = a[tid] + b[tid]

    ptx = vector_add.ptx
    runner = GPURunner(ptx, "vector_add")

    N = 1024
    a_gpu = cp.array(np.random.rand(N).astype(np.float32))
    b_gpu = cp.array(np.random.rand(N).astype(np.float32))
    c_gpu = cp.zeros(N, dtype=np.float32)

    block_size = 256
    grid_size  = (N + block_size - 1) // block_size
    runner(a_gpu, b_gpu, c_gpu, np.int32(N), grid=(grid_size,), block=(block_size,))

    np.testing.assert_allclose(c_gpu.get(), a_gpu.get() + b_gpu.get(), rtol=1e-5)
"""
from __future__ import annotations

from typing import Any, Tuple


class GPURunner:
    """
    Thin wrapper around a CuPy RawModule for launching compiled PTX kernels.

    Parameters:
        ptx       : PTX source string (output from PTXEmitter.emit_function)
        fn_name   : name of the .visible .entry in the PTX
        backend   : 'ptx' (default) or 'cubin'
    """

    def __init__(self, ptx: str, fn_name: str):
        try:
            import cupy as cp
        except ImportError:
            raise ImportError(
                "CuPy is required to run GPU kernels. "
                "Install via: pip install cupy-cuda12x  (adjust for your CUDA version)"
            )

        import os
        import tempfile

        # Write PTX to a temp file and load via path — CuPy's path loader calls
        # cuModuleLoad which handles PTX natively (no backend= needed).
        tmp = tempfile.NamedTemporaryFile(suffix='.ptx', mode='w', delete=False)
        tmp.write(ptx)
        tmp.flush()
        tmp.close()
        self._ptx_path = tmp.name

        self._mod = cp.RawModule(path=self._ptx_path)
        self._fn  = self._mod.get_function(fn_name)
        self._fn_name = fn_name

    def __del__(self):
        import os
        try:
            os.unlink(self._ptx_path)
        except Exception:
            pass

    def __call__(
        self,
        *args: Any,
        grid: Tuple[int, ...],
        block: Tuple[int, ...],
    ) -> None:
        """
        Launch the kernel.

        args  : kernel arguments (CuPy arrays or scalars)
        grid  : grid dimensions  e.g. (num_blocks,) or (bx, by)
        block : block dimensions e.g. (256,) or (16, 16)
        """
        self._fn(grid=grid, block=block, args=args)

    def __repr__(self) -> str:
        return f"<GPURunner '{self._fn_name}'>"

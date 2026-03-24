"""
gpucc/runtime/cupy_runner.py — CuPy-based GPU kernel runner

Compiles PTX → cubin via ptxas, then loads via CuPy RawModule.

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

    runner(a_gpu, b_gpu, c_gpu, np.int32(N), grid=(4,), block=(256,))
    np.testing.assert_allclose(c_gpu.get(), a_gpu.get() + b_gpu.get(), rtol=1e-5)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, Tuple


def _find_ptxas() -> str:
    """Return path to ptxas, checking common CUDA install locations."""
    for candidate in ('ptxas', '/usr/local/cuda/bin/ptxas'):
        try:
            subprocess.run([candidate, '--version'], capture_output=True, check=True)
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError(
        "ptxas not found. Make sure the CUDA toolkit is installed "
        "(on Colab it lives at /usr/local/cuda/bin/ptxas)."
    )


def _ptx_to_cubin(ptx: str, sm_version: str) -> str:
    """
    Compile a PTX string to a cubin file using ptxas.
    Returns the path to the cubin (inside a temp dir stored in the returned tuple).
    Caller owns the tmpdir and must clean it up.
    """
    ptxas = _find_ptxas()
    tmpdir = tempfile.mkdtemp()
    ptx_path   = os.path.join(tmpdir, 'kernel.ptx')
    cubin_path = os.path.join(tmpdir, 'kernel.cubin')

    with open(ptx_path, 'w') as f:
        f.write(ptx)

    result = subprocess.run(
        [ptxas, f'-arch={sm_version}', ptx_path, '-o', cubin_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f'ptxas compilation failed:\n{result.stderr}')

    return tmpdir, cubin_path


class GPURunner:
    """
    Compiles PTX → cubin (via ptxas) and wraps the kernel launch.

    Parameters:
        ptx     : PTX source string from PTXEmitter
        fn_name : .visible .entry name in the PTX
    """

    def __init__(self, ptx: str, fn_name: str):
        try:
            import cupy as cp
        except ImportError:
            raise ImportError(
                "CuPy is required. Install via: pip install cupy-cuda12x"
            )

        sm = f"sm_{cp.cuda.Device(0).compute_capability}"

        # Compile PTX → cubin; keep tmpdir alive until __del__
        self._tmpdir, cubin_path = _ptx_to_cubin(ptx, sm)

        # Load the cubin — RawModule(path=.cubin) loads directly, no compilation
        self._mod = cp.RawModule(path=cubin_path)
        self._fn  = self._mod.get_function(fn_name)
        self._fn_name = fn_name

    def __del__(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def __call__(
        self,
        *args: Any,
        grid: Tuple[int, ...],
        block: Tuple[int, ...],
    ) -> None:
        self._fn(grid=grid, block=block, args=args)

    def __repr__(self) -> str:
        return f"<GPURunner '{self._fn_name}'>"

"""
gpucc/runtime/cupy_runner.py — CuPy-based GPU kernel runner

Compiles PTX → cubin via ptxas, then loads via the CUDA driver API directly
(cupy.cuda.function.Module.load_file) — bypasses CuPy's compilation layer.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, Tuple


def _find_ptxas() -> str:
    for candidate in ('ptxas', '/usr/local/cuda/bin/ptxas'):
        try:
            subprocess.run([candidate, '--version'], capture_output=True, check=True)
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError(
        "ptxas not found — make sure the CUDA toolkit is installed "
        "(on Colab: /usr/local/cuda/bin/ptxas)."
    )


class GPURunner:
    """
    Compiles PTX → cubin (ptxas) and launches the kernel via CUDA driver API.

    Parameters:
        ptx     : PTX source string from PTXEmitter
        fn_name : .visible .entry name in the PTX
    """

    def __init__(self, ptx: str, fn_name: str):
        try:
            import cupy as cp
            from cupy.cuda import function as cuda_fn
        except ImportError:
            raise ImportError("CuPy required. pip install cupy-cuda12x")

        sm = f"sm_{cp.cuda.Device(0).compute_capability}"
        ptxas = _find_ptxas()

        # Compile PTX → cubin in a temp dir
        tmpdir = tempfile.mkdtemp()
        try:
            ptx_path   = os.path.join(tmpdir, 'kernel.ptx')
            cubin_path = os.path.join(tmpdir, 'kernel.cubin')

            with open(ptx_path, 'w') as f:
                f.write(ptx)

            result = subprocess.run(
                [ptxas, f'-arch={sm}', ptx_path, '-o', cubin_path],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f'ptxas failed:\n{result.stderr}')

            # Load cubin via cuModuleLoad — no CuPy compilation layer involved
            mod = cuda_fn.Module()
            mod.load_file(cubin_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        self._mod     = mod
        self._fn      = mod.get_function(fn_name)
        self._fn_name = fn_name

    def __call__(
        self,
        *args: Any,
        grid: Tuple[int, ...],
        block: Tuple[int, ...],
    ) -> None:
        # cuda_fn.Function.__call__(grid3, block3, args_tuple)
        g = tuple(grid)  + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))
        self._fn(g, b, args)

    def __repr__(self) -> str:
        return f"<GPURunner '{self._fn_name}'>"

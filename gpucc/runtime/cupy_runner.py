"""
gpucc/runtime/cupy_runner.py — CuPy-based GPU kernel runner

Compiles PTX → cubin via ptxas, loads via CUDA driver API, and launches
using cuLaunchKernel with manually-packed ctypes args — bypasses all of
CuPy's compilation and arg-handling layers.
"""
from __future__ import annotations

import ctypes
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
    Compiles PTX → cubin (ptxas), loads via CUDA driver, launches via
    cuLaunchKernel with explicitly-packed ctypes args.
    """

    def __init__(self, ptx: str, fn_name: str):
        try:
            import cupy as cp
            from cupy.cuda import function as cuda_fn
        except ImportError:
            raise ImportError("CuPy required. pip install cupy-cuda12x")

        sm = f"sm_{cp.cuda.Device(0).compute_capability}"
        ptxas = _find_ptxas()

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

            mod = cuda_fn.Module()
            mod.load_file(cubin_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        self._mod     = mod
        self._fn      = mod.get_function(fn_name)   # cuda_fn.Function
        self._fn_name = fn_name

    def __call__(
        self,
        *args: Any,
        grid: Tuple[int, ...],
        block: Tuple[int, ...],
    ) -> None:
        import numpy as np
        import cupy as cp
        from cupy.cuda import driver

        # Pad grid/block to 3-tuples
        g = tuple(grid)  + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))

        # Pack each arg into the correct ctypes type.
        # CuPy arrays → uint64 device pointer.
        # NumPy scalars → matching C type (must match PTX .param type).
        cvals = []
        for a in args:
            if isinstance(a, cp.ndarray):
                cvals.append(ctypes.c_uint64(a.data.ptr))
            elif isinstance(a, (np.int8, np.int16, np.int32)):
                cvals.append(ctypes.c_int32(int(a)))
            elif isinstance(a, (np.uint8, np.uint16, np.uint32)):
                cvals.append(ctypes.c_uint32(int(a)))
            elif isinstance(a, (np.int64, np.uint64)):
                cvals.append(ctypes.c_int64(int(a)))
            elif isinstance(a, np.float32):
                cvals.append(ctypes.c_float(float(a)))
            elif isinstance(a, np.float64):
                cvals.append(ctypes.c_double(float(a)))
            elif isinstance(a, int):
                cvals.append(ctypes.c_int32(a))
            elif isinstance(a, float):
                cvals.append(ctypes.c_float(a))
            else:
                raise TypeError(f"Unsupported kernel arg type: {type(a)}")

        # Build void** (array of pointers to each ctypes value)
        arg_ptrs = (ctypes.c_void_p * len(cvals))()
        for i, cv in enumerate(cvals):
            arg_ptrs[i] = ctypes.cast(ctypes.pointer(cv), ctypes.c_void_p)

        # Launch via CUDA driver API directly
        driver.launchKernel(
            self._fn.ptr,
            g[0], g[1], g[2],
            b[0], b[1], b[2],
            0,   # sharedMemBytes
            0,   # stream (NULL = default stream)
            ctypes.addressof(arg_ptrs),
            0,   # extra
        )

    def __repr__(self) -> str:
        return f"<GPURunner '{self._fn_name}'>"

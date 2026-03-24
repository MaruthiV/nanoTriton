"""
gpucc/transforms/__init__.py — Optimization pass runner

run_passes(fn, opt_level) applies the enabled passes in order:
  0 : no optimization
  1 : constant folding + DCE
  2 : constant folding + DCE + loop unrolling   (default)
  3 : all of the above + coalescing analysis (warnings only)
"""
from __future__ import annotations

import copy

from gpucc.ir import IRFunction


def run_passes(fn: IRFunction, opt_level: int = 2) -> IRFunction:
    """
    Run optimization passes on fn.  Returns a (possibly new) IRFunction.
    The original fn is not modified (deep copy is made first).
    """
    if opt_level == 0:
        return fn

    # Work on a copy so the original KernelHandle.ir is unchanged
    fn = copy.deepcopy(fn)

    if opt_level >= 1:
        from gpucc.transforms.constant_fold import constant_fold
        from gpucc.transforms.dce import dead_code_elimination
        fn = constant_fold(fn)
        fn = dead_code_elimination(fn)

    if opt_level >= 2:
        from gpucc.transforms.loop_unroll import loop_unroll
        fn = loop_unroll(fn)
        # Re-run fold + DCE after unrolling to clean up constants
        from gpucc.transforms.constant_fold import constant_fold
        from gpucc.transforms.dce import dead_code_elimination
        fn = constant_fold(fn)
        fn = dead_code_elimination(fn)

    if opt_level >= 3:
        from gpucc.transforms.coalesce_check import check_coalescing
        check_coalescing(fn)  # emits warnings, does not mutate fn

    return fn

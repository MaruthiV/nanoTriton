#!/usr/bin/env python3
"""
demo.py — gpucc compiler demo

Run with:  python demo.py
           python demo.py --matmul     (pipeline demo only)
           python demo.py --opts       (optimization diff only)
           python demo.py --vector     (vector_add secondary demo)
"""
from __future__ import annotations

import copy
import sys
import textwrap
import time

# ── ANSI colors ───────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"
STRIKE  = "\033[9m"

def bold(s):   return f"{BOLD}{s}{RESET}"
def dim(s):    return f"{DIM}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"
def green(s):  return f"{GREEN}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def blue(s):   return f"{BLUE}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def white(s):  return f"{WHITE}{s}{RESET}"
def strike(s): return f"{STRIKE}{RED}{s}{RESET}"

def pause(secs: float = 0.8) -> None:
    """Flush stdout then sleep so each section lands before the next starts."""
    sys.stdout.flush()
    time.sleep(secs)

def scroll(line: str, delay: float = 0.025) -> None:
    """Print one line, flush immediately, then sleep — creates scrolling effect."""
    print(line)
    sys.stdout.flush()
    time.sleep(delay)


def header(title: str) -> None:
    width = 66
    print()
    print(bold(blue("─" * width)))
    print(bold(blue(f"  {title}")))
    print(bold(blue("─" * width)))

def subheader(title: str) -> None:
    print(f"\n{bold(cyan(title))}")
    print(cyan("·" * len(title)))


# ── PTX annotation engine ─────────────────────────────────────────────────────

# Maps PTX instruction patterns to source-level descriptions.
# Checked in order — first match wins.
_ANNOTATIONS = [
    # Thread / block intrinsics
    ("%tid.x",    "thread_id(0)          ← which thread am I in the block?"),
    ("%tid.y",    "thread_id(1)"),
    ("%ctaid.x",  "block_id(0)           ← which block am I in the grid?"),
    ("%ctaid.y",  "block_id(1)"),
    ("%ntid.x",   "block_size(0)         ← how many threads per block?"),
    ("%ntid.y",   "block_size(1)"),
    # Kernel parameters
    ("ld.param.u64",  "load kernel param (pointer to array)"),
    ("ld.param.u32",  "load kernel param (scalar int32)"),
    # Address calculation — the tricky 32→64 bit boundary
    ("cvta.to.global.u64", "convert pointer → global address space"),
    ("mul.wide.u32",       "index × elem_size → 64-bit byte offset   ← i32→i64 widening"),
    ("add.u64",            "base_addr + byte_offset = final address"),
    # Memory
    ("ld.global.f32",  "load float32 from GPU global memory"),
    ("st.global.f32",  "store float32 to GPU global memory"),
    # Float arithmetic
    ("add.f32",    "floating-point add"),
    ("sub.f32",    "floating-point subtract"),
    ("mul.f32",    "floating-point multiply   ← acc += A[row,ki]*B[ki,col]"),
    ("div.rn.f32", "floating-point divide"),
    # Integer arithmetic
    ("add.u32",    "integer add"),
    ("mul.lo.u32", "integer multiply (low 32 bits)"),
    # Comparison / branch
    ("setp.",      "compare → 1-bit predicate register"),
    ("@%",         "conditional branch (predicated)"),
    # Loop back-edge: bare bra after the predicated branch
    ("bra ",       "unconditional jump   ← loop back-edge / else branch"),
    # Misc
    ("ret;",       "kernel return"),
    ("mov.u32",    "copy / move register"),
]

def _strip_comment(line: str) -> str:
    """Remove existing // comment from a PTX line."""
    if "//" in line:
        return line[:line.index("//")].rstrip()
    return line.rstrip()

def annotate_ptx(ptx: str, *, show_all: bool = False) -> str:
    """
    Return a formatted string with color-coded PTX + dim source annotations.
    Lines with no annotation are shown dim (boilerplate) unless show_all=True.
    """
    lines = ptx.split("\n")
    output_lines = []
    COL = 46  # column to align annotations

    for raw_line in lines:
        stripped = _strip_comment(raw_line)
        inner = stripped.lstrip()

        # Block labels: e.g. "BB1_loop_header:"
        if inner.endswith(":") and not inner.startswith(".") and " " not in inner:
            output_lines.append(f"\n{bold(yellow(stripped))}")
            continue

        # Non-instruction lines: directives, blank lines, braces
        if (inner.startswith(".")
                or inner.startswith("//")
                or inner == ""
                or inner == "{"
                or inner == "}"):
            if inner.startswith(".visible") or inner.startswith(".entry"):
                output_lines.append(bold(white(stripped)))
            elif inner.startswith(".") and not inner.startswith(".reg") and not inner.startswith(".param"):
                output_lines.append(dim(stripped))
            else:
                output_lines.append(dim(stripped))
            continue

        # Find annotation
        annotation = None
        for pattern, desc in _ANNOTATIONS:
            if pattern in inner:
                annotation = desc
                break

        if annotation:
            padded = stripped.ljust(COL)
            ann_str = dim(f"← {annotation}")
            output_lines.append(f"{white(padded)}  {ann_str}")
        elif show_all:
            output_lines.append(dim(stripped))
        else:
            output_lines.append(dim(stripped))

    return "\n".join(output_lines)


# ── IR diff (optimization before/after) ───────────────────────────────────────

def _ir_lines(fn) -> list[str]:
    from gpucc.ir_printer import print_function
    return print_function(fn).split("\n")


def show_opt_diff(fn_before, fn_after) -> None:
    """
    Side-by-side diff of IR before and after optimization.
    Removed lines shown in red strikethrough, surviving lines in white.
    """
    before = _ir_lines(fn_before)
    after_set = set(_ir_lines(fn_after))

    print(f"\n  {bold('Before optimization')}  {dim('(removed lines in red)')}:\n")
    for line in before:
        stripped = line.rstrip()
        if not stripped:
            print()
            continue
        # Is this line present in the after IR?
        if stripped in after_set:
            print(f"  {dim(stripped)}")
        else:
            # Highlight removed instruction
            print(f"  {strike(stripped)}  {dim(red('← eliminated'))}")

    print(f"\n  {bold('After optimization')}  {dim('(what the PTX emitter sees)')}:\n")
    for line in _ir_lines(fn_after):
        stripped = line.rstrip()
        if not stripped:
            print()
            continue
        print(f"  {green(stripped)}")


# ── Demo sections ─────────────────────────────────────────────────────────────

def demo_matmul() -> None:
    """Primary demo: matmul pipeline — source → IR → annotated PTX."""
    from gpucc import kernel, float32, int32, N, M, K
    from gpucc.ir_printer import print_function

    header("DEMO: Python → GPU Assembly   (matmul)")
    pause(0.4)

    # ── Source ────────────────────────────────────────────────────────────────
    subheader("Stage 1 — Source  (what you write)")

    src = textwrap.dedent("""\
        @kernel
        def matmul(A: float32[M, K], B: float32[K, N], C: float32[M, N],
                   m: int32, k: int32, n: int32):
            row = block_id(0) * block_size(0) + thread_id(0)
            col = block_id(1) * block_size(1) + thread_id(1)
            acc = 0.0
            for ki in range(k):
                acc = acc + A[row, ki] * B[ki, col]
            C[row, col] = acc
    """)
    for line in src.splitlines():
        if line.startswith("@"):
            scroll(f"  {bold(cyan(line))}")
        elif "def " in line:
            scroll(f"  {bold(white(line))}")
        elif "#" in line:
            scroll(f"  {dim(line)}")
        else:
            scroll(f"  {line}")

    pause()

    # ── Compile ───────────────────────────────────────────────────────────────
    @kernel
    def matmul(A: float32[M, K], B: float32[K, N], C: float32[M, N],
               m: int32, k: int32, n: int32):
        row = block_id(0) * block_size(0) + thread_id(0)
        col = block_id(1) * block_size(1) + thread_id(1)
        acc = 0.0
        for ki in range(k):
            acc = acc + A[row, ki] * B[ki, col]
        C[row, col] = acc

    # ── IR ────────────────────────────────────────────────────────────────────
    subheader("Stage 2 — IR  (compiler's internal representation)")
    print(dim("  The compiler broke your 10-line function into typed instructions"))
    print(dim("  and explicit basic blocks. Each %v is a virtual register.\n"))

    ir_text = print_function(matmul.ir)
    for line in ir_text.splitlines():
        stripped = line.rstrip()
        if "function " in stripped:
            scroll(f"  {bold(white(stripped))}")
        elif stripped.endswith(":") and not stripped.startswith(" "):
            scroll(f"\n  {bold(yellow(stripped))}")
        elif "= tid[" in stripped or "= ctaid[" in stripped or "= ntid[" in stripped:
            scroll(f"  {cyan(stripped)}")
        elif "= load" in stripped or "store." in stripped:
            scroll(f"  {green(stripped)}")
        elif "cbranch" in stripped or "jump" in stripped:
            scroll(f"  {yellow(stripped)}")
        elif "= fadd" in stripped or "= fmul" in stripped:
            scroll(f"  {white(stripped)}")
        elif stripped == "" or stripped.startswith("─"):
            scroll(f"  {dim(stripped)}", delay=0.005)
        else:
            scroll(f"  {dim(stripped)}")

    pause()

    # ── PTX ───────────────────────────────────────────────────────────────────
    subheader("Stage 3 — PTX  (GPU assembly your compiler generated)")
    print(dim("  This is what runs on the Nvidia GPU. Every line came from"))
    print(dim("  your 10-line Python function. Annotations show the source.\n"))

    ptx = matmul.compile(opt_level=2)
    for line in annotate_ptx(ptx).splitlines():
        scroll(line, delay=0.018)

    pause(0.4)

    # ── Stats ─────────────────────────────────────────────────────────────────
    ptx_lines = [l for l in ptx.splitlines() if l.strip() and not l.strip().startswith(".") and l.strip() != "{" and l.strip() != "}"]
    print(f"\n  {dim('─'*50)}")
    print(f"  {bold('Source lines:')}  {cyan('10')}    "
          f"{bold('PTX instructions:')}  {cyan(str(len(ptx_lines)))}    "
          f"{dim('(6× expansion)')}")


def demo_opts() -> None:
    """Show optimization passes as a before/after IR diff."""
    from gpucc import kernel, float32, int32, N
    from gpucc.transforms.constant_fold import constant_fold
    from gpucc.transforms.dce import dead_code_elimination

    header("DEMO: Optimization Passes — Before / After")

    print(dim("\n  Kernel with a constant expression and dead code:\n"))

    src = textwrap.dedent("""\
        @kernel
        def scaled_add(a: float32[N], b: float32[N], c: float32[N], n: int32):
            tid = thread_id()
            scale = 2.0 * 3.0      # constant expression — should be folded
            unused = 99 + 1        # never stored anywhere — should be eliminated
            if tid < n:
                c[tid] = (a[tid] + b[tid]) * scale
    """)
    for line in src.splitlines():
        if "constant expression" in line or "should be" in line:
            print(f"  {yellow(line)}")
        else:
            print(f"  {line}")

    @kernel
    def scaled_add(a: float32[N], b: float32[N], c: float32[N], n: int32):
        tid = thread_id()
        scale = 2.0 * 3.0
        unused = 99 + 1
        if tid < n:
            c[tid] = (a[tid] + b[tid]) * scale

    # Run passes manually so we can capture before/after
    import gpucc.ir as ir_mod
    fn_before = copy.deepcopy(scaled_add.ir)
    fn_after  = copy.deepcopy(scaled_add.ir)
    fn_after  = constant_fold(fn_after)
    fn_after  = dead_code_elimination(fn_after)

    # ── Pass 1: Constant folding ───────────────────────────────────────────────
    subheader("Pass 1 — Constant folding")
    print(dim("  Expressions with all-constant operands are evaluated at compile"))
    print(dim("  time. The result replaces the instruction entirely.\n"))

    fn_fold_only = copy.deepcopy(scaled_add.ir)
    fn_fold_only = constant_fold(fn_fold_only)

    from gpucc.ir_printer import print_function
    before_lines = print_function(scaled_add.ir).split("\n")
    after_fold   = set(print_function(fn_fold_only).split("\n"))

    for line in before_lines:
        stripped = line.rstrip()
        if not stripped:
            print()
            continue
        if stripped not in after_fold and ("fmul" in stripped or "iadd" in stripped or "2.0" in stripped or "3.0" in stripped or "99" in stripped):
            print(f"  {strike(stripped)}  {dim(red('← folded to constant'))}")
        elif stripped in after_fold and ("copy" in stripped and ("6.0" in stripped or "100" in stripped or "6" in stripped)):
            print(f"  {green(stripped)}  {dim(green('← result: compile-time constant'))}")
        else:
            print(f"  {dim(stripped)}")

    # ── Pass 2: DCE ───────────────────────────────────────────────────────────
    subheader("Pass 2 — Dead code elimination")
    print(dim("  Values that are computed but never used (no STORE, no CBRANCH,"))
    print(dim("  no downstream use) are removed entirely.\n"))

    before_fold_lines = print_function(fn_fold_only).split("\n")
    after_both_set    = set(print_function(fn_after).split("\n"))

    for line in before_fold_lines:
        stripped = line.rstrip()
        if not stripped:
            print()
            continue
        if stripped not in after_both_set and ("unused" in stripped or ("copy" in stripped and "100" in stripped)):
            print(f"  {strike(stripped)}  {dim(red('← dead — result never used'))}")
        else:
            print(f"  {dim(stripped)}")

    # ── PTX comparison ────────────────────────────────────────────────────────
    subheader("Result — PTX instruction count")

    ptx_raw = scaled_add.compile(opt_level=0)
    ptx_opt = scaled_add.compile(opt_level=2)

    def count_instrs(ptx):
        return sum(
            1 for l in ptx.splitlines()
            if l.strip()
            and not l.strip().startswith(".")
            and l.strip() not in ("{", "}")
            and not l.strip().startswith("//")
            and not l.strip().endswith(":")
        )

    n_raw = count_instrs(ptx_raw)
    n_opt = count_instrs(ptx_opt)

    print(f"\n  Without optimization:  {red(bold(str(n_raw)))} PTX instructions")
    print(f"  With optimization:     {green(bold(str(n_opt)))} PTX instructions")
    print(f"  {dim(f'({n_raw - n_opt} instructions eliminated)')}\n")

    print(dim("  Key lines from optimized PTX:"))
    for line in ptx_opt.splitlines():
        clean = line.split("//")[0].rstrip()
        # Match mov.f32 loading the folded constant 6 (not register names containing 6)
        if ("mov.f32" in clean and ", 6;" in clean):
            print(f"  {white(clean)}  {dim('← 2.0 * 3.0 folded at compile time (no runtime multiply)')}")
        elif "mul.f32" in clean and "scale" not in clean:
            pass  # skip — the actual multiply is the acc update, not folded


def demo_vector_add() -> None:
    """Secondary demo: vector_add for contrast with matmul."""
    from gpucc import kernel, float32, int32, N

    header("DEMO: vector_add  (secondary example — simpler for contrast)")

    print(dim("\n  The classic 'hello world' of GPU kernels.\n"))
    print(f"  {bold('Source:')}\n")
    src = textwrap.dedent("""\
        @kernel
        def vector_add(a: float32[N], b: float32[N], c: float32[N], n: int32):
            tid = thread_id()
            if tid < n:
                c[tid] = a[tid] + b[tid]
    """)
    for line in src.splitlines():
        print(f"    {line}")

    @kernel
    def vector_add(a: float32[N], b: float32[N], c: float32[N], n: int32):
        tid = thread_id()
        if tid < n:
            c[tid] = a[tid] + b[tid]

    print(f"\n  {bold('Annotated PTX:')}\n")
    ptx = vector_add.ptx
    print(annotate_ptx(ptx))

    ptx_lines = [l for l in ptx.splitlines()
                 if l.strip() and not l.strip().startswith(".")
                 and l.strip() not in ("{", "}")]
    print(f"\n  {dim(f'Source: 4 lines  →  PTX: {len(ptx_lines)} instructions')}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args or "--matmul" in args:
        demo_matmul()
    if not args or "--opts" in args:
        demo_opts()
    if "--vector" in args:
        demo_vector_add()

    if not args:
        print()
        print(bold(blue("─" * 66)))
        print(f"  {bold('To run on GPU:')}  open notebooks/gpu_validation.ipynb on Colab")
        print(f"  {bold('Run tests:    ')}  python -m pytest tests/ -v")
        print(f"  {bold('Sections:     ')}  python demo.py --matmul | --opts | --vector")
        print(bold(blue("─" * 66)))
        print()


if __name__ == "__main__":
    main()

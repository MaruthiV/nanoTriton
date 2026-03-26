# nanoTriton

A mini GPU kernel compiler built from scratch in Python. Write a simple Python function, and nanoTriton compiles it all the way down to NVIDIA GPU assembly — no external compiler frameworks required.

---

## What is this?

Modern GPU frameworks like PyTorch or CUDA let you run code on a GPU, but most developers treat them as a black box. This project opens that box.

**nanoTriton is a compiler** — it takes a Python function annotated with GPU types and produces PTX assembly, which is the low-level instruction language that NVIDIA GPUs understand. Think of it like this:

> You write Python. A normal Python interpreter runs it on your CPU. nanoTriton instead *compiles* it into GPU instructions that run in parallel across thousands of GPU threads.

The name is inspired by [Triton](https://github.com/openai/triton), OpenAI's production GPU compiler. nanoTriton is a from-scratch, educational reimplementation of those core ideas — no shortcuts, no magic.

**Why does this matter?** Writing GPU code normally requires learning CUDA C++, understanding memory hierarchies, and managing thread synchronization manually. Triton-style compilers aim to let you express GPU kernels in a higher-level language and have the compiler handle the hard parts. nanoTriton shows exactly how that works, step by step.

### What you write

```python
@kernel
def vector_add(a: float32[N], b: float32[N], c: float32[N], n: int32):
    tid = block_id(0) * block_size(0) + thread_id(0)
    if tid < n:
        c[tid] = a[tid] + b[tid]
```

### What nanoTriton generates

```ptx
.visible .entry vector_add(.param .u64 param_a, .param .u64 param_b,
                            .param .u64 param_c, .param .u32 param_n) {
    .reg .f32 %f<14>;
    .reg .u32 %r<10>;
    .reg .u64 %rd<23>;
    .reg .pred %p<11>;

    ld.param.u64 %rd0, [param_a];
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mul.lo.u32 %r6, %r4, %r5;
    mov.u32 %r7, %tid.x;
    add.u32 %r8, %r6, %r7;
    setp.lt.s32 %p10, %r8, %r3;
    @%p10 bra BB1_then;
    ...
}
```

4 lines of Python → 31 PTX instructions, ready to run on a GPU.

---

## Compilation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         nanoTriton Pipeline                             │
└─────────────────────────────────────────────────────────────────────────┘

  @kernel
  def vector_add(a: float32[N], ...):       ← You write this
      ...
        │
        ▼
┌───────────────────┐
│   AST Parsing     │  Python's built-in `ast` module parses the function
│  (frontend/)      │  into a syntax tree (just like a Python interpreter
└───────────────────┘  would, but we intercept it before execution)
        │
        ▼
┌───────────────────┐
│  Type Resolution  │  Annotations like float32[N] are lowered to concrete
│  (ast_to_ir.py)   │  GPU types: pointers, 32-bit ints, 32-bit floats, etc.
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  IR Generation    │  The syntax tree is translated into an Intermediate
│  (ir.py)          │  Representation — a list of simple typed instructions
└───────────────────┘  operating on virtual registers (like a simplified
        │              assembly language with infinite registers)
        ▼
┌───────────────────┐
│  Optimization     │  Optional passes run over the IR:
│  (transforms/)    │    • Constant folding  (evaluate 3+4 → 7 at compile time)
└───────────────────┘    • Dead code elim.   (remove instructions nobody reads)
        │                • Loop unrolling     (unroll small loops 4× for speed)
        │                • Coalescing check   (warn about bad memory patterns)
        ▼
┌───────────────────┐
│ Register Alloc.   │  Each virtual register is mapped to a real PTX register
│ (codegen/)        │  name (%f0, %r3, %rd7, %p1) grouped by type
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  PTX Emission     │  The final IR walks top-to-bottom and emits PTX text:
│  (ptx_emit.py)    │  NVIDIA's portable GPU assembly language
└───────────────────┘
        │
        ▼
  kernel.ptx                                ← You get this (or run it directly)
        │
        ▼  (optional, requires GPU + CuPy)
┌───────────────────┐
│  GPU Execution    │  ptxas assembles PTX → binary, then CUDA driver loads
│  (runtime/)       │  and launches it on the GPU
└───────────────────┘
        │
        ▼
  Result on GPU                             ← Verified against NumPy
```

---

## Project Structure

```
nanoTriton/
├── gpucc/
│   ├── __init__.py              # Public API: @kernel, float32, int32, N, M, K
│   ├── ir.py                    # IR data structures (VReg, Instruction, BasicBlock)
│   ├── ir_printer.py            # Human-readable IR dumps for debugging
│   ├── types.py                 # GPU type system (Int32, Float32, Pointer, Bool)
│   ├── annotations.py           # Type annotation sentinels (float32[N], etc.)
│   ├── frontend/
│   │   ├── decorator.py         # @kernel entry point → KernelHandle
│   │   └── ast_to_ir.py         # Python AST → IR translation (~1000 lines)
│   ├── transforms/
│   │   ├── __init__.py          # Pass pipeline runner (opt_level 0–3)
│   │   ├── constant_fold.py     # Constant folding
│   │   ├── dce.py               # Dead code elimination
│   │   ├── loop_unroll.py       # Loop unrolling (default 4×)
│   │   └── coalesce_check.py    # Memory coalescing analysis
│   ├── codegen/
│   │   ├── regalloc.py          # Virtual → physical register mapping
│   │   └── ptx_emit.py          # PTX code generator
│   └── runtime/
│       └── cupy_runner.py       # ptxas + CUDA driver launcher
├── examples/
│   ├── vector_add.py            # Vector addition kernel
│   └── matmul.py                # Naive matrix multiplication kernel
├── tests/                       # 50+ pytest tests (no GPU required)
├── notebooks/
│   └── gpu_validation.ipynb     # Google Colab notebook (runs on real GPU)
└── demo.py                      # Interactive pipeline demo (colorized output)
```

---

## Quick Start

### Inspect the compiler output (no GPU needed)

```bash
# Clone and run
git clone <repo>
cd nanoTriton

# See the full pipeline: Python → IR → PTX
python examples/vector_add.py
python examples/matmul.py

# Interactive demo with annotations and optimization comparison
python demo.py
python demo.py --opts    # Show before/after optimization
```

### Run tests

```bash
python -m pytest tests/ -v
```

All 50+ tests run without a GPU. They validate IR correctness, optimization passes, and end-to-end PTX generation.

### Run on a real GPU (Google Colab recommended)

Open `notebooks/gpu_validation.ipynb` in Google Colab with a T4 GPU runtime. It:
1. Compiles and runs `matmul` — validates against NumPy
2. Compiles and runs `vector_add` — validates + benchmarks
3. Demonstrates optimization: compares `opt_level=0` vs `opt_level=2` PTX

Local GPU execution requires CuPy (`pip install cupy-cuda12x`) and an NVIDIA GPU.

---

## Example Kernels

### Vector Addition

```python
from gpucc import kernel, float32, int32, N
from gpucc.frontend.intrinsics import thread_id, block_id, block_size

@kernel
def vector_add(a: float32[N], b: float32[N], c: float32[N], n: int32):
    tid = block_id(0) * block_size(0) + thread_id(0)
    if tid < n:
        c[tid] = a[tid] + b[tid]

# Inspect the IR
print(vector_add.ir_text)

# Get PTX (opt_level=2: fold + DCE + unroll)
ptx = vector_add.compile(opt_level=2)
print(ptx)
```

### Matrix Multiplication

```python
from gpucc import kernel, float32, int32, M, K, N

@kernel
def matmul(A: float32[M, K], B: float32[K, N], C: float32[M, N],
           m: int32, k: int32, n: int32):
    row = block_id(0) * block_size(0) + thread_id(0)
    col = block_id(1) * block_size(1) + thread_id(1)
    acc = 0.0
    for ki in range(k):
        acc = acc + A[row, ki] * B[ki, col]
    C[row, col] = acc
```

---

## Optimization Passes

Control optimization with `opt_level` when calling `.compile()`:

| Level | Passes |
|-------|--------|
| `0` | None (raw IR → PTX) |
| `1` | Constant folding + Dead code elimination |
| `2` | Fold + DCE + Loop unrolling + Fold + DCE (default) |
| `3` | All of above + Coalescing analysis warnings |

**Constant folding** — evaluates expressions at compile time:
```
%v = iadd 3, 4   →   %v = copy 7
```

**Dead code elimination** — removes instructions whose results are never used.

**Loop unrolling** — unrolls counted `for` loops with known bounds (4× factor):
```python
for ki in range(64)   →   16 iterations of 4-unrolled body in PTX
```

---

## The IR (Intermediate Representation)

Between Python and PTX, nanoTriton uses its own typed IR. You can inspect it at any time:

```
function vector_add(a: float32*, b: float32*, c: float32*, n: int32)
─────────────────────────────────────────────────────────────────────
BB0_entry:
  %v0:a           = param[0] a (float32*)
  %v4             = ctaid[x]
  %v5             = ntid[x]
  %v6             = imul %v4, %v5
  %v7             = tid[x]
  %v8             = iadd %v6, %v7
  %v9:tid         = copy %v8
  %v10            = lt bool %v9:tid, %v3:n
                   cbranch %v10 → BB1_then | BB2_merge

BB1_then:
  %v11            = load float32 %v0:a[%v9:tid]
  %v12            = load float32 %v1:b[%v9:tid]
  %v13            = fadd float32 %v11, %v12
                   store float32 %v2:c[%v9:tid] = %v13
                   jump → BB2_merge

BB2_merge:
  ret
```

Every virtual register is typed. Control flow is explicit via basic blocks. This makes the compiler stages easy to inspect and debug.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Source language | Python 3.10+ |
| AST parsing | Python's built-in `ast` module |
| IR | Custom dataclasses (pure Python, no dependencies) |
| Output assembly | NVIDIA PTX 7.0 |
| Assembler | `ptxas` (NVIDIA, part of CUDA toolkit) |
| GPU execution | CuPy + CUDA driver API (optional) |
| Tests | pytest + NumPy |

No LLVM. No external compiler frameworks. The entire compilation pipeline is ~3000 lines of Python.

---

## Limitations

This is an MVP compiler — it handles real programs correctly but doesn't yet support:

- **Shared memory** — no `.shared` space; all memory accesses go to global memory
- **Synchronization** — no `__syncthreads()` equivalent
- **Atomic operations** — no atomicAdd, etc.
- **Advanced register allocation** — trivial 1-to-1 mapping (ptxas handles spilling)
- **Tiling / memory optimization** — matmul is naive (no blocked algorithms)
- **Multiple kernels per module** — one `@kernel` per compilation unit

---

## Inspiration

This project is inspired by [OpenAI Triton](https://github.com/openai/triton), which compiles Python kernels to efficient GPU code for use in deep learning. nanoTriton strips that idea down to its core to make the compiler machinery understandable and hackable.

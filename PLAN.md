# SparseDirectSolver.jl — porting plan from CUDSS.jl / cuDSS

Goal: a portable sparse direct linear solver (LLᵀ/LLᴴ, LDLᵀ/LDLᴴ, LDU) for GPUs,
written in Julia on top of KernelAbstractions.jl (KA) and GPUArrays.jl, offering
the feature set of CUDSS.jl v0.8 (and the underlying cuDSS 0.8) wherever that
feature set is meaningful outside CUDA, and beating cuDSS where the target
workload allows it.

Principles fixed by the project owner:

* **Feature parity across backends from day one.** CUDA.jl, AMDGPU.jl, oneAPI.jl
  and Metal.jl are first-class targets; the KA CPU backend is the reference.
  Every milestone is tested on all of them. Tolerated differences are hardware
  limits only (Metal has no Float64; consumer Intel GPUs emulate Float64).
* **Vendor BLAS/LAPACK wherever it pays.** Dense work on fronts large enough for
  BLAS-3 goes to the libraries each GPU package ships (cuBLAS/cuSOLVER,
  rocBLAS/rocSOLVER, oneMKL, Metal Performance Shaders). KA kernels do the
  irregular work (assembly, extend-add, gather/scatter, permutations, SpMV), the
  tiny-front regime where vendor libraries have nothing to offer, and serve as
  fallbacks for whatever a vendor lacks, so the *feature* set is identical on
  every backend even where the *speed* is not. "Pure Julia" means "no
  closed-source sparse solver", not "no vendor dense kernels".

Primary downstream consumer: MadNLPGPU (`CUDSSSolver`), i.e. MadNLP, MadIPM and
MadNCL: symmetric indefinite or condensed KKT systems with a fixed pattern,
hundreds of refactorizations per solve, inertia queries, iterative refinement,
user permutations, uniform batches (two-stage Schur KKT), later Schur complement
mode. Milestones are ordered so that MadNLP can switch as early as possible.

Decisions taken (2026-10-01):

1. CliqueTrees.jl is the ordering dependency.
2. Vendor BLAS/LAPACK on every backend where it pays, KA fallbacks for parity.
3. Keep the CUDSS.jl parameter strings and phase names as a flat handle-style
   layer that a future C interface can wrap; keep it small.
4. v1 cut line = M0–M6 (SPD, LDLᵀ, IR, uniform batch, MadNLP-ready).
5. No multi-GPU (single node or MGMN) for now.

Research input: `RESEARCH.md` (state of the art of GPU sparse direct solvers for
IPM KKT systems, 2026-10-01). What it changed in this plan:

* **Workload model.** ACOPF/KKT matrices have no block structure and very sparse
  factors: deep, narrow elimination trees with thousands of tiny supernodes.
  Generic supernodal/multifrontal GPU solvers lost to single-threaded MA57 on
  them; cuDSS keeps a separate algorithm for this regime. The bound is launch
  latency and per-front overhead, not BLAS-3 throughput.
* **Three regimes instead of two** in the numeric phase: fused
  subtree-per-workgroup kernels for the bottom of the tree (A), fused
  per-front size-binned level-batched kernels for the middle (B), vendor
  BLAS/LAPACK for the few root fronts (C). Vendor batched calls are an option
  for B only where measurements show they win.
* **GPU-tuned amalgamation, subtree partition and size bins** become first-class
  outputs of the analysis, with an ordering cost model (fill *and* critical
  path) to choose between AMD and nested dissection automatically.
* **Pivoting**: in-front 1×1/2×2 Bunch–Kaufman restricted to the fully-summed
  block, perturbation with a user-selectable sign policy (cuDSS cannot do this),
  an extended inertia report (n₊, n₋, n₀, n_perturbed). A posteriori threshold
  pivoting with optional delayed pivots is a later robustness milestone.
* **Refinement and mixed precision**: FGMRES-IR with the factors as
  preconditioner as an option next to plain IR; Float32 factors with Float64
  refinement as the mixed-precision mode and the only route on Metal.
* **Portability constraints** taken as hard: no stable subgroup primitives in
  KA, no inter-workgroup forward progress on Metal (likely Intel), no graphs or
  dynamic parallelism. Level/subtree scheduling is the portable baseline;
  sync-free solves and graph capture live only in the CUDA/ROCm extensions.
* **Benchmark harness and cuDSS baseline measurements** move into M0.

Status: empty repository apart from `PLAN.md` and `RESEARCH.md`.

---

## 1. Feature inventory and disposition

Legend: **port** (same semantics), **reinterpret** (same parameter name, portable
semantics), **accept+ignore** (accepted for drop-in compatibility, warns once),
**defer** (after v1), **not planned**.

### 1.1 Matrix descriptors and value types

| cuDSS / CUDSS.jl | Plan | Notes |
| --- | --- | --- |
| CSR input `rowPtr/colVal/nzVal`, `INT ∈ {Int32, Int64}` | port | Accept each backend's CSR type (`CuSparseMatrixCSR`, `ROCSparseMatrixCSR`, `oneSparseMatrixCSR`) through extensions, plus an in-package `CSR{T,INT,VI,VT}` for Metal, CPU and raw arrays. |
| CSC input (MadNLP passes `colPtr/rowVal` as CSR of Aᵀ with view `'U'`) | port | Treated as CSR of Aᵀ with `view`/`solve_mode` flipped; no copy. |
| Structures `"G" "S" "H" "SPD" "HPD"` | port | Same strings. |
| Views `'L' 'U' 'F'` | port | For `'F'` on symmetric types only one triangle is read, as in cuDSS. |
| Index base `'Z' 'O'` | port | Handled in the symbolic maps; values never copied. |
| `T ∈ {Float32, Float64, ComplexF32, ComplexF64}` | port | Generic in `T`. Metal: Float32/ComplexF32 factors (see mixed precision, §3.8). |
| Double-double (`CUDSS_R_64F_64F`) | not planned | |
| `offsetType` (Int64 row pointers, Int32 columns) | port | Separate eltype parameter for `rowPtr`. |
| Dense RHS/solution: vector, matrix, col/row-major, leading dimension | port | Row-major kept as a `transposed` flag. |
| `CudssMatrix(T, n; nbatch)` + `cudss_update` descriptors | reinterpret | Core takes plain arrays; a thin `MatrixDescriptor` + `update!` keeps allocation-free buffer swapping for MadNLP-style code. |
| Host-memory input (hybrid execute) | defer (M12) | |
| Distributed (`CUDSS_MFORMAT_DISTRIBUTED`, row-1d) | not planned | |

### 1.2 Phases

All CUDSS.jl phase strings are kept: `"reordering"`, `"symbolic_factorization"`,
`"analysis"`, `"factorization"`, `"refactorization"`, `"solve_fwd_perm"`,
`"solve_fwd"`, `"solve_diag"`, `"solve_bwd"`, `"solve_bwd_perm"`,
`"solve_refinement"`, `"solve"`, plus `"solve_fwd_schur"` and
`"solve_bwd_schur"`. Named functions `analyze!`, `factorize!`, `refactorize!`,
`solve!` are thin wrappers.

### 1.3 Configuration parameters (`cudss_set` → `setparam!`)

| Name | Plan | Portable semantics |
| --- | --- | --- |
| `reordering_alg` | reinterpret | `"default"`: automatic choice between ND (Metis extension) and AMD by the cost model of §2.3; `"algo3"` AMD, `"algo4"` ND, `"algo5"` natural. `"algo1"/"algo2"` (BTF_COLAMD/COLAMD with global pivoting): accepted, falls back to symmetric-pattern LU with a one-time warning (§3.3). |
| `factorization_alg` | reinterpret | `"default"` auto; `"algo1"` forces the very-sparse-factor path (regimes A/B only, no vendor calls); `"algo2"` forces vendor calls for everything above the subtree regime. Mirrors cuDSS keeping a separate algorithm for very sparse factors. |
| `solve_alg` | reinterpret | `"default"` level-batched; `"algo1"` partitioned-inverse diagonal blocks (M11); sync-free variant selected automatically on CUDA/ROCm when available. |
| `matching_alg` | port (M9) | `"default"` off; algo1..5 = MC64 jobs 1..5; algo6 auto → job 5. |
| `solve_mode` | port (M5) | 0 = A, 1 = Aᵀ, 2 = Aᴴ. |
| `ir_n_steps`, `ir_tol` | port (M5) | `ir_tol` honored (cuDSS ignores it). Default `ir_n_steps = 0` in the handle layer (cuDSS parity; MadNLP runs its own refinement loop), refinement on with early exit in the LinearAlgebra layer (§3.1). |
| `pivot_type` | port (partial) | `'A'` auto, `'N'` none, `'D'` diagonal, `'L'` local block, `'B'` Bunch–Kaufman 1×1/2×2 (default for `S`/`H`). `'C'/'R'` global: not planned. |
| `pivot_threshold` | port (M4/M7) | Threshold for in-front pivot acceptance (LDLᵀ and LU). |
| `pivot_epsilon`, `pivot_epsilon_alg` | port (M4) | Static or scaled perturbation. Defaults match cuDSS: 1e-5 (Float32), 1e-13 (Float64). |
| `max_lu_nnz` | port (M1) | Checked after symbolic analysis. |
| `hybrid_memory_mode`, `hybrid_device_memory_limit` | defer (M12) | Host-resident panels streamed per level. |
| `use_cuda_register_memory` | reinterpret (M12) | Pinned host memory through the backend extension. |
| `hybrid_execute_mode` | defer (M12) | Small levels on the KA CPU backend. |
| `host_nthreads` | reinterpret | Julia threads for the host symbolic phase / CPU backend. |
| `nd_nlevels`, `nd_ubfactor` | port (M1) | ND ordering parameters. |
| `ubatch_size`, `ubatch_index` | port (M6) | Uniform batch. |
| `use_superpanels` | reinterpret | Supernode amalgamation on/off. |
| `device_count`, `device_indices` | not planned | Raise "not supported". |
| `schur_mode` | port (M8) | |
| `deterministic_mode` | port | Assembly is deterministic by construction (§3.4); the flag switches the forward solve to its atomic-free variant. |

### 1.4 Data parameters (`cudss_get` → `getparam` / `getparam!`)

| Name | Plan | Notes |
| --- | --- | --- |
| `info` | port | 0 ok; `k > 0` first failed pivot (1-based, original numbering); vector for batches; settable (reset before refactorization, as CUDSS.jl does). |
| `lu_nnz` | port | From the symbolic phase. |
| `npivots` | port | Perturbed pivots (static perturbation yields the inertia of A+E, so this must be read together with `inertia`). |
| `inertia` | port | `(npos, nneg)` from D including 2×2 blocks; correct under matching (fixes the cuDSS 0.8 defect MadNLP works around). |
| `perm_reorder_row/col`, `perm_row/col`, `perm_matching` | port | Returned as vectors; `getparam!(buf, …)` writes into a user buffer, replacing the C-style set-buffer-then-get protocol. |
| `diag` | port | Diagonal of D (LDLᵀ) / U (LU) / L (Cholesky). |
| `scale_row/col` | port (M9) | |
| `user_perm` | port (M1) | Host or device vector. |
| `memory_estimates` | port (M1) | `Int64[16]` with a documented layout. |
| `hybrid_device_memory_min` | defer (M12) | |
| `nsuperpanels` | port | Supernodes after amalgamation. |
| `user_schur_indices`, `schur_shape`, `schur_matrix` | port (M8) | Dense or CSR export; Hermitian case returns one triangle with a correct nnz. |
| `user_nd_partition_tree`, `nd_partition_tree` | port (M1) | Same binary-tree encoding as cuDSS, so orderings can be cached between runs. Extra: `"etree"`, `"supernodes"`. |
| `user_host_interrupt` | port | `Threads.Atomic{Bool}` polled between levels. |
| `ir_n_steps` (data) | port (M5) | Steps actually performed. |
| `ubatch_mask` | port (M6) | |
| `flops` | port (M1) | From supernode sizes. |
| `comm_device`, `comm_host` | not planned | MGMN. |

### 1.5 Generic LinearAlgebra interface

`lu`, `lu!`, `ldlt`, `ldlt!`, `cholesky`, `cholesky!`, `ldiv!`, `\`, the
`Symmetric`/`Hermitian` wrappers, uniform-batch auto-detection via
`length(nzVal) ÷ length(colVal)`, and the `fresh_factorization` flag that turns
`lu!` et al. into refactorizations: ported unchanged. Added: `inertia(F)`,
`logabsdet(F)`, `diag(F)`, `nnz(F)`.

### 1.6 Advanced cuDSS features

| Feature | Plan |
| --- | --- |
| Uniform batch, strided layouts, 3-D arrays, `nrhs > 1` | port (M6): one more grid dimension in every kernel; serves ExaModels multi-scenario models. The cuDSS multi-RHS batch bug (CUDSS.jl *Known issues*) becomes a regression test. |
| Non-uniform batch (`CudssBatchedSolver`) | port (M10): batch packed into one block-diagonal system, processed as a forest of elimination trees. |
| Multi-GPU single node, MGMN, comm layers | not planned. The assembly tree keeps subtree ownership (ND top-level split) so this can be added later without redesign. |
| Multi-threaded host mode | reinterpret: Julia threads in analysis, KA CPU backend for numerics. |
| Logging | reinterpret: Julia `Logging` + `SDS_LOG_LEVEL`. |
| Device memory handler, CUDA graphs, NVTX | graph capture of refactorize+solve through the CUDA/ROCm extensions (§3.9); the rest is not applicable. |
| Deterministic mode | port. |

### 1.7 Parameters beyond cuDSS (motivated by the research)

| Name | Kind | Semantics |
| --- | --- | --- |
| `pivot_sign` | data | Per-row expected sign (`Int8`: +1, −1, 0 unknown). When a pivot must be perturbed, the replacement takes this sign instead of the sign of the tiny value. MadNLP passes +1 for primal and −1 for dual rows. |
| `pivot_stats` | data | `(npos, nneg, nzero, nperturbed, n2x2)` from a per-front reduction, so MadNLP can raise its regularization instead of trusting a perturbed factorization. |
| `ir_mode` | config | `"ir"` (default) or `"fgmres"`: FGMRES with the factorization as preconditioner (Krylov.jl extension). |
| `factor_precision` | config | Element type of the factors (`Float32` or `Float64`); refinement runs in the input precision. Required on Metal. |
| `amalgamation` | config | Relaxed-amalgamation parameters (max explicit-zero fraction, target panel width 8–32 columns). |
| `schedule` | config | `"auto"`, `"subtree+level"` (portable baseline), `"syncfree"` (CUDA/ROCm only). |

---

## 2. Architecture

### 2.1 Layering

```
user API: LinearAlgebra generics  +  handle-style layer (CUDSS.jl strings, C-wrappable)
  └─ DirectSolver / BatchedDirectSolver        options, symbolic, numeric state, info
       ├─ symbolic (host, Julia, once per pattern)
       │    pattern → ordering (CliqueTrees, cost model) → etree → column counts
       │    → GPU-tuned supernodes → assembly tree → subtree partition + size bins
       │    → level schedule → static memory layout → device maps
       ├─ numeric (device, per factorization; allocation-free, host-sync-free)
       │    regime A  fused subtree-per-workgroup kernels          (KA)
       │    regime B  fused per-front size-binned level kernels    (KA; vendor batched optional)
       │    regime C  vendor potrf/sytrf/getrf + trsm + syrk/gemm  (extensions; KA fallback)
       │    + KA kernels for assembly, extend-add, gather/scatter, permutation
       ├─ solve (device): permute/scale → subtree/level fwd → diag → bwd → unpermute → IR/FGMRES
       └─ extensions: CUDA, AMDGPU, oneAPI, Metal (dense + sparse adapters + graph capture
                      + sync-free solves where the hardware allows), Metis, Krylov
```

Only the pattern (`rowPtr`, `colVal`) travels to the host, during analysis.
Values stay on the device; refactorization is a fixed kernel sequence.

### 2.2 Why supernodal multifrontal, and why three regimes

Multifrontal gives contiguous dense fronts, which batch well and are the natural
home for in-front pivoting; its usual drawback (dynamic contribution-block
memory) disappears with a fixed pattern because every front's location and the
memory high-water mark are computed once at analysis. Schur complement mode is
"do not factorize the root front" (M8), hybrid memory mode is "only the current
level's fronts on the device" (M12), non-uniform batches are a forest (M10), and
LU on the symmetric pattern of A+Aᵀ with pivoting restricted to the fully-summed
block reuses the Cholesky symbolic machinery (M7).

The ACOPF evidence says the time goes into thousands of tiny fronts and many
tree levels, so the numeric phase is organized by front size:

* **Regime A, leaf subtrees.** One workgroup assembles, factors and extend-adds
  an entire subtree serially in local memory. This collapses the deep, narrow
  bottom of the tree into one launch per subtree batch, with no inter-workgroup
  synchronization, so it is portable to every backend.
* **Regime B, mid-level fronts.** One launch per (level, size bin); one
  workgroup per front doing a shared-memory blocked POTRF/LDLᵀ-BK/GETRF of the
  fully-summed block, TRSM, SYRK/GEMM Schur update and extend-add through the
  precomputed maps, all in one fused kernel. Vendor batched calls are used
  instead only when the M2 benchmarks show they win for a bin on a backend.
* **Regime C, root fronts.** A handful of large fronts: vendor
  `potrf`/`sytrf`/`getrf`, `trsm`, `syrk`/`herk`/`gemm` through the generic
  LinearAlgebra entry points and the extensions; KA tiled kernels as fallback.

This mirrors cuDSS keeping a distinct algorithm for very sparse factors, and
CHOLMOD's subtree streaming. The split points are analysis outputs (§2.3), not
runtime decisions.

### 2.3 Symbolic engine (host)

1. **Pattern**: expand the user's triangle, symmetrize (`G`: A ∪ Aᵀ), drop
   duplicates, rebase indices, check squareness; build the full-pattern CSR map
   used by the residual SpMV of refinement.
2. **Ordering** via CliqueTrees.jl: `permutation(graph; alg)` with `AMD()`/`MMD()`
   (pure Julia) or `METIS()`/`ND` (Metis extension, honoring
   `nd_nlevels`/`nd_ubfactor`); user permutation; natural; ND-tree export/import
   in the cuDSS encoding. **Automatic choice** computes both AMD and ND
   candidates (cheap for KKT sizes) and picks by a cost model combining
   predicted fill/flops with predicted critical path and level count: ND gives
   bushier, shallower trees; AMD often gives lower fill on power grids. Schur
   mode constrains the ordering (§3.6). Matching composes a column permutation
   in (M9).
3. **Elimination tree**, postorder, **column counts** (Gilbert–Ng–Peyton), nnz(L),
   flops → `lu_nnz`, `flops`, `max_lu_nnz`. CliqueTrees' `supernodetree` gives
   the fundamental supernode partition; etree, column-count and amalgamation
   code is ours.
4. **GPU-tuned amalgamation**: relaxed amalgamation (Ashcraft–Grimes) with
   GPU parameters: accept more explicit zeros than a CPU solver would (bounded
   by the `amalgamation` option, default ≈25% extra factor storage) to reach
   8–32-column panels, which is the main lever against tiny-front overhead.
   `max_width` caps merging only: a fundamental supernode wider than that (a
   dense separator) stays one front for regime C; splitting it into a chain of
   panels multiplies the update-stack footprint and adds levels (issue #48).
5. **Assembly-tree partition** into regimes: leaf subtrees whose live front set
   fits a local-memory budget (a few `Val`-selected budgets, since KA local
   memory is static), mid-level fronts binned by (rows, cols) size class, root
   fronts above a vendor threshold. Per-level launch lists per bin.
6. **Static memory layout**: offsets for every front and contribution block,
   update-stack high-water mark per level, level chunking under a memory
   budget; no allocator at run time.
7. **Device maps**: destination of every `nzVal` entry inside its front;
   child→parent relative indices for extend-add; gather lists for the solve;
   subtree descriptors for regime A; all as `INT` device vectors.
8. **Storage layout**: all L panels (and U panels for `G`) in one device array,
   each panel a contiguous column-major `f×w` block (leading dimension `f`) so a
   `reshape(view(...))` is a valid strided matrix for vendor BLAS; D separate;
   batch stride for uniform batches. The front's first `w` columns *are* the
   factor panel; only the contribution block lives on the update stack.

### 2.4 Numeric phase (device)

Per subtree batch (A), per level and bin (B), or per root front (C):

```
assemble:  zero panel+CB; scatter A via map; extend-add children's CBs
factor:    F11 = L11 L11ᵀ   |   L11 D L11ᵀ (BK 1×1/2×2 in-block)   |   P L11 U11 (in-block)
trsm:      F21 ← F21 L11⁻ᵀ (D⁻¹)                     [LU: also F12 ← L11⁻¹ P F12]
update:    F22 ← F22 − F21 (D) F21ᵀ                   [LU: F22 ← F22 − F21 F12]
```

In regimes A and B all four steps run inside one KA kernel per workgroup, with
pivot decisions, perturbations and sign counts accumulated in local memory and
written to per-front stats arrays. In regime C the vendor routine factors the
fully-summed block; its diagonal is checked on the device and only blocks
violating the pivot tolerance are redone by the KA kernel with the perturbation
policy (vendor LAPACK cannot apply `pivot_epsilon`/`pivot_sign` inline). Vendor
`sytrf` is complex *symmetric*, not Hermitian, so `"H"` root fronts use the KA
kernel. The whole phase is a fixed sequence of launches with no allocation and
no host synchronization (§3.9).

Kernel design rules (from the KA constraints in §2.7): 1-D workgroups,
shared-memory reductions, no subgroup/shuffle primitives in the portable path;
warp-level fast paths may be added later behind the CUDA/ROCm extensions once
KernelInterface/KernelIntrinsics stabilize.

### 2.5 Solve phase

Permutation and scaling kernels, then forward/diagonal/backward sweeps on the
same subtree/level schedule: regime A subtrees solved by one workgroup each,
regime B/C fronts by batched TRSV/GEMV (TRSM/GEMM for multiple RHS; vendor on
root fronts). The forward sweep has write conflicts between fronts of one
level: default `Atomix.@atomic` accumulation; `deterministic_mode = 1` and
backends without float atomics use the front-based update-stack variant. The
backward sweep is conflict-free.

Solve latency matters as much as factorization for IPM loops (MA57 averages
about six refinement solves per factorization on OPF matrices), so two
accelerations are planned: partitioned-inverse diagonal blocks for
refinement-heavy loops (`solve_alg = "algo1"`, M11) and a sync-free
ready-flag sweep in the CUDA/ROCm extensions, selected only where the hardware
guarantees forward progress (M11).

Refinement: plain IR with a KA CSR SpMV residual, or FGMRES with the factors as
preconditioner (Krylov.jl extension). Refinement runs in the input precision
even when factors are Float32 (§3.8).

### 2.6 Backends, extensions and the dense layer

The dense layer is a small interface (`gemm!`, `syrk!/herk!`, `trsm!`, `potrf!`,
`sytrf!`, `getrf!`, `laswp!`, batched variants) with:

1. **Generic LinearAlgebra entry points** each backend already routes to its
   vendor library on its own array type: `mul!`, triangular `ldiv!`/`rdiv!`,
   `cholesky!`, `lu!`. No code in our extensions.
2. **Vendor-specific calls** in each extension where no generic entry point
   exists: `syrk!/herk!`, batched `gemm/trsm/getrf/potrf`, `sytrf!`, `laswp!`.
3. **KA fallback kernels** (the regime A/B kernels and tiled large-front
   kernels) for everything a backend lacks, so every feature works everywhere.

Expected coverage, to be verified by the M0 capability audit (which also checks
strided `reshape(view(...))` acceptance, float atomics, and the reported MPS LU
defect above 128×128 on macOS 27):

| Op | CUDA (cuBLAS/cuSOLVER) | AMDGPU (rocBLAS/rocSOLVER) | oneAPI (oneMKL) | Metal (MPS) |
| --- | --- | --- | --- | --- |
| gemm, trsm, syrk/herk | yes | yes | yes | gemm yes (Float32); trsm via `MPSMatrixSolveTriangular`; syrk → gemm |
| potrf, getrf | yes | yes | yes | LU/Cholesky in MPS (audit; LU bug report) |
| sytrf (Bunch–Kaufman) | yes (real/complex-symmetric) | yes | yes | no → KA |
| batched gemm/trsm | yes (pointer + strided) | yes | gemm yes, trsm audit | no → KA |
| batched getrf/potrf | yes | audit | getrf yes, potrf audit | no → KA |
| Float64 / complex | yes | yes | hardware-dependent / yes | no / no |

Extensions: `…CUDAExt`, `…AMDGPUExt` (dense bindings, sparse adapters, pinned
memory, graph capture, sync-free solve), `…OneAPIExt`, `…MetalExt` (dense
bindings, adapters), `…MetisExt` (ND), `…KrylovExt` (FGMRES-IR). Core depends
on KernelAbstractions, GPUArrays(Core), Adapt, Atomix, LinearAlgebra,
SparseArrays, CliqueTrees.

### 2.7 Portability constraints (hard, from the research)

| Constraint | Consequence |
| --- | --- |
| KA has no stable portable shuffle/ballot/subgroup reduction (KernelInterface 0.4 and KA 0.10 still moving; KernelIntrinsics.jl is a proof of concept tested on CUDA/ROCm only). | Portable kernels use 1-D workgroups and shared-memory reductions. Warp-level variants only behind backend extensions, later. |
| Metal (and likely Intel) guarantee no inter-workgroup forward progress or launch order. | Sync-free SpTRSV and persistent DAG schedulers are CUDA/ROCm-only, selected by a capability check. Subtree + level scheduling is the portable baseline. |
| No dynamic parallelism, no graphs in KA; one launch per level is the portable floor. | Subtree fusion and level merging are first-class analysis goals. CUDA/HIP graph capture of refactorize+solve through the extensions. |
| Metal has no Float64. | `factor_precision = Float32` with Float64 refinement on the CPU or in double-single arithmetic; acceptable for condensed/well-conditioned systems only. |
| Float32 atomics have codegen regressions on CUDA; atomics cost everywhere. | No atomics in assembly or extend-add (owner-pull, deterministic). Atomics only in the default forward solve, with an atomic-free variant. |

---

## 3. Design details

### 3.1 Public API: two thin layers

**Handle-style layer** (mirrors CUDSS.jl; plain arrays, strings and scalars
only, so a C shim can wrap it later):

```julia
solver = DirectSolver(A, "S", 'U'; index='O')            # ≅ CudssSolver
solver = DirectSolver(rowPtr, colVal, nzVal, "G", 'F')    # uniform batch if nzVal is longer / 2-D
bsolver = BatchedDirectSolver([A1, A2, …], "SPD", 'L')    # ≅ CudssBatchedSolver

setparam!(solver, "ir_n_steps", 2)        # ≅ cudss_set, same strings
getparam(solver, "inertia")               # ≅ cudss_get, returns plain values
getparam!(buf, solver, "diag")            # non-allocating variant
update!(solver, A) ; update!(solver, rowPtr, colVal, nzVal)      # ≅ cudss_update
execute!("analysis", solver, x, b; asynchronous=true)            # ≅ cudss(phase, …)
```

Strings map to integer enums internally; status is reported through exceptions
in Julia and would become return codes in a C shim. `x`, `b` are backend arrays
of shape `(n,)`, `(n, nrhs)` or `(n, nrhs, nbatch)` (or strided vectors).
`MatrixDescriptor(T, n; nbatch)` + `update!(desc, buffer)` keeps
allocation-free buffer swapping. Defaults in this layer follow cuDSS
(`ir_n_steps = 0`, cuDSS pivot epsilons), because MadNLP drives its own
refinement loop and expects the inertia of the factorization it asked for.

**Julia-native layer**: `lu`, `ldlt`, `cholesky` (+ `!` variants), `ldiv!`, `\`,
`Symmetric`/`Hermitian`, `inertia`, `logabsdet`, `diag`, implemented on top of
the handle-style layer; refinement is on by default here (two steps with early
exit on `ir_tol`), a documented difference from cuDSS for standalone users.
Nothing is duplicated between the layers. MadNLPGPU implements its
`AbstractLinearSolver` (`factorize!`, `solve_linear_system!`, `inertia`) on the
handle-style layer.

### 3.2 Types

```julia
abstract type AbstractDirectSolver{T,INT} <: LinearAlgebra.Factorization{T} end
struct Symbolic{INT, VI<:AbstractVector{INT}}   # host tree data, schedule, device maps
struct Numeric{T, VT<:AbstractVector{T}}        # panels, D, update stack, stats arrays
mutable struct DirectSolver{T,INT,…} <: AbstractDirectSolver{T,INT}
    A::CSR{…}; structure::Symbol; view::Char; options::Options
    symbolic::Union{Nothing,Symbolic}; numeric::Union{Nothing,Numeric}
    fresh_factorization::Bool; nbatch::Int; backend::KA.Backend
end
```

`setparam!` validates names against the two tuples CUDSS.jl exports plus the
§1.7 additions.

### 3.3 Pivoting

* `SPD/HPD`: none; first non-positive pivot → `info`.
* `S`/`H`: Bunch–Kaufman 1×1/2×2 pivots restricted to the fully-summed block of
  each front (`pivot_type = 'B'`, default), `pivot_threshold` for acceptance;
  if no acceptable pivot exists, static perturbation with the `pivot_sign`
  policy; per-front flags for perturbed, tiny and 2×2 pivots; no delayed pivots
  in v1. Vendor `sytrf` only on `S` root fronts, with post-check.
* `G`: in-block threshold partial pivoting (regimes A/B) or vendor `getrf` +
  `laswp` (regime C) with perturbation by post-check (SuperLU_DIST GESP style);
  refinement recovers accuracy.
* Global column/row pivoting with dynamic fill (cuDSS `GLOBAL_COL/ROW` +
  COLAMD/BTF): not planned; it does not batch on a GPU.
* Later (M13): a posteriori threshold pivoting (SSIDS v2) with optional delayed
  pivots and host re-analysis when delays occur, for the unreduced K2 systems
  where static pivoting is not enough.

Inertia: a 1×1 pivot contributes its sign, a 2×2 block with negative determinant
contributes one of each sign; per-front reduction plus global sum; `inertia`
keeps the cuDSS pair, `pivot_stats` adds n₀, n_perturbed and n_2×2.

### 3.4 Determinism

Assembly (scatter of A and extend-add) is done by the owner workgroup of each
parent front pulling from its children in a fixed order: no atomics, bitwise
reproducible by construction. Only the default forward solve uses atomics.

### 3.5 Uniform batch

Every array gets a batch stride (`nzVal`, panels, D, update stack, RHS, stats);
kernels take the batch index as one more grid dimension; `ubatch_index`/
`ubatch_mask` restrict the range; pivots and `info` are per member; `nrhs > 1`
works at any batch size.

### 3.6 Schur complement mode

Schur indices are ordered last and form the root supernode; the rest is ordered
fill-reducingly. Factorization stops before the root front, which then *is* `S`
(dense); its symbolic CSR pattern is known, so `schur_shape` and the sparse
export are exact. `solve_fwd_schur`/`solve_diag`/`solve_bwd_schur` sweep the
non-root supernodes with the condensed RHS in the last `n_s` entries.

### 3.7 Non-uniform batch

Systems packed into one block-diagonal matrix; per-system orderings computed in
parallel on the host, concatenated into a forest; values and RHS copied into
packed buffers on `update!`. Same `nrhs` across systems in v1.

### 3.8 Mixed precision

`factor_precision = Float32` stores panels, D and the update stack in Float32
while the input, residual and refinement run in the input precision. Plain IR
is expected to stall on ill-conditioned late-IPM systems, so FGMRES-IR
(`ir_mode = "fgmres"`) is the recommended companion. Float64 stays the default
everywhere except Metal, where Float32 factors are the only option.

### 3.9 Refactorization contract

Everything from analysis is reused: permutation, scaling, etree, supernodes,
front sizes, extend-add maps, layout, schedule. The numeric phase and the solve
are pure "values in, factors/solution out" kernel sequences with no allocation
and no host synchronization, so the CUDA and ROCm extensions can capture the
refactorize+solve sequence in a graph and replay it per IPM iteration.

---

## 4. Repository layout

```
Project.toml                  deps: KernelAbstractions, GPUArrays, GPUArraysCore, Adapt, Atomix,
                              LinearAlgebra, SparseArrays, CliqueTrees
                              weakdeps: CUDA, AMDGPU, oneAPI, Metal, Metis, Krylov
src/SparseDirectSolver.jl
src/types.jl                  enums, Options, Symbolic, Numeric, DirectSolver, BatchedDirectSolver
src/options.jl                setparam!/getparam tables (CUDSS.jl names + §1.7)
src/matrix.jl                 CSR container, adapters, view/index handling, MatrixDescriptor
src/symbolic/pattern.jl       symmetrize, expand, dedup, rebase, full-pattern map
src/symbolic/ordering.jl      CliqueTrees orderings, cost model, user perm, Schur-constrained, ND-tree I/O
src/symbolic/etree.jl         elimination tree, postorder, column counts, nnz/flops
src/symbolic/supernodes.jl    fundamental supernodes, GPU-tuned amalgamation
src/symbolic/schedule.jl      assembly tree, subtree partition, size bins, level lists, chunking
src/symbolic/layout.jl        static offsets, update-stack high-water mark, memory estimates
src/symbolic/maps.jl          scatter maps, relative indices, subtree descriptors
src/dense/interface.jl        dense-op interface + dispatch (generic → vendor → KA)
src/dense/fallback/*.jl       KA tiled potrf, ldlt, getrf, trsm, gemm, syrk, laswp (regime C fallback)
src/numeric/subtree.jl        regime A fused subtree kernels (chol / ldlt-bk / lu)
src/numeric/front.jl          regime B fused per-front kernels, size-binned
src/numeric/root.jl           regime C driver (vendor + post-check + KA redo)
src/numeric/assembly.jl       KA scatter / extend-add kernels
src/numeric/factorize.jl      phase driver, stats reduction, info/inertia/pivot_stats
src/numeric/refactorize.jl    src/numeric/schur.jl
src/solve/permute.jl  sweeps.jl  partinv.jl  refinement.jl
src/matching/mc64.jl          weighted bipartite matching + scalings (host)
src/batch/uniform.jl  nonuniform.jl
src/generic.jl                LinearAlgebra interface
src/hybrid.jl                 (M12)
ext/SparseDirectSolverCUDAExt.jl  …AMDGPUExt.jl  …OneAPIExt.jl  …MetalExt.jl  …MetisExt.jl  …KrylovExt.jl
test/                         CPU backend always + every GPU backend present; ported CUDSS.jl tests
bench/                        harness (NREL opf_matrices, pglib-opf KKT/condensed dumps, CUTEst subset),
                              cuDSS/CHOLMOD/MA57 baselines, per-phase timings, supernode statistics
docs/
```

---

## 5. Milestones

Every milestone ends with the test suite green on the KA CPU backend and on
CUDA, AMDGPU, oneAPI and Metal (Float32), plus a `CHANGELOG.md` entry. Sizes are
rough: S ≈ days, M ≈ 1–2 weeks, L ≈ 3–4 weeks, XL ≈ more.

| # | Milestone | Size | Definition of done |
| --- | --- | --- | --- |
| M0 | Scaffolding, audit, baselines | M | Package, four backend extensions with CSR adapters, CI matrix (GitHub Actions CPU; Buildkite juliagpu queue for CUDA/AMDGPU/oneAPI/Metal; self-hosted `kkt`), `Options` with the full name tables, error types, Aqua. **Capability audit** script filling the §2.6 table per backend and eltype (kept as a test). **Benchmark harness**: NREL opf_matrices, pglib-opf KKT and condensed matrices dumped from MadNLP, a CUTEst subset; cuDSS analysis/factorization/solve times, `flops`, supernode statistics and MA57 refinement counts recorded as the baseline every later milestone is measured against. |
| M1 | Symbolic engine | L | Pattern, orderings with the AMD/ND cost model, etree, column counts, GPU-tuned amalgamation, subtree partition, size bins, level lists, static layout, device maps, `memory_estimates`, `flops`, `lu_nnz`, `nsuperpanels`, ND-tree I/O, `max_lu_nnz`. Validated against CHOLMOD's nnz(L) and etree; supernode/level statistics compared with the cuDSS baseline on the harness matrices. A CPU reference numeric factorization (plain Julia, same layout) for testing. |
| M2 | Numeric kernels | L | Regime A fused subtree kernels, regime B fused per-front kernels (Cholesky first, LDLᵀ/LU hooks), regime C vendor bindings in all four extensions with KA tiled fallbacks, assembly/extend-add kernels. Unit tests against the CPU reference on every backend; per-bin micro-benchmarks deciding where vendor batched calls replace regime B. |
| M3 | SPD/HPD end-to-end (**v0.1**) | M | All phases, single and multi RHS, `info`, `cholesky`/`cholesky!`/`ldiv!`/`\`, `Hermitian` wrapper, async flag; subtree/level solve sweeps. Ported `test_cudss.jl` subsets pass on all backends. **Target**: within 1.5× of cuDSS Cholesky refactorization+solve on condensed pglib-opf systems on CUDA, running on AMD and Intel. |
| M4 | Symmetric indefinite LDLᵀ/LDLᴴ (**MadNLP-ready**) | L | In-front Bunch–Kaufman, `pivot_threshold`, `pivot_epsilon(_alg)`, `pivot_sign`, `inertia`, `pivot_stats`, `npivots`, `diag`, `solve_diag`, `ldlt`/`ldlt!`; vendor `sytrf` with post-check on root fronts. MadNLPGPU gets a `SparseDirectSolver` option; validated on OPF/ExaModels KKTs (MadNLP K2/K2r, MadIPM, MadNCL settings) against the cuDSS path, including refinement counts against MA27/MA57. |
| M5 | Solve extras | M | IR (`ir_n_steps`, `ir_tol`, actual steps), FGMRES-IR extension, all solve sub-phases, `solve_mode`, `perm_*` getters, `user_host_interrupt`, logging. |
| M6 | Uniform batch (**v1 cut line**) | M | Strided/3-D arrays, `ubatch_size/index/mask`, per-member `info`/stats, `nrhs > 1` at any batch size, generic API auto-detect, strided-batched vendor calls on root fronts. MadNLP two-stage Schur KKT and ExaModels multi-scenario models run on it. |
| M7 | General LU (LDU) | L | Symmetric-pattern multifrontal LU with in-block pivoting and GESP-style perturbation, `lu`/`lu!`, `perm_row/col`, transpose/adjoint `solve_mode`; optional up-looking row-per-workgroup kernel for extremely sparse factors (Ginkgo style) if the harness shows a gap. |
| M8 | Schur complement mode | M | `schur_mode`, `user_schur_indices`, `schur_shape`, dense and CSR `schur_matrix`, `solve_*_schur`; `test_schur_cudss.jl` ported and enabled. |
| M9 | Matching and scaling | L | MC64-style jobs (max product first), `perm_matching`, `scale_row/col`, inertia correct with matching. |
| M10 | Non-uniform batch | M | `BatchedDirectSolver`, block-diagonal packing, `test_nonuniform_batch_cudss.jl` ported. |
| M11 | Performance | XL, ongoing | Partitioned-inverse solve option, sync-free sweeps and graph capture in the CUDA/ROCm extensions, level merging, amalgamation and bin tuning per backend, warp-level fast paths once KernelInterface stabilizes, benchmark tracking vs cuDSS, CHOLMOD and MA57. **Target**: match or beat cuDSS refactorization+solve on ACOPF KKT matrices on CUDA; parity ratios reported for AMD/Intel. |
| M12 | Hybrid modes | L | Host-resident panels (`hybrid_memory_mode`, limits, pinned memory), CPU-backend execution of small levels (`hybrid_execute_mode`, `host_nthreads`). |
| M13 | Robustness extras | L | A posteriori threshold pivoting with optional delayed pivots and host re-analysis; `factor_precision = Float32` + FGMRES-IR mixed precision (also the Metal Float64 story); CPU MA27/MA57 fallback guidance documented for hard nonconvex NLPs. |

Not planned: multi-GPU (MG, MGMN), global pivoting, double-double, distributed
input. The assembly tree keeps per-subtree ownership so MG can be added later
by splitting the ND top level.

---

## 6. Risks and mitigations

| # | Risk | Mitigation |
| --- | --- | --- |
| R1 | Tiny-front latency dominates on KKT matrices (the failure mode of SSIDS/STRUMPACK/SuperLU on ACOPF). | Regime A subtree fusion and regime B fused per-front kernels; GPU-tuned amalgamation; level merging; measured against the cuDSS baseline from M0 onward. |
| R2 | Vendor coverage gaps (batched ops on oneAPI/Metal, no `hetrf`, no Float64 on Metal). | Vendor calls are confined to root fronts; regimes A/B are KA everywhere; mixed precision for Metal. |
| R3 | KA subgroup/shuffle API churn (KA 0.10, KernelInterface 0.4). | Portable kernels avoid subgroup primitives entirely; warp-level variants are optional and extension-local. |
| R4 | No forward-progress guarantees on Metal/Intel. | Sync-free algorithms only behind a capability check in CUDA/ROCm extensions; subtree+level is the baseline. |
| R5 | Local-memory budgets: fronts or subtrees that do not fit a `Val` budget. | Analysis assigns regimes by measured budgets; anything that does not fit regime A falls to B, anything above B's largest bin to C. |
| R6 | Vendor LAPACK blocks cannot apply `pivot_epsilon`/`pivot_sign` inline. | Post-check + KA redo of offending blocks (rare); root fronts are few. |
| R7 | Static pivoting reports the inertia of A+E; silent accuracy loss on badly scaled matrices (a reported cuDSS failure mode). | `pivot_stats` and `npivots` always available; refinement with residual check; matching + scaling (M9); APTP/delayed pivots (M13); CPU fallback guidance. |
| R8 | Level-set processing raises peak memory vs postorder. | Peak computed at analysis; chunking under a budget; hybrid memory later. |
| R9 | No global pivoting → highly unsymmetric indefinite `G` matrices lose accuracy. | Static pivoting + IR (cuDSS default too); matching (M9); `info` never hides failure. |
| R10 | Host ordering quality/speed (no multithreaded ND like cuDSS). | Cost-model choice AMD vs ND; Metis for ND; ND-tree caching; threads for non-uniform batches. |
| R11 | Mixed precision stalls on ill-conditioned late-IPM systems. | FGMRES-IR as the companion; Float64 default outside Metal. |
| R12 | 32-bit overflow in nnz(L). | `INT` type parameter end to end; host uses `Int` and checks before casting. |
| R13 | API surface (26 config + 27 data names + §1.7). | Names kept verbatim; each implemented, accepted-and-ignored with a warning, or raises "not supported"; never silently misinterpreted. |

---

## 7. Verification strategy

* **Oracle**: the KA CPU backend runs the same code (fused kernels + CPU BLAS),
  plus the plain-Julia CPU reference factorization from M1, so every GPU test
  also runs on CPU and failures are debuggable in plain Julia.
* **Backends**: every test runs on CUDA, AMDGPU, oneAPI and Metal (Float32) in
  CI; the M0 capability audit documents which implementation each op took.
* **Reference solvers**: SparseArrays/CHOLMOD (`cholesky`, `ldlt`), UMFPACK
  (`lu`) for residuals, inertia (vs `eigvals` on small matrices), nnz(L),
  permutations; CUDSS.jl when CUDA is available, through a shared harness
  parametrized on the solver module (`Random.seed!(666)` kept); MA57 refinement
  counts as the robustness yardstick for LDLᵀ.
* **Ported tests**: `test_cudss.jl`, `test_uniform_batch_cudss.jl`,
  `test_nonuniform_batch_cudss.jl`, `test_schur_cudss.jl`; documented cuDSS bugs
  become regression tests (multi-RHS uniform batch, deterministic mode in
  batches, inertia with matching, silently perturbed pivots).
* **Matrices**: NREL opf_matrices, pglib-opf KKT and condensed systems dumped
  from MadNLP (K2, K2r, LiftedKKT, HyKKT), a CUTEst subset, SuiteSparse via
  MatrixDepot.jl, the hand-built Schur examples from the CUDSS.jl docs.
* **Performance**: `bench/` compares against cuDSS (through CUDSS.jl), CHOLMOD
  and MA57 per phase, with supernode and level statistics; results tracked per
  milestone against the M0 baseline and the M3/M11 targets.

---

## 8. References

* `RESEARCH.md` — state of the art, workload evidence, portability hazards, and
  the sources behind the design choices above.
* CUDSS.jl (`../CUDSS.jl`): `src/interfaces.jl`, `src/types.jl`, `src/generic.jl`,
  `docs/src/*.md`, `test/*.jl` — the API contract mirrored here.
* cuDSS documentation: Advanced Features, Release Notes 0.1–0.8, Types/Functions.
* Świrydowicz et al., GPU-resident sparse direct solvers for ACOPF (IJEPES 2024)
  and the GPU linear-solver review (Parallel Computing 2022) — why generic
  supernodal GPU solvers lose on KKT matrices and refactorization wins.
* Pacaud, Shin, Montoison, Schanen, Anitescu — condensed-space IPMs on GPUs;
  MadIPM and MadNCL papers — the regularization and pivot-epsilon settings the
  solver must serve.
* Rennich, Stosic, Davis — CHOLMOD subtree algorithm; ShyLU/Tacho progress
  report (2025) — level-set scheduling and SpTRSV variants; Karsavuran et al. —
  amalgamation thresholds; Duff, Hogg, Lopez — a posteriori threshold pivoting
  (SSIDS v2); Ginkgo sync-free LU/Cholesky; SuperLU_DIST GESP and mixed
  precision.
* KernelAbstractions.jl, KernelInterface 0.4 discussion, KernelIntrinsics.jl /
  KernelForge.jl — the portable-primitive situation.
* CliqueTrees.jl, LinearSolve.jl `SupernodalLUFactorization`, PureUMFPACK.jl —
  pure-Julia CPU prior art; MadNLPGPU `cudss.jl` — downstream parameter usage;
  KrylovPreconditioners.jl — KA kernels and extension layout in the ecosystem.

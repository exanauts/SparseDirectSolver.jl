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
| CSC input (MadNLP passes `colPtr/rowVal` as CSR of Aᵀ with view `'U'`) | port | Treated as CSR of Aᵀ with the view flipped; no copy. For complex Hermitian input the stored matrix is `conj(A)`, which the conjugated solve of `solve_mode` handles (T16). |
| Structures `"G" "S" "H" "SPD" "HPD"` | port | Same strings. |
| Views `'L' 'U' 'F'` | port | For `'F'` on symmetric types only one triangle (the lower) is read, as in cuDSS. For `"G"` the view is ignored and the full matrix is read, as in cuDSS (issue #84; until that fix lands `"G"` requires `'F'`). |
| Index base `'Z' 'O'` | port | Handled in the symbolic maps; values never copied. |
| `T ∈ {Float32, Float64, ComplexF32, ComplexF64}` | port | Generic in `T`. Metal: Float32/ComplexF32 factors (see mixed precision, §3.8). |
| Double-double (`CUDSS_R_64F_64F`) | not planned | |
| `offsetType` (Int64 row pointers, Int32 columns) | defer | `CSR{T,INT,VI,VT}` has one index type for `rowptr` and `colval`; mixed types raise `InvalidValueError` (T02). A separate `rowptr` type parameter is mechanical to add if a consumer needs it. |
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
`solve!` are thin wrappers. As implemented: `"solve"` is the sweeps plus
refinement and the six sub-phases compose to it bitwise (T16); `"solve_diag"`
is the identity for Cholesky and LU outside Schur mode; in Schur mode only
`"solve_fwd_schur"`, `"solve_diag"` and `"solve_bwd_schur"` exist and the other
solve phases raise `NotSupportedError` (T20, §3.6).

### 1.3 Configuration parameters (`cudss_set` → `setparam!`)

| Name | Plan | Portable semantics |
| --- | --- | --- |
| `reordering_alg` | reinterpret | `"default"`: automatic choice between ND (Metis extension) and AMD by the cost model of §2.3; `"algo3"` AMD, `"algo4"` ND, `"algo5"` natural. `"algo1"/"algo2"` (BTF_COLAMD/COLAMD with global pivoting): accepted, falls back to symmetric-pattern LU with a one-time warning (§3.3). |
| `factorization_alg` | reinterpret | `"default"` auto; `"algo1"` runs the fronts above the regime-B bins on the KA tiled kernels (no vendor calls); `"algo2"` sends every front outside regime A to the regime-C path. Mirrors cuDSS keeping a separate algorithm for very sparse factors (T07, T10). |
| `solve_alg` | reinterpret | `"default"` level-batched; `"algo1"` partitioned-inverse diagonal blocks (M11); sync-free variant selected automatically on CUDA/ROCm when available. |
| `matching_alg` | port (T21) | `"default"` off; algo1..5 = MC64 jobs 1..5; algo6 auto → job 5. Host MC64 at reordering, from the first batch member, reused by refactorizations. `"G"` factors `(Dr A Dc)[:, q]`; the symmetric structures are scaled only (`D A D`, inertia of `A`) and take the 2×2 pivot pairs from the matching cycles for jobs 5/6. Not combined with Schur mode. |
| `solve_mode` | port (M5) | 0 = A, 1 = Aᵀ, 2 = Aᴴ. |
| `ir_n_steps`, `ir_tol` | port (T16) | `ir_tol` honored (cuDSS ignores it); the early-exit test costs one host synchronization per step and runs only when `ir_tol > 0`. Default `ir_n_steps = 0` in the handle layer (cuDSS parity; MadNLP runs its own refinement loop); the LinearAlgebra layer sets `ir_n_steps = 2` and keeps `ir_tol = 0` (§3.1). With `ir_mode = "fgmres"`, `ir_n_steps` is the maximum number of FGMRES iterations (T18). |
| `pivot_type` | port (partial) | `'A'` auto, `'N'` none, `'D'` diagonal, `'L'` local block (treated as Bunch–Kaufman inside the block), `'B'` Bunch–Kaufman 1×1/2×2 (default for `S`/`H`). For `"G"`, `'N'`/`'D'` mean diagonal pivots. `'C'/'R'` global: not planned. |
| `pivot_threshold` | port (T14, T19) | Threshold for in-front pivot acceptance (LDLᵀ and LU), tested against the whole remaining front column. With `pivot_pairs = "default"` it also bounds the partner acceptance of the 2×2 pivot pairs at analysis time (§2.3 step 2), so a value set after `"analysis"` changes the in-front test but not the pairs. |
| `pivot_epsilon`, `pivot_epsilon_alg` | port (M4) | Static or scaled perturbation. Defaults match cuDSS: 1e-5 (Float32), 1e-13 (Float64). |
| `max_lu_nnz` | port | Accepted and validated; the check against `nnz_stored` after the symbolic analysis is not wired in yet (T07/T13 open issue). |
| `hybrid_memory_mode`, `hybrid_device_memory_limit` | defer (M12) | Host-resident panels streamed per level. |
| `use_cuda_register_memory` | reinterpret (M12) | Pinned host memory through the backend extension. |
| `hybrid_execute_mode` | defer (M12) | Small levels on the KA CPU backend. |
| `host_nthreads` | reinterpret | Julia threads for the host symbolic phase / CPU backend. |
| `nd_nlevels`, `nd_ubfactor` | port (T05) | ND is `METIS_NodeND` through CliqueTrees; `nd_ubfactor` is passed through, `nd_nlevels` is read as cuDSS documents it, a *minimum* number of dissection levels, which NodeND's full recursion meets (a level-capped ND was 4–13× slower with more fill). It becomes meaningful with the partition-tree export (T24). |
| `ubatch_size`, `ubatch_index` | port (M6) | Uniform batch. |
| `use_superpanels` | reinterpret | Supernode amalgamation on/off. |
| `device_count`, `device_indices` | not planned | Raise "not supported". |
| `schur_mode` | port (M8) | |
| `deterministic_mode` | port | Assembly is deterministic by construction (§3.4); the flag switches the forward solve to its atomic-free variant. Complex `T` always uses that variant (no complex atomics on any backend, issue #36). |

### 1.4 Data parameters (`cudss_get` → `getparam` / `getparam!`)

| Name | Plan | Notes |
| --- | --- | --- |
| `info` | port | 0 ok; `k > 0` first failed pivot (1-based, original numbering); vector for batches; settable (reset before refactorization, as CUDSS.jl does). |
| `lu_nnz` | port | From the symbolic phase. |
| `npivots` | port | Perturbed pivots (static perturbation yields the inertia of A+E, so this must be read together with `inertia`). Valid when `info == 0`; after a failed factorization the per-front counts past the failure are stale. |
| `inertia` | port | `(npos, nneg)` from D including 2×2 blocks; correct under matching (the symmetric structures are scaled, never permuted, by the matching). Cholesky reports `(n, 0)`, LU `(0, 0)`. On a uniform batch, the members outside the last `ubatch_index`/`ubatch_mask` keep the counts of their last reduction. |
| `perm_reorder_row/col`, `perm_row/col`, `perm_matching` | port | Returned as 1-based vectors; `getparam!(buf, …)` writes into a host or device buffer, replacing the C-style set-buffer-then-get protocol. `perm_row` is the composed local pivot order of LU; on a uniform batch it is batch member 1, as cuDSS returns vector outputs (issue #92). With CSC input the permutations and scalings refer to the stored `Aᵀ`. |
| `diag` | port | Diagonal of D (LDLᵀ, in factor order) / U (LU) / L (Cholesky); of the scaled matrix when matching is on. A uniform batch returns the members concatenated (`n·nbatch`), where cuDSS returns member 1. |
| `scale_row/col` | port (M9) | |
| `user_perm` | port (M1) | Host or device vector. |
| `memory_estimates` | port | `Int64[16]`, slots 1–6 as cuDSS (permanent/peak device, host, hybrid), slot 11 the per-front statistics; the matching buffers are not counted yet (T21 follow-up). |
| `hybrid_device_memory_min` | defer (M12) | |
| `nsuperpanels` | port | Supernodes after amalgamation. |
| `user_schur_indices`, `schur_shape`, `schur_matrix` | port (T20) | Dense or CSR export of the exact symbolic pattern (on `A + Aᵀ` for `"G"`, so explicit zeros are possible); symmetric structures return one triangle with the diagonal and a correct nnz. `schur_matrix` is set as a Julia destination (dense, `MatrixDescriptor`, `CSR`, `(csr, view)`) and filled by `getparam`. |
| `user_nd_partition_tree`, `nd_partition_tree` | port (M1) | Same binary-tree encoding as cuDSS, so orderings can be cached between runs. Extra: `"etree"`, `"supernodes"`. |
| `user_host_interrupt` | port (T16) | Polled (host read, no synchronization) before every launch group of the factorization, at the start of the analysis phases and before every refinement step. An interrupted factorization returns the solver to the analyzed state. |
| `ir_n_steps` (data) | port (T16) | One name for the configured value and the steps performed: after a solve `getparam` returns the steps performed (also on interrupt), otherwise the configured value. |
| `ubatch_mask` | port (M6) | |
| `flops` | port (M1) | From supernode sizes. |
| `comm_device`, `comm_host` | not planned | MGMN. |

### 1.5 Generic LinearAlgebra interface

`lu`, `lu!`, `ldlt`, `ldlt!`, `cholesky`, `cholesky!`, `ldiv!`, `\`, the
`Symmetric`/`Hermitian` wrappers, uniform-batch auto-detection via
`length(nzVal) ÷ length(colVal)`, and the `fresh_factorization` flag that turns
`lu!` et al. into refactorizations: ported unchanged. Added: `inertia(F)`,
`logabsdet(F)`, `diag(F)`, `nnz(F)`. The `Symmetric`/`Hermitian` wrappers exist
for the backend sparse types of the extensions (`CuSparseMatrixCSR`); the
in-package `CSR` is not an `AbstractMatrix`, and on the CPU backend the generic
layer takes a `CSR` (overriding `cholesky(::SparseMatrixCSC)` would be piracy
of CHOLMOD's method). The CUDA methods carry the same signatures as CUDSS.jl,
so loading both packages overwrites one set (T13 open issue, relevant while
MadNLPGPU migrates).

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
| `amalgamation` | config | Relaxed-amalgamation parameters `(max_width, zero_fraction, min_width)`; `max_width` caps merging only (§2.3 step 4). |
| `schedule` | config | `"auto"`, `"subtree+level"` (portable baseline), `"syncfree"` (CUDA/ROCm only, not implemented yet: raises). |
| `pivot_pairs` | config | `"default"`: for `"S"`/`"H"` the analysis pairs every candidate row whose pivot is structurally zero in the ordering with a partner, so that in-front Bunch–Kaufman can form the 2×2 block (§2.3 step 2); `"all"` pairs every candidate (about 2× nnz(L) on KKT systems); `"none"`. Candidates are decided from the values present at analysis time: an all-zero `nzval` gives no pairs. |
| `pivot_pair_tolerance` | config | Relative tolerance below which a diagonal counts as a pair candidate (default `1e-6`). |
| regime thresholds | `Options` keywords only | `regime_c_width`, `regime_c_rows`, `subtree_budgets`, `subtree_parallelism`, `subtree_max_fronts`, `memory_budget` select the three regimes (§2.3 step 5); they are keywords of `Options(...)`, not `setparam!` strings, until there is a reason to expose them. Measured on the 78k-bus condensed KKT (PR #107): the width boundary 64 of regime B is right, `regime_c_rows` matters as much as the width, `subtree_budgets = [16384]` beats the 48 KiB class through occupancy, and `subtree_max_fronts` is neutral there. |

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
       │    regime C  Cholesky: vendor potrf + trsm + syrk/herk (extensions; KA fallback)
       │              LDLᵀ: KA blocked in-front pivoting + vendor trsm/gemm; LU: KA kernel (BLAS-3 in T25)
       │    + KA kernels for assembly, extend-add, gather/scatter, permutation
       ├─ solve (device): permute/scale → subtree/level fwd → diag → bwd → unpermute → IR/FGMRES
       └─ extensions: CUDA, AMDGPU, oneAPI, Metal (dense + sparse adapters + graph capture
                      + sync-free solves where the hardware allows), Metis, Krylov
```

Only the pattern (`rowPtr`, `colVal`) travels to the host, during analysis;
for `"S"`/`"H"` and for matching the values of the first batch member are
copied to the host once at analysis as well (2×2 pivot pairs, MC64). Values
otherwise stay on the device; refactorization is a fixed kernel sequence.
Implemented extensions: CUDA, Metis, Krylov; AMDGPU, oneAPI and Metal are T23.

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
  workgroup per front doing the assembly (zero, scatter of A, owner-pull
  extend-add), the factorization of the fully-summed block (`F₁₁` in local
  memory for the wider width classes), TRSM, SYRK/GEMM Schur update into the
  packed contribution block, all in one fused kernel. Vendor batched calls are
  not used for regime B; the per-bin benchmark that would justify them is still
  owed (issue #60).
* **Regime C, root fronts.** A handful of large fronts. Cholesky: vendor
  `potrf`, `trsm`, `syrk`/`herk` through the dense interface, then a pack
  step. LDLᵀ: the fully-summed block is factored in blocks of 32 columns by a
  KA kernel that reproduces the reference pivot sequence, with one vendor GEMM
  per block for the trailing columns and the contribution block (#89). LU: the
  fused KA kernel, one workgroup per front; BLAS-3 for `L₂₁`/`U₁₂` is a T25
  item. Vendor `sytrf`/`getrf` are not used (§3.3).

This mirrors cuDSS keeping a distinct algorithm for very sparse factors, and
CHOLMOD's subtree streaming. The split points are analysis outputs (§2.3), not
runtime decisions.

### 2.3 Symbolic engine (host)

1. **Pattern**: expand the user's triangle, symmetrize (`G`: A ∪ Aᵀ), drop
   duplicates, rebase indices, check squareness; build the full-pattern CSR map
   used by the residual SpMV of refinement.
2. **Ordering** via CliqueTrees.jl: `permutation(graph; alg)` with `AMD()`
   (AMD.jl, a hard dependency) /`MMD()` or `METIS_NodeND` (Metis extension,
   `nd_ubfactor` passed through, `nd_nlevels` a minimum); user permutation;
   natural; ND-tree export/import in the cuDSS encoding (T24). **Automatic
   choice** computes both AMD and ND candidates (cheap for KKT sizes) and picks
   by a cost model `flops × (1 + nlevels/n)` on the column etree: ND gives
   bushier, shallower trees; AMD often gives lower fill on power grids (and
   wins on the KKT and random harness matrices). That model is wrong for
   large KKT systems (issue #108, PR #107): at n ≈ 7e5 it weighs depth at
   0.2% and scores column-etree depth, which does not predict the supernodal
   schedule depth that governs GPU time, so it picks AMD where ND halves the
   schedule depth (65 → 29) and is 36% faster on the solve; until T25 scores
   the schedule depth, GPU users should set `reordering_alg = "algo4"`. Schur
   mode constrains the
   ordering (§3.6). Matching (T21) composes a column permutation for `"G"` and
   only scales the symmetric structures. **2×2 pivot pairs** for `"S"`/`"H"`
   (`pivot_pairs`): rows whose diagonal is absent or negligible are candidates;
   `"default"` pairs the candidates whose pivot is structurally zero in the
   ordering (an augmenting-path test on the leading blocks, iterated with
   re-ordering), or the 2-cycles of the symmetric matching when `matching_alg`
   is job 5/6; the ordering runs on the graph compressed by the pairs and is
   expanded partner-first, and the symbolic factorization gives both columns of
   a pair the union structure so that they share a fundamental supernode
   (MA57/HSL_MA97 compressed-graph ordering, Duff & Pralet 2005). Cost on
   MadNLP K2 systems: 1.3–1.4× nnz(L) for `"default"`, about 2× for `"all"` and
   for the matching pairs (#66, #96). Not applied with `user_perm` or the
   natural ordering.
3. **Elimination tree**, postorder, **column counts** (Gilbert–Ng–Peyton), nnz(L),
   flops → `lu_nnz`, `flops`, `max_lu_nnz`. Etree, column counts, fundamental
   supernodes and amalgamation are our code (CliqueTrees is used for the
   orderings only). The supernodal numbering is a topological reordering of the
   etree composed with the ordering (`SupernodePartition.perm = perm[order]`,
   same fill); every later phase uses it.
4. **GPU-tuned amalgamation**: relaxed amalgamation (Ashcraft–Grimes) with
   GPU parameters: accept more explicit zeros than a CPU solver would (bounded
   by the `amalgamation` option, default ≈25% extra factor storage) to reach
   8–32-column panels, which is the main lever against tiny-front overhead.
   `max_width` caps merging only: a fundamental supernode wider than that (a
   dense separator) stays one front for regime C; splitting it into a chain of
   panels adds tree levels and update-stack traffic (issue #48).
5. **Assembly-tree partition** into regimes: leaf subtrees whose packed serial
   stack (fronts and contribution blocks as packed triangles, minus a small
   reserve for control words) fits one of four local-memory classes (8–48 KiB,
   `subtree_budgets` mapped to `Val` kernel instances; 48 KiB is CUDA's static
   shared-memory limit, per-backend caps are T23, issue #60) **and** whose
   flops are at most `1/subtree_parallelism` of the total (default 4096), so
   that a KKT tree never runs on one workgroup (#81); mid-level fronts binned
   by width class (8, 16, 32, 64) and row class; root fronts above
   `regime_c_width`/`regime_c_rows`. Regime-A subtrees all run first, one
   launch per budget class; the B/C level loop counts levels above the
   subtrees. Launch lists per (level, bin).
6. **Static memory layout**: offsets for every front and contribution block,
   update-stack high-water mark; the block placement is offline over the known
   lifetimes (best of first fit and two size-ordered placements, #54), level
   chunking under `memory_budget` only bounds the bytes produced per chunk
   until hybrid memory exists; no allocator at run time. The lifetimes are
   schedule steps, so the placement assumes level-synchronous execution: an
   out-of-level-order numeric phase (dependency counters, streams, fused
   kernels, experiment 5) needs private blocks (4× the stack on the 78k-bus
   KKT) or lifetimes computed on its own execution order (issue #109).
7. **Device maps**: destination of every `nzVal` entry inside its front
   (owner-pull grouping `amap_ptr`/`amap_src`, conjugation as a negative
   offset); child→parent relative indices for extend-add; gather lists for the
   solve; subtree descriptors and local-memory layouts (`local_front`,
   `local_cb`) for regime A; all as `INT` device vectors with one
   overflow-checked conversion from the host `Int` analysis.
8. **Storage layout**: all L panels (and U panels for `G`) in one device array,
   each panel a contiguous column-major `f×w` block (leading dimension `f`) so a
   `reshape(view(...))` is a valid strided matrix for vendor BLAS; D separate;
   batch stride for uniform batches. The front's first `w` columns *are* the
   factor panel; only the contribution block lives on the update stack, in
   packed lower-triangular storage (`m(m+1)/2` entries; regime C packs after
   the vendor `syrk` through a workspace of the largest `m²`), since the stack
   is otherwise 4–9× the factor on KKT matrices (issue #48). LU stores `Uᵀ` in
   a second buffer with the `L` layout (`ufactor`, `ustack`), so every map and
   the sweeps are reused and the factor and stack are twice the Cholesky size.
   The regime-C LDLᵀ path keeps a per-launch-group workspace of at most twice
   the largest front for its blocked panel.

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
written to per-front stats arrays. In regime C, Cholesky uses the vendor
`potrf` (through an asynchronous `potrf_info!` whose status stays on the
device), `trsm` and `syrk`/`herk`. For the pivoted factorizations the vendor
`sytrf`/`getrf` are **not** used: they choose their own pivot order on `F₁₁`
alone, so the device factor would not equal the CPU reference, which is the
test contract of T15/T19 (the pivot search is cooperative across the workgroup,
the tie-breaking is the reference's). LDLᵀ root fronts run a blocked KA panel
factorization (32 columns per block, lazily updated) with one vendor GEMM per
block for the trailing columns and the contribution block (#89); LU root
fronts run the fused KA kernel, BLAS-3 pending (T25, #75). On the pglib K2
dumps the device and the reference can still differ by rounding (max|L| up to
1e16 unscaled), which flips threshold and tie decisions: there the contract is
equal inertia and `nperturbed` within a tolerance (#86). The whole phase is a
fixed sequence of launches with no allocation and no host synchronization
(§3.9).

Kernel design rules (from the KA constraints in §2.7): 1-D workgroups,
shared-memory reductions, no subgroup/shuffle primitives in the portable path,
the right-hand-side and batch dimensions folded into the 1-D group index, at
most 32 kernel arguments; warp-level fast paths may be added later behind the
CUDA/ROCm extensions once KernelInterface/KernelIntrinsics stabilize.

### 2.5 Solve phase

Permutation and scaling kernels, then forward/diagonal/backward sweeps on the
same subtree/level schedule: regime A subtrees solved by one workgroup each,
regime B/C fronts by batched TRSV/GEMV (TRSM/GEMM for multiple RHS; vendor on
root fronts). The forward sweep has write conflicts between fronts of one
level: default `Atomix.@atomic` accumulation; `deterministic_mode = 1`,
complex `T` and backends without float atomics use the pull variant, whose
per-front buffers live at the fronts' gather-list rows (`length(rowval) × nrhs`
entries), not on the update stack. The backward sweep is conflict-free. LDLᵀ
and LU apply each supernode's local pivot order inside the sweeps (LAPACK
`sytrs` style), so the symbolic structure and the maps are pivot-independent.

Solve latency matters as much as factorization for IPM loops (MA57 averages
about six refinement solves per factorization on OPF matrices), so two
accelerations are planned: partitioned-inverse diagonal blocks for
refinement-heavy loops (`solve_alg = "algo1"`, M11) and a sync-free
ready-flag sweep in the CUDA/ROCm extensions, selected only where the hardware
guarantees forward progress (M11). Both were prototyped in PR #107
(`bench/solve_proto_*.jl`): per-front `L₁₁` inverses turn each TRSV into a
GEMV, and one fused dependency-counter kernel per direction over every front
of width ≤ 256 solves the 78k-bus condensed KKT in 3.5 ms against 3.8 ms for
cuDSS and 18 ms for the level-batched sweeps, on CUDA and (8 ms) on a Radeon
VII; the shallow ND ordering (#108) is a precondition. On lap3d_40 and apache2
the same approach regresses, so T25 picks the strategy per schedule.

Refinement: plain IR with a KA CSR SpMV residual (gather, no atomics; the
workspace is allocated by the first refining solve, so the handle layer with
`ir_n_steps = 0` pays nothing), or FGMRES with the factors as preconditioner
(Krylov.jl extension): one vector holding every right-hand side and batch
member, joint stopping test, one host synchronization per iteration, so only
plain IR with `ir_tol = 0` is synchronization-free. The residual is measured
against the unscaled `A` when matching is on. Refinement runs in the input
precision even when factors are Float32 (§3.8).

### 2.6 Backends, extensions and the dense layer

The dense layer is a small interface (`gemm!`, `syrk!/herk!`, `trsm!`, `potrf!`,
`sytrf!`, `getrf!`, `laswp!`, batched variants) with:

1. **Generic LinearAlgebra entry points** each backend already routes to its
   vendor library on its own array type: `mul!`, triangular `ldiv!`/`rdiv!`,
   `cholesky!`, `lu!`. No code in our extensions. This `:generic` path is the
   allocating reference implementation (§3.9).
2. **Vendor-specific calls** (`:vendor`) in each extension where no generic
   entry point exists: `syrk!/herk!`, the asynchronous `potrf_info!`, batched
   `gemm/trsm/getrf/potrf` with pointer arrays prebuilt at analysis, `sytrf!`.
   On the KA CPU backend the "vendor" library is host BLAS/LAPACK through
   LinearAlgebra. `laswp!` is KA-only (CUDA.jl has no binding).
3. **KA fallback kernels** (`:ka`: the regime A/B kernels and tiled
   large-front kernels) for everything a backend lacks, so every feature works
   everywhere. Batched ops have no `:generic` path. The `:ka` fallbacks still
   allocate per call; making them allocation-free is a T23 item (issue #53).

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

Extensions: `…CUDAExt` (weak dependencies `CUDACore`, `cuSPARSE`, `cuBLAS`,
`cuSOLVER` of CUDA.jl 6: dense bindings, sparse adapters; pinned memory, graph
capture and the sync-free solve are T25/T26), `…AMDGPUExt`, `…OneAPIExt`,
`…MetalExt` (T23), `…MetisExt` (ND), `…KrylovExt` (FGMRES-IR). Core depends on
KernelAbstractions 0.9, GPUArrays(Core), Adapt, Atomix, LinearAlgebra,
SparseArrays, CliqueTrees, AMD, Metis (ordering only through the extension).
The CUDA column of the table above was verified by the T03 capability audit
(all yes, `herk` only for complex `T`, no complex atomics).

### 2.7 Portability constraints (hard, from the research)

| Constraint | Consequence |
| --- | --- |
| KA has no stable portable shuffle/ballot/subgroup reduction (KernelInterface 0.4 and KA 0.10 still moving; KernelIntrinsics.jl is a proof of concept tested on CUDA/ROCm only). | Portable kernels use 1-D workgroups and shared-memory reductions. Warp-level variants only behind backend extensions, later. |
| Metal (and likely Intel) guarantee no inter-workgroup forward progress or launch order. | Sync-free SpTRSV and persistent DAG schedulers are CUDA/ROCm-only, selected by a capability check. Subtree + level scheduling is the portable baseline. |
| No dynamic parallelism, no graphs in KA; one launch per level is the portable floor. | Subtree fusion and level merging are first-class analysis goals. CUDA/HIP graph capture of refactorize+solve through the extensions. |
| Metal has no Float64. | `factor_precision = Float32` with Float64 refinement on the CPU or in double-single arithmetic; acceptable for condensed/well-conditioned systems only. |
| Float32 atomics have codegen regressions on CUDA; atomics cost everywhere. | No atomics in assembly or extend-add (owner-pull, deterministic). Atomics only in the default forward solve, with an atomic-free variant. |
| KernelAbstractions 0.9 CPU backend: a 2-D ndrange allocates per launch, launches with more than about 32 arguments are not specialized and allocate, `@localmem` is heap-allocated under coverage (issue #53). | Allocation-free phases use 1-D ndranges with the extra dimension folded into the group index and kernels of at most 32 arguments; the CPU allocation tests carry an explicit budget for the backend's own allocations. |
| `@localmem` is static shared memory; CUDA rejects more than 48 KiB per kernel, Metal allows 32 KiB. | Regime-A local sizes are a fixed ladder of `Val` instances (8–48 KiB); per-backend caps are a capability (T23, #60). |

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
the handle-style layer; refinement is on by default here (two steps,
`ir_tol = 0` so that `ldiv!` adds no host synchronization; a user can set
`ir_tol` for early exit), a documented difference from cuDSS for standalone
users.
Nothing is duplicated between the layers. MadNLPGPU implements its
`AbstractLinearSolver` (`factorize!`, `solve_linear_system!`, `inertia`) on the
handle-style layer.

### 3.2 Types

```julia
abstract type AbstractDirectSolver{T,INT} <: LinearAlgebra.Factorization{T} end
struct Symbolic{INT, VI<:AbstractVector{INT}}   # partition, schedule, layout, device maps
struct Numeric{T, VT, VS, VI, VK, …}            # panels (and Uᵀ panels), D, update stack(s),
                                                # regime-C workspace, Int64 stats and totals,
                                                # Int32 status, piv and pivot_kind, psign, aux,
                                                # host launch plan
mutable struct DirectSolver{T,INT,M,B,SY,NU,WS,RF} <: AbstractDirectSolver{T,INT}
    A::CSR{…}; structure; view; options::Options; backend
    ordering; host_symbolic; symbolic; numeric; workspace; refinement; schur; matching
    stage; fresh_factorization::Bool; info; nbatch::Int
end
```

The type parameters are derived from the backend at construction so the phases
are type-stable; `host_numeric` copies a device `Numeric` back for the oracle
comparisons.

`setparam!` validates names against the two tuples CUDSS.jl exports plus the
§1.7 additions.

### 3.3 Pivoting

* `SPD/HPD`: none; first non-positive pivot → `info`.
* `S`/`H`: Bunch–Kaufman 1×1/2×2 pivots restricted to the fully-summed block of
  each front (`pivot_type = 'B'`, default), `pivot_threshold` for acceptance
  (tested against the whole remaining front column); if no acceptable pivot
  exists, the best 1×1 of the block, then the Bunch–Kaufman choice if it is not
  tiny, then static perturbation `d ← sign·ε` with the `pivot_sign` policy;
  per-front counts of perturbed, zero and 2×2 pivots; no delayed pivots in v1.
  In-front pivoting can only pair columns of one fully-summed block, so the
  analysis puts every zero or negligible diagonal in the block of a partner
  (`pivot_pairs`, §2.3 step 2); without that, KKT dual rows are width-1 leaf
  supernodes and get growth `1/δ` or a perturbed zero pivot (#64). The same
  algorithm runs in every regime on the device, pivot for pivot as the CPU
  reference; vendor `sytrf` is not used (§2.4).
* `G`: in-block threshold partial pivoting in every regime (the KA kernel,
  same sequence as the reference), static perturbation of tiny pivots;
  refinement recovers accuracy. Vendor `getrf` + `laswp` are not used.
  `pivot_sign` is ignored for `"G"`.
* Badly scaled matrices: static in-front pivoting cannot bound the growth on
  unscaled MadNLP K2 systems (max|L| 1e14–1e16, #71); MC64 scaling and the
  matching-based pairs (T21) bring them to 0–6 perturbed pivots and residuals
  below cuDSS's `algo5`, so a posteriori pivoting (M13) is not needed before
  the MadNLP integration.
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
panels and `info` are interleaved per front, the rest is member-major; kernels
take the batch index as one more (folded) grid dimension; `ubatch_index`/
`ubatch_mask` restrict the range; pivots and `info` are per member; `nrhs > 1`
works at any batch size. Regime C uses strided-batched vendor `gemm` and
pointer-array `trsm`/`potrf`/`getrf` prebuilt at analysis; the solve's vendor
`trsm` runs member by member (cuBLAS has no strided `trsm`). Not supported on a
batch: row-major right-hand sides, `logabsdet`, Schur mode. The 2×2 pivot pairs
and the matching come from the first member's values.

### 3.6 Schur complement mode

Schur indices are ordered last and form the root supernode; the rest is ordered
fill-reducingly (no 2×2 pivot pairs, no matching). The Schur root is a regime-C
front alone in the last launch group, assembled by its own launch sequence
(`assemble_schur!`) after the regular drivers and never factored; it *is* `S`
(dense), and its exact symbolic pattern (on `A + Aᵀ` for `"G"`) gives
`schur_shape` and the sparse export. `solve_fwd_schur`/`solve_diag`/
`solve_bwd_schur` follow cuDSS (read `B`, write `X`, condensed RHS in the last
`n_s` entries); LU's diagonal is applied inside `solve_bwd_schur`, so
`solve_diag` is the identity for `"G"`. `"solve"` and the regular sub-phases
raise in Schur mode; `lu_nnz`/`flops` count the dense root as if factored.

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
refactorize+solve sequence in a graph and replay it per IPM iteration (graph
capture itself is T25, #82). This holds for the dense implementations `:auto`,
`:vendor` and `:ka` (the latter still allocates until T23, #53); `:generic`
(host LinearAlgebra) is the reference path and may allocate. Exceptions by
design: the `ir_tol > 0` early-exit test and every FGMRES iteration read on the
host; `getparam` of statistics and `info` are the phase-boundary reads.

---

## 4. Repository layout

As of T21 (the planned `src/numeric/root.jl`, `refactorize.jl`, `partinv.jl`,
`src/batch/`, `src/hybrid.jl` do not exist: their content lives in the files
below or is still to come).

```
Project.toml                  deps: KernelAbstractions 0.9, GPUArrays, GPUArraysCore, Adapt, Atomix,
                              LinearAlgebra, SparseArrays, CliqueTrees, AMD
                              weakdeps: CUDACore, cuSPARSE, cuBLAS, cuSOLVER (CUDA.jl 6), Metis, Krylov
src/SparseDirectSolver.jl
src/errors.jl  logging.jl     error hierarchy; SDS_LOG_LEVEL phase logging
src/types.jl  options.jl      enums, Options, setparam!/getparam tables (CUDSS.jl names + §1.7)
src/matrix.jl                 CSR container, adapters, view/index handling, MatrixDescriptor
src/symbolic/pattern.jl       symmetrize, expand, dedup, rebase, full-pattern map
src/symbolic/ordering.jl      CliqueTrees orderings, cost model, user perm, Schur-constrained
src/symbolic/pairs.jl         2×2 pivot candidate pairs, compressed-graph ordering
src/symbolic/etree.jl         elimination tree, postorder, column counts, nnz/flops
src/symbolic/supernodes.jl    fundamental supernodes, GPU-tuned amalgamation
src/symbolic/schedule.jl      regimes, subtree partition, size bins, level lists, chunking
src/symbolic/layout.jl        static offsets, offline update-stack placement, memory estimates
src/symbolic/maps.jl          scatter maps, relative indices, local layouts, symbolic_analysis
src/matching/mc64.jl          MC64 jobs 1–5, scalings, matching pairs (host)
src/dense/interface.jl        dense-op interface + dispatch (impl = :auto/:vendor/:generic/:ka)
src/dense/vendor.jl  capabilities.jl   host BLAS/LAPACK bindings, capability probes
src/dense/fallback/*.jl       KA potrf, getrf, trsm, gemm/syrk/herk, laswp
src/reference/cholesky.jl  ldlt.jl  lu.jl   CPU reference multifrontal factorizations (the oracle)
src/numeric/storage.jl        Numeric, allocation, host_numeric
src/numeric/assembly.jl       KA zero / scatter / owner-pull extend-add / pack kernels
src/numeric/subtree.jl        regime A fused subtree kernels (Cholesky)
src/numeric/front.jl          regime B fused per-front Cholesky kernel, regime C Cholesky driver
src/numeric/ldlt.jl  ldlt_c.jl   LDLᵀ/LDLᴴ kernels (regimes A/B), blocked regime-C panel + GEMM path
src/numeric/lu.jl             LU kernels (regimes A/B/C)
src/numeric/factorize.jl      phase driver, stats reduction, info/inertia/pivot_stats
src/numeric/batch.jl  schur.jl  extract.jl   uniform batch pointer arrays, Schur root, factor extraction
src/solve/permute.jl  sweeps.jl  refinement.jl   permutation/scaling, sweeps, IR and the FGMRES operators
src/solver.jl                 DirectSolver, execute!, phases, parameters
src/generic.jl                LinearAlgebra interface
ext/SparseDirectSolverCUDAExt.jl  …MetisExt.jl  …KrylovExt.jl   (AMDGPU, oneAPI, Metal: T23)
test/                         CPU backend always + CUDA when present; ported CUDSS.jl tests under test/ported
bench/                        harness, cuDSS/CHOLMOD/UMFPACK baselines, cuDSS comparison per feature,
                              phase profiles, pglib-opf KKT dumps (ExaModelsPower), pivot-pair and
                              refinement tables
PERFORMANCE.md  STATE.md      gap analysis against cuDSS and the experiment plan; owner state review
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

**Status (2026-10-07)**: M0–M3 and M5–M9 are done; M4 is done except the
MadNLPGPU integration (External task); M2 and M4 differ from the definitions
above in that vendor batched calls are not used in regime B and vendor
`sytrf`/`getrf` are not used in regime C (§2.4). CI runs the CPU backend and
CUDA, plus a non-required AMD leg that passes everything but the ROC
constructor forwarders (T23); oneAPI/Metal wait for T23. The M3 target (1.5×
cuDSS on condensed pglib systems) is not met by the library: 2.5×
factorization / 3.9× refactorization / 4.4× solve (geometric means, RTX 4080);
LDLᵀ 4.4× / 6.3×. The remaining gap on KKT systems is launch- and level-bound
(#82); bench-level prototypes (PR #107) reach 1.3× cuDSS on the refactorization
and parity on the solve of a 78k-bus condensed KKT, and a MadNLP end-to-end run
on CUDA and AMD exists in `bench/e2e/`, so M11 is a matter of moving measured
designs into `src/`. Measurements and the experiment plan are in
`PERFORMANCE.md`.

---

## 6. Risks and mitigations

| # | Risk | Mitigation |
| --- | --- | --- |
| R1 | Tiny-front latency dominates on KKT matrices (the failure mode of SSIDS/STRUMPACK/SuperLU on ACOPF). | Regime A subtree fusion and regime B fused per-front kernels; GPU-tuned amalgamation; level merging; measured against the cuDSS baseline from M0 onward. |
| R2 | Vendor coverage gaps (batched ops on oneAPI/Metal, no `hetrf`, no Float64 on Metal). | Vendor calls are confined to root fronts; regimes A/B are KA everywhere; mixed precision for Metal. |
| R3 | KA subgroup/shuffle API churn (KA 0.10, KernelInterface 0.4). | Portable kernels avoid subgroup primitives entirely; warp-level variants are optional and extension-local. |
| R4 | No forward-progress guarantees on Metal/Intel. | Sync-free algorithms only behind a capability check in CUDA/ROCm extensions; subtree+level is the baseline. |
| R5 | Local-memory budgets: fronts or subtrees that do not fit a `Val` budget. | Analysis assigns regimes by measured budgets; anything that does not fit regime A falls to B, anything above B's largest bin to C. |
| R6 | Vendor LAPACK blocks cannot apply `pivot_epsilon`/`pivot_sign` inline, and choose their own pivot order. | Resolved by not using them: the pivot search runs in KA kernels in every regime with the reference sequence; vendor BLAS-3 only for the trailing updates (§2.4). |
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
* **Backends**: every test runs on the CPU backend (GitHub runners) and on
  CUDA (self-hosted runner) in CI, and on the AMD runner as a non-required leg
  (99.4% pass without the AMDGPU extension, PR #107); the leg becomes required
  with T23, oneAPI and Metal join then. The
  T03 capability audit documents which implementation each op took. The suite
  runs in parallel workers with a duration history (#90); compile time is the
  dominant cost (#91).
* **Reference solvers**: SparseArrays/CHOLMOD (`cholesky`; its `ldlt` is
  simplicial and not a performance reference), UMFPACK (`lu`) for residuals,
  inertia (vs `eigvals` on small matrices), nnz(L), permutations, also as the
  `--solver=cholmod` CPU row of the benchmark harness; CUDSS.jl when CUDA is
  available, through a shared harness parametrized on the solver module
  (`Random.seed!(666)` kept); MA57 refinement counts as the robustness
  yardstick for LDLᵀ (not measured yet, no HSL). Device factors are compared to
  the CPU reference pivot for pivot on well-conditioned generators and by
  inertia and perturbation counts on real KKT data (#86).
* **Ported tests**: `test_cudss.jl`, `test_uniform_batch_cudss.jl`,
  `test_nonuniform_batch_cudss.jl`, `test_schur_cudss.jl`; documented cuDSS bugs
  become regression tests (multi-RHS uniform batch, deterministic mode in
  batches, inertia with matching, silently perturbed pivots).
* **Matrices**: pglib-opf KKT (K2) and condensed systems dumped from MadNLP
  through ExaModelsPower (case14, case118, case1354 at several iterations),
  SuiteSparse via MatrixDepot.jl, generated Laplacians, the hand-built Schur
  examples from the CUDSS.jl docs; NREL opf_matrices, a CUTEst subset and the
  K2r/LiftedKKT/HyKKT variants are not in the harness yet.
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

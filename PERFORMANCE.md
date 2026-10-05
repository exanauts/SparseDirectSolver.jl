# SparseDirectSolver.jl vs cuDSS: Where the Gap Is and How to Close It for Repeated IPM KKT Solves

**Bottom line:** For MadNLP-style repeated KKT solves, SparseDirectSolver.jl (SDS) will not beat cuDSS by out-BLASing it on large fronts. Its realistic edge comes from three places. First, robust indefinite numerics that cuDSS lacks: true 2×2 Bunch–Kaufman pivots, per-row pivot signs, and correct inertia. Second, a refactorize+solve path that is device-resident, launch-minimal and graph-captured, which removes the host synchronizations and launches that dominate at OPF scale. Third, a GPU-resident or cached analysis phase. The immediate blockers are not raw FLOP rate. They are (a) the serial, reference-faithful pivot search and the one-workgroup-per-front regime-C LDLᵀ (issue #75), and (b) missing matching and scaling, without which K2 systems produce max|L| of 1e14–1e16 (issue #71).

## Tracked issues

Performance issues carry the GitHub label `performance` ([list](https://github.com/exanauts/SparseDirectSolver.jl/issues?q=label%3Aperformance)). Keep this table in step with them: add a row when an issue is opened, and update the status when its PR merges or it closes.

| issue | what | experiment | status |
| --- | --- | --- | --- |
| #81 (PR) | regime-A subtrees ran a whole KKT tree on one workgroup; flop limit `subtree_parallelism` | 0 | merged |
| #82 | KKT refactorization and solve are level-bound after #81: 54–59 launches per refactorization, 62–85 per solve | 5, 6 | open |
| #75 | device LDLᵀ: serial pivot search, one workgroup per regime-B/C front, no vendor `sytrf` | 1, 2 | open, triaged; step 1 (cooperative pivot search) and step 2 (regime B in local memory) in PRs from `perf/exp1-pivot-search`, `perf/exp1-regime-b-local` |
| #60 | regime-A follow-ups: CUDA timings (partly answered by experiment 0) and per-backend local-memory caps | 7 | open, triaged |
| #25 | T25 performance pass (task) | 5, 6, 7 | open |

Related accuracy issues that gate the K2 results: #71 (max\|L\| 1e14–1e16 on the K2 dumps, needs scaling) and #67 (pivot-pair matching), both experiment 3 / T21.

## TL;DR
- **Current state:** SDS is a KernelAbstractions multifrontal solver. Symbolic analysis runs on the host (AMD or METIS ND, amalgamation, MA57-style pivot pairs). Numeric factorization runs in three regimes: fused subtree kernels, level-batched fused front kernels, and vendor potrf/trsm/syrk. SPD Cholesky and LDLᵀ with in-front Bunch–Kaufman and static perturbation both work on CUDA. LU, batching, matching and scaling, mixed precision and the non-CUDA backends do not exist yet. The LDLᵀ is deliberately slow: it reproduces the CPU reference pivot for pivot.
- **cuDSS weak spots to exploit:** analysis (reordering) always runs on the host and is synchronous. It is the documented bottleneck in MadNLP/ExaModels: Pacaud, Shin, Montoison, Schanen and Anitescu (arXiv:2405.14236) report that "the analysis phase is four times slower for cuDSS compared to CHOLMOD". QOCO-GPU reports the same bottleneck. Symmetric-indefinite pivoting is diagonal-only within a supernode plus epsilon perturbation, with no LBLᵀ. The defaults (no matching) give relres 0.26–170 on the K2 dumps. Hybrid, MG and MGMN modes force synchronous phases.
- **Plan:** fix the LDLᵀ kernels first (parallel APTP-style pivoting, vendor-backed regime C), then matching and scaling, then a graph-captured, allocation-free refactor+solve with a partitioned-inverse/supernodal solve. Only after that, tune amalgamation and front bins, then mixed precision and batching. Success means matching cuDSS iteration counts in MadNLP and beating it on refactor+solve wall time per IPM iteration for pglib cases up to ~10k buses.

## Key Findings

### 1. SparseDirectSolver.jl today
- **Scope and API.** SDS is "a portable sparse direct solver (LLᵀ/LLᴴ, LDLᵀ/LDLᴴ, LDU) for GPUs, written in Julia on KernelAbstractions.jl and GPUArrays.jl". It mirrors CUDSS.jl's phases and parameter names so that MadNLP can switch mechanically. Status is v0.1 under construction. It requires Julia ≥ 1.13 and is unregistered. Development is agent-driven: 45 commits, with strictly sequential tasks T01–T27 in TASKS.md. [github](https://github.com/exanauts/SparseDirectSolver.jl)
- **Symbolic (host).** AMD, or METIS nested dissection through an extension. [github](https://github.com/exanauts/SparseDirectSolver.jl) Elimination tree, column counts, fundamental supernodes with GPU-tuned amalgamation; issue #64 gives the default as (32, 0.25, 8), i.e. max merged width, zero-fill budget and a min-width rule. [github](https://github.com/exanauts/SparseDirectSolver.jl/issues/64) Static factor layout and device assembly maps follow. Contribution blocks are placed offline over their lifetimes (PR #54). The update stack is still ~4.5× the factor size, and 8.7× on a 60k×25k KKT generator. [github](https://github.com/exanauts/SparseDirectSolver.jl/pull/56)
- **Indefinite-specific analysis.** Structurally zero pivots get a partner (`pivot_pairs`, MA57-style compressed-graph ordering) so that in-front pivoting can form 2×2 blocks. [github](https://github.com/exanauts/SparseDirectSolver.jl) Measured on the CPU reference with the default pairing: case118 K2 goes from 147 to 3 perturbed pivots at 1.25× nnz(L), and case1354 K2 from 2060 to 33 at 1.40×. Pairing everything ("all") costs 1.8–2.2× nnz(L) and is worse. [github](https://github.com/exanauts/SparseDirectSolver.jl/pull/72)
- **Numeric regimes.**
  - **A:** fused subtree-per-workgroup kernels for leaf subtrees, selected by local-memory budget. The budgets are 8/16/32/48 KiB static shared memory classes, capped at 48 KiB by ptxas. Metal needs 32 KiB and AMD could use 64 KiB, so per-backend caps are pending (issue #60). [github](https://github.com/exanauts/SparseDirectSolver.jl/issues/60)
  - **B:** level-batched fused per-front kernels for width ≤ 64. [github](https://github.com/exanauts/SparseDirectSolver.jl/pull/59)
  - **C:** width > 64, dense potrf/trsm/syrk through cuBLAS/cuSOLVER for Cholesky. [github](https://github.com/exanauts/SparseDirectSolver.jl/pull/50)
  - The CUDA timing table that should justify these thresholds has not been measured yet (#60). [github](https://github.com/exanauts/SparseDirectSolver.jl/issues/60)
- **LDLᵀ (T15).** In-front Bunch–Kaufman 1×1/2×2 with `pivot_threshold`, static perturbation with a user-chosen sign per row (`pivot_sign`, which cuDSS cannot do), and inertia and pivot stats read back from the device. [github](https://github.com/exanauts/SparseDirectSolver.jl) It is bit-faithful to the CPU reference, and that is the performance problem. Issue #75 lists three costs:
  1. A serial pivot search by one work-item, O(w²·f) per front when the BK choice fails threshold. On a 3000/1000 KKT generator this takes 13.5 s against 12.9 s for the reference (KA CPU backend).
  2. Every regime-B/C front is one 128-thread workgroup with no vendor `sytrf`. "C only" takes 7.7 s against 2.1 s for the reference.
  3. F₁₁ is not staged in local memory in regime B. [github](https://github.com/exanauts/SparseDirectSolver.jl/issues/75)
- **Solve.** Supernodal forward, diagonal and backward sweeps with multiple RHS and IR (`ir_n_steps`), allocation-free after analysis. [github](https://github.com/exanauts/SparseDirectSolver.jl) Refinement with 5 steps gives 1e-15..1e-18 on case118 K2, which beats the cuDSS bar, but only 1e-5..1e-7 on case1354 K2 until matching and scaling land (PR #79). [github](https://github.com/exanauts/SparseDirectSolver.jl/pull/79)
- **Accuracy blocker.** On the K2 dumps, max|L| reaches 1e14–1e16 and factor error reaches 3.6 with every pivot-pair mode. This needs scaling (T21) or delayed pivots (M13) (#71). [github](https://github.com/exanauts/SparseDirectSolver.jl/pull/73)
- **Baselines.** The cuDSS baseline was run on an RTX 4080 with cuDSS 0.8.0: 2 Laplacians, 5 SuiteSparse matrices, and pglib case14/118/1354 K2 and condensed dumps, 39 rows in total. With cuDSS defaults, every K2 KKT gives relres 0.26–170, with 1036 perturbed pivots on case118 K2. `matching_alg="algo5"` (plus 5 IR steps on case1354) brings the residuals to 1e-13…4e-7 (PR #44). [github](https://github.com/exanauts/SparseDirectSolver.jl/pull/44) The README ships an SDS/cuDSS ratio plot per feature and phase. [github](https://github.com/exanauts/SparseDirectSolver.jl) I could not retrieve per-matrix SDS timing ratios, so I treat the performance gap as unquantified. Measuring it is experiment 0.
- **Roadmap.** T16 (IR) is done. Next come T17 batches, T18 FGMRES-IR via Krylov.jl, T19 LU, T21 MC64-style matching and scaling, T23 backends, T25 performance pass, T27 APTP with delayed pivots and Float32 factorization with Float64 refinement, and an external MadNLPGPU integration (#28). [github +3](https://github.com/exanauts/SparseDirectSolver.jl/issues/18) The T25 summary lists a partitioned-inverse solve (`solve_alg="algo1"`), a CUDA sync-free forward sweep, CUDA graph capture of refactorize+solve, level merging, and amalgamation/bin tuning. [github](https://github.com/exanauts/SparseDirectSolver.jl/issues/25) An owner note adds parallel pivot search, F₁₁ in local memory, and KA pivoting plus vendor trsm/gemm for regime-C root fronts; the vendor `sytrf` deliverable was dropped. [github](https://github.com/exanauts/SparseDirectSolver.jl/pull/77)
- **Julia-side cost.** JIT dominates CI: test_dense takes 143 s on the first run and 4.9 s on the second. This is a deliberate consequence of `Val`-specialized kernels (PR #61). [github](https://github.com/exanauts/SparseDirectSolver.jl/pull/61)

### 2. What is public about cuDSS
- **Phases.** Reordering and symbolic analysis, factorization or refactorization, and solve with optional IR. [nvidia](https://docs.nvidia.com/cuda/archive/12.8.1/cudss/getting_started.html) Reordering runs on the host and symbolic factorization, numeric factorization and solve run on the GPU, so "the analysis phase is always synchronous". [nvidia](https://docs.nvidia.com/cuda/archive/13.0.1/cudss/general.html) The default reordering is a custom METIS-like ND. [nvidia](https://docs.nvidia.com/cuda/cudss/doc_output/tips_and_tricks.html) The alternatives are AMD, COLAMD, BTF_COLAMD and NONE (0.8.0). [nvidia](https://docs.nvidia.com/cuda/cudss/migration_guide.html)
- **Pivoting.** With ND or AMD ordering, pivoting is local, within the supernode diagonal block. Symmetric indefinite matrices use `CUDSS_PIVOT_DIAGONAL`, which searches only the diagonal entries; unsymmetric matrices use complete block pivoting. [nvidia](https://docs.nvidia.com/cuda/cudss/advanced_features.html) Global pivoting requires the COLAMD orderings and is "significantly slower". Small pivots are replaced by `pivot_epsilon`, optionally scaled (`PIVOT_EPSILON_ALG_SCALED`). [nvidia](https://docs.nvidia.com/cuda/cudss/doc_output/tips_and_tricks.html) [nvidia](https://forums.developer.nvidia.com/t/cudss-is-sometimes-wrong-where-cusparse-and-umfpack-succeed/342132) Matching and scaling options: MAX_DIAG_COUNT, MAX_MIN_DIAG(_ALT), MAX_DIAG_SUM, MAX_DIAG_PRODUCT (algo5), AUTO; the default is none. [nvidia](https://docs.nvidia.com/cuda/cudss/migration_guide.html) Pacaud, Shin et al. note that cuDSS lacks the LBLᵀ factorization that NLP solvers normally use. [researchgate](https://www.researchgate.net/publication/394921161_GPU_Implementation_of_Second-Order_Linear_and_Nonlinear_Programming_Solvers)
- **Modes.**
  - Hybrid memory mode keeps the factors in host memory. [nvidia](https://docs.nvidia.com/cuda/cudss/advanced_features.html) [nvidia](https://docs.nvidia.com/cuda/cudss/advanced_features.html)
  - Hybrid execute mode uses the CPU for low-parallelism parts and is recommended for small and medium matrices. [nvidia](https://docs.nvidia.com/cuda/archive/13.0.0/cudss/release_notes.html) It cannot be combined with batches. [nvidia](https://docs.nvidia.com/cuda/cudss/advanced_features.html)
  - MG (single node) and MGMN (NCCL/MPI comm layer) modes exist, as does MT (threading layer for host work). [nvidia](https://docs.nvidia.com/cuda/cudss/advanced_features.html) [nvidia](https://docs.nvidia.com/cuda/archive/13.1.0/cudss/doc_output/index.html)
  - Uniform and non-uniform batches.
  - Superpanels and `FACTORIZATION_ALG_1` for very sparse factors. [nvidia](https://docs.nvidia.com/cuda/cudss/doc_output/release_notes.html)
  - Hybrid, MG and MGMN modes make all phases synchronous. [nvidia](https://docs.nvidia.com/cuda/cudss/advanced_features.html) [nvidia](https://docs.nvidia.com/cuda/archive/13.0.1/cudss/general.html)
- **Reuse.** Reusing only the permutation (`CUDSS_DATA_USER_PERM`) makes later factorizations "significantly slower". Full speed needs the saved ND partition tree. [nvidia](https://docs.nvidia.com/cuda/cudss/advanced_features.html)
- **CUDA graphs.** Graph capture requires a user device memory handler that uses `cudaMallocAsync`. Analysis cannot be captured. [nvidia](https://docs.nvidia.com/cuda/archive/13.0.1/cudss/general.html)
- **Evidence on weaknesses.** Pacaud and Shin (arXiv:2403.15913, CDC 2024) find that "cuDSS spends most of its time in the symbolic analysis" and that "ma27 is approximately twice as fast during the pre-processing". On their N = 50,000 instance, initialization took 15.4 s for ma27 against 27.9 s for Lifted-KKT and 29.7 s for HyKKT. On an A30 GPU, that same N = 50,000 instance (3,350,067 rows) took 20.187 s for SYM, 0.432 s for FAC and 0.165 s for SOLVE, i.e. "less than 0.5 seconds to recompute the factorization of the largest instance". Over the full solve, "Lifted-KKT and HyKKT solve the problem respectively 26x and 18x faster than HSL ma27 on the largest instance (N = 50, 000)" (totals of 33.8 s and 34.5 s against 125.5 s). QOCO-GPU (arXiv:2603.29197) reports that "for larger problems, up to 75% of qoco-gpu's runtime is spent in the setup phase, where the dominant cost is the reordering step in cuDSS's analysis phase."

### 3. Techniques most likely to matter, ranked for repeated IPM KKT solves
1. **Parallel indefinite pivoting.** Use APTP (Duff–Hogg–Lopez; SSIDS): factor a block optimistically with BLAS-3, check the threshold a posteriori, and fail or delay only the bad columns. It keeps TPP-level robustness and was designed for GPU/multicore. [researchgate](https://www.researchgate.net/publication/339832172_A_New_Sparse_LDLT_Solver_Using_A_Posteriori_Threshold_Pivoting) [researchgate](https://www.researchgate.net/publication/335908712_Exploring_Benefits_of_Linear_Solver_Parallelism_on_Modern_Nonlinear_Optimization_Applications) For SDS this means replacing the serial `_choose_pivot` with a warp-parallel argmax BK search over the panel plus an a-posteriori check. Pivots that fail inside the front fall back to perturbation now and to delayed pivots in T27. This is the biggest lever, because it is what lets regime C use vendor trsm/gemm (or syrk on L·D) for most of the flops.
2. **Matching and scaling for K2.** Use MC64-style symmetric scaling (max diag product, as in cuDSS algo5) or cheaper equilibration (Ruiz) computed once per pattern on the host, with the scaling recomputed on the device each iteration. cuDSS needs algo5 to be accurate on K2, so SDS must have it for parity, and per-iteration device-side equilibration is a differentiator. [github](https://github.com/exanauts/SparseDirectSolver.jl/pull/44) Recomputing the matching itself per iteration is a host cost, so cache the matching and refresh it only when IR stalls.
3. **Device-resident, launch-minimal refactor+solve.** At pglib scale (n ≈ 1e4–1e6), elimination-tree depth times launches per level dominates. Merge levels: a persistent kernel walks a level window with grid-wide or atomic-counter sync, i.e. sync-free dependency counters on supernodes (Liu et al. 2016 style). [ssslab](https://www.ssslab.cn/assets/papers/2016-liu-sptrsv.pdf) Capture the whole refactorize+solve+IR sequence into one CUDA graph. cuDSS itself cannot capture analysis, and hybrid execute (its small-matrix mode) forces syncs, which is exactly SDS's opening. [nvidia](https://docs.nvidia.com/cuda/archive/13.0.1/cudss/general.html)
4. **Triangular solve.** Use supernodal level-set SpTRSV with batched TRSV/GEMV at the bottom levels and streams at the top. Invert the diagonal blocks so TRSV becomes GEMV (the partitioned inverse of Alvarado–Pothen–Schreiber). Tacho uses exactly this split. Yamazaki, Rajamanickam and Ellingwood (ICPP '20), whose paper also covers "an algorithmic variant called the partitioned inverse", report that their Kokkos supernodal solver "can be 12.4× or 19.5× faster than the vendor optimized implementation in NVIDIA's CuSPARSE library" on V100/P100. In IPMs the solve is called 2–6× per factorization (IR, inertia correction retries, second-order correction), so solve latency matters as much as factorization.
5. **Small-front batching and bins.** STRUMPACK uses custom kernels for fronts with dim(F₁₁) ≤ 32 and cuBLAS/cuSOLVER above that. [escholarship](https://escholarship.org/content/qt7tv84567/qt7tv84567_noSplash_d41501c8913db5b2aa4fc426284a01c2.pdf) CHOLMOD-GPU batches many small subtree operations to hide launch overhead. [researchgate](https://www.researchgate.net/publication/274079177_Accelerating_sparse_cholesky_factorization_on_GPUs) SuperLU_DIST added variable-size batched GETRF/TRSM plus a batched scatter, and a GPU-resident solve (GPURES, v9.3). [researchgate](https://www.researchgate.net/publication/382142370_Batched_Sparse_Direct_Solver_Design_and_Evaluation_in_SuperLU_DIST) [github](https://github.com/xiaoyeli/superlu_dist/releases/tag/v9.3.0) SDS already has the architecture (regimes A/B). What is missing is measured thresholds and size-sorted bins.
6. **Mixed precision.** Factor in FP32 and refine in FP64 with IR or GMRES-IR. In SuperLU_DIST this gives 1.33–1.6× when factorization dominates. Li et al. (SuperLU_DIST Version-8 release paper, ACM TOMS) report for FP32 SpLU with FP64 IR that "the IR time is usually under 10% of the factorization time." For IPM K2 near convergence (κ ≫ 1/ε₃₂) plain IR will fail, so put FGMRES-IR (T18) first and gate FP32 on the IPM phase (early iterations) or on the condensed SPD systems. Tensor-core FP16/TF32 is only plausible for the root fronts.
7. **Condensed / reduced KKT (MadNLP-specific).** HyKKT and LiftedKKT turn the system into SPD Cholesky, where cuDSS is already strong and pivoting disappears. [pnnl](https://www.pnnl.gov/publications/hykkt-hybrid-direct-iterative-method-solving-kkt-linear-systems) [nrel](https://www.nrel.gov/media/docs/libraries/grid/mihai-anitescu.pdf?sfvrsn=e138a999_7) In SDS terms the SPD path is the place to chase raw speed against cuDSS, and LDLᵀ on K2 is where to win on robustness. The Świrydowicz et al. 2021 review found that no tested package delivered significant GPU acceleration on ACOPF KKT systems. [arxiv](https://arxiv.org/pdf/2106.13909) Their later GPU-resident work showed that static pivoting with FGMRES refinement is viable. [arxiv](https://arxiv.org/html/2401.13926) Both results support the "cheap static factorization + strong refinement" design.
8. **GPU or cached analysis.** Reordering is cuDSS's documented bottleneck. [arxiv](https://arxiv.org/pdf/2403.15913) Three options: cache the ND tree across MadNLP solves of the same model (or multi-period instances), run AMD/ND multithreaded in Julia, or build a GPU symbolic phase (etree, column counts) from the host permutation. This matters for MPC and online re-solve workloads, not for a single long IPM run.
9. **Multi-GPU and hybrid memory.** Low priority for KKT/OPF. The factors fit on GH200/H100-class GPUs even for Eastern Interconnection-scale cases, which MadNLP+cuDSS already solves on GH200. [nvidia](https://developer.nvidia.com/blog/nvidia-cudss-library-removes-barriers-to-optimizing-the-us-power-grid)

### 4. Julia-specific design considerations
- **CUDA graphs from Julia.** CUDA.jl exposes `capture`/`instantiate`/`update`/`launch`. [juliagpu](https://juliagpu.org/post/2021-06-10-cuda_3.3/) `@captured` re-records and calls `cuGraphExecUpdate` on every call, which Molly.jl measured as about 4× the cost of a bare `cuGraphLaunch`. [github](https://github.com/JuliaMolSim/Molly.jl/pull/289) Cache the `CuGraphExec` per solver and replay it directly. All buffers must be preallocated (SDS already is after analysis), kernels must be compiled before capture, and GC is disabled during capture. [github](https://github.com/JuliaGPU/CUDA.jl/issues/3310) Values go in through a fixed device buffer that `update!` copies into, so the captured pointers stay valid.
- **KernelAbstractions portability.** Static `@localmem` caps differ (CUDA 48 KiB static, Metal 32 KiB, AMD 64 KiB LDS), so budgets must come from a backend query (#60). [github](https://github.com/exanauts/SparseDirectSolver.jl/issues/60) Warp-level primitives such as shuffles and argmax reductions, which a fast pivot search needs, are not portable in KA. Implement them via local-memory reductions, or use backend-specific `@static` paths in extensions. Graph capture is CUDA-only (HIP graphs exist but are not wrapped equivalently), so design a "launch plan" abstraction with a graph backend and a plain-launch backend.
- **Compile latency.** The `Val`-specialized kernels make the first factorization cost minutes, which is bad for MadNLP users. Limit the specialization set (a few width classes), use PrecompileTools workloads on the CPU backend, and consider `@device_override`-free generic paths for rare sizes.
- **Type stability and allocation.** Keep integer types uniform (Int32 on device). Keep the `pivot_stats` and inertia readback to a single host transfer per factorization. MadNLP needs inertia every iteration, and that one sync is unavoidable unless inertia correction moves to the device.

## Experiment 0 results (2026-10-02, RTX 4080)

Measured with `bench/profile_phases.jl` (CUPTI trace of one warm refactorization and solve per harness matrix; `bench/profile/phase_split.md`, the state before the fix in `bench/profile/phase_split_main_9d280d0.md`) and `bench/compare.jl` (`bench/comparison/comparison.md`).

**The gap was not launch overhead.** GPU busy time (sum of kernel durations) equals wall time on every harness row, so the host never starves the device. The time goes into kernels that run on far too few thread blocks.

**Root cause on every pglib KKT dump: one thread block.** The regime-A rule took a front into a fused subtree whenever its whole subtree's stack fit the local-memory budget, with no parallelism criterion. KKT trees of small fronts fit as a whole, so the entire factorization (3134 supernodes on case1354 condensed, 19.6k on case1354 K2) and the forward solve ran in one 128-thread workgroup on one of 76 SMs, at about 12 µs per front.

**Fix (branch `perf/exp0-baseline-split`).** A new analysis tuning knob `subtree_parallelism` (default 4096): a regime-A subtree may do at most 1/4096 of the factorization flops. 4096 was best or near-best in a sweep over 128..4096 and a work floor never helped. Refactorization and solve, before and after:

| matrix | refactorization | solve |
| --- | --- | --- |
| case118 condensed, Cholesky | 3.9 → 1.2 ms | 2.7 → 0.8 ms |
| case118 K2, LDLᵀ | 13.8 → 2.0 ms | 7.7 → 0.9 ms |
| case1354 condensed, Cholesky | 39.9 → 3.0 ms | 19.7 → 1.6 ms |
| case1354 condensed, LDLᵀ | 56.3 → 9.6 ms | 22.6 → 1.2 ms |
| case1354 K2, LDLᵀ | 146 → 7.2 ms | 78.7 → 2.6 ms |
| SuiteSparse and Laplacian matrices | unchanged within noise (lap2d_300 36 → 31 ms) | unchanged |

SDS/cuDSS geometric means over the harness (`comparison.md`), before → after:

| feature | factorization | refactorization | solve |
| --- | --- | --- | --- |
| Cholesky, Float64 | 5.64× → 2.47× | 9.06× → 3.89× | 8.93× → 4.36× |
| LDLᵀ, static pivoting | 20.4× → 6.37× | 29.8× → 9.90× | 13.4× → 3.96× |
| LDLᵀ + 2 IR steps, K2 dumps | 23.4× → 3.18× | 36.0× → 5.45× | 33.3× → 7.57× |

A side effect: the default solve (`deterministic_mode = 0`) now uses its atomic regime-B forward sweep on the small KKT and test matrices too, so two solves are no longer bitwise identical there. That was always the documented contract (bitwise reproducibility needs `deterministic_mode = 1`), but before the fix these matrices never reached the atomic path. MadNLP should set `deterministic_mode = 1` if it relies on repeatable solves.

**What is left, by measurement:**
1. **KKT refactor+solve is now level-bound.** 30–60 kernels per refactorization and 35–85 per solve, the longest kernel under 15% of busy time, regime-B fronts on one or two blocks per level. This is where the launch minimization of experiment 5 (level merging, graph capture) now pays, together with experiment 6 for the solve (issue #82). Remaining refactorization gap on case1354: 5× (Cholesky) and 9–14× (LDLᵀ).
2. **LDLᵀ regime B/C is the single largest gap on everything else** (lap2d_300 43×, bcsstk17 59×, lap3d_40 250×, apache2 186× vs cuDSS): `front_ldlt_kernel` runs one front per workgroup on 1–2 blocks (issue #75). Experiments 1–2 unchanged in priority.
3. **Large Cholesky** (lap3d_40 4.4×, apache2 2.4×, solve 11–13×): 1222/6608 kernels per refactorization, `extend_add` on 2 blocks. Level merging and wider extend-add grids.

Revised order: experiments 1–2 (LDLᵀ kernels, #75) and 5–6 (launches and solve) now have the largest measured payoff; experiment 3 (matching and scaling) remains the accuracy blocker on K2 (relres unchanged by this fix).

## Experiments 1–2 results (2026-10-05, RTX 4080)

Issue #75 in three steps, one PR each, measured on the same machine with nothing else running. Refactorization: best of 5 warm runs of `bench/profile_phases.jl` (`*`: the `bench/compare.jl` median, for the matrices the profile skips); cuDSS from `bench/comparison/cudss.csv`; SDS Cholesky from the profile of `main`. The contract does not move: `piv` and `pivot_kind` equal `ref_ldlt!` on the test matrices on the CPU backend and CUDA, D and the panels within `panel_tol`.

### Step 1: cooperative pivot search

The pivot search of `front_ldlt_kernel!` and `subtree_ldlt_kernel!` ran on work item 1. Its column maxima are now workgroup reductions in up to three passes (λ and r with the column maximum; σ and the column-r maxima of Bunch–Kaufman; the fallback scan, one lane per candidate column), with the reference's tie-breaking (first maximum), so every decision is the serial one. Work item 1 takes the pivot from the values before the interchange, which keeps five barriers per step, and folds only the lanes that can hold data. The reference's fallback (`_best_1x1`) skips columns that cannot win and stops the threshold test at the first violation; the result is the same. The regime-A LDLᵀ kernel reserves 1024 B of local memory (`SUBTREE_LOCAL_RESERVE_LDLT`) for the reduction slots.

LDLᵀ refactorization, `main` → step 1:

| matrix | main ms | step 1 ms | step 1 / cuDSS | step 1 / SDS Cholesky |
| --- | --- | --- | --- | --- |
| lap2d_300 | 228 | 192 | 37.4× | 6.12× |
| lap3d_40 | 14,777* | 14,494* | 248× | 64.7× |
| HB/bcsstk17 | 206 | 167 | 49.5× | 6× |
| Boeing/bcsstk38 | 137 | 106 | 32.1× | 4.69× |
| GHS_psdef/apache2 | 92,576* | 97,296* | 199× | 94.5× |
| kkt_pglib_opf_case118_ieee_condensed_1 | 1.77 | 1.82 | 4.66× | 1.99× |
| kkt_pglib_opf_case118_ieee_condensed_10 | 1.72 | 1.79 | 5.13× | 2.04× |
| kkt_pglib_opf_case118_ieee_condensed_20 | 1.87 | 1.82 | 5.18× | 2.08× |
| kkt_pglib_opf_case118_ieee_k2_1 | 1.69 | 1.69 | 4.66× |  |
| kkt_pglib_opf_case118_ieee_k2_10 | 1.62 | 1.68 | 4.74× |  |
| kkt_pglib_opf_case118_ieee_k2_20 | 1.6 | 1.66 | 4.16× |  |
| kkt_pglib_opf_case1354_pegase_condensed_1 | 10.9 | 7.2 | 10.5× | 2.46× |
| kkt_pglib_opf_case1354_pegase_condensed_10 | 10.9 | 7.3 | 11.4× | 2.5× |
| kkt_pglib_opf_case1354_pegase_condensed_20 | 11 | 7.32 | 10.4× | 2.51× |
| kkt_pglib_opf_case1354_pegase_k2_1 | 8.31 | 6.6 | 8.08× |  |
| kkt_pglib_opf_case1354_pegase_k2_10 | 7.15 | 6.05 | 7.32× |  |
| kkt_pglib_opf_case1354_pegase_k2_20 | 9.1 | 6.71 | 8.32× |  |
| kkt_pglib_opf_case14_ieee_condensed_1 | 0.446 | 0.503 | 2.61× | 1.38× |
| kkt_pglib_opf_case14_ieee_condensed_10 | 0.443 | 0.52 | 2.26× | 1.53× |
| kkt_pglib_opf_case14_ieee_condensed_11 | 0.448 | 0.463 | 1.59× | 1.41× |
| kkt_pglib_opf_case14_ieee_k2_1 | 0.669 | 0.697 | 2.49× |  |
| kkt_pglib_opf_case14_ieee_k2_10 | 0.722 | 0.708 | 2.98× |  |
| kkt_pglib_opf_case14_ieee_k2_15 | 0.97 | 1.03 | 3.2× |  |

On the K2 dumps the inertia and `nperturbed` are those of `main` on all nine. On the T15 matrix `kkt_matrix(3000, 1000, 1e-8)` (fallback scans on most steps of its two large root fronts) CUDA goes from 33.9 s to 8.3 s; the KA CPU backend stays at 7.0 s against 2.6 s for `ref_ldlt!` (it runs the work items of a workgroup one after another, so a parallel search saves nothing there). The medium matrices gain 16–34%; the smallest KKT dumps (case14, w ≤ 16) lose up to 17% to the reductions' barriers; the two matrices with very large regime-C fronts (apache2, lap3d_40) are within ±5%, still one workgroup per front.

### Step 2: regime B in local memory

Staging only F₁₁ in `@localmem` (as `front_cholesky_kernel!`) gained nothing (lap2d_300 190 → 192 ms): the rows below F₁₁ are updated in global memory at every step and the contribution-block update was two thirds of regime B. The step became: `front_ldlt_kernel!` takes the width class `W` of its launch group and keeps the packed `W×W` F₁₁ in local memory from `W = 32` (classes 8 and 16 keep the panel in global memory: the staging phases cost more than they save on KKT fronts); the pivot columns stay unscaled (`L D`) until the end of the front, where `_lt_finalize!` applies the reference's divisions and 2×2 transforms with the pivot recomputed from D, so a step has no scale phase; the update reduces the next column as it writes it (pass 1), and work items own the rows below F₁₁ when there are at least `WG` of them (one division per row); four barriers per step. Contribution blocks with `m ≥ 64` are updated in 16×16 tiles with `W₂₁ = L₂₁D` and `L₂₁` staged 16 columns at a time in the F₁₁ buffer. Regime-C fronts run the same kernel with the panel in global memory and gain from the same changes.

LDLᵀ refactorization, `main` → step 1 → step 2:

| matrix | main ms | step 1 ms | step 2 ms | step 2 / cuDSS | step 2 / SDS Cholesky |
| --- | --- | --- | --- | --- | --- |
| lap2d_300 | 228 | 192 | 116 | 22.5× | 3.68× |
| lap3d_40 | 14,777* | 14,494* | 8,753* | 150× | 39.1× |
| HB/bcsstk17 | 206 | 167 | 115 | 34.2× | 4.14× |
| Boeing/bcsstk38 | 137 | 106 | 72.9 | 22× | 3.22× |
| GHS_psdef/apache2 | 92,576* | 97,296* | 50,736* | 104× | 49.3× |
| kkt_pglib_opf_case118_ieee_condensed_1 | 1.77 | 1.82 | 1.91 | 4.89× | 2.09× |
| kkt_pglib_opf_case118_ieee_condensed_10 | 1.72 | 1.79 | 1.87 | 5.35× | 2.12× |
| kkt_pglib_opf_case118_ieee_condensed_20 | 1.87 | 1.82 | 1.88 | 5.36× | 2.15× |
| kkt_pglib_opf_case118_ieee_k2_1 | 1.69 | 1.69 | 1.8 | 4.97× |  |
| kkt_pglib_opf_case118_ieee_k2_10 | 1.62 | 1.68 | 1.76 | 4.97× |  |
| kkt_pglib_opf_case118_ieee_k2_20 | 1.6 | 1.66 | 1.74 | 4.36× |  |
| kkt_pglib_opf_case1354_pegase_condensed_1 | 10.9 | 7.2 | 7.51 | 11× | 2.56× |
| kkt_pglib_opf_case1354_pegase_condensed_10 | 10.9 | 7.3 | 7.56 | 11.8× | 2.59× |
| kkt_pglib_opf_case1354_pegase_condensed_20 | 11 | 7.32 | 7.56 | 10.7× | 2.59× |
| kkt_pglib_opf_case1354_pegase_k2_1 | 8.31 | 6.6 | 6.95 | 8.51× |  |
| kkt_pglib_opf_case1354_pegase_k2_10 | 7.15 | 6.05 | 6.27 | 7.58× |  |
| kkt_pglib_opf_case1354_pegase_k2_20 | 9.1 | 6.71 | 6.84 | 8.48× |  |
| kkt_pglib_opf_case14_ieee_condensed_1 | 0.446 | 0.503 | 0.484 | 2.51× | 1.33× |
| kkt_pglib_opf_case14_ieee_condensed_10 | 0.443 | 0.52 | 0.517 | 2.26× | 1.52× |
| kkt_pglib_opf_case14_ieee_condensed_11 | 0.448 | 0.463 | 0.515 | 1.77× | 1.57× |
| kkt_pglib_opf_case14_ieee_k2_1 | 0.669 | 0.697 | 0.728 | 2.6× |  |
| kkt_pglib_opf_case14_ieee_k2_10 | 0.722 | 0.708 | 0.727 | 3.06× |  |
| kkt_pglib_opf_case14_ieee_k2_15 | 0.97 | 1.03 | 1.07 | 3.33× |  |

K2 dumps: inertia and `nperturbed` unchanged on all nine. T15 matrix on CUDA: 8.3 → 5.6 s. Experiment-1 criterion still not met (lap2d 3.7×, bcsstk17 4.1×, case1354 condensed 2.6× the Cholesky time); KKT dumps within +3–6% of step 1.

## Recommendations: Prioritized Experiment Plan

| # | Experiment | Payoff | Effort | Key measurement | Success criterion |
|---|---|---|---|---|---|
| 0 | **Baseline split.** SDS vs cuDSS (default, algo5, algo5 + hybrid execute) on the T04 harness plus larger pglib (case2869, case9241, case13659 if available) K2 and condensed dumps; Nsight Systems per phase | Essential | Low | analysis/factor/refactor/solve time; launches per refactor; host syncs; nnz(L), flops; GFLOP/s per front-width bucket (Nsight Compute on regime B/C kernels) | Reproducible table; launch count per refactor known |
| 1 | **Parallel BK/APTP pivot search** in regimes B/C (warp/workgroup argmax, F₁₁ in local memory, blocked a-posteriori check) | Very high | Med | LDLᵀ refactor time vs SDS Cholesky on the same pattern; nperturbed, inertia vs reference | LDLᵀ within 1.5× of SDS Cholesky flops-time; inertia unchanged on all K2 dumps |
| 2 | **Regime-C LDLᵀ via vendor BLAS**: KA pivoting of an F₁₁ panel (nb = 32–64), then cuBLAS trsm + gemm (L·D·Lᵀ update) for F₂₁/F₂₂; multi-workgroup per root front | Very high | Med | achieved TFLOP/s on fronts > 256; share of factor time in root fronts | ≥ 50% of cuBLAS DGEMM peak on fronts > 512; "C only" faster than reference on CPU and ≥ 5× on GPU |
| 3 | **Matching + scaling (T21)**: MC64 max-product symmetric scaling cached per pattern; device Ruiz equilibration per iteration as a cheap variant | Very high (accuracy) | Med | max\|L\|, factor error, nperturbed, relres after 0/2/5 IR on case1354 K2 | relres ≤ cuDSS algo5 (≤ 4e-7) with ≤ 5 IR steps; nperturbed ≤ cuDSS's 0–14 [github](https://github.com/exanauts/SparseDirectSolver.jl/pull/73) |
| 4 | **MadNLP end-to-end** (#28) on pglib via ExaModelsPower: K2 with SDS-LDLᵀ vs cuDSS-LDLᵀ, plus condensed/Lifted with SDS-Cholesky vs cuDSS-Cholesky | Essential | Low–Med | IPM iterations, inertia-correction count, time per iteration split (factor/solve/other) | iterations within ±2 of cuDSS, same objective to 1e-6 (the #28 criterion); [github](https://github.com/exanauts/SparseDirectSolver.jl/issues/28) fewer inertia corrections thanks to 2×2 pivots |
| 5 | **Launch minimization**: level merging into persistent kernels with atomic dependency counters (regimes A/B); one graph for refactor+solve+IR, replayed via a cached exec (not `@captured`) | High at small/medium n | Med | launches/refactor, CPU-side time, GPU idle gaps (Nsight timeline) | ≥ 3× fewer launches; refactor+solve faster than cuDSS for n ≤ 1e5 |
| 6 | **Solve path**: partitioned-inverse diagonal blocks (`solve_alg="algo1"`), batched GEMV for small supernodes, sync-free forward sweep; fused permute/scale/IR residual kernels | High | Med | solve latency (1 RHS and 2–6 RHS), backward error | solve ≤ cuDSS solve on all harness matrices; backward error unchanged within 10× |
| 7 | **Amalgamation and bin tuning**: sweep (max_width, zero_fraction, min_width), regime thresholds, local-memory budgets; size-sorted regime-A subtrees | Med | Low | flops vs time Pareto; stack memory/nnz(L) | 10–30% factor-time gain without > 15% nnz(L) growth |
| 8 | **Mixed precision**: FP32 factor + FGMRES-IR (T18 first); enable by IPM phase or by condensed SPD | Med | Med | IR iterations, time to 1e-10 relres, failures near convergence | ≥ 1.3× factor speedup with no change in IPM iteration count |
| 9 | **Uniform batching** (T17) for multi-scenario/MPC: shared symbolic, batched fronts across instances | Med–High for batch workloads | Med | throughput vs cuDSS uniform batch | ≥ cuDSS batch throughput |
| 10 | **Analysis caching / GPU symbolic**: serialize analysis per pattern; multithreaded AMD/ND; device etree/colcounts | Med (MPC/online) | Med–High | analysis time vs cuDSS (default and MT) | analysis ≤ cuDSS on the harness; zero cost when reusing a cached pattern |
| 11 | **Delayed pivots / APTP fallback** (T27) and portability (AMDGPU on MI250/MI300) | Med | High | relres on the hard delayed-pivot case; AMD timings | T27 tests; portability as a unique selling point (cuDSS is NVIDIA-only) |

**Benchmark matrices.** MadNLP KKT dumps (K2 and condensed, early, middle and converged iterates) from pglib-opf via ExaModelsPower: case118, case1354_pegase, case2869_pegase, case9241_pegase, case13659_pegase, and ACTIVSg25k/70k if available. Add COPS and optimal-control instances from the MadNLP/HybridKKT benchmarks. From SuiteSparse: TSOPF_RS_*, rajat21, and an SPD set (e.g. thermal2, G3_circuit, audikw_1, Serena) for Cholesky parity.

**Measurement protocol.**
- Fixed GPU clocks; warm JIT; median of ≥ 10 runs.
- Report time per phase, and per IPM iteration in MadNLP.
- Use Nsight Systems for timelines, syncs and launch gaps.
- Use Nsight Compute for per-kernel FLOP/s and DRAM bytes, bucketed by front width (≤ 32, 33–64, 65–256, > 256).
- Use `CUDA.@profile` for quick launch counts.
- Accuracy metrics: relres and backward error, max|L|, nperturbed, n2x2, inertia vs reference.
- Memory metrics: stack and factor bytes.

**Overall success criteria vs cuDSS.**
1. **Accuracy parity at defaults:** SDS without manual tuning reaches what cuDSS needs algo5+IR to reach.
2. **Speed:** refactor+solve per IPM iteration at or below cuDSS on pglib ≤ 10k buses, and within 1.5× on the largest cases.
3. **Robustness:** fewer MadNLP inertia corrections, from 2×2 pivots and signed perturbation.
4. **Portability:** a working AMD backend, which cuDSS cannot match.

## Caveats
- I could not retrieve SDS's per-matrix timing ratios (bench/comparison) or the full T04 table, so the size of the current speed gap is not quantified here. Experiment 0 is mandatory before committing effort.
- Issue #75's timings are from the KA CPU backend and show algorithmic cost, not GPU performance. [github](https://github.com/exanauts/SparseDirectSolver.jl/issues/75)
- cuDSS internals (supernodal vs multifrontal, kernel design) are not public. The claims here rely on NVIDIA docs and release notes.
- SDS is developed largely by automated agents and moves daily (PRs dated Oct 1–2, 2026). Re-check task status before planning around it.
- The speedup numbers cited from other codes (Tacho/Kokkos SpTRSV, SuperLU_DIST mixed precision) come from different hardware and matrix classes. Treat them as indicative only.
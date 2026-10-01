# GPU Sparse Direct Solvers for IPM KKT Systems: State of the Art and a Design for a Portable Pure-Julia cuDSS Replacement

**Bottom line:** A competitive, vendor-agnostic cuDSS replacement for MadNLP-class KKT workloads should start as a *fixed-pattern, CPU-analyzed, level-batched multifrontal (or supernodal) solver*: Cholesky first, then LDLᵀ with static (perturbation) pivoting confined to each front's fully-summed block plus exact inertia counting, with iterative refinement/FGMRES on top. The win for power-flow KKT matrices comes from beating cuDSS on *many tiny supernodes* (subtree-per-workgroup fusion, batched small dense kernels, precomputed scatter maps) rather than from dense BLAS-3 throughput, and the main portability hazards are KernelAbstractions' immature subgroup/fence support, the lack of inter-workgroup forward-progress guarantees on Metal (and likely Intel), and Metal's lack of FP64.

## TL;DR
- **cuDSS is a classic three-phase supernodal-style solver whose GPU advantage on KKT systems comes from GPU-resident refactorization, not exotic pivoting:** reordering (custom METIS-like nested dissection by default, or AMD/COLAMD/BTF+COLAMD) runs on the host; symbolic factorization, numeric factorization and solves run on the GPU; indefinite systems get only *local* (within-supernode) diagonal pivoting plus epsilon replacement of tiny pivots, with no 2×2 Bunch-Kaufman support, an inertia query, uniform/non-uniform batching, and hybrid-memory/MG/MGMN modes.
- **No open, portable solver currently matches cuDSS on IPM KKT workloads:** generic GPU solvers (SuperLU_DIST, STRUMPACK, SSIDS, PaStiX, cuSOLVER QR) were no faster than single-threaded MA57 on ACOPF matrices; refactorization-style GPU LU (KLU+cuSolverRf/GLU, Ginkgo's sync-free up-looking LU) gave 2.4–3.4× factorization speedups, which Świrydowicz et al. (IJEPES 155, 2024) project to "1.6–1.9× overall speedup in ACOPF"; and condensed-space IPMs (LiftedKKT/HyKKT) + cuDSS Cholesky/LDLᵀ gave ~10× end-to-end speedups on large OPF instances.
- **Recommended path:** CPU analysis (AMD.jl/Metis.jl + elimination tree + relaxed supernode amalgamation + static level schedule), then a KernelAbstractions numeric phase with (i) fused subtree-per-workgroup kernels for the tiny-front bottom of the tree, (ii) size-binned batched dense kernels for mid levels, (iii) vendor BLAS/LAPACK for the few large root fronts; Phase 1 Cholesky, Phase 2 LDLᵀ with static pivoting + inertia, Phase 3 LU with static pivoting (SuperLU_DIST-style), and only CUDA/ROCm-specific sync-free triangular solves.

## Key Findings

### 1. What is publicly known about cuDSS internals

**Pipeline and phase placement.** NVIDIA documents a "multi-stage execution with three main phases: analysis (consisting of reordering and symbolic factorization), numerical factorization and solving," with optional refactorization and solve sub-phases (forward/backward substitution, permutations, iterative refinement).\[1\]\[2\] When hybrid execution is off, "reordering (a major part of the analysis phase) is executed on the host, while symbolic factorization (another part of the analysis phase), numerical factorization and solve are executed on the GPU."\[3\]\[4\] The analysis phase is always synchronous; factorization and solve are asynchronous (and CUDA-Graph-capturable) unless hybrid memory, hybrid execution or MGMN modes are enabled.\[5\]

**Reordering.** The default is "a custom METIS-like (nested dissection) code"; NVIDIA states reordering stays on the host "as it is known to be extremely hard to accelerate the underlying graph algorithms on a GPU," and warns that for some matrices reordering dominates total time.\[6\] Since v0.8.0 the options are `NESTED_DISSECTION` (default), `AMD`, `COLAMD`, `BTF_COLAMD`, and `NONE`;\[7\] multi-threaded reordering exists for the default algorithm, and users can supply their own ND partition tree.\[2\]\[7\]\[8\]\[9\] The third-party license list (SuiteSparse AMD, COLAMD, METIS, HSL) confirms which classical codes are embedded.\[2\] Schwan, Kuhn & Jones's block-tridiagonal Cholesky study (arXiv 2601.03754, FP64 on RTX 3080/RTX 5090) observed "jumps in computation time at powers-of-two boundaries," which suggests cuDSS exploits the ND separator tree, and reported that their structure-specialized code "consistently achieves approximately 2× speedup" over cuDSS because cuDSS cannot exploit dense block structure; this is inference by outside authors, not NVIDIA documentation.

**Factorization scheme.** NVIDIA does not document whether cuDSS is supernodal or multifrontal. The docs refer to "the relatively small set of columns (usually called a supernode)," expose `CUDSS_CONFIG_FACTORIZATION_ALG` and `SOLVE_ALG` variants, a `USE_SUPERPANELS` option, and v0.7.0 "re-introduced factorization algorithm CUDSS_ALG_1 with improved performance for matrices with very sparse factors" — i.e., NVIDIA itself maintains a separate code path for the very-sparse-factor regime typical of power-grid KKT matrices.\[7\]\[10\]\[11\] Supported factorizations are Cholesky, LDLᵀ and LU, in single, double and double-double precision, with 32/64-bit indices.\[1\]\[2\]\[4\]

**Pivoting.** For the ND/AMD orderings cuDSS uses *local partial pivoting*: it searches "for the pivot element within the diagonal sub-block of the relatively small set of columns (usually called a supernode)"; with COLAMD/BTF_COLAMD it uses *global* pivoting over full columns or rows.\[11\]\[12\] Since v0.8.0, `CUDSS_PIVOT_AUTO` maps symmetric/Hermitian indefinite matrices to `CUDSS_PIVOT_DIAGONAL` and general matrices to `CUDSS_PIVOT_LOCAL_BLOCK`.\[7\]\[12\] SPD matrices are not pivoted.\[11\] Tiny pivots are replaced by a pivot epsilon (static perturbation), optionally scaled (`PIVOT_EPSILON_ALG_SCALED`), and the number of perturbed pivots is reported (`CUDSS_DATA_NPIVOTS`).\[6\]\[11\]\[12\]\[13\] An NVIDIA forum reply states the default epsilon is 1e-5 (single) and 1e-13 (double) and that cuDSS does zero iterative-refinement steps by default.\[13\] Pacaud et al. note the 2×2 pivoting used by sparse indefinite CPU solvers "is not supported,"\[4\] and the MadNCL authors note cuDSS does not let the user choose the sign of the replacement epsilon.\[14\]

**Inertia.** `CUDSS_DATA_INERTIA` returns the positive and negative inertia indices (two integers), which MadNLP uses for inertia-corrected regularization.\[12\] With static perturbation this is the inertia of the *perturbed* factorization, so it must be read together with `NPIVOTS`.

**Batching, multi-GPU, memory.** cuDSS supports uniform batching (same sparsity pattern; `UBATCH_SIZE`/`UBATCH_INDEX`) and non-uniform batching,\[1\] a Schur-complement mode, deterministic mode, hybrid host/device memory (factors partly on the host, "not normally a speed optimization"), hybrid host/device execution, single-node multi-GPU (MG), and multi-node (MGMN) with a user-pluggable OpenMPI/NCCL layer; MG/MGMN do not support the COLAMD-family reorderings or batching.\[2\]\[10\]\[15\]\[16\]

**Strengths:** GPU-resident refactorization that reuses analysis; low launch overhead on very sparse problems (as the MadNLP results show); inertia; batching; scale-out modes. **Limitations:** closed source and NVIDIA-only; no 2×2/Bunch-Kaufman or delayed pivots; no user control over the epsilon sign; host-side, synchronous analysis; accuracy failures on badly scaled matrices when pivots are silently perturbed (a user-reported case on the NVIDIA forum); refinement off by default.\[13\]

### 2. Evidence from IPM/KKT workloads

- **Generic GPU solvers fail on ACOPF KKT matrices.** Świrydowicz et al. benchmarked SuperLU, STRUMPACK, SPRAL-SSIDS, PaStiX and cuSolverSp on ACOPF systems and found "none of the GPU-accelerated packages was substantially better than MA57," with GPU acceleration often slower than CPU. For example, on a 238K-row case MA57 took 0.8 s versus 4.8 s for SSIDS on the GPU.\[17\] Their diagnosis is that power-grid KKT matrices have "no inherent block structure," so supernodal and multifrontal codes that rely on dense blocks do poorly; instead, "we require fine-grained scheduling of individual variable eliminations."\[17\]
- **Refactorization works.** KLU (AMD ordering, CPU) for the first system, then cuSolverRf/GLU or Ginkgo's GPU LU for all later systems with the pivot sequence frozen, gave "matrix factorization speedup of 2.4–3.4× and triangular solve speedup of 3.2–5.8×," projecting to 1.6–1.9× overall ACOPF speedup; Cojean et al. (Ginkgo, IJHPCA 2024) separately report that "ACOPF is overall 1.8 − 2.4× faster when using the Ginkgo linear solver on the AMD MI250X GPU instead of MA57 on the AMD EPYC 7A53 CPU." Typically only 1–2 refinement iterations (FGMRES-style, using the factors as a preconditioner) were needed. IPM regularization acts like static-pivoting perturbation, which is what makes pivot reuse safe. Ginkgo's LU is an up-looking factorization mapping each row to a warp/wavefront, with sync-free "ready flag" dependency resolution that relies on NVIDIA/AMD scheduling thread blocks in monotonic order with forward-progress guarantees.\[17\] LU gives no inertia, so HiOp had to use its inertia-free IPM option.\[17\]
- **Condensed-space IPMs + cuDSS.** HyKKT (Golub–Greif) factorizes K_γ = K + γGᵀG with Cholesky and runs CG on the Schur complement; non-optimized HyKKT "outperforms MA57 by 10× on largest problems."\[3\]\[18\] LiftedKKT relaxes equalities so the condensed system is SPD. With MadNLP + ExaModels + cuDSS, Pacaud et al. report "a remarkable tenfold acceleration in solving large-scale OPF instances," but warn that a single dense row densifies the condensed matrix and that condensed systems are more ill-conditioned (in a controlled, structured way).\[4\]\[19\] NVIDIA's blog (Nov 2024) reports more than 10× speedup over the previous state of the art on an AMD EPYC 7443 CPU for the Eastern Interconnection ("approximately 70K nodes," a 674K-dimensional system), cutting time to solution from more than 3 minutes to under 20 seconds on an A100. In practice MadNLP still often uses cuDSS **LDLᵀ** on these "SPD-by-construction" condensed matrices for robustness.\[3\] MadIPM uses LDLᵀ with fixed primal-dual regularization (1e-8, −1e-8), and MadNCL sets the cuDSS pivot epsilon to 1e-10.\[14\]\[20\]\[21\]
- **Implication for your solver:** the target workload is quasi-definite or SPD-after-regularization with a fixed pattern and hundreds of refactorizations. Static pivoting + inertia + cheap refinement covers MadNLP/MadIPM/MadNCL's needs; full Bunch-Kaufman with delayed pivots is a robustness feature for the *unreduced* K2 system, not a prerequisite for parity with cuDSS.

### 3. Survey of other GPU sparse direct solvers

| Solver | Factorization scheme | GPU strategy | Pivoting | Portability | License |
|---|---|---|---|---|---|
| **cuDSS** (NVIDIA) | Supernodal-style (exact scheme undocumented), Cholesky/LDLᵀ/LU | Host reordering; GPU symbolic + numeric + solve; batching; MG/MGMN; hybrid memory | Local (in-supernode) diagonal/partial, or global with COLAMD; epsilon perturbation; no 2×2 | NVIDIA only | Proprietary |
| **CHOLMOD** (SuiteSparse) | Left-looking supernodal Cholesky | cuBLAS on large supernodes; "subtree" algorithm (Rennich, Stosic & Davis 2016) streams etree subtrees fully onto GPU and batches small dense ops (up to ~2× vs CPU) | None (SPD) | CUDA only | GPL for the supernodal module (verify) |
| **SuperLU_DIST** 8.x/9.x | Right-looking supernodal LU, 2D/3D communication-avoiding | GEMM offload + GPU Schur updates; multi-GPU NVIDIA & AMD; GPU SpTRSV | Static pivoting (MC64-style + tiny-pivot perturbation) + iterative refinement; mixed-precision FP32 LU + FP64 IR | CUDA, HIP | BSD |
| **STRUMPACK** | Multifrontal (+ HSS/BLR compression) | GPU subtrees that fit in memory; batched dense on fronts; ~10× on 24 V100s vs CPU | Partial pivoting inside fronts (+ static MC64 matching) | CUDA, HIP (SYCL work) | BSD\[22\] |
| **MUMPS** | Multifrontal, distributed | No mature public GPU numeric phase that I could verify in this research | Threshold partial, delayed pivots, 2×2 | CPU (MPI) | CeCILL-C |
| **PaStiX 6** | Right-looking supernodal (+ BLR) | StarPU/PaRSEC runtime tasks offloaded to GPU for POTRF/SYTRF/HETRF/GETRF; no GPU kernels for low-rank blocks | Static pivoting within supernodes | CUDA (via runtimes) | LGPL (verify) |
| **Tacho** (Trilinos/ShyLU) | Multifrontal Cholesky/LDLᵀ | Originally Kokkos tasking; now *level-set scheduling* for factorization and SpTRSV, four SpTRSV variants trading setup cost and stability for speed | Pivoting only within fronts (LDLᵀ; skew-symmetric LDLᵀ sequential) | Kokkos: CUDA, HIP, SYCL, OpenMP | BSD-2-Clause\[23\]\[24\] |
| **Basker** (ShyLU) | Hierarchical parallel sparse LU (BTF + ND) | Threaded/Kokkos CPU, aimed at circuit matrices | Partial pivoting | Kokkos CPU | BSD |
| **Ginkgo** (1.5+/1.6) | Up-looking LU and Cholesky; GPU parallel symbolic Cholesky; symbolic LU for near-symmetric patterns | Row-per-warp sync-free numeric factorization and triangular solves; full refactorization reuse | None in the numeric phase (relies on a prior CPU pivot order/scaling) | CUDA, HIP, SYCL, OpenMP | BSD-3-Clause |
| **SPRAL SSIDS** | Multifrontal LDLᵀ | v1: GPU-only; v2: CPU/GPU, task-based | Threshold partial pivoting (v1), *a posteriori threshold pivoting* (APTP, v2), delayed pivots, 2×2 | CUDA | BSD-3 (verify)\[25\]\[26\]\[27\] |
| **HSL MA97 / MA57 / MA86** | Multifrontal (MA97, MA57) / supernodal (MA86) | CPU only | Threshold 1×1/2×2, delayed pivots | CPU | HSL license |
| **symPACK** | Fan-out supernodal Cholesky | GPU-capable, outperforms PaStiX in its paper | None (SPD) | CUDA + UPC++ | BSD (verify) |
| **PanguLU** (SC23 best paper) | Regular 2D blocking with blocks stored sparse (no extra fill); distributed | Selects among block-wise sparse BLAS kernels per block pattern | Static (pre-pivoting) | CUDA (+CPU) | AGPL-3.0 |
| **Caracal** (SC25) | Two-level coarse/fine block sparse LU, GPU-resident | Static fine-grained scheduling over multiple streams; memory pool/caching; up to 21% of A100 peak, 7× over SuperLU_DIST, 94× over PanguLU, 16× over PaStiX on one GPU | Static | CUDA | Research code |
| **GLU / GLU3.0** | Left-looking, column-level-scheduled LU for circuit matrices | Three kernel modes keyed to columns per level; 13× (arith. mean) over GLU2.0 | Relies on CPU pre-pivoting | CUDA | Open source (verify) |
| **cuSolverSp / cuSolverRf** | QR/Cholesky/LU (Sp); refactorization (Rf) | Rf: GPU numeric refactorization with user-supplied L/U pattern & permutations | Rf: frozen pivot sequence | CUDA; **deprecated** in favor of cuDSS | Proprietary\[28\] |
| **rocSOLVER / rocALUTION** | rocSOLVER provides dense batched LAPACK; rocALUTION is mainly iterative/ILU | Dense batched potrf/getrf (useful as building blocks) | — | ROCm | MIT\[29\]\[30\] |
| **Julia** | CliqueTrees.jl (pure-Julia supernodal multifrontal Cholesky with BLAS/LAPACK calls and Julia ordering); PureUMFPACK.jl; LinearSolve's pure-Julia KLU and supernodal LU | CPU only; no GPU sparse direct solver found | — | CPU | MIT-style |

Takeaway: the solvers that do well in production on GPUs either (a) have large fronts and use dense BLAS-3 (STRUMPACK, SuperLU_DIST, CHOLMOD, PaStiX, symPACK, Caracal) or (b) are fine-grained and refactorization-based (Ginkgo, GLU, cuSolverRf). Power-grid KKT matrices need (b)'s latency profile with (a)'s robustness for the top of the tree. cuDSS appears to cover both regimes with separate factorization algorithms.

### 4. Algorithmic components: state of the art on GPU

**Fill-reducing ordering — keep on CPU, run once.** cuDSS, Ginkgo, KLU-based refactorization, Tacho and CHOLMOD all order on the host. GPU symbolic analysis exists (Ginkgo's parallel symbolic Cholesky; GSoFa for unsymmetric LU symbolic factorization),\[31\]\[32\] but with a fixed pattern its cost is amortized over hundreds of IPM iterations. In the Świrydowicz study, KLU's AMD gave the sparsest factors for ACOPF.\[17\] Recommendation: offer AMD (AMD.jl or a pure-Julia port, as in CliqueTrees.jl/PureUMFPACK.jl) and METIS ND (Metis.jl). Auto-select by predicted fill/flops *and by predicted critical path/level count*: ND produces bushier, shallower trees (more parallelism), while AMD often gives lower fill on power grids.

**Symbolic factorization, etree, supernodes.** Use the standard pipeline: etree, postorder, column counts, fundamental supernodes, then *relaxed amalgamation* (Ashcraft–Grimes style). Karsavuran et al. (LBNL, arXiv 2409.14009) stopped merging "when the cumulative increase in factor matrix storage went beyond 25%." For GPU work, amalgamation is the main lever against tiny-supernode overhead. Tune it differently than on CPU (accept more explicit zeros to reach 8–32-column panels), and precompute every scatter/extend-add index map once. CHOLMOD's GPU path only sends descendant supernodes of at least 256 rows × 32 columns to the GPU, which shows how small the profitable granularity is for cuBLAS-style kernels.\[33\]\[34\]

**Supernodal vs multifrontal; looking direction.** Multifrontal (Tacho, SSIDS, STRUMPACK, MUMPS) gives contiguous dense fronts, which batch well and are the natural home for in-front pivoting and delayed pivots. It costs contribution-block memory and extend-add traffic. Right-looking supernodal (SuperLU_DIST, PaStiX) updates in place and suits static pivoting and static data structures. Left-looking (CHOLMOD) reduces writes but serializes gathers. Up-looking row-wise (Ginkgo) suits very sparse factors. For a fixed pattern, the multifrontal memory high-water mark and every front's location can be precomputed, removing the dynamic-allocation drawback (Tacho needed a memory pool and respawning tasks precisely because it lacked this).\[35\]

**Scheduling small supernodes.** The options, roughly ordered by maturity:
1. Level-set scheduling of the assembly tree with one or a few batched launches per level (Tacho's current approach; CHOLMOD batches within subtrees).\[23\]
2. Subtree-to-device streaming (CHOLMOD).\[36\] The GPU-internal analogue is *subtree-to-workgroup*: one workgroup factors a whole leaf subtree serially in shared memory, which collapses the deep, narrow bottom of the tree into one launch.
3. Static multi-stream fine-grained scheduling (Caracal).\[37\]
4. Sync-free dataflow with ready flags (Ginkgo, SFLU). This needs forward-progress guarantees.\[17\]

The ACOPF evidence says (2)+(1) matter most: tiny fronts and many levels make launch latency, not FLOPs, the bound.

**Indefinite pivoting on GPU.**
- *Static pivoting + perturbation + iterative refinement* (SuperLU_DIST GESP: replace tiny pivots with roughly √ε‖A‖; cuDSS epsilon replacement): fully static structure, ideal for refactorization.\[38\]\[39\] The risk is silent loss of accuracy, so refinement is mandatory.
- *Bunch-Kaufman / 1×1–2×2 pivoting restricted to the fully-summed block of each front/supernode* (cuDSS's "local" pivoting, Tacho, PaStiX): keeps the structure static. Stability is only as good as the pivot candidates inside the block; fall back to perturbation when no acceptable pivot exists.
- *Threshold partial pivoting with delayed pivots* (MA57/MA97/MUMPS/SSIDS v1): robust, but delays change front sizes at run time and break the fixed-pattern design.
- *A posteriori threshold pivoting* (SSIDS v2): factor a block optimistically without pivoting, test the threshold condition afterwards, and backtrack only the failed columns. This has "the same numerical robustness as the TPP strategy" and is much more parallel.\[26\]\[40\] It is the best-known GPU-friendly compromise if delays are ever needed.
- *Avoid pivoting by reformulation* (LiftedKKT, HyKKT, quasi-definite regularization in MadIPM/MadNCL): the dominant approach in today's GPU IPMs.\[4\]\[20\]

**Inertia.** With LDLᵀ, count the signs of D: a 1×1 pivot contributes its sign, and a 2×2 block with negative determinant contributes one positive and one negative. Use a per-front reduction followed by a global sum. Report both perturbed-pivot counts and near-zero pivots, because static perturbation means you get the inertia of A+E. LU gives no inertia, which is why HiOp needed an inertia-free IPM with LU-based GPU solvers.\[17\]

**Triangular solves.** SpTRSV has arithmetic intensity O(1) and more dependency than SpMV.\[41\] The approaches:
- *Level-set* (batched TRSV/GEMV per level): portable, but bound by global synchronizations.
- *Reduced-synchronization*: Park et al. report 1.6× over level-based scheduling on CPU.\[41\]
- *Sync-free/ready-flag* (Ginkgo, Liu et al.): fastest on NVIDIA/AMD, not portable.
- *Partitioned inverse*: turns the solve into a sequence of SpMVs; offered as a Tacho variant. More setup cost and some stability risk, but attractive when each factorization is followed by several solves (refinement, HyKKT CG).\[23\]\[42\]
- *Multi-GPU via NVSHMEM*: Ding et al. reached 6.1× over cuSPARSE csrsv2.\[41\]

For KKT refinement loops the solve count per factorization is high (cuDSS defaults to 0 refinement steps; MA57 averaged about 6 on OPF matrices), so solve latency matters as much as factorization.\[13\]\[17\]

**Refactorization.** Everything from analysis can be reused: permutation, scaling, etree, supernode partition, front sizes, extend-add maps, memory layout and the level schedule.\[17\] Ginkgo, cuSolverRf, KLU and cuDSS all do this. The numeric phase should be a pure "values-in, factors-out" kernel sequence with no allocation or host synchronization. That makes it capturable in a CUDA/HIP graph on those backends, which KernelAbstractions cannot express portably.

**Mixed precision.** SuperLU_DIST ships FP32 LU + FP64 iterative refinement.\[43\] For ill-conditioned late-IPM KKT systems, FP32 factors plus plain IR will likely stall. GMRES-IR (factors as preconditioner, as in the FGMRES approach of Świrydowicz et al.) is the safer variant. Keep FP64 as the default, and make FP32 + GMRES-IR an option for the SPD condensed path and for Metal, which lacks hardware FP64.

### 5. Recent (2020–2026) work most relevant to you
- Świrydowicz et al., "Linear solvers for power grid optimization problems: a review of GPU-accelerated linear solvers" (Parallel Computing, 2022) and "GPU-resident sparse direct linear solvers for ACOPF analysis" (IJEPES 155, 2024; KLU+cuSolverRf/GLU and Ginkgo; first GPU-native sparse direct solver on both AMD and NVIDIA).\[44\]\[45\]
- Świrydowicz, Koukpaizan, Alam, Regev, Saunders, Peleš, "Iterative methods in GPU-resident linear solvers for nonlinear constrained optimization" (Parallel Computing, 2024).\[46\]
- Regev et al., HyKKT (Optim. Methods Softw. 38(2), 2023).\[47\]
- Shin, Anitescu, Pacaud, ExaModels/MadNLP condensed-space IPM for OPF (EPSR 236, 2024); Pacaud, Shin, Montoison, Schanen, Anitescu, condensed-space IPMs on GPUs (Math. Prog. Computation, 2026; arXiv 2405.14236); the NVIDIA cuDSS power-grid blog (Nov 2024).\[4\]\[19\]\[48\]\[49\]
- Shin et al., "GPU Implementation of Second-Order Linear and Nonlinear Programming Solvers" (arXiv 2508.16094; MadIPM, pivoting-free IPMs); Montoison, Pacaud, Saunders, Shin et al., MadNCL (arXiv 2510.05885).\[20\]\[50\]\[51\]
- Li, Lin, Liu, Sao, SuperLU_DIST v8 (TOMS, 2023); Li & Liu, "Parallel Sparse and Data-Sparse Factorization-based Linear Solvers" (2026 survey chapter).\[41\]\[43\]
- Fu et al., PanguLU (SC23); Ren et al., Caracal (SC25);\[52\]\[53\] the ShyLU node/Tacho progress report (arXiv 2506.05793, 2025); Ginkgo's parallel symbolic Cholesky (SC-W 2023).\[23\]\[31\]
- Portability in Julia: KernelForge.jl/KernelIntrinsics.jl (arXiv 2603.18695) shows that warp shuffles and acquire/release fences can be layered portably over KernelAbstractions, matching CUB/cuBLAS on A40 for scan, mapreduce and matvec.\[54\]

## Recommended Design

### Architecture
1. **Analysis (CPU, Julia, once per pattern):**
   - Symmetric scaling/matching (optional; MC64-like for LU).
   - Ordering via AMD.jl, Metis.jl, or a pure-Julia AMD port.
   - Etree, postorder, column counts, relaxed supernode amalgamation tuned for the GPU.
   - Assembly-tree partition into (a) *leaf subtrees* small enough for one workgroup's shared memory, (b) *mid-level fronts* binned by (nrows, ncols) size class, (c) a handful of *root fronts*.
   - All index maps (A→front scatter, child→parent extend-add, front→L storage) as flat device arrays.
   - Memory offsets for every front (static, no allocator at run time).
   - The level schedule.
2. **Numeric refactorization (device, KernelAbstractions):**
   - **Kernel A (subtree-per-workgroup):** each workgroup assembles, factors and extend-adds an entire small subtree serially in local memory. This removes most levels and launches and needs no inter-workgroup synchronization, so it is portable.
   - **Kernel B (level-batched fronts):** one launch per (level, size-bin), one workgroup per front, doing a register/shared-memory blocked dense POTRF/SYTRF (fully-summed block) + TRSM + SYRK/GEMM Schur update + extend-add via precomputed maps.
   - **Kernel C (root fronts):** call vendor dense LAPACK/BLAS via package extensions (CUSOLVER/CUBLAS in CUDA.jl, rocSOLVER/rocBLAS in AMDGPU.jl, oneMKL in oneAPI.jl), with a pure-Julia KA fallback.
3. **Pivoting (LDLᵀ):**
   - Bunch-Kaufman (or Bunch-Kaufman-rook) restricted to each front's fully-summed block, with 1×1 and 2×2 pivots.
   - If no acceptable pivot exists, static perturbation with a *user-selectable sign policy* (e.g., sign matching the expected inertia from primal/dual block membership, an improvement over cuDSS).
   - No delayed pivots in v1; record per-front flags for perturbed, tiny and 2×2 pivots.
4. **Inertia + diagnostics:** device reduction over D blocks, returning (n₊, n₋, n₀, n_perturbed), so MadNLP's inertia correction can raise δ_w/δ_c instead of trusting a perturbed factorization.
5. **Solve:**
   - Supernodal forward/backward substitution using the same subtree/level schedule (batched TRSV/GEMV, multiple RHS supported).
   - Optional partitioned-inverse diagonal blocks for refinement-heavy loops.
   - CUDA/ROCm-only sync-free variant behind a capability check.
   - Built-in iterative refinement and FGMRES/GMRES-IR with the factorization as preconditioner (default on, unlike cuDSS).
6. **API:**
   - Mirror CUDSS.jl/MadNLP's `AbstractLinearSolver` (analyze, factorize, refactorize, solve, inertia).
   - Support uniform batches (same pattern, many value sets), which maps directly to "one more grid dimension" in every kernel and serves ExaModels' batched/multi-scenario models.

### Reuse vs write from scratch
- **Reuse:**
  - SparseArrays (CSC on host); AMD.jl and Metis.jl (ordering).
  - CliqueTrees.jl (pure-Julia etree/supernodal Cholesky; a good CPU reference and possible source of the symbolic phase).\[55\]
  - GPUArrays/KA for array plumbing and Atomix.jl for atomics.\[56\]
  - Vendor dense routines for root fronts: CUBLAS `getrf_strided_batched!`; oneAPI.jl's `potrf/getrf/potrs _batched` wrappers.\[57\]\[58\]
  - MadNLP's existing KA kernels for KKT assembly.
- **Write from scratch:**
  - GPU-tuned amalgamation and schedule generation.
  - Fused small-front kernels (dense LDLᵀ/BK in shared memory, Cholesky, extend-add).
  - Inertia reduction; the supernodal SpTRSV.
  - Refinement drivers; the batched interface; benchmarking against cuDSS.
- **Vendor gaps:** I could not confirm which batched rocSOLVER/cuSOLVER Cholesky routines AMDGPU.jl and CUDA.jl currently wrap. Metal's MPS has only non-batched, FP32 LU/Cholesky (with a reported correctness bug for LU larger than 128×128 on macOS 27).\[59\] For small fronts you need your own kernels anyway.

### KernelAbstractions-specific risks
- **Warp/subgroup primitives:** KA exposes `@synchronize` (workgroup barrier) and Atomix atomics, but no stable portable shuffle, ballot or subgroup reductions.\[56\]\[60\] KernelInterface's subgroup API (`sub_group_barrier`, `shfl_down`) is still being redefined as of KA 0.10 / KernelInterface 0.4: "Which work-items form a sub-group, and which sub-groups are partial, is unspecified."\[61\]\[62\] KernelIntrinsics.jl provides shuffles, votes and acquire/release `@access`, but its authors call it "a proof of concept" with only CUDA and ROCm tested.\[54\]\[63\] **Design for shared-memory reductions with 1-D workgroups first; add warp-level fast paths behind backend extensions.**
- **Forward progress:** Apple states that Metal gives no guarantee threadgroups make concurrent forward progress or launch in order, and a Triton Apple-GPU issue reports decoupled look-back losing data.\[64\]\[65\] Sync-free SpTRSV and persistent-kernel DAG schedulers are therefore CUDA/ROCm-only. Use level or subtree scheduling as the portable baseline.
- **No dynamic parallelism or graphs in KA:** launch overhead per level is unavoidable portably. This is why subtree-per-workgroup fusion and level merging are first-class design goals. Use CUDA/HIP graph capture via backend extensions for the refactorize+solve sequence.
- **Precision:** Metal has no FP64. A Metal backend needs FP32 factors + FP64-on-CPU (or double-single) refinement, which is only acceptable for well-conditioned/condensed systems.
- **Atomics performance:** CUDA.jl notes Float32 atomic codegen regressions under default semantics.\[66\] Avoid atomics in extend-add by scheduling children per level, or by giving each parent a deterministic reduction order (which also yields bitwise reproducibility, matching cuDSS's deterministic mode).\[2\]

### Phased roadmap
1. **Phase 0 (2–4 weeks):**
   - Benchmark harness: NREL opf_matrices, pglib-opf KKT/condensed matrices dumped from MadNLP, CUTEst subsets.
   - Measure cuDSS analysis/factorization/solve times, flop counts (`CUDSS_DATA_FLOPS`) and supernode statistics.
   - Build the CPU symbolic pipeline and a CPU reference numeric factorization.
2. **Phase 1, Cholesky (SPD condensed KKT, HyKKT, LiftedKKT):**
   - Kernels A/B/C; level scheduling; supernodal SpTRSV; uniform batching.
   - Target: within 1.5× of cuDSS Cholesky refactorization+solve on pglib-opf condensed systems on NVIDIA, and working on AMD/Intel.
3. **Phase 2, LDLᵀ with static pivoting + inertia (MadNLP K2/K2r, MadIPM, MadNCL):**
   - In-front 1×1/2×2 BK, signed perturbation policy, inertia/perturbation counts.
   - Refinement/FGMRES on by default; refinement counts validated against MA27/MA57 iteration counts.
4. **Phase 3, LU with static pivoting:** MC64-style matching/scaling + symmetric-pattern ordering of A+Aᵀ; multifrontal or right-looking supernodal LU with tiny-pivot replacement (SuperLU_DIST GESP); Ginkgo-style up-looking kernel for extremely sparse factors.
5. **Phase 4, robustness/perf extras:**
   - APTP-style in-front a posteriori pivoting with optional delayed pivots (re-analysis on the host when delays occur).
   - FP32 + GMRES-IR; CUDA/ROCm sync-free solve paths; graph capture; multi-GPU via ND top-level splitting.

## Caveats
- cuDSS's internal algorithms (supernodal vs multifrontal, kernel design) are not publicly documented. Statements beyond the documentation (e.g., ND-tree exploitation) are outside inference. The pivot-epsilon defaults come from an NVIDIA forum reply, not the reference manual.\[13\]
- The ACOPF benchmark of generic GPU solvers versus MA57 dates from roughly 2021–2023 hardware and software versions. SuperLU_DIST, STRUMPACK and others have improved since.
- Several license entries (CHOLMOD supernodal GPL, PaStiX LGPL, SSIDS/symPACK BSD, GLU) are from general knowledge and marked "verify." MUMPS GPU status and the exact AMDGPU.jl/CUDA.jl batched-LAPACK wrappers were not confirmed.
- KernelAbstractions' subgroup API was changing at the time of writing (KA 0.10 / KernelInterface 0.4 unreleased).\[67\] Re-check before committing to warp-level designs.
- Static pivoting reports the inertia of a perturbed matrix. For hard nonconvex or degenerate NLPs, a CPU fallback (MA27/MA57 via MadNLP) remains the prudent safety net until delayed-pivot support exists.

## Sources

1. [NVIDIA cuDSS (Preview): A high-performance CUDA Library for Direct Sparse Solvers — NVIDIA cuDSS](https://docs.nvidia.com/cuda/cudss/index.html)
2. [NVIDIA cuDSS (Preview): A high-performance CUDA Library for Direct Sparse Solvers — NVIDIA cuDSS](https://docs.nvidia.com/cuda/cudss)
3. [Condensed-space methods for nonlinear programming on ...](https://arxiv.org/pdf/2405.14236)
4. [Condensed Interior-Point Methods for Scalable Nonlinear Programming on GPUs](https://arxiv.org/html/2405.14236)
5. [cuDSS General Description — NVIDIA cuDSS](https://docs.nvidia.com/cuda/archive/13.0.1/cudss/general.html)
6. [cuDSS Tips and tricks — NVIDIA cuDSS](https://docs.nvidia.com/cuda/archive/13.1.0/cudss/doc_output/tips_and_tricks.html)
7. [Release Notes — NVIDIA cuDSS](https://docs.nvidia.com/cuda/cudss/release_notes.html)
8. [cuDSS 0.8.0 Migration Guide — NVIDIA cuDSS](https://docs.nvidia.com/cuda/cudss/migration_guide.html)
9. [nvmath.bindings.cudss.ReorderingAlg — nvmath-python](https://docs.nvidia.com/cuda/nvmath-python/1.0.0/bindings/generated/nvmath.bindings.cudss.ReorderingAlg.html)
10. [docs.nvidia.com](https://docs.nvidia.com/cuda/nvmath-python/0.8.0/bindings/generated/nvmath.bindings.cudss.ConfigParam.html)
11. [cuDSS Advanced Features — NVIDIA cuDSS](https://docs.nvidia.com/cuda/cudss/doc_output/advanced_features.html)
12. [cuDSS Data Types — NVIDIA cuDSS](https://docs.nvidia.com/cuda/cudss/types.html)
13. [cuDSS is sometimes wrong where cuSparse and Umfpack succeed](https://forums.developer.nvidia.com/t/cudss-is-sometimes-wrong-where-cusparse-and-umfpack-succeed/342132)
14. [MadNCL: A GPU Implementation of Algorithm NCL for Large-Scale, Degenerate Nonlinear Programs](https://arxiv.org/html/2510.05885)
15. [NVIDIA cuDSS GPU Solver - OpenSeesMatlab Documentation](https://openseesmatlab.readthedocs.io/en/latest/getting_started/extensions/cudss_solver/)
16. [Solving Large-Scale Linear Sparse Problems with NVIDIA cuDSS](https://developer.nvidia.com/blog/solving-large-scale-linear-sparse-problems-with-nvidia-cudss/)
17. [GPU-Resident Sparse Direct Linear Solvers for Alternating Current Optimal](https://arxiv.org/pdf/2306.14337)
18. [HyKKT: A Hybrid Direct and Iterative Method for Solving KKT Linear Systems](https://web.stanford.edu/group/SOL/talks/22ICCOPT-shaked-regev.pdf)
19. [MadSuite](https://madsuite.org/)
20. [GPU Implementation of Second-Order Linear and Nonlinear Programming Solvers](https://arxiv.org/pdf/2508.16094)
21. [feat(ipm): normal equations factored by NVIDIA cuDSS on the device, opt-in, licence checked (#489) by Chirag6722 · Pull Request #683 · thegoodengineers/SANKHYA](https://github.com/thegoodengineers/SANKHYA/pull/683)
22. [High performance sparse multifrontal solvers on modern ...](https://www.sciencedirect.com/science/article/am/pii/S0167819122000059)
23. [ShyLU node: On-node Scalable Solvers and Preconditioners Recent Progresses and Current Performance](https://arxiv.org/pdf/2506.05793)
24. [Tacho (Software)](https://www.osti.gov/biblio/code-94709)
25. [Google Scholar](https://scholar.google.com/scholar_lookup?doi=10.1145/2756548)
26. [A new sparse symmetric indefinite solver using A Posteriori Threshold Pivoting](https://www.researchgate.net/publication/329101422_A_new_sparse_symmetric_indefinite_solver_using_A_Posteriori_Threshold_Pivoting)
27. [H2020-FETHPC-2014: GA 671633 NLAFET Working Note 21](https://www.nlafet.eu/wp-content/uploads/2017/03/NLAFET-WN21-Duff-Hogg-Lopez.pdf)
28. [1\. Introduction — cuSOLVER 13.4 documentation](https://docs.nvidia.com/cuda/cusolver/index.html)
29. [3.3. LAPACK Functions — rocSOLVER Documentation](https://rocm.docs.amd.com/projects/rocSOLVER/en/docs-5.7.1/api/lapack.html)
30. [rocSOLVER API — rocSOLVER Documentation](https://rocm.docs.amd.com/projects/rocSOLVER/en/docs-5.1.3/api/index.html)
31. [Parallel Symbolic Cholesky Factorization](https://dl.acm.org/doi/fullHtml/10.1145/3624062.3624253)
32. [Scalable Sparse Symbolic LU Factorization on GPUs](https://arxiv.org/pdf/2007.00840)
33. [User Guide for CHOLMOD: a sparse Cholesky factorization and](https://fossies.org/linux/SuiteSparse/CHOLMOD/Doc/CHOLMOD_UserGuide.pdf)
34. [(PDF) Accelerating Sparse Cholesky Factorization on GPUs](https://www.researchgate.net/publication/304531882_Accelerating_Sparse_Cholesky_Factorization_on_GPUs)
35. [Tacho: Memory-scalable task parallel sparse cholesky factorization](https://www.sandia.gov/research/publications/details/tacho-memory-scalable-task-parallel-sparse-cholesky-factorization-2018-08-03/)
36. [Accelerating sparse Cholesky factorization on GPUs - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S016781911630059X)
37. [Caracal: A GPU-Resident Sparse LU Solver with Lightweight Fine-Grained Scheduling](https://dl.acm.org/doi/full/10.1145/3712285.3759792)
38. [Sparse linear solvers Laura Grigori ALPINES INRIA and LJLL, UPMC](https://people.eecs.berkeley.edu/~demmel/cs267_Spr15/Lectures/lecture15_SparseDirectSolvers_short_Grigori.pdf)
39. [SuperLU\_DIST: A scalable distributed-memory sparse direct solver for unsymmetric linear systems](https://escholarship.org/uc/item/83z8696r)
40. [A New Sparse \$LDL^T\$ Solver Using A Posteriori Threshold Pivoting](https://www.researchgate.net/publication/339832172_A_New_Sparse_LDLT_Solver_Using_A_Posteriori_Threshold_Pivoting)
41. <https://arxiv.org/pdf/2602.14289>
42. [An Experimental Study of Two-Level Schwarz Domain Decomposition Preconditioners on GPUs](https://arxiv.org/pdf/2304.04876)
43. [Newly Released Capabilities in the Distributed-Memory SuperLU Sparse Direct Solver - Oak Ridge National Laboratory](https://impact.ornl.gov/en/publications/newly-released-capabilities-in-the-distributed-memory-superlu-spa/)
44. [GPU-resident sparse direct linear solvers for alternating current optimal power flow analysis](https://www.ornl.gov/publication/gpu-resident-sparse-direct-linear-solvers-alternating-current-optimal-power-flow)
45. [GPU Accelerated Security Constrained Optimal Power Flow](https://link.springer.com/article/10.1007/s11081-026-10085-6)
46. [Iterative methods in GPU-resident linear solvers for nonlinear constrained optimization](https://ouci.dntb.gov.ua/en/works/962dOvW9/)
47. [Condensed interior-point methods for scalable nonlinear programming on GPUs](https://link.springer.com/article/10.1007/s12532-026-00335-0)
48. [Simulation / Modeling / Design](https://developer.nvidia.com/blog/nvidia-cudss-library-removes-barriers-to-optimizing-the-us-power-grid)
49. [GitHub - MadNLP/MadNLP.jl: A solver for nonlinear programming with GPU support · GitHub](https://github.com/MadNLP/MadNLP.jl)
50. [MadNCL: A GPU Implementation of Algorithm NCL for ...](https://www.arxiv.org/pdf/2510.05885)
51. [GPU Implementation of Second-Order Linear and Nonlinear Programming Solvers](https://arxiv.org/html/2508.16094v1)
52. [Awards - SC23 - SC Conference](https://sc23.supercomputing.org/program/awards/)
53. [Parallel Sparse and Data-Sparse Factorization-Based Linear Solvers](https://link.springer.com/chapter/10.1007/978-3-032-33428-2_8)
54. [High-Performance Portable GPU Primitives for Arbitrary Types and Operators in Julia](https://arxiv.org/pdf/2603.18695)
55. [Pure-Julia Sparse Cholesky - Numerics - Julia Programming Language](https://discourse.julialang.org/t/pure-julia-sparse-cholesky/131293)
56. [Atomic operations with Atomix.jl · KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/examples/atomix/)
57. [Using getrf\_batched to find matrix inverses](https://discourse.julialang.org/t/using-getrf-batched-to-find-matrix-inverses/131449)
58. [oneAPI.jl 1.5: Ponte Vecchio support and oneMKL improvements](https://www.juliabloggers.com/oneapi-jl-1-5-ponte-vecchio-support-and-onemkl-improvements/)
59. [Metal.jl](https://zenodo.org/records/21834491)
60. [API · KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/api/)
61. [KernelInterface: don't promise how work-items form sub-groups by maleadt · Pull Request #815 · JuliaGPU/KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl/pull/815)
62. [Implement KernelInterface, and support KernelAbstractions 0.10 by maleadt · Pull Request #3314 · JuliaGPU/CUDA.jl](https://github.com/JuliaGPU/CUDA.jl/pull/3314)
63. [GitHub - epilliat/KernelIntrinsics.jl: Julia functions for invoking GPU-specific memory access instructions](https://github.com/epilliat/KernelIntrinsics.jl)
64. [AppleGPU: a program polling a flag with \`tl.load\` (volatile or not) never sees another program's store in the same launch; a decoupled-lookback sort loses elements · Issue #147 · triton-lang/triton-ext](https://github.com/triton-lang/triton-ext/issues/147)
65. <https://developer.apple.com/forums/thread/672532>
66. [Expose fast math for KernelAbstractions CUDA kernels by maleadt · Pull Request #3282 · JuliaGPU/CUDA.jl](https://github.com/JuliaGPU/CUDA.jl/pull/3282)
67. [KernelInterface 0.4: a better API for writing kernels · Issue #810 · JuliaGPU/KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl/issues/810)

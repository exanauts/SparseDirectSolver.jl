# SparseDirectSolver vs cuDSS across seven GPU platforms — October 2026 campaign

Benchmark: refactorization of the condensed ACOPF KKT system of
`pglib_opf_case78484_epigrids` (n = 674562, nnz(L) ≈ 14.3M after the tuned
ordering), the operation an interior-point method repeats ~200 times per solve.
Every number below is a median of gated runs (info = 0, elementwise factor
delta ~1e-9 vs the stock factorization, residual matching stock), interleaved
A/B against a baseline in the same process. Methodology and knob mechanisms:
`bench/ARCH-TUNING.md`.

## Headline results

| device | arch | f64:f32 | cuDSS refact | SDS fused refact | ratio | SDS config (per-arch defaults) |
|---|---|---|---|---|---|---|
| Quadro GV100 | sm_70 | 1:2 | 24.3 ms | **24.1 ms** | **0.99×** | host wides, 16K budgets, graph |
| NVIDIA H200 | sm_90 | 1:2 | 11.1 ms | **12.0 ms** | 1.08× | host wides, 16K, 96-wide panels, graph |
| NVIDIA A100-80GB | sm_80 | 1:2 | 17.3–18.7 ms | 19–21 ms | ~1.1× | host wides, 16K |
| RTX Pro 6000 (Blackwell) | sm_120 | 1:64 | 13.5 ms | 15.8 ms | 1.17× | merged wides, 8K |
| NVIDIA L40S | sm_89 | 1:64 | 12.2 ms | 15.2 ms | 1.25× | merged wides, 8K, graph |
| RTX 3080 (10 GB) | sm_86 | 1:64 | — | 32.4 ms | — | merged wides (kernel-level; e2e needs ~17 GB) |
| AMD Radeon VII | gfx906 | 1:4 | n/a | 40.0 ms | — | merged wides, 8K, workgroup 256, spin cap 16 |

Solve (triangular sweeps): SDS 3.61 ms vs cuDSS 3.77 ms on the GV100 — ahead.
End-to-end MadNLP (`SparseCondensedKKTSystem`, tol 1e-6): identical iteration
counts and objectives as cuDSS on every platform tested; AMD end-to-end linear
algebra improved 36.5 → 34.0 s from the per-arch defaults alone.

Within one day of per-architecture tuning the campaign moved: L40S 34.1 → 15.2,
H200 13.6 → 12.0, Radeon VII 66 → 40.0, GV100 29.3 → 24.1 (the division-free
and triangle-enumeration kernel work, PR #125, is included in all rows).

## Why the times look the way they do

**The workload is scheduling-bound, not flop-bound.** One refactorization is
~1.2 GFLOP spread over ~92k mostly tiny fronts (median supernode width ~7–16).
The instruction stream is dominated by index arithmetic, gathers/scatters,
barriers, dependency counters and kernel-launch latency — all full-rate on
every chip — with f64 FMAs a small minority. Two measurements pin this:

- H200 has ~50× the f64 throughput of an L40S, yet is only 1.27× faster for SDS
  and 1.1× for cuDSS. Both solvers' times track *core clock*, not FLOP rate.
- cuDSS across chips orders as A100 (1.4 GHz) 17.3 > H200 (2.0 GHz) 11.1 ≈
  L40S (2.5 GHz) 12.2: the signature of a latency-bound computation.

**Consequently, f64 on "fp32 chips" is nearly free for this problem class.**
On 1:64 parts (L40S, 3080, workstation Blackwell) the pure arithmetic is
~1–2 ms of a ~15 ms budget. The caveat marks the boundary: wherever the work
becomes dense-block-shaped the penalty returns (the stock path, which leans on
dense vendor kernels, spreads 3× between H200 and L40S where the fused pipeline
spreads 1.27×; large-supernode 3D problems would be far worse). Sparse
small-front power-grid KKTs are close to the best case for cheap consumer GPUs.

**Tensor-core f64 cannot close the remaining gaps.** A role-ablation ceiling
measurement on the A100 shows that making the GEMM-shaped tile-syrk infinitely
fast saves ~0.75 ms, and freeing *all* narrow-front in-kernel compute saves
~5 ms of ~21. DMMA would at best double the speed of that subset, bounding the
realistic gain at ≤10% on Ampere/Hopper — and the chips furthest from cuDSS
(sm_86/89/120) have no f64 tensor cores at all. The DMMA kernel was therefore
not built.

**Where cuDSS's remaining edge lives.** Feeding cuDSS any external permutation
(even its own, extracted and fed back, bit-identical fill) collapses its
refactorization ~12× — its speed is scheduler metadata generated inside its
reorderer, not ordering quality (our fill is slightly better than its own:
14.3M vs 15.55M by its own count). The residual 8–25% on Hopper/Ada/Blackwell
is that scheduling craftsmanship against clock-insensitive dependency-hop
latency in our counter-based design; the L40S ratio (1.25×) is inflated by
cuDSS being anomalously strong on high-clock mature-toolchain Ada — our own
times on L40S and Blackwell are nearly identical (15.2 vs 15.8).

## What moved the needle, per class (measured, gated)

- 2048-threads/SM NVIDIA (sm_70/80/90): host vendor wides; CUDA-graph replay
  of the whole refactorization (−0.5 ms on H200); on Hopper only, 96-wide
  in-kernel panels (−1.8 ms; occupancy-free in 228 KB/SM shared memory).
- 1536-threads/SM NVIDIA (sm_86/89/120): in-kernel blocked wide fronts with a
  single merged launch (1.6–2.1×), 8 KB subtree budget (~−1 ms), graphs.
- AMD gfx906: merged wides, 8 KB budgets, workgroup 256 (wave64 wants 4 waves
  per workgroup: −20%), spin cap 16 (−1 ms).
- Everywhere: division-free scale-first Cholesky column step and packed-triangle
  trailing-update enumeration (PR #125: 29.3 → 24.4 ms on GV100 alone); the
  nseps = 4 / seed = 3 METIS ordering (−6%, PR #124) with an on-disk
  permutation cache so METIS runs once per matrix pattern.

Tried and rejected with measurements: inverse-based TRSM (2×), multi-stream
wides, subtree/mega-kernel overlap, spin-cap tuning on NVIDIA, tile-64 on every
chip, 96-panels on ≤100 KB/SM parts, nseps beyond 4, Scotch ordering (+20–60%
fill), cuDSS-extracted orderings.

## LDL^T in the fused pipeline (added 2026-10-11)

The mega-kernel gained a no-pivot LDL^T mode (real element types; the MadNLP
case): a compile-time `Val` switches the column phases, D stays on the shared
diagonal in flight, the factor keeps the stock LDL layout (unit-lower panels +
separate D), and the solve applies D between the unit-triangular sweeps. Wide
fronts factor through the VERBATIM Cholesky vendor path where the front's
pivots share a sign (A = CC^T = LDL^T with L = C diag(C)^-1, d = diag(C)^2;
the herk contribution block is bit-identical), with the in-kernel wide roles
as the mixed-sign fallback. Gated against stock pivot-N (info 0, factor
~1e-7, D ~5e-10, 20/20 repeats):

| device | fused LDL | fused Cholesky | LDL multiplier | cuDSS LDL |
|---|---|---|---|---|
| Quadro GV100 | 28.6 ms | 25.8 ms | **1.106** | 33.8 ms (1.25x of its own Cholesky) |
| AMD Radeon VII | 41.6 ms | 39.6 ms | **1.051** | n/a |

## CPU: the fast path (added 2026-10-11)

`bench/e2e/SDSCPU.jl`: a task-parallel CPU refactorization over the stock
schedule - subtree groups as chunked tasks running the regime-A algorithm in
plain scalar Julia, regime-B/C fronts assembling themselves per task (cache
local) then LAPACK potrf + BLAS trsm/syrk with BLAS pinned to one thread, and
a single-thread specialization with no task machinery at all. CPU-specific
amalgamation: max_width 96 (-17% at 24 threads vs the GPU champion 48; the
GPU prefers 48 by +39%). Measured and ruled out with the reasons recorded:
DAG scheduling (correct, 6x slower: the CB stack's slot reuse IS the cache
locality) and single-socket NUMA pinning (2.5x slower: the fronts phase wants
both sockets' bandwidth). The phase is memory-bound past ~8 threads.
MA27/MA57 reference numbers come via libcoinhsl: the libHSL 2025 build STUBS
the HSL Archive codes (its ma27 symbols return success with zeroed output;
the subset build returns -26).

## The full picture

Refactorization of the 78k condensed KKT; one row per hardware; best
validated configuration each; all gated.

| hardware | SDS LL^T | SDS LDL^T | best alternative (LL^T / LDL^T) |
|---|---|---|---|
| NVIDIA H200 | 12.0 | - | cuDSS 11.1 / - |
| NVIDIA L40S | 15.2 | - | cuDSS 12.2 / - |
| RTX Pro 6000 | 15.8 | - | cuDSS 13.5 / - |
| NVIDIA A100 | 19-21 | - | cuDSS 17.3-18.7 / - |
| Quadro GV100 | 24.1 | 28.6 | cuDSS 24.3 / 33.8 |
| RTX 3080 (kernel-level, 10 GB) | 32.4 | - | - |
| AMD Radeon VII | 39.6 | 41.6 | no vendor solver exists |
| 2x Xeon 8160, 24 threads | ~85* | - | - / MA86 76.5 |
| 2x Xeon 8160, 8 threads | 106 | - | - / MA86 151 |
| 2x Xeon 8160, 1 thread | 449 | - | CHOLMOD 966 / MA86 963 |

More 1-thread references: MA57 1909 ms, MA27 2798 ms (both LDL^T, libcoinhsl;
MA27 on the augmented k2 system of the same problem: 3709 ms at n = 2.09M).
*Projected from the measured interleaved -16.8% (amalgamation 96) on the
quiet-box 101.5 ms; every other entry is directly measured. Solve times where
they matter: SDS GPU 3.4-3.6 ms, cuDSS 3.7-4.1 ms, SDS CPU 81 ms at 24
threads, MA86 168 ms at 24 threads.

## Correctness note

All of the above sits on the completion-signal fix (`left == 0`): the fused
kernel's last-finisher test had used the atomic's *new* value since PR #107,
signaling one block early — invisible on 2048-thread parts, 20/20 failures on
sm_86/89/120 and ~25% on gfx906, including one *silent* wrong factor with
info = 0. Validated 0-failure on all seven platforms after the fix. Gate on
factor deltas, not info alone.

## Reproduction

One command on a fresh GPU machine (NVIDIA or AMD, autodetected):

```bash
julia +1.13 --startup-file=no -i -e 'include(download("https://raw.githubusercontent.com/exanauts/SparseDirectSolver.jl/custom-reordering/bench/e2e/fresh_machine.jl"))'
```

Kernel-level benchmarks: the scripts listed in `bench/ARCH-TUNING.md` against
`bench/data/kkt_pglib_opf_case78484_epigrids_condensed_10.mtx`. Per-arch
defaults: `MadNLPSDS.default_tuning`; override any knob with
`madnlp(m; sds_tuning = MadNLPSDS.SDSTuning(; ...))`.

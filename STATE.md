# State of SparseDirectSolver.jl (2026-11-09)

Snapshot of `main` at baf3b2f, after T36 (PR #187, `solve_alg = "algo1"` on
`DirectSolver`), the chore PR #189 (#188: the CPU allocation checks run in a
child process, the KernelAbstractions cap is lifted) and the bench PR #190 (the
cuDSS 1.0 comparison for the 0.3.0 release). It replaces the snapshot
of 2026-11-04; the history is in git (`git log -- STATE.md`). Sources: the Report blocks of `TASKS.md`, the open and
closed issues, `PLAN.md` §5, `bench/comparison/comparison.md`,
`bench/ARCH-TUNING.md`, `PERFORMANCE.md` and `bench/PERFORMANCE-2026-10.md`.

## 1. What exists

| Milestone | Tasks | State |
| --- | --- | --- |
| M0 scaffolding, baselines | T01–T04 | done |
| M1 symbolic | T05–T07, T22 | done; ND-tree export/import and the ordering cache are T30 (parked, #117); `max_lu_nnz` is accepted and stored but not checked (#159 warns once) |
| M2 numeric kernels | T09–T11, T23, T31, T32 | done; regime A/B/C Cholesky, LDLᵀ, LU; CUDA, AMDGPU, oneAPI and Metal extensions; `:ka` fallbacks allocation-free and ahead of `:generic` under `:auto` (T32). Deviations from PLAN §5: no vendor batched calls in regime B, no vendor `sytrf`/`getrf` in regime C; the unblocked `ka_potrf!`/`ka_trsm!` on wide fronts are #183 |
| M3 SPD/HPD v0.1 | T12, T13 | done; the 1.5× cuDSS target is not met through the handle API (§2) |
| M4 LDLᵀ MadNLP-ready | T14, T15, External | T14/T15 done; the MadNLP extension `SDSSolver` (`ext/SparseDirectSolverMadNLPExt.jl`, tested since #176) is the in-repository stand-in for the External MadNLPGPU task (#28, open) |
| M5 solve extras | T16, T18 | done |
| M6 uniform batch | T17 | done |
| M7 general LU | T19 | done; BLAS-3 for `L₂₁`/`U₁₂` is #75, unscheduled |
| M8 Schur | T20 | done |
| M9 matching/scaling | T21 | done |
| M10 non-uniform batch | T29 | parked (#116) |
| M11 performance | T33–T36 done; T37, T38 open; T24–T26 superseded | the fused pipeline (`src/fused.jl`, `FusedSolver`) is the MadNLP extension's path; since T36 its solve runs on `DirectSolver` under `solve_alg = "algo1"` (one solve driver, `fused_solve!`, for both solvers), `"default"` stays the level sweeps; T37 moves the fused factorization under `factorization_alg`, T38 adds graph capture of the stock path |
| M12 hybrid modes | T27 | parked (#27); `use_cuda_register_memory` and `host_nthreads` are stored, no effect, warn once (#159) |
| M13 robustness extras | T28 | parked (#115) |

Every Report carries `[!]` (done with documented deviations); none is `[~]`.
Test markers: one `@test_skip` (the device allocation counter on non-CUDA
backends, `test/test_api.jl`); no `@test_broken`. The CPU allocation checks
(`test/test_allocations.jl`, #188) measure in a child process, one per element
type, budgets unchanged.

## 2. Where the numbers stand

`bench/comparison/comparison.md`, cuDSS 1.0.0 (CUDSS.jl 1.0.0) vs SDS at
352271c on an H200 through Slurm (2026-11-09, PR #190; cuDSS matching set
explicitly: off except on the algo5 row), geometric means of the SDS/cuDSS time
ratio, all matrices / the condensed and K2 KKT dumps only:

| feature | matrices | analysis | factorization | refactorization | solve |
| --- | --- | --- | --- | --- | --- |
| Cholesky, Float64 | 15 / 10 | 0.58× / 0.40× | 3.13× / 1.88× | 3.84× / 2.44× | 3.84× / 2.83× |
| Cholesky, 16 right-hand sides | 15 / 10 | 0.58× / 0.40× | 3.16× / 1.92× | 3.83× / 2.44× | 3.20× / 2.75× |
| LDLᵀ, static pivoting | 24 / 19 | 1.07× / 1.01× | 5.64× / 4.73× | 6.87× / 6.00× | 4.66× / 4.20× |
| LDLᵀ + 2 refinement steps, K2 dumps | 9 | 2.64× | 4.20× | 5.46× | 10.24× |
| Uniform batch of 8, Cholesky | 10 | 0.40× | 1.15× | 1.76× | 2.60× |
| LU, unsymmetric | 2 | 1.17× | 13.82× | 15.19× | 5.46× |
| Schur complement, Cholesky | 10 | 0.43× | 2.01× | 2.54× | 4.25× |
| LDLᵀ + matching (algo5), K2 dumps | 9 | 0.74× | 3.57× | 4.21× | 4.21× |
| Cholesky, fused pipeline (`FusedSolver`) | 15 / 10 | 0.74× / 0.50× | 1.38× / 1.21× | 1.62× / 1.48× | 1.92× / 1.62× |
| Cholesky, `solve_alg = "algo1"` | 15 / 10 | 0.62× / 0.43× | 3.41× / 2.10× | 4.01× / 2.60× | 1.83× / 1.41× |
| LDLᵀ no pivoting, `"algo1"`, condensed dumps | 10 | 0.46× | 4.23× | 5.24× | 1.39× |

On the SuiteSparse and Laplacian matrices alone (apache2, bcsstk17/38,
lap2d_300, lap3d_40) the stock Cholesky is 9.5× behind on refactorization and
7.1× on solve, the fused pipeline 1.9× / 2.7×, `"algo1"` 3.1× on solve: the
wide fronts' vendor dense sweeps. On the 78k-bus condensed KKT (now in the
harness) the fused refactorization is 16.6 vs 10.4 ms (1.60×) and its solve
2.34 vs 1.63 ms (1.44×); the stock refactorization 71.9 ms, the `"algo1"` solve
2.33 vs the level sweeps' 10.3 ms; static-pivoting LDLᵀ refactorization 354 vs
15.8 ms; the batch of 8 100 vs 57.7 ms. Against the October run (cuDSS 0.8.0,
95ee682) the SDS handle API moved within ±4% on factorization and up to +15% on
solve, the fused pipeline got 17% faster, and cuDSS 1.0 sped up its solves
(Schur 0.69×, matching 0.60×, refinement 0.77× of the 0.8 times) and the
matching analysis (0.58×), so several SDS solve ratios read worse than in
October without a regression on our side. Residual notes: cuDSS 1.0 with algo5
fixed its case1354 K2 residuals (2e-2 → 5e-6); LU without matching is unusable
on rajat21/TSOPF on both sides, as before; `"algo1"` on bcsstk38 is 38× the
level sweeps' residual.

The fused pipeline after T34 (`fused_probe.jl` medians, T34 Report, now
superseded by the comparison above: 78k 18.6 vs 11.2 ms and 2.28 vs 1.67 ms;
lap3d_40 27.5 vs 14.4 ms and 6.5 vs 1.4 ms, the wide fronts near the root).
T35 made the fused factorization
bitwise reproducible in both modes at no cost (the parent-pull extend-add
replaced the atomic one everywhere); `deterministic_mode = 1` selects the pull
forward sweep, 4–8% slower on the solve, so the default stays `0`. With the
#174 handle binding, graph replay of the fused refactorization and solve with
vendor dense calls works (78k: 16.75 → 16.11 ms and 2.42 → 2.30 ms);
`SDSTuning.graph_max_n` still defaults to `0`, an owner decision for sm_90.
The factor gate of `bench/ARCH-TUNING.md` is restated (T34/T35, #177, #181):
bitwise run-to-run, fused vs stock within the larger eps-perturbation response
of the two paths, backward error within 5%.

T36 (H200 sm_90, `DirectSolver` default options, 1 RHS, medians of 21, T36
Report): `solve_alg = "algo1"` beats the level sweeps on every harness matrix,
78k condensed KKT 10.14 → 2.12 ms (cuDSS 1.40), case1354 0.886 → 0.269 (0.193),
bcsstk38 1.540 → 0.327 (0.353), lap2d_300 3.21 → 1.60 (0.53), lap3d_40 11.72 →
6.99 (1.25), apache2 70.53 → 47.88 (3.29); the inverse launch adds 0.2–1.3% to a
refactorization and +23% device memory on the 78k (`memory_estimates` slot 14).
Residuals are 1.1–3.3× the level sweeps' on the condensed KKTs and 35× on
bcsstk38 (explicit inverses of ill-conditioned `L₁₁` blocks; refinement removes
it), measured on sm_90 only, so `"default"` stays the level sweeps. On lap3d_40
and apache2 the vendor dense sweeps of the fronts wider than 256 are where the
remaining gap to cuDSS sits.

The handle-API gap on KKT systems is launch- and level-bound (#82).
`bench/comparison` is regenerated by hand (`bench/compare.jl`, one solver per
process, each on an exclusive GPU through Slurm; the SuiteSparse matrices load
from local copies on `kkt`, an uncommitted workaround for the 403 downloads
that belongs upstream); the report has an all-matrices and a KKT-only overview.

## 3. Open tasks and their order

The next task is the one whose issue the owner opened (`AGENTS.md`,
`TASKS.md` intro); the file order is the fallback. Open:

1. **T37**: the fused factorization on `DirectSolver` through
   `factorization_alg`; `FusedSolver` becomes a wrapper; #109's placement
   decision. Closes #109. No issue yet (the owner opens it to start).
2. **T38**: graph capture of the stock refactorization and solve; verifies
   #110. Closes #82, #110. No issue yet.
3. Parked behind T38, numbers unchanged, no scheduling decision: T27 hybrid
   memory (#27, M12), T28 robustness extras (#115, M13), T29 non-uniform
   batch (#116, M10), T30 ND tree and ordering cache (#117, M1).
4. External, MadNLPGPU integration (#28, M4): independent of the chain, owner
   decision of 2026-10-09 to integrate against finished kernels.

T24–T26 are superseded (`[-]`): their kernels exist in `src/fused.jl`
(T33–T35); T36 integrated the solve, T37–T38 are the rest. The T36 Report
proposes three `PLAN.md` edits for the owner (§1.3 `solve_alg` row, §2.5
multi-segment wording and numbers, §2.7 `forward_progress`); not applied yet.

## 4. Open issues and what to do with each

| Issue | Milestone / labels | State and decision |
| --- | --- | --- |
| #27, #115, #116, #117 (T27–T30) | M12, M13, M10, M1, `task` | Parked; the issue bodies say so and name no predecessor until the owner schedules one. Milestones agree with PLAN §5. |
| #28 (External) | M4, `task`, `external` | Independent of the task chain (the issue body says so); agrees with PLAN. |
| #82 KKT level-bound | M11, triaged, performance | Solve half done by T36 (PR #187, one dependency-counter segment per run of narrow plan entries); T37 takes the factorization half; closed by T38. Body is the PR #81 profile; still the right criterion (≥ 3× fewer launches). |
| #109 update-stack placement under out-of-level-order execution | M11, triaged | Closed by T37 (private slots `pcb` are reported since T35; the placement decision for the `DirectSolver` path remains). |
| #110 regime-C host synchronization | M11, triaged | Verified and closed by T38 (host-synchronization count per refactorization). |
| #183 unblocked `ka_potrf!`/`ka_trsm!` 100–1000× slower on wide fronts | M2, found-by-agent | The `:auto` path on oneAPI (`potrf`/`getrf`) and Metal (`trsm`, `potrf` below 64) since T32. No task owns it; decide between a blocked `:ka` kernel and vendor bindings on those backends. |
| #163 merge the regime-A/B kernel bodies | design-record, performance | Proposal from the housekeeping pass; unscheduled. Candidate for after T38, or fold into the compile-time work (#91). |
| #167 in-front Bunch–Kaufman for the fused pipeline | design-record, fused, madnlp | Record of the closed prototypes #135/#139. `"algo1"` (T36) accepts `pivot_type = 'N'` only and raises `NotSupportedError` otherwise (2×2 D blocks and the local pivot order are not in the fused sweeps); unscheduled. |
| #142 `_tri_decode` integer isqrt not timed on CUDA/AMDGPU | performance, bench | One A/B on the regime-A/B factorization; T36 did not take it, T37's measurement campaign can. |
| #148 oneAPI.jl shims (uncaught oneMKL exceptions, batched trsm pointer types, host sync per call) | upstream, oneapi | Waits on oneAPI.jl; item (c) is a standing deviation from PLAN §2.4 on oneAPI. |
| #91 compile time | triaged, performance, ci | Standing; the test split per element type (`SPLIT_FILES`) and the parallel runner are the mitigations in place. The allocation child of #188 adds one cold compile per element type (about 6 min each on the loaded `kkt` runner, inside the wall time of the other long files). |
| #168 Metal follow-ups (#143, #137, #138) | M2, on-hold, metal | Needs Apple hardware; no CI leg. |

Closed since the last snapshot: #186 (T36, PR #187) and #188 (PR #189; the
diagnosis of the KernelAbstractions ≥ 0.10.6 yield and the reproducer are in
the issue). Not reported upstream, by owner decision.

Agreement with `PLAN.md`: every open issue is consistent with the plan text
after the housekeeping of PR #185 (§1.3 rows for #159, §3.1 for #144, §2.5/§3.9
for #174, §2.6/§5 for T32 and #183, §5 status for T36–T38; the M1 row names
the unwired `max_lu_nnz` check, #159). #24–#26 were closed as superseded and
the dependency lines of #27, #115–#117 and #28 rewritten on 2026-11-04.

## 5. Standing deviations from `PLAN.md`

All in the plan text already, listed here because they shape what a task may
assume:

1. Regime B never uses vendor batched calls; regime C uses no vendor
   `sytrf`/`getrf` (KA in-front pivoting with the reference sequence).
2. The `:auto` dense order is `:vendor`, then `:ka`, never `:generic`
   (T32); on oneAPI, LAPACK stays on `:ka` and every oneMKL call
   synchronizes the host (#148); Metal is `Float32` only.
3. `schedule = "syncfree"` is accepted and raises (not implemented);
   `"subtree+level"` is rejected (#159). `hybrid_*`, `factor_precision ≠ T`,
   `use_cuda_register_memory`, `host_nthreads` and `max_lu_nnz` have no
   effect (raise or warn once).
4. The fused solve is on `DirectSolver` as `solve_alg = "algo1"` (T36): set
   before the analysis; real `"SPD"`/`"HPD"` and `"S"`/`"H"` with
   `pivot_type = 'N'`, `nrhs = 1`, `solve_mode = 0`, `nbatch = 1`, no Schur
   mode, backends with `forward_progress` (CUDA, AMDGPU, oneAPI; the CPU
   backend and Metal refuse); everything else `NotSupportedError`, never a
   silent fallback. `"algo1"` set after the analysis raises too. The fused
   factorization is reached only through `FusedSolver` until T37; it refuses
   the same scope plus complex `T`.
5. `update!` is public but not exported (#144): `SparseDirectSolver.update!`.
6. Graph replay is opt-in (`graph_max_n = 0`); captured ranges may contain
   vendor dense calls only because the CUDA extension binds the handles to
   the capture stream (#174); a captured vendor `potrf` bakes in the task's
   cuSOLVER workspace pointer, so T38 must size the workspace for every live
   solver before capturing (`capture_graph` docstring, `bench/ARCH-TUNING.md`).
7. T36 measured on sm_90 only; sm_89 and AMD numbers of the multi-segment
   solve, and the accuracy criterion for making `"algo1"` the default
   (`"algo1"` + one refinement step vs level sweeps without), are open
   follow-ups in the T36 Report, not issues.

## 6. Tests, CI, process

- CI (`ci.yml`): the CPU suite on the self-hosted `kkt` runner at the Julia
  compat minimum, the GPU suite on the `cuda` and `amdgpu` runners
  (`SDS_TEST_CPU=0 SDS_TEST_SKIP=test_aqua`), oneAPI on Aurora through the
  ALCF GitLab mirror (`alcf.yml`/`alcf-report.yml`), Aqua, the docs build
  with PR previews. Every run tests all four element types. The test
  environment does not carry CUDA; the CUDA job adds it, and a local GPU run
  adds it the same way without committing `test/Project.toml`.
- KernelAbstractions compat is `0.10.3` again, unbounded above (#188, PR
  #189). KA ≥ 0.10.6 `yield`s after CPU launches of ≥ 50 µs, and `@allocated`
  is process-wide, so the CPU allocation checks of the numeric and solve
  phases run in `test/allocations_child.jl`, a plain `julia` per element type
  spawned by `test/test_allocations.jl`; the budgets (`ka_cpu_alloc_budget`)
  carry no allowance. GPU checks (`device_allocated`) stay in their files. A
  CompatHelper PR for a new KA release needs no special handling anymore.
- Known log noise: the complex atomic probe (`_probe_atomic`) prints
  "exception during kernel execution" once per process on CUDA and the KA CPU
  backend; the device runtime prints it, the probe returns `false` as designed
  (T36 Report).
- The reviewer (`claude-review.yml`, Fable 5.1) reviews a `task/**` or
  `claude:pr` PR once; a re-review only when the `claude:pr` label is
  re-added. The owner squash-merges; PLAN/TASKS housekeeping goes to `main`
  directly (the ruleset's PR requirement is bypassed by the owner).
- Benchmarks (`bench/`) have their own environment with CUDSS.jl; timing runs
  on `kkt` go through Slurm (`srun --gres=gpu:1`, exclusive H200), everything
  else runs on the host. `bench/ARCH-TUNING.md` holds the fused pipeline's
  ground rules and knob tables, `PERFORMANCE.md` the experiment record,
  `bench/PERFORMANCE-2026-10.md` the GV100 record of the MadNLP end-to-end
  runs. The PR #107 prototypes and the GPU METIS port live at tag
  `prototypes-2026-10`.
- Documentation: `https://exanauts.github.io/SparseDirectSolver.jl/dev/`,
  built on the `cuda` runner for every push to `main`.

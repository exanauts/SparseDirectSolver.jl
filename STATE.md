# State of SparseDirectSolver.jl after T21 (2026-10-07)

Owner review between T21 (PR #97) and T22. Sources: every PR verdict and inline
review comment (#29–#97), every Report block in `TASKS.md`, every issue, and
`PERFORMANCE.md`. The chain is paused: issue #22 (T22) carries `on-hold`.

## 1. What exists

| Milestone | Tasks | State |
| --- | --- | --- |
| M0 scaffolding, baselines | T01–T04 | done; cuDSS baseline on the RTX 4080 recorded (#44) |
| M1 symbolic | T05–T07 (+T24 pending) | done; 2×2 pivot pairs added out of plan (#69, #72); offline stack placement (#54); packed contribution blocks (T11) |
| M2 numeric kernels | T09–T11 (+T23 pending) | done; regime A/B/C Cholesky; T23 backends not started |
| M3 SPD/HPD v0.1 | T12, T13 | done; public API, ported CUDSS.jl tests |
| M4 LDLᵀ MadNLP-ready | T14, T15, External | T14/T15 done; device LDLᵀ made fast out of plan (#87–#89); MadNLP integration not started |
| M5 solve extras | T16, T18 | done (IR, `solve_mode`, interrupt, logging, FGMRES-IR) |
| M6 uniform batch | T17 | done |
| M7 general LU | T19 | done |
| M8 Schur | T20 | done |
| M9 matching/scaling | T21 | PR #97 open, approved, CI on the CI-fix head running |
| M10–M13 | T22, T25–T27 | not started |

Every Report carries status `[!]`: done with documented deviations. None is
`[~]`. Three test markers exist: `@test_broken` on the SPD 100× refinement
claim (T16, by design), one `@test_skip` for the device allocation counter on
non-CUDA backends, and a comment in `test_reference_ldlt.jl` about a case that
is asserted after one refinement step instead of unrefined.

**Accuracy on real data (K2 dumps, after #97):** case118 0 perturbed pivots and
relres 5.6e-14 before refinement; case1354 5–6 perturbed, relres ≤ 3.9e-6
before and 1e-15 after 5 IR steps. Both beat cuDSS with `algo5`
(0/14 perturbed, 5.4e-13/7.7e-3). This closes #71 and the question whether
M13 (delayed pivots) must move before the MadNLP task: it need not.

**Speed against cuDSS (RTX 4080, after #87–#89):** Cholesky geometric mean
2.5× factorization, 3.9× refactorization, 4.4× solve. LDLᵀ 4.4× / 6.3×. On the
KKT dumps the remaining gap is launch- and level-bound (#82), 2–12× on
refactorization and 3–6× on the solve. The large-front LDLᵀ gap is closed to
1.3–2.1× of SDS Cholesky (experiment 1 criterion 1.5× met on 2 of 7 matrices).

## 2. Open issues and what to do with each

| Issue | Label | Decision |
| --- | --- | --- |
| #53 KA/`:generic` fallbacks allocate | triaged | closes with T23 (owner note exists). Nothing now. |
| #60 regime-A CUDA timings, per-backend local-memory cap | triaged | item 1 (the `bench/regimes.jl` table) is still owed and was never run; experiment 0 answered part of it. Run it once on the 4080 during the pause, or close item 1 as superseded by `PERFORMANCE.md`. Item 2 closes with T23 (owner note exists). |
| #67 fixed τ, greedy pairs | triaged | closed by #97 (`Closes #67` in its body). |
| #71 K2 growth needs scaling | triaged | closed by #97 with the acceptance table above. |
| #82 KKT level-bound | triaged, performance | T25 experiments 5–6. Largest remaining performance item for MadNLP. See §5. |
| #84 `"G"` refuses views `'L'`/`'U'` | triaged | small host-only fix, decided (cuDSS ignores the view for general). Do during the pause, with the T21 note for #86. |
| #86 device LDLᵀ ≠ reference on K2 dumps | triaged | no code change; contract documented. Check after #97: if scaling removed the divergence, close; else close as documented. |
| #91 compile time | triaged, performance | real cost: CI 21 → 34 min, first LDLᵀ factorization 20–36 s per (T, INT), CUDA `test_numeric_ldlt` up to 30 min. Needs an owner decision on levers 1–4 of the issue (dynamic class barrier on CPU, fewer classes, run-time flags, fewer eltypes per testset). Recommend lever 1 plus lever 4 for the non-numeric test files. Give it a task slot or fold into T25. |
| #96 matching pairs cost 2.1× nnz(L) | untriaged | triage needed before T22 starts. Recommendation: accept for now (accuracy is the point of `algo5`; 1.7× over the default pairs), note under T25, and let the pair selection refinement the issue suggests be a T25 item measured with `bench/pivot_pairs.jl`. |

Closed but worth remembering: #92 (`perm_row` member 1 by design, documented in #94); #68 (Julia 1.11 budgets, not planned); #37 (raw cuBLAS binding; its per-call host pointer-array upload was flagged for T09 and is now prebuilt at analysis, done in T17).

## 3. Reviewer findings never acted on

Every blocking finding was fixed before merge. The non-blocking ones below
were acknowledged and left; grouped by what they risk. Items marked **(check)**
were verified as still absent on `main` today.

### Correctness or semantics for MadNLP

- **Pairs from analysis-time values** (#69): `pivot_pairs` decides from `nzval` at analysis. An all-zero buffer gives no pairs, `undef` gives arbitrary ones. MadNLPGPU builds its solver before the first KKT evaluation, so the External task hits this. The promised docstring sentence never landed **(check)**. Needs either a structure-only fallback or an analysis after the first KKT assembly in the integration.
- **`pivot_threshold` shapes the ordering** (#72): with `pivot_pairs = "default"` partners are accepted down to `u·max|aᵢₖ|` at analysis; MadNLP sets `pivot_threshold` per solver after construction. Not documented in the options table **(check)**.
- **Statistics after a failed factorization** (#76): `inertia`/`npivots`/`pivot_stats` reduce stale per-front counts when `info ≠ 0`; after a masked batch refactorization the inactive members are stale too (#80). Not documented **(check)**. MadNLP reads inertia after every factorization.
- **`pivot_epsilon = 0`** (#65): an exactly zero pivot divides silently with `info = 0`. `nzero` is counted but nothing fails **(check)**.
- **`cholesky(::CuSparseMatrixCSR)` overlap with CUDSS.jl** (T13 Report, PR #63): both packages define the same methods; loading both overwrites one. No issue was ever opened. Matters the day MadNLPGPU loads both during migration.
- **`_packed` index in `INT`** (#59): overflows for fronts wider than 46340 with `Int32` maps while the block still fits. Unrealistic today, one-line guard.
- **`work_len` offsets truncated to `INT`** (#89): same class, no overflow check.
- **`csr_of_transpose` on oversized `SparseMatrixCSC` buffers** (#31): silent wrong `nnz` on the zero-copy path MadNLP will use **(check)**.

### Performance left on the table

- `_lt_cb_tile` O(nt) scan per work item per tile (#88): root fronts with thousands of rows.
- `refine!` permutes the workspace capacity columns, not `nrhs` (#79): wasted launches in the IPM loop.
- Schur export re-copies the CSR pattern host→device on every `getparam("schur_matrix")` (#95): two copies per IPM iteration.
- `memory_estimates` ignores the matching buffers (#97): values, weights, scale vectors; the reviewer asked for an issue, none opened.
- `FactorPreconditioner.scaling::Any` (#97): dynamic dispatch per FGMRES iteration.
- MC64 warm start `u[i]` always 0 (#97): more Dijkstra work on jobs 4/5.
- Second host copy of the values for symmetric job 5 (#97).
- `subtree_parallelism` disables regime A entirely below 4096 flops (#81): microseconds, but one regime-B launch per level on tiny matrices.
- `:ka` batched `potrf` uses a 2-D ndrange (#80): allocates on the CPU backend only.
- `unmatched` and the round cap of the pairing loop are not surfaced in `Ordering.stats` (#72, deferred to "T15 diagnostics", which never happened) **(check)**.

### Tests and docs

- `test_numeric_ldlt.jl` still uses `panel_tol`; T19 moved LU to `growth_tol` because FMA on CUDA breaks bitwise equality on high-growth fronts (the #86 fragility) **(check)**.
- Per-testset `Random.seed!(666)` missing in several LDLᵀ/LU testsets (#76, #85).
- The blocked regime-B branch of `ldlt_blocked_path` is not covered by any default-analysis test (#89).
- `DirectSolver(::SparseMatrixCSC)` and `update!(::SparseMatrixCSC)` untested (#63).
- `@test_broken r1 <= r0/100` can flip to an unexpected pass on a backend with different rounding (#79).
- Stale docstrings: `factorize_ldlt!`/`factorize!` signatures and "no vendor calls" (#89), `W = 0` meaning in `ldlt.jl` (#88), sweep docstring headers missing `transpose` (#85), the shared docstring on `LDLT_BLOCKED_MIN_*` (#89), `perm_row/perm_col` with scalings and CSC input (#97), `pivot_sign` silently ignored for `"G"` (#85).
- Cosmetic: `size(solver, d ≤ 0)` and `size(op, d ≤ 0)` return a size instead of throwing (#63, #83); `sin` shadows `Base.sin` (#97).

## 4. Deviations from PLAN.md the owner has accepted implicitly

PLAN.md was last edited for #50 and #69. The Reports suggest about thirty
wording changes; the ones that change behaviour and are now the de-facto
design:

1. Regime C for `"S"`, `"H"` and `"G"` runs KA in-front pivoting with the reference sequence; vendor `sytrf`/`getrf` are not used (T15, T19, #75). BLAS-3 only for the trailing update (#89, LDLᵀ) and not yet for LU.
2. `nd_nlevels` is a minimum, not a level cap; ND is `METIS_NodeND` (T05).
3. `max_width` caps merging only; wide fundamental supernodes stay whole (T06, #50).
4. Regime thresholds, budgets and `memory_budget` are `Options` keywords, not parameter strings (T07).
5. Contribution blocks and regime-A fronts are packed lower triangles; four local-size classes (T11).
6. Deterministic forward buffers at the gather-list rows; complex `T` always deterministic (T12, #36).
7. `Symmetric`/`Hermitian` wrappers only for the vendor sparse types; CPU backend takes `CSR` (T13).
8. The LinearAlgebra layer refines with `ir_n_steps = 2` and `ir_tol = 0` (no early exit, no host sync) (T16).
9. FGMRES is not synchronization-free (T18).
10. LU stores `Uᵀ` in a second buffer with the `L` layout (T19).
11. Schur mode: root assembled by `assemble_schur!`; `"solve"` and the sub-phases raise in Schur mode; no batches, no pivot pairs (T20).
12. Matching: computed once at reordering from the first batch member; symmetric structures get scaling only; pairs from the matching only for jobs 5/6; not with Schur mode (T21).
13. Pairs: `pivot_pairs` `"default"`/`"all"`/`"none"`, `pivot_pair_tolerance`; candidates from values at analysis (#69, #72, T21).
14. CSC input: handled through `solve_mode` conjugation (T16), `"G"` view rule to change with #84.
15. `offsetType` (Int64 `rowptr` with Int32 `colval`) is not representable (T02); never decided.

Recommendation: one PLAN.md refresh pass during the pause, taking the
"Suggested plan changes" of each Report in order. It is the only document the
implementer reads as design truth, and it is now behind on all of the above.

## 5. Suggested order for the pause

1. Triage #96 (accept, note under T25). Merge #97 when CI is green (the pipeline does it; T22 stays held by `on-hold`).
2. Small PR, host only: #84 fix, the T21 note for #86, the four **(check)** docstring items of §3 (pairs from values, `pivot_threshold`, statistics validity, `csr_of_transpose` guard), the `INT` overflow guards, `growth_tol` in the LDLᵀ test, the missing seeds. One session with Julia, no GPU.
3. PLAN.md refresh (§4).
4. Decide #91. It costs every CI run and every first factorization.
5. Decide whether to run the External MadNLP task before T22–T24. The code it needs exists: `"S"` LDLᵀ with pivot signs and inertia, refinement, matching. The two integration hazards are in §3 (pairs from values, `pivot_threshold` after analysis). Running it now would surface the real integration findings before the performance pass; T22 (non-uniform batch) and T24 (ND tree export) are not on MadNLP's path.
6. Then T25 with #82 and #75's remainder (fallback scan, extend-add width), and #60 item 1 either measured or closed.

## 6. Process notes

- The `CLAUDE_GH_PAT` secret is still unset: T17, T19 and T20 all outlived the one-hour App token and were published by the fallback with the workflow token, each needing a manual "Approve and run".
- The branch ruleset still lists the two `lts` required checks removed by #74 unless you already edited it.
- Squash merges of stacked PRs (#88 on #87, #89 on #88) needed manual rebases; stack performance PRs serially or merge main between them.
- `bench/comparison/comparison.{md,png}` and `PERFORMANCE.md` are regenerated by hand and are current as of #89; #97 changes the LDLᵀ + refinement rows (matching on) and they have not been re-run.

## 7. Addendum (2026-10-09): what the pause merged

Out-of-chain PRs merged since this review, all by hand and on `main`:

| PR | what | state of the items above |
| --- | --- | --- |
| #98 | this file | — |
| #99 | `PLAN.md` refreshed to the implemented design | §4 done; the fifteen deviations are now the plan text |
| #100 | tests hardening: `INT` overflow guards, `growth_tol` in the LDLᵀ test, missing seeds | §3 tests items done |
| #101 | semantics and docstrings: `"G"` ignores the view (#84), pairs from values, `pivot_threshold` after analysis, statistics validity, `csr_of_transpose` guard | §3 **(check)** items done, #84 closed |
| #102 | `bench/comparison` re-run with matching on | §6 "not re-run" item done |
| #103 | `SDS_TEST_ELTYPES`: PR CI runs `Float64,ComplexF32`, the weekly run all four (#91) | §5 item 4 decided: fewer element types, no CPU-specific barrier |
| #104 | #96 accepted: the fill of matching-based pairs is documented, not fixed | #96 still waits for the `triaged` label |
| #105 | Documenter site, built on the cuda runner | — |
| #106 | CompatHelper | — |
| #107 | performance prototypes by a collaborator and their agent (see `PERFORMANCE.md`, "Experiments 5–6 prototypes"): partitioned-inverse fused solve at cuDSS parity, split/fused factorization at 1.31× cuDSS, ordering study, `subtree_max_fronts`, MadNLP end-to-end on CUDA and AMD (`bench/e2e/`), the `amdgpu` CI leg running as continue-on-error at 99.4% pass | §5 item 5 (MadNLP before T22–T24) now has a starting point; T23 is mostly ROC constructor forwarders |

New issues from #107, all `found-by-agent` + `performance`, none `triaged` yet (they hold nothing while #22 carries `on-hold`, but they gate the chain once it resumes): #108 (ordering chooser picks AMD on large KKT: score the supernodal schedule depth), #109 (update-stack placement assumes level-synchronous execution), #110 (regime-C Cholesky reportedly host-synchronizes per front; verify). Owner notes for them are under T23, T24, T25 and the External task in `TASKS.md`.

Still open from §5–§6: #96 and #108–#110 untriaged; #60 item 1 unmeasured; `CLAUDE_GH_PAT` unset; the two `lts` required checks in the ruleset (unless already removed); the External MadNLP task before T22–T24 is still the recommendation, now with `bench/e2e/MadNLPSDS.jl`, `reordering_alg = "algo4"` and the phase-coupled knobs as known inputs; T25 absorbs #82, #75's remainder and the #107 designs.

## 8. Addendum (2026-10-09): task restructuring after PR #107

The former T25 held six deliverables from the #107 prototypes; it is split and
the post-v1 tasks are renumbered so the pipeline's `TNN → TNN+1` chaining
still holds. New order in `TASKS.md`: **T22** ordering chooser scoring the
supernodal schedule depth (#108, host only, precondition for the fused solve);
**T23** AMDGPU/oneAPI/Metal extensions (unchanged); **T24** partitioned-inverse
fused solve `solve_alg = "algo1"` (#82, solve half); **T25** split
factorization (tiled SYRK, chunked TRSM, split regime-C fronts, longest-first
regime A); **T26** segmented fused factorization with dependency counters,
with #109 (stack lifetimes) and #110 (verify the regime-C host sync) inside
it, plus the #75 remainder and the #96 re-measurement; **T27** hybrid memory
(was T26); **T28** robustness extras (was T27); **T29** non-uniform batch (was
T22); **T30** ND partition-tree export and ordering cache (was T24); **T31**
oneAPI and Metal extensions with the allocation-free `:ka` fallbacks (#112,
#53), the remainder of T23 after PR #113 delivered the AMDGPU extension alone;
External unchanged and last (owner decision: integrate against finished
kernels; the §5 item 5 recommendation is withdrawn). GitHub: issues #22–#27
renamed to the new titles, #22 loses `on-hold`, new task issues for T28–T31;
#96, #108–#110 and #112 labelled `triaged` (#108 → T22, #109/#110 → T26,
#112 → T31). The chain resumes with T22 once PR #113 (T23) has merged.

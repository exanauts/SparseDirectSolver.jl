# Per-architecture tuning of the fused factorization

How the October 2026 campaign tuned the fused refactorization prototype per GPU
architecture: the knob inventory, the measurement discipline, the step-by-step
procedure for a new device, the mechanisms behind each knob, and the measured
champions. Workload throughout: the 78k-bus ACOPF condensed KKT
(`pglib_opf_case78484_epigrids`, n = 674562, nnz(L) ≈ 14.3M), tuned ordering
(METIS nseps = 4, seed = 3).

## Ground rules (these produced every number below)

1. **Gate every configuration.** A trial is valid only with `info == 0`, the
   residual matching stock, and the factor gate of #177 as restated in T35:
   * (a) **reproducibility**: under `deterministic_mode = 1` the factor, `D`
     and the solution of 20 refactorizations of the same values are bitwise
     equal (run-to-run delta exactly 0). Under `deterministic_mode = 0` the
     run-to-run spread of the atomic extend-add must stay within (b);
   * (b) **accuracy**: the elementwise factor delta of the fused path against
     the stock factorization on the same analysis is no larger than the stock
     path's own response to an eps-level perturbation of A (each stored entry
     times `1 + δ`, |δ| ≤ eps; the #177 table: 5.5e-8 case118 `"S"`, 7.0e-8
     case1354 `"S"`, 9.8e-8 78k `"S"`, 3.6e-8 78k `"SPD"`, 4.6e-8 case1354
     `"SPD"`, 4.9e-11 case118 `"SPD"`), with the strict ≤ 1e-8 kept for the
     well-conditioned harness matrices (bcsstk17/38, lap2d_300, lap3d_40;
     ~1e-13 in practice). The fused `"S"` factor carries a constant offset
     against the stock one (its LDLᵀ kernels round differently: 1.9e-8 /
     2.9e-8 / 1.1e-7 on case118 / case1354 / 78k); determinism does not
     remove it, (b) bounds it.

   An `info == 0` check alone is insufficient: one race produced a silently
   wrong factor (Δ ≈ 1e-5) with clean info.
2. **Interleave with the baseline.** Shared boxes drift by 2× within an hour;
   A/B/A/B in one process is the only defensible comparison. Three pairs
   minimum; treat sub-2% differences as noise.
3. **Correctness before speed on a new chip.** Run 20 gated refactorizations
   (`bench/e2e/ldl_fused_race.jl`) before believing any timing. This is what caught
   the sm_86/89/Blackwell/gfx906 completion-signal race that sm_70/80/90 hide.
4. **Occupancy arithmetic before submitting sweeps.** Most structural knobs are
   decided by `blocks/SM = min(smem_per_SM ÷ smem_per_block,
   threads_per_SM ÷ workgroup)`; compute it first and you can predict the sign
   of the result (every prediction below held).

## Knob inventory

| knob | where | kind | mechanism |
|---|---|---|---|
| wide-front strategy (`wk`/`mseg`) | plan builder | runtime | host vendor calls vs in-kernel blocked wides + single merged launch; trades ~100 host launches against in-kernel spin residency |
| subtree budget (`subtree_budgets`) | solver option | runtime | regime-A local-memory class; smaller class = more subtree workgroups resident per SM/CU |
| subtree parallelism / amalgamation | solver options | runtime | subtree formation and supernode merging; saturated on every tested arch except amal 32→48 (small win on big-SM parts) |
| ordering (`nd_nseps`, `nd_seed`, `user_perm`) | solver/wrapper | runtime | fill/flops; arch-independent; every instance recomputes (no caching); pin with `user_perm` |
| panel width (`pw`) | plan + kernel `Val` | runtime per-plan | fronts ≤ pw factor in-kernel; packed panel = pw(pw+1)/2 · 8B of shared memory **for every block of the kernel** (one kernel = one shared size) |
| tile size (`tl`) | plan + kernel `Val` | runtime per-plan | syrk tile; shared halves = 2·32·tl · 8B; accumulators = tl²/WGF registers per thread |
| workgroup size (`SDSTuning.workgroup`) | tuning bundle | runtime (>= 128, multiple of 32: the tile roles' register accumulators) | waves per workgroup; latency hiding on wave64 AMD; occupancy on 1536-thread parts |
| spin backoff cap (`SDSTuning.spin_cap`) | tuning bundle | runtime | dependency-counter wake-up latency vs polling traffic |
| CUDA graph replay (`SDSTuning.graph_max_n`) | tuning bundle | runtime | refactorization + inverses and the solve sweeps captured once, replayed as one launch (only without vendor calls, #174) |
| fused-solve workgroup (`SDSTuning.solve_workgroup`) | tuning bundle | runtime | work items per front in `fwd2`/`bwd2`: shorter per-front serial work near the root vs idle threads on small fronts |
| per-front wides (`SDSTuning.wide_host_min`) | tuning bundle | runtime | merged plans: fronts wider than it on the host vendor path at a segment boundary |

## Procedure for a new architecture

1. **Correctness:** `bench/e2e/ldl_fused_race.jl` (20 gated refactorizations).
   Any failure: stop, bisect kernel-vs-src-vs-config before timing anything.
2. **Phase profile:** `profile_refact()` — sync-inflated but reveals the split
   (resets / subtrees / mega kernel / host wides / stats). The dominant phase
   picks the first lever: host-wides share ≥50% ⇒ try merged wides first.
3. **Wide strategy:** `mseg_ab.jl`. Decision so far follows one attribute:
   `MAX_THREADS_PER_MULTIPROCESSOR < 2048` (sm_86/89, Blackwell workstation)
   or AMD ⇒ merged wins big (1.6–2.1×); 2048-thread parts (sm_70/80/90) keep
   host wides (merged costs ~1.5 ms there).
4. **Knob sweep:** `tune_platform.jl` (budgets, sp, amal, cw, rows, flip-check
   of the wide strategy), baseline-interleaved. Expect budgets to be the only
   mover: pick the class whose size ≈ `LDS_per_SM ÷ desired_workgroups`.
5. **Structural (only where shared memory allows):** `pw_ab.jl`, `tl_ab.jl`.
   Run the occupancy arithmetic first; on ≤100 KB/SM parts pw=96/tl=64 lose by
   construction, on 228 KB (Hopper) they are free.
6. **Bundle knobs:** `SDSTuning.workgroup` ∈ {128, 256} and `SDSTuning.spin_cap`
   ∈ {128, 16} (runtime, per plan; workgroup has a hard 128 floor). WGF=256 was −20% on wave64 AMD, flat-to-pending on
   NVIDIA; the spin cap matters only on residency-thin parts.
7. **Graphs:** `graph_ab.jl` — small (~1%) and free; keep it on everywhere.
8. **Stack and re-validate:** combine the winners, three gated interleaved
   pairs (`combo_ab.jl`), then 20 gated refactorizations of the final config.

## Mechanisms, compressed

- **2048- vs 1536-thread SMs is the single most predictive attribute.** It
  decides the wide strategy, the subtree budget, and whether the completion
  race (pre-fix) manifested at all.
- **One fused kernel means one shared-memory size.** Any panel/tile growth
  taxes *all* roles' occupancy; it only pays where SM shared memory grew
  faster than the panel (Hopper).
- **Launch overhead doesn't scale with silicon.** The faster the chip, the
  larger the share of host launches and dependency-hop latency; hence merged
  wides and graphs matter more on newer parts, flops-side knobs less.
- **Wave64 wants bigger workgroups.** 128 threads = 2 waves on AMD leaves too
  little in-flight work per workgroup; 256 = 4 waves was −20%.

## Measured champions (fused refactorization, gated, interleaved)

| device | arch | threads/SM | config | SDS | cuDSS |
|---|---|---|---|---|---|
| Quadro GV100 | sm_70 | 2048 | host wides, bud 16K, graph | **24.1 ms** | 24.3 ms |
| A100-80GB | sm_80 | 2048 | host wides, bud 16K | 19–21 ms | 17.3–18.7 ms |
| H200 | sm_90 | 2048 | pending the structural matrix | 13.6 ms so far | 11.1 ms |
| L40S | sm_89 | 1536 | merged, bud 8K, graph | **15.2 ms** | 12.2 ms |
| RTX Pro 6000 | Blackwell | 1536 | merged, bud 8K | **15.8 ms** | 13.5 ms |
| RTX 3080 | sm_86 | 1536 | merged | 32.4 ms (10 GB: kernel-level only) | — |
| Radeon VII | gfx906 | wave64 | merged, bud 8K, WGF 256, cap 16 | **40.0 ms** | — |

All of this is bundled: `SparseDirectSolver.Fused.SDSTuning` holds every knob as a runtime
field, `default_tuning(backend)` is the single per-architecture defaults table,
and `madnlp(m; ..., sds_tuning = SDSTuning(; workgroup = 256))` overrides it.
The portable kernel takes workgroup size and spin cap per plan — nothing is
compile-time or environment-dependent in the wrapper path.

## T34: the fused pipeline on an H200 (sm_90), November 2026

Measured on `kkt` (H200, sm_90, an exclusive Slurm GPU), not on the sm_89 part
the task was written for; the sm_89 defaults are the T33 ones (see below).
Gate as above (info 0, 20 gated refactorizations, interleaved rounds). On the
pglib KKT dumps the fused-vs-stock delta is 1.4e-8 to 1.2e-7 (`"SPD"` 78k;
`"S"` case118/case1354/78k), the same size as the stock path's own
eps-perturbation response and its CPU-vs-CUDA delta on those matrices, and the
backward error of the fused factor equals the stock one to three digits (#177,
measured on this branch), so the rounding-level gate of ground rule 1 holds
there; T34 left every delta at the base's level. The well-conditioned matrices
stay at ≤ 1e-12.

What changed, and the medians of three interleaved rounds (refactorization):

| step | 78k | lap3d_40 | lap2d_300 | bcsstk17 | bcsstk38 |
|---|---|---|---|---|---|
| base (c38e013) | 19.95 | 33.53 | 7.00 | 5.57 | 5.85 |
| 1: `invert_blocked_kernel!` (inverses of L11) | 19.17 | 28.09 | 5.62 | 4.81 | 4.48 |
| 2: gpu-scope `device_fence` | 18.94 | 27.74 | 5.83 | 4.73 | 4.41 |
| 3: one zero + scatter launch for the C groups | 18.82 | 27.48 | 5.68 | 4.63 | 4.37 |

- **Inverses of L11** (#155): `invert_l11_kernel!` spent its time in the fronts
  of width 65..256 (78k: 1.5 of 1.85 ms; lap3d_40: 4.8 ms), where one thread
  walks a column with an O(w²) dependent chain. `invert_blocked_kernel!` (one
  workgroup per front and 32-column block, block forward substitution, the
  diagonal-block inverses in local memory) takes 3 µs on case118 and removes
  that time. Writing the inverse from the PANEL roles (the task's proposal) was
  measured too and dropped: no faster than the blocked kernel over every front
  (78k 18.42 vs 18.25 ms), and the root front's inverse lands on the critical
  path of the counter kernel (+22 µs on case118).
- **Fences** (#154): `SDS.device_fence(A)`, `fence.sc.gpu` through the CUDA
  extension. 1–2% on sm_90 (T33 measured 3.5–8% on sm_89); never slower.
- **Launches**: `czero_scatter_kernel!` replaces `panel_zero_kernel!` and the
  per-C-group `scatter_A!` launches (lap3d_40: 37 → 1). Within noise on the
  device; it and a statically dispatched counter-kernel launch (the plan
  carries the kernel object and its `Val`s) cut the host allocation of a
  refactorization of the 18³ Laplacian from 40.8 KB to 26.7 KB.
- **Subtree budgets / parallelism**: not re-swept (16K stays on 2048-thread
  parts; the subtree launch is one launch, 6.5 µs on case118).

Fused solve (`SDSTuning.solve_workgroup`, coalesced gather in `bwd2_kernel!`,
ms; the base row from the per-process A/B above, the others three in-process
interleaved rounds; the solves of case118/case1354 are launch-bound and noisy):

| | 78k | lap3d_40 | lap2d_300 | bcsstk17 | bcsstk38 | case1354 |
|---|---|---|---|---|---|---|
| base (64, strided gather) | 3.45 | 16.9 | 1.51 | 0.81 | 0.76 | 0.46 |
| coalesced gather, 64 | 2.70 | 11.5 | 1.47 | 0.56 | 0.51 | 0.25 |
| coalesced gather, 128 | **2.23** | 6.48 | 1.19 | 0.34 | 0.32 | 0.24 |
| coalesced gather, 256 | 2.95 | **4.34** | 1.19 | **0.26** | **0.25** | 0.24 |

`bwd2_kernel!` read L21 column by column per work item (32 cache lines per
warp load); 8 work items now share a column. A larger workgroup shortens the
per-front serial work on the fronts near the root; 128 is the 78k optimum
and the sm_90 default (`default_tuning`), 256 wins on the 3-D problems.

Per-front wide strategy (`wide_host_min`, merged plans): on sm_90 `:host`
stays best (lap3d_40: host 27.3, merged 65.7, merged + host above 128 39.3,
above 256 52.6 ms; 78k: 18.4, 19.4, 18.1, 19.3; lap2d_300: 5.45, 8.43, 6.10,
6.41). Off by default; it targets sm_89-class parts (T33: lap3d_40 merged
91.0 vs host 79.3 ms), where it was not measured.

CUDA-graph replay (`graph_max_n`): the solve sweeps gain 3–6% on the small
matrices (case118 0.171 → 0.160, case1354 0.544 → 0.420 (noisy), bcsstk17 0.717
→ 0.696, lap2d_300 1.370 → 1.320 ms), the refactorization is mixed (case118
0.514 vs 0.588, case1354 1.059 vs 1.004 ms; the fronts with host wides are not
captured). Not a clear win: `graph_max_n = 0` in every default. **Capture
constraint (#174):** cuSOLVER `potrf` invalidates a stream capture on CUDA.jl
6.5 / cuSOLVER 12.3.5 (`cusolverDnSetStream` on the capture stream), so the
pipeline captures only when no vendor call falls in the range: no host-wide
front in the plan (`isempty(fp.wides)`) for the refactorization, no front
wider than 256 for the solve; otherwise it launches eagerly. The permutations
of the solve stay outside the graph, so the graph does not depend on the
caller's vectors; a new values buffer captures again.

## Scripts

The campaign's one-off sweep drivers (`race_repro.jl`, `fact_lib.jl`'s
`profile_refact`, `mseg_ab.jl`, `pw_ab.jl`, `tl_ab.jl`, `graph_ab.jl`,
`combo_ab.jl`, `tune_platform.jl`, `h200_matrix.jl`, `e2e/amd_tune.jl`) were
not shipped; the committed equivalents are `e2e/ldl_fused_race.jl` (gated
correctness), `e2e/ldl_fused_test.jl` (fused vs stock + same-strategy ratio)
and `e2e/amd_sweep.jl` (knob sweep). Read the procedure above with those
substitutions; CUDA-graph replay (step 7's `graph_ab.jl`) retired with the
consolidation cleanup (no graph kwarg at HEAD). All gates take the
matrix from `bench/data/` and the tuned permutation from `bench/perm_cand3.bin` (not committed).

# Per-architecture tuning of the fused factorization

How the October 2026 campaign tuned the fused refactorization prototype per GPU
architecture: the knob inventory, the measurement discipline, the step-by-step
procedure for a new device, the mechanisms behind each knob, and the measured
champions. Workload throughout: the 78k-bus ACOPF condensed KKT
(`pglib_opf_case78484_epigrids`, n = 674562, nnz(L) ≈ 14.3M), tuned ordering
(METIS nseps = 4, seed = 3).

## Ground rules (these produced every number below)

1. **Gate every configuration.** A trial is valid only with `info == 0`,
   elementwise factor delta ~1e-9 against the stock factorization, and the
   residual matching stock. An `info == 0` check alone is insufficient: one
   race produced a silently wrong factor (Δ ≈ 1e-5) with clean info.
2. **Interleave with the baseline.** Shared boxes drift by 2× within an hour;
   A/B/A/B in one process is the only defensible comparison. Three pairs
   minimum; treat sub-2% differences as noise.
3. **Correctness before speed on a new chip.** Run 20 gated refactorizations
   (`bench/race_repro.jl 20`) before believing any timing. This is what caught
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
| ordering (`nd_nseps`, `nd_seed`, `user_perm` + on-disk perm cache) | solver/wrapper | runtime | fill/flops; arch-independent; pay METIS once, cache the permutation |
| panel width (`pw`) | plan + kernel `Val` | runtime per-plan | fronts ≤ pw factor in-kernel; packed panel = pw(pw+1)/2 · 8B of shared memory **for every block of the kernel** (one kernel = one shared size) |
| tile size (`tl`) | plan + kernel `Val` | runtime per-plan | syrk tile; shared halves = 2·32·tl · 8B; accumulators = tl²/WGF registers per thread |
| workgroup size (`SDS_WGF`) | env (const at include) | per-process | waves per workgroup; latency hiding on wave64 AMD; occupancy on 1536-thread parts |
| spin backoff cap (`SDS_SPIN_CAP`) | env (const at include) | per-process | dependency-counter wake-up latency vs polling traffic |
| CUDA graph replay (`graph`) | run_fused kwarg | runtime | whole refactorization captured once, replayed as one launch |

## Procedure for a new architecture

1. **Correctness:** `race_repro.jl 20` (kernel-level, <2 GB device memory).
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
6. **Process knobs:** `SDS_WGF` ∈ {128, 256} and `SDS_SPIN_CAP` ∈ {128, 16}
   (one process per value). WGF=256 was −20% on wave64 AMD, flat-to-pending on
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

All of this is bundled: `MadNLPSDS.SDSTuning` holds every knob as a runtime
field, `default_tuning(backend)` is the single per-architecture defaults table,
and `madnlp(m; ..., sds_tuning = SDSTuning(; workgroup = 256))` overrides it.
The portable kernel takes workgroup size and spin cap per plan — nothing is
compile-time or environment-dependent in the wrapper path.

## Scripts

`race_repro.jl` (gated correctness), `profile_refact` in `fact_lib.jl` (phase
split), `mseg_ab.jl` / `pw_ab.jl` / `tl_ab.jl` / `graph_ab.jl` / `combo_ab.jl`
(paired A/Bs), `tune_platform.jl` (knob sweep), `h200_matrix.jl` (structural
matrix), `e2e/amd_tune.jl` (AMD sweep). All print per-trial gates; all take the
matrix from `bench/data/` and the tuned permutation from `bench/perm_cand3.bin`.

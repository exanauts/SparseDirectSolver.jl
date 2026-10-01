# Benchmarks

Baselines every later milestone is measured against (PLAN.md §5 M0, §7). The
`bench/` environment is separate from the package and the test suite; results
(`bench/results/`) and matrix dumps (`bench/data/`) are gitignored.

| File | Purpose |
| --- | --- |
| `matrices.jl` | module `BenchMatrices`: generated Laplacians (`lap2d_300`: 300×300 grid, n = 90 000; `lap3d_40`: 40³ grid, n = 64 000), SuiteSparse matrices through MatrixDepot (`HB/bcsstk17`, `Boeing/bcsstk38`, `GHS_psdef/apache2`, `Rajat/rajat21`, `TSOPF/TSOPF_RS_b39_c7`), loader for KKT dumps `bench/data/kkt_<case>_<kind>_<iter>.{mtx,jld2}` with `kind ∈ {k2, condensed}` |
| `harness.jl` | module `BenchHarness`: `time_phases` (median per phase over `nruns` after warm-up), CHOLMOD/UMFPACK CPU reference, CSV writer |
| `cudss_baseline.jl` | runs every matrix × structure (`SPD` and `S` for symmetric matrices, `G` for unsymmetric ones) and writes `bench/results/<solver>_baseline.csv` |
| `report.jl` | prints the CSV files as Markdown tables |
| `dump_madnlp_kkt.jl` | dumps MadNLP K2 and condensed KKT matrices of pglib-opf cases |

## Commands

```bash
# once: instantiate the benchmark environment (CUDA, CUDSS, MatrixDepot, ...)
julia --project=bench -e 'using Pkg; Pkg.instantiate()'

# cuDSS baseline (needs a functional CUDA GPU); SuiteSparse matrices are
# downloaded by MatrixDepot on first use (DATADEPS_ALWAYS_ACCEPT is set)
julia --project=bench bench/cudss_baseline.jl
julia --project=bench bench/cudss_baseline.jl --no-suitesparse --nruns=5 --only=lap2d_300,lap3d_40

# CPU reference (CHOLMOD for SPD/S, UMFPACK for G), no GPU needed
julia --project=bench bench/cudss_baseline.jl --solver=cholmod

# print the results
julia --project=bench bench/report.jl                     # every CSV in bench/results/
julia --project=bench bench/report.jl bench/results/cudss_baseline.csv

# MadNLP KKT dumps (separate environment; prints install hints and exits 0 without it)
julia --project=bench/kkt -e 'using Pkg; Pkg.add(["MadNLP", "ExaModels", "ExaModelsPower"])'
julia --project=bench/kkt bench/dump_madnlp_kkt.jl pglib_opf_case118_ieee pglib_opf_case1354_pegase --iters=1,10,20
```

## What is measured

For each matrix and structure, after one warm-up run, five runs each of
analysis, factorization, refactorization (same values) and solve (`nrhs = 1`,
`b = rand(n)` with seed 666). A fresh solver is created per run (not timed);
GPU phases are bracketed by `CUDA.synchronize()`. The CSV holds the medians in
seconds, `lu_nnz`, `flops` (cuDSS `CUDSS_DATA_FLOPS`, read through the C API
since CUDSS.jl has no getter) and `nsuperpanels`, the relative residual of the
last solve and a status (`ok` or the error message; a failing matrix does not
stop the run). Symmetric structures pass the lower triangle (view `'L'`).

The KKT dumps store the lower triangle (MatrixMarket `symmetric`) with
explicitly stored zeros kept, so all iterations of a case share one pattern
(refactorization workload). K2 dumps run as `S`, condensed dumps as `SPD` and `S`.
The file name records the iteration of the last factorization; if MadNLP
converges before the requested iteration, the file is named after the
iteration where it stopped.

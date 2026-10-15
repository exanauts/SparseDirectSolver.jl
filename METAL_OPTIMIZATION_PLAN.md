# Metal Optimization Plan

## Goal

Improve Float32 performance on Apple GPUs without changing the portable sparse
symbolic analysis, factor layout, or public API. The primary target is warmed
Cholesky refactorization and solve on fixed-pattern systems. Correctness and
phase ordering take precedence over individual dense-kernel timings.

The reference workload is `lap2d_300` (`n = 90_000`, `nnz(A) = 448_800`) so
results can be compared with the existing RTX 4080 and cuDSS records. Smaller
Laplacians remain in the benchmark to locate the CPU/Metal crossover.

## Current Baseline

M4 Max, Metal.jl 1.11.1, Float32, warmed median:

| workload | refactorization | solve | relative residual |
| --- | ---: | ---: | ---: |
| `lap2d_120`, `n = 14_400` | 16.8 ms | 1.59 ms | 1.56e-4 |
| `lap2d_300`, `n = 90_000` | 142 ms | 5.19 ms | 1.08e-3 |
| `lap2d_300`, two IR steps | 142 ms | 15.1 ms | 4.63e-4 |

Published RTX 4080 Float64 results for `lap2d_300`:

| implementation | refactorization | solve | relative residual |
| --- | ---: | ---: | ---: |
| SparseDirectSolver CUDA | 32.9 ms | 3.48 ms | 1.78e-12 |
| cuDSS | 3.96 ms | 0.675 ms | 1.84e-12 |

The precision difference must be shown with every CUDA comparison.

## Available Libraries

### Metal.jl / Metal Performance Shaders

Use the existing low-level Metal.jl wrappers for:

- `MPSMatrixDecompositionCholesky`
- `MPSMatrixSolveTriangular`
- `MPSMatrixMultiplication`
- `MPSMatrix`, including explicit buffer offsets and descriptors

Do not use the high-level `cholesky!` and triangular `ldiv!` methods in the
numeric phase. They allocate scratch matrices, synchronize for status, and do
not accept the panel views used by the solver.

### AppleAccelerate.jl

Use Accelerate/libSparse only as an Apple CPU baseline and possible future
Float64 fallback. It is not a Metal GPU sparse solver and is not part of the
initial implementation dependency set.

### MLX

Treat MLX as a source of optimized Metal kernel designs, not as a dependency.
Reconsider only if direct MPS integration is slower and zero-copy `MTLBuffer`
interop can be demonstrated in a small prototype.

## Phase 1: Measure Regime-C Dense Operations

Create a standalone benchmark that uses the actual solver panel layout and
compares, by front shape:

- current KA POTRF
- low-level MPS Cholesky
- current KA TRSM
- low-level MPS triangular solve
- current Metal.jl panel GEMM/SYRK
- low-level MPS matrix multiplication

Shapes must be taken from `lap2d_300` and at least one condensed KKT matrix.
Measure complete operation cost, including descriptor construction and command
submission, both cold and warmed.

Record for every shape:

- dimensions and leading dimension
- median and minimum runtime
- host allocations
- Metal allocations
- synchronization behavior
- maximum relative error
- factorization status behavior on an indefinite input

Gate: no operation enters the solver unless it is correct on panel offsets,
asynchronous, and faster than KA at a stable size threshold.

## Phase 2: Offset-Aware MPS Bindings

Add Metal extension methods that construct `MPSMatrix` directly over the
solver's existing factor and workspace buffers:

```text
factor MtlVector
  + panel byte offset
  + rows, columns
  + rowBytes / matrixBytes
  -> MPSMatrix
```

Precompute and retain during analysis/allocation:

- matrix descriptors per regime-C front shape
- MPS kernel objects per operation and shape
- status buffers
- any required scratch descriptors and buffers

No per-refactorization device allocation is allowed. Host object allocation
must be measured and either eliminated or shown negligible relative to the
operation.

The bindings stay in `SparseDirectSolverMetalExt`; numeric core code continues
to call the dense interface.

## Phase 3: Ordered KA and MPS Execution

Prototype command ordering before enabling MPS in the solver:

1. KA assembly writes a front.
2. MPS factors or updates that front.
3. KA packs and extend-adds the result.
4. A later KA kernel consumes it.

Verify visibility without intermediate host synchronization over at least 100
repetitions and with Metal private storage. Use one of:

- one shared command buffer/queue accepted by both KA and MPS;
- explicit command-buffer commits on one queue, relying on queue order;
- Metal events/fences if separate encoders require them.

Gate: bitwise-stable status and residuals across repeated runs. If the ordering
cannot be made reliable without one synchronization per front, MPS POTRF/TRSM
is rejected.

## Phase 4: Selective Regime-C Dispatch

Enable MPS independently per operation and only above measured thresholds.
Expected initial policy:

| operation | small/medium front | large front |
| --- | --- | --- |
| POTRF | KA | MPS if ordered and faster |
| TRSM | KA | MPS if ordered and faster |
| GEMM/SYRK | Metal.jl generic/native | MPS if faster |
| assembly/extend-add | KA | KA |

Thresholds belong to Metal backend capabilities, not global options. Preserve
explicit `impl = :ka` as a correctness/reference route.

Uniform batches remain on the current KA/per-member path until MPS batch
correctness and status handling are demonstrated separately.

## Phase 5: Reduce Launch Count

Dense acceleration alone cannot remove the hundreds of sparse launch groups.
After MPS integration, profile launch and GPU time separately and pursue:

- merge adjacent same-level Metal launch groups where dependencies permit;
- preserve fused level solves for all Metal fronts;
- increase subtree fusion subject to the 32 KiB threadgroup-memory limit;
- test Apple-specific workgroup sizes for regime A and B;
- batch assembly kernels for consecutive regime-C fronts;
- use Metal command batching without forcing early flushes between KA kernels.

Every launch-count change must retain owner-pull determinism and cannot assume
inter-threadgroup forward progress.

## Phase 6: Accuracy Strategy

Same-precision refinement is insufficient for `lap2d_300`. Keep performance
and accuracy results separate until mixed precision exists.

Near-term:

- report raw Float32 residuals;
- report same-precision IR residual and total solve time;
- document matrices that fail the standard Float32 tolerance.

Future work:

- Float32 Metal factors and solves;
- Float64 residual on the CPU using unified memory or an explicit conversion;
- FGMRES refinement with Float32 factors;
- AppleAccelerate/libSparse Float64 fallback for systems that do not converge.

## Verification Matrix

Run after each enabled MPS operation:

```bash
SDS_TEST_CPU=0 SDS_TEST_ELTYPES=Float32 \
SDS_TEST_ONLY="test_dense,test_numeric_cholesky_a,test_numeric_cholesky_b,test_numeric_cholesky_c,test_solve,test_ubatch,test_schur" \
PTR_NUM_JOBS=1 julia +1.13.1 --project=. -e 'using Pkg; Pkg.test()'
```

Before merging:

- full Float32 Metal suite;
- full Float32 CPU suite;
- Aqua;
- CUDA CI to verify backend capability defaults do not alter CUDA dispatch;
- `git diff --check`;
- no committed `Manifest.toml` or local GPU dependency in `test/Project.toml`.

Correctness gates:

- no test failures;
- exact pivot sequences and inertia where specified;
- failed batch members report the correct original pivot;
- 100 repeated refactorization/solve cycles without races;
- explicit user schedule and `impl` choices remain authoritative.

## Performance Gates

Use warmed medians with at least seven samples and synchronize after each
phase. Every timing is accepted only with a recorded residual.

Minimum useful improvement for merging an optimization:

- at least 10% lower `lap2d_300` refactorization time, or
- at least 10% lower solve time without slowing refactorization by more than
  3%, or
- at least 15% lower combined refactorization-plus-solve time.

Report:

- M4 Max before/after;
- M4 CPU before/after;
- published RTX 4080 SDS and cuDSS context;
- precision and residual differences;
- schedule composition and launch count.

## Stop Conditions

Do not merge an approach when it:

- synchronizes once per front;
- allocates device storage per refactorization or solve;
- requires scalar indexing;
- weakens failed-pivot reporting;
- depends on unsupported inter-threadgroup progress;
- regresses CPU, CUDA, or ROCm defaults;
- adds MLX or another dependency without a measured advantage over direct MPS.

## Planned Order of Work

1. Build panel-shape benchmark and capture `lap2d_300` front distribution.
2. Prototype offset-aware MPS GEMM, then POTRF, then TRSM.
3. Prove KA/MPS command ordering over repeated runs.
4. Enable the fastest correct operations behind Metal capabilities.
5. Re-profile launch count and implement one launch-reduction experiment.
6. Run the verification matrix and publish the before/after table.

## Execution Record

Implemented and measured on the M4 Max:

- Captured `lap2d_300`: 113 regime-C fronts, widths 7–300; 18 fronts have
  width at least 64.
- Added an offset-aware low-level MPS Cholesky prototype over padded solver
  panels. MPS crosses the KA POTRF near width 64 and is 5–14× faster at widths
  128–256 in the isolated benchmark.
- Rejected MPS TRSM: it was correct but made end-to-end refactorization slower.
- Enabled MPS POTRF for single-matrix Float32 fronts of width at least 64.
- Cached MPS descriptors, decomposition kernels and the status buffer binding
  during numeric allocation; no panel or status device allocation occurs in a
  refactorization.
- Proved queue ordering with 100 repeated refactorization/solve cycles using
  private Metal storage.
- Preserved KA POTRF for narrow fronts and every uniform batch.
- Preserved KA diagonal validation after MPS so failed pivots retain the
  package's LAPACK-style first-column semantics.
- Fused every Metal front in the level solve and selected 128 work items per
  Metal solve workgroup, without changing other backends.

Final `lap2d_300` result before the full regression gate:

| phase | initial Metal | final Metal | speedup |
| --- | ---: | ---: | ---: |
| refactorization | 142 ms | 62 ms | 2.29× |
| solve | 5.19 ms | 5.18 ms | unchanged |

Published RTX 4080 context for the same matrix (Float64, versus Metal
Float32): SparseDirectSolver CUDA records 32.9 ms refactorization and 3.48 ms
solve, so final Metal is 1.89× and 1.49× slower respectively. cuDSS records
3.96 ms and 0.675 ms, so final Metal is 15.7× and 7.7× slower. Metal's raw
Float32 residual is about 9.3e-4; these timings are not precision parity.

The final MPS route creates command-buffer wrapper objects and measured about
2.1 MB of host allocation per refactorization. The MPS matrix descriptors,
kernels, panel storage and status storage are preallocated. Removing the
remaining wrapper allocation requires an upstream-style reusable command-buffer
encoding API and is follow-up work rather than a correctness blocker.

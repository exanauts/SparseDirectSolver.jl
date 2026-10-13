# GPU-hybrid METIS NodeND — loader for the ordering prototype (bench-level,
# stock src untouched).
#
#   MetisExact.jl  bitwise-faithful Julia translation of METIS 5.1.0's NodeND
#                  path (all-default options, the PR #107 configuration);
#                  validated perm- and iperm-identical to libmetis on lap2d/
#                  lap3d and the case78484 KKT dumps up to n = 2.09M.
#   GpuND.jl       KernelAbstractions device kernels for the pipeline stages
#                  (hashed-handshake HEM matching incl. 2-hop, device-resident
#                  contraction), stage-validated against the exact port.
#   Hybrid.jl      nodend_hybrid(backend, A): device coarsening chain + exact
#                  host initial separator / FM refinement / MMD, threaded
#                  sub-threshold subtrees. Fill quality 1.003–1.015 nnz_L vs
#                  exact METIS on the benchmark set.
#
# Usage (any KA backend; CUDABackend for the GPU, CPU() works for smoke):
#   include("bench/ordering/gpu_nodend.jl")
#   perm, iperm = Hybrid.nodend_hybrid(backend, A)   # A symmetric sparse
#
# MetisExact ccalls libc rand() and the binary's genmmd, so libmetis must be
# initialized once per process (Metis.jl's jll carries it):

using Metis

include(joinpath(@__DIR__, "MetisExact.jl"))
include(joinpath(@__DIR__, "GpuND.jl"))
include(joinpath(@__DIR__, "GpuRefine.jl"))
include(joinpath(@__DIR__, "Hybrid.jl"))
include(joinpath(@__DIR__, "GpuBatch.jl"))

MetisExact.init_libmetis!(Metis.libmetis)

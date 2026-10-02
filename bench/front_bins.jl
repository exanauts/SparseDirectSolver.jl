# Regime B vs regime C per size bin (TASKS.md T10): synthetic batches of
# assembled fronts of one bin, factored (a) by the fused regime-B kernel, one
# launch for the batch (`front_cholesky!`), and (b) by the regime-C path, per
# front `potrf`/`trsm`/`syrk` (`herk`) through the dense interface (vendor on a
# GPU). Every front of a batch has `f = rows`, `w = width` and a contribution
# block. Prints a Markdown table of median times and the crossover per width.
#
#   julia --project=. bench/front_bins.jl [options]
#
# Options:
#   --backend=cpu|cuda   KA CPU backend (default) or CUDA; CUDA must be installed in an environment on
#                        the load path, e.g. the default one: julia -e 'using Pkg; Pkg.add("CUDA")'
#   --T=Float64          element type (Float32, Float64, ComplexF32, ComplexF64)
#   --nb=256             fronts per batch
#   --nruns=5            timed runs per measurement (median), after one warm-up
#   --impl=auto          dense implementation of the regime-C path (auto, vendor, generic, ka)
#   --sizes=w:f,...      (w, f) shapes to run instead of the bin corners (w ∈ 8,16,32,64, f ∈ 64…512)

using LinearAlgebra
using Random
using Statistics: median
using KernelAbstractions
using SparseDirectSolver
const SDS = SparseDirectSolver

function parse_args(args)
    o = Dict{String, String}("backend" => "cpu", "T" => "Float64", "nb" => "256", "nruns" => "5",
                             "impl" => "auto", "sizes" => "")
    for a in args
        m = match(r"^--(backend|T|nb|nruns|impl|sizes)=(.*)$", a)
        m === nothing && error("unknown argument \"$a\"; see the header of bench/front_bins.jl")
        o[m[1]] = m[2]
    end
    return o
end

const OPTS = parse_args(ARGS)

if OPTS["backend"] == "cuda"
    @eval using CUDA
    const BACKEND = CUDABackend()
    device_sync() = CUDA.synchronize()
    to_dev(x) = CuArray(x)
elseif OPTS["backend"] == "cpu"
    const BACKEND = CPU()
    device_sync() = nothing
    to_dev(x) = copy(x)
else
    error("unknown backend $(OPTS["backend"])")
end

# a batch of `nb` assembled fronts of shape (w, f): panels, contribution blocks, descriptors
function make_batch(::Type{T}, nb, w, f) where {T}
    m = f - w
    factor = zeros(T, nb * f * w)
    stack = zeros(T, nb * m * m)
    for s in 1:nb
        M = rand(T, f, f)
        F = M * M' + f * I
        p0 = (s - 1) * f * w
        reshape(view(factor, (p0 + 1):(p0 + f * w)), f, w) .= tril(F[:, 1:w])
        c0 = (s - 1) * m * m
        reshape(view(stack, (c0 + 1):(c0 + m * m)), m, m) .= tril(F[(w + 1):end, (w + 1):end])
    end
    front_ptr = Int32.(1 .+ (0:nb) .* (f * w))
    cb_ptr = Int32.(1 .+ (0:(nb - 1)) .* (m * m))
    return (; factor, stack, front_ptr, cb_ptr, nrows = fill(Int32(f), nb), ncols = fill(Int32(w), nb),
            nodes = Int32.(1:nb))
end

function time_median(run!, reset!, nruns)
    reset!(); run!(); device_sync()                         # warm-up (compilation)
    t = Float64[]
    for _ in 1:nruns
        reset!(); device_sync()
        t0 = time_ns()
        run!()
        device_sync()
        push!(t, (time_ns() - t0) / 1e9)
    end
    return median(t)
end

function bench_bin(::Type{T}, nb, w, f, W, impl, nruns) where {T}
    h = make_batch(T, nb, w, f)
    factor0, stack0 = to_dev(h.factor), to_dev(h.stack)
    factor, stack = similar(factor0), similar(stack0)
    info = to_dev(zeros(Int32, nb))
    d = map(to_dev, (; h.nodes, h.front_ptr, h.nrows, h.ncols, h.cb_ptr))
    reset!() = (copyto!(factor, factor0); copyto!(stack, stack0); nothing)
    runb!() = SDS.front_cholesky!(factor, stack, info, d.nodes, 1, nb, d.front_ptr, d.nrows, d.ncols, d.cb_ptr;
                                  width = W)
    tb = time_median(runb!, reset!, nruns)
    Fb = Array(factor)
    p = (potrf = SDS.select_impl(:potrf, factor, impl), trsm = SDS.select_impl(:trsm, factor, impl),
         herk = SDS.select_impl(T <: Real ? :syrk : :herk, factor, impl))
    m = f - w
    runc!() = for s in 1:nb
        SDS._factor_panel_c!(factor, stack, info, s, (s - 1) * f * w + 1, f, w, (s - 1) * m * m + 1, p)
    end
    tc = time_median(runc!, reset!, nruns)
    Fc = Array(factor)
    all(iszero, Array(info)) || @warn "nonzero info in bin ($w, $f)"
    err = maximum(abs, Fb - Fc) / maximum(abs, Fc)
    return tb, tc, err, p
end

function main()
    T = eval(Symbol(OPTS["T"]))
    nb, nruns, impl = parse(Int, OPTS["nb"]), parse(Int, OPTS["nruns"]), Symbol(OPTS["impl"])
    shapes = isempty(OPTS["sizes"]) ?
             [(w, f) for w in SDS.REGIME_B_WIDTHS for f in (64, 128, 256, 512) if f > w] :
             [Tuple(parse.(Int, split(s, ':'))) for s in split(OPTS["sizes"], ',')]
    Random.seed!(666)
    println("backend $(OPTS["backend"]), T = $T, $nb fronts per batch, median of $nruns runs, ",
            "Julia $(VERSION), $(Threads.nthreads()) threads")
    println()
    println("| w | f | B fused (ms) | C per front (ms) | C impl | C / B | B µs/front | max rel diff |")
    println("| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    crossover = Dict{Int, Int}()
    for (w, f) in shapes
        W = SDS.REGIME_B_WIDTHS[findfirst(>=(w), SDS.REGIME_B_WIDTHS)]
        tb, tc, err, p = bench_bin(T, nb, w, f, W, impl, nruns)
        tc < tb && !haskey(crossover, w) && (crossover[w] = f)
        println("| $w | $f | $(round(1e3 * tb; digits = 3)) | $(round(1e3 * tc; digits = 3)) | $(p.potrf) | ",
                "$(round(tc / tb; digits = 2)) | $(round(1e6 * tb / nb; digits = 2)) | ",
                "$(round(err / eps(real(T)); digits = 1)) eps |")
    end
    println()
    for w in sort!(unique(first.(shapes)))
        println("w = $w: ", haskey(crossover, w) ? "regime C faster from f = $(crossover[w])" :
                                                   "regime B faster for every f measured")
    end
end

main()

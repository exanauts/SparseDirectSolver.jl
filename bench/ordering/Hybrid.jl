# Hybrid NodeND v1: the coarsening chain runs through the GpuND kernels
# (device-formulated matching + contraction), everything else through the
# bitwise-exact host port (initial separator, FM node refinement, split, MMD,
# recursion). Quality gate: nnz_L / schedule depth vs MetisExact on the same
# graphs — the cost of parallel matching is the ONLY delta by construction.
module Hybrid

using SparseDirectSolver.KernelAbstractions
const KA = KernelAbstractions
using ..MetisExact
using ..GpuND
using Random
const ME = MetisExact

export nodend_hybrid

# device coarsening chain following CoarsenGraph's control flow. The device
# arrays stay resident across levels (each level's contraction output feeds
# the next level's matching directly); only the per-level host GraphT copies
# needed by the host refinement are downloaded.
function coarsen_gpu!(backend, ctrl::ME.CtrlT, graph::ME.GraphT; seed::Integer = 1,
                      keep_levels::Union{Nothing, Vector{Any}} = nothing)
    ctrl.maxvwgt = trunc(Int, 1.5 * graph.tvwgt / ctrl.CoarsenTo)
    dev(x, T) = (y = KA.allocate(backend, T, length(x)); copyto!(y, T.(x)); y)
    n = graph.nvtxs
    dx = dev(view(graph.xadj, 1:(n + 1)), Int32)
    da = dev(view(graph.adjncy, 1:graph.nedges), Int32)
    dw = dev(view(graph.adjwgt, 1:graph.nedges), Int32)
    dv = dev(view(graph.vwgt, 1:n), Int32)
    level = 0
    while true
        match = GpuND.gpu_match(backend, dx, da, dw, dv, ctrl.maxvwgt;
                                seed = seed + level)
        cn, cx, ca, cw, cv, cmap = GpuND.gpu_contract(backend, dx, da, dw, dv, match)
        # the finer level's CSR + its fine->coarse map, for device uncoarsening
        keep_levels === nothing ||
            push!(keep_levels, (dx = dx, da = da, dv = dv, n = graph.nvtxs, cmap = cmap))
        cgraph = ME.CreateGraph()
        cgraph.nvtxs = cn
        cgraph.nedges = length(ca)
        cgraph.xadj = Vector{Int}(Array(cx))
        cgraph.adjncy = Vector{Int}(Array(ca))
        cgraph.adjwgt = Vector{Int}(Array(cw))
        cgraph.vwgt = Vector{Int}(Array(cv))
        ME.SetupGraph_tvwgt!(cgraph)
        graph.cmap = Vector{Int}(Array(cmap))
        cgraph.finer = graph
        graph.coarser = cgraph
        graph = cgraph
        dx, da, dw, dv = cx, ca, cw, cv            # stay on the device
        level += 1
        (graph.nvtxs > ctrl.CoarsenTo &&
         graph.nvtxs < ME.COARSEN_FRACTION * (graph.finer::ME.GraphT).nvtxs &&
         graph.nedges > graph.nvtxs ÷ 2) || break
    end
    return graph
end

# below this size a subgraph's bisection runs entirely on the host exact
# port: small graphs pay GPU launch/transfer latency without filling the
# device, and the deep recursion has thousands of them
const GPU_MIN_NVTXS = Ref(50_000)

# device uncoarsening: project + round-based parallel refinement per level
# (GpuRefine.jl) instead of the host FM walk. Off by default until the fill
# quality gate passes on the benchmark set.
const GPU_REFINE = Ref(false)
# :rounds — device wavefront refinement at every level;
# :project — device projection only, one exact host balance+FM at the finest
const GPU_REFINE_MODE = Ref(:rounds)
const GPU_REFINE_TRACE = Ref(false)      # print per-level separator weights

function refine_chain_gpu!(backend, ctrl::ME.CtrlT, graph::ME.GraphT, cgraph::ME.GraphT,
                           levels::Vector{Any})
    # chain[k] is the graph whose CSR/cmap is levels[k]; chain[end] = cgraph
    chain = ME.GraphT[graph]
    while chain[end] !== cgraph
        push!(chain, chain[end].coarser::ME.GraphT)
    end
    # coarse half: EXACT host project/balance/refine below the size threshold
    # (serial FM wins where graphs are tiny and every vertex is an aggregate;
    # job_35: device rounds left the n=108 level at sides 90k/55k)
    k = length(chain) - 1
    while k >= 1 && chain[k].nvtxs < REFINE_HOST_BELOW[]
        g = chain[k]
        ME.Project2WayNodePartition!(ctrl, g)
        ME.FM_2WayNodeBalance!(ctrl, g)
        ME.FM_2WayNodeRefine1Sided!(ctrl, g, ctrl.niter)
        GPU_REFINE_TRACE[] &&
            println("    host  n=", g.nvtxs, "  sep=", g.pwgts[3], "  sides=",
                    g.pwgts[1], "/", g.pwgts[2])
        k -= 1
    end
    k == 0 && return nothing                   # whole chain refined on host
    # fine half: device projection + balance + wavefront rounds
    src = chain[k + 1]
    cw = KA.allocate(backend, Int32, src.nvtxs)
    copyto!(cw, Int32.(view(src.where_, 1:src.nvtxs)))
    for j in k:-1:1
        L = levels[j]
        fw = KA.allocate(backend, Int32, L.n)
        Main.GpuRefine.project_where!(backend, fw, cw, L.cmap, L.n)
        if GPU_REFINE_MODE[] == :rounds
            Main.GpuRefine.balance_rounds!(backend, L.dx, L.da, L.dv, fw, L.n,
                                           Float64(ctrl.ubfactors))
            pw, np = Main.GpuRefine.refine_rounds!(backend, L.dx, L.da, L.dv, fw, L.n,
                                                   Float64(ctrl.ubfactors); niter = ctrl.niter)
            GPU_REFINE_TRACE[] &&
                println("    dev   n=", L.n, "  sep=", pw[3], "  sides=", pw[1], "/", pw[2],
                        "  passes=", np)
        end
        cw = fw
    end
    ME.Allocate2WayNodePartitionMemory!(graph)
    copyto!(view(graph.where_, 1:graph.nvtxs), Int.(Array(cw)))
    graph.coarser = nothing
    ME.Compute2WayNodePartitionParams!(ctrl, graph)
    if GPU_REFINE_MODE[] == :project
        ME.FM_2WayNodeBalance!(ctrl, graph)
        ME.FM_2WayNodeRefine1Sided!(ctrl, graph, ctrl.niter)
    end
    return nothing
end

# device refinement runs only at levels at least this large; smaller levels
# use the exact serial walk above (they cost microseconds on the host)
const REFINE_HOST_BELOW = Ref(50_000)

# L1-shaped bisection with the device coarsening
function node_bisection!(backend, ctrl::ME.CtrlT, graph::ME.GraphT, niparts::Int)
    if graph.nvtxs < GPU_MIN_NVTXS[]
        ME.MlevelNodeBisectionL2!(ctrl, graph, niparts)
        return nothing
    end
    ctrl.CoarsenTo = graph.nvtxs ÷ 8
    ctrl.CoarsenTo > 100 && (ctrl.CoarsenTo = 100)
    ctrl.CoarsenTo < 40 && (ctrl.CoarsenTo = 40)
    levels = GPU_REFINE[] ? Any[] : nothing
    cgraph = coarsen_gpu!(backend, ctrl, graph; keep_levels = levels)
    niparts = max(1, cgraph.nvtxs <= ctrl.CoarsenTo ? niparts ÷ 2 : niparts)
    ME.InitSeparator!(ctrl, cgraph, niparts)
    if levels === nothing
        ME.Refine2WayNode!(ctrl, graph, cgraph)
    else
        refine_chain_gpu!(backend, ctrl, graph, cgraph, levels)
    end
    return nothing
end

# MlevelNodeBisectionMultiple semantics: nseps tries, keep the lightest separator
function node_bisection_multiple!(backend, ctrl::ME.CtrlT, graph::ME.GraphT)
    if ctrl.nseps == 1 || graph.nvtxs < (ctrl.compress ? 1000 : 2000)
        node_bisection!(backend, ctrl, graph, ME.LARGENIPARTS)
        return nothing
    end
    bestwhere = Vector{Int}(undef, graph.nvtxs)
    mincut = graph.tvwgt
    for i in 0:(ctrl.nseps - 1)
        node_bisection!(backend, ctrl, graph, ME.LARGENIPARTS)
        if i == 0 || graph.mincut < mincut
            mincut = graph.mincut
            i < ctrl.nseps - 1 && copyto!(bestwhere, graph.where_)
        end
        mincut == 0 && break
    end
    if mincut != graph.mincut
        copyto!(graph.where_, bestwhere)
        ME.Compute2WayNodePartitionParams!(ctrl, graph)
    end
    return nothing
end

_copy_ctrl(c::ME.CtrlT) = ME.CtrlT(c.ctype, c.rtype, c.iptype, c.nseps, c.niter, c.ufactor,
                                   c.compress, c.ccorder, c.pfactor, c.no2hop, c.seed,
                                   c.CoarsenTo, c.maxvwgt, c.ubfactors, copy(c.pijbm))

function dissect!(backend, ctrl::ME.CtrlT, graph::ME.GraphT, order::Vector{Int}, lastvtx::Int)
    node_bisection_multiple!(backend, ctrl, graph)
    nbnd = graph.nbnd
    bndind = graph.bndind
    label = graph.label
    for i in 0:(nbnd - 1)
        lastvtx -= 1
        order[label[bndind[i + 1] + 1] + 1] = lastvtx
    end
    lgraph, rgraph = ME.SplitGraphOrder(ctrl, graph)
    tasks = Task[]
    for (g, lv) in ((lgraph, lastvtx - rgraph.nvtxs), (rgraph, lastvtx))
        if g.nvtxs > ME.MMDSWITCH && g.nedges > 0
            if g.nvtxs < GPU_MIN_NVTXS[] && ME.RNG_TASK[] && Threads.nthreads() > 1
                c = _copy_ctrl(ctrl)
                let g = g, lv = lv, c = c
                    push!(tasks, Threads.@spawn begin
                        Random.seed!(Random.TaskLocalRNG(), UInt32(hash((g.nvtxs, lv)) % UInt32))
                        ME.MlevelNestedDissection!(c, g, order, lv)
                    end)
                end
            else
                dissect!(backend, ctrl, g, order, lv)
            end
        else
            ME.MMDOrder!(ctrl, g, order, lv)
        end
    end
    foreach(wait, tasks)
    return nothing
end

"""
    nodend_hybrid(backend, A; seed = -1) -> (perm, iperm) 1-based

NodeND with the coarsening chain on the GpuND kernels and the rest on the
exact host port. Compression as in METIS_NodeND.
"""
function nodend_hybrid(backend, A; seed::Int = -1)
    n = size(A, 1)
    xadj = Vector{Int}(undef, n + 1)
    adjncy = Int[]
    rws = MetisExact.SparseArrays.rowvals(A)
    xadj[1] = 0
    for j in 1:n
        for p in MetisExact.SparseArrays.nzrange(A, j)
            i = rws[p]
            i != j && push!(adjncy, i - 1)
        end
        xadj[j + 1] = length(adjncy)
    end
    ctrl = ME.CtrlT(ME.METIS_CTYPE_SHEM, ME.METIS_RTYPE_SEP1SIDED, ME.METIS_IPTYPE_EDGE,
                    1, 10, 200, true, false, 0.0, false, seed, 100, 0,
                    Float32(Float64(Float32(1.0 + 0.001 * 200)) + 0.0000499), Float32[])
    ME.RNG_TASK[] = Threads.nthreads() > 1      # threaded subtrees want task RNG
    ME.InitRandom(ctrl.seed)
    nnvtxs = 0
    graph = nothing
    cptr = Int[]
    cind = Int[]
    if ctrl.compress
        cptr = Vector{Int}(undef, n + 1)
        cind = Vector{Int}(undef, n)
        graph = ME.CompressGraph(n, xadj, adjncy, nothing, cptr, cind)
        if graph === nothing
            ctrl.compress = false
        else
            nnvtxs = graph.nvtxs
            cfactor = 1.0 * n / nnvtxs
            cfactor > 1.5 && ctrl.nseps == 1 && (ctrl.nseps = 2)
        end
    end
    graph === nothing && (graph = ME.SetupGraph(n, xadj, adjncy, nothing))
    iperm = Vector{Int}(undef, n)
    perm = Vector{Int}(undef, n)
    dissect!(backend, ctrl, graph, iperm, graph.nvtxs)
    if ctrl.compress
        for i in 0:(nnvtxs - 1)
            perm[iperm[i + 1] + 1] = i
        end
        l = 0
        for ii in 0:(nnvtxs - 1)
            i = perm[ii + 1]
            for j in cptr[i + 1]:(cptr[i + 2] - 1)
                iperm[cind[j + 1] + 1] = l
                l += 1
            end
        end
    end
    for i in 0:(n - 1)
        perm[iperm[i + 1] + 1] = i
    end
    return perm .+ 1, iperm .+ 1
end

end # module

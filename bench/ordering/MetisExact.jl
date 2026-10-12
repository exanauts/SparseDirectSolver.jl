module MetisExact

using SparseArrays
using Libdl
using Random

# Faithful Julia translation of METIS 5.1.0 (METIS_jll 5.1.4+0 = upstream
# 5.1.0 at amontoison/METIS e827ffe, build-system patches only), NodeND path
# with OMETIS defaults, idx_t = Int32 semantics, 0-based ids kept internally
# (C `a[i]` with value index i becomes `a[i + 1]`). Goal: bitwise-identical
# perm/iperm to libmetis before anything is moved to the GPU.
#
# Source of truth: ~/git/scratch/vixen-metis/metis-src/METIS/{libmetis,GKlib}.
# Every section names the C file and function it translates.

export InitRandom, irand, irandInRange, irandArrayPermute!

# ---------------------------------------------------------------------------
# GKlib/random.c — METIS_jll is built WITHOUT -DUSE_GKRAND (GKlib option
# default OFF, Yggdrasil does not set it), so the Mersenne-Twister code is
# compiled out and gk_randinit/gk_randint32 are libc srand()/rand(). We call
# the same libc, which is exact by construction on any host, and matches the
# shipped binary because both resolve to the process' C library.

# Two RNG modes. :libc reproduces the shipped binary bitwise (process-global
# libc rand). RNG_TASK[] = true switches to Julia's task-local RNG so
# independent subtrees can run on threads deterministically (each task seeds
# its own stream); orderings are then METIS-quality but not bitwise-libmetis.
const RNG_TASK = Ref(false)

function isrand(seed)
    if RNG_TASK[]
        Random.seed!(Random.TaskLocalRNG(), UInt32(seed % UInt32))
    else
        ccall(:srand, Cvoid, (Cuint,), Cuint(seed % UInt32))
    end
    return nothing
end
irand() = RNG_TASK[] ? Int(rand(Random.TaskLocalRNG(), UInt32) & 0x7fffffff) :
                       Int(ccall(:rand, Cint, ()))
irandInRange(max) = Int(mod(irand(), max))

# libmetis/util.c InitRandom
InitRandom(seed) = isrand(seed == -1 ? 4321 : seed)

# gk_mkrandom.h randArrayPermute: p is 1-based storage of 0-based values when
# flag == 1; swaps are position swaps so the caller's base does not matter.
function irandArrayPermute!(n::Int, p::AbstractVector, nshuffles::Int, flag::Int)
    if flag == 1
        for i in 1:n
            p[i] = i - 1
        end
    end
    if n < 10
        for _ in 1:n
            v = irandInRange(n) + 1
            u = irandInRange(n) + 1
            p[v], p[u] = p[u], p[v]
        end
    else
        for _ in 1:nshuffles
            v = irandInRange(n - 3) + 1
            u = irandInRange(n - 3) + 1
            # the template swaps (v+k, u+2), (v+1, u+3), (v+2, u+0), (v+3, u+1)
            p[v], p[u + 2] = p[u + 2], p[v]
            p[v + 1], p[u + 3] = p[u + 3], p[v + 1]
            p[v + 2], p[u] = p[u], p[v + 2]
            p[v + 3], p[u + 1] = p[u + 1], p[v + 3]
        end
    end
    return p
end


# ---------------------------------------------------------------------------
# GKlib/gk_mksort.h GK_MKQSORT — the glibc-derived quicksort. Tie order is
# observable downstream (compression keys, matching orders), so the exact
# algorithm is translated, pointers becoming 1-based indices.

const _QSORT_MAX_THRESH = 4

function gk_qsort!(v::AbstractVector, n::Int, lt::F) where {F}
    n == 0 && return v
    if n > _QSORT_MAX_THRESH
        lo = 1
        hi = n
        stack = Vector{Tuple{Int, Int}}()         # (_lo, _hi); C pushes a sentinel
        push!(stack, (0, 0))
        while true
            mid = lo + ((hi - lo) >> 1)
            if lt(v[mid], v[lo])
                v[mid], v[lo] = v[lo], v[mid]
            end
            jump = false
            if lt(v[hi], v[mid])
                v[mid], v[hi] = v[hi], v[mid]
            else
                jump = true
            end
            if !jump && lt(v[mid], v[lo])
                v[mid], v[lo] = v[lo], v[mid]
            end
            left = lo + 1
            right = hi - 1
            while true
                while lt(v[left], v[mid])
                    left += 1
                end
                while lt(v[mid], v[right])
                    right -= 1
                end
                if left < right
                    v[left], v[right] = v[right], v[left]
                    if mid == left
                        mid = right
                    elseif mid == right
                        mid = left
                    end
                    left += 1
                    right -= 1
                elseif left == right
                    left += 1
                    right -= 1
                    break
                end
                left <= right || break
            end
            if right - lo <= _QSORT_MAX_THRESH
                if hi - left <= _QSORT_MAX_THRESH
                    lo, hi = pop!(stack)
                    lo == 0 && break              # popped the sentinel: done
                else
                    lo = left
                end
            elseif hi - left <= _QSORT_MAX_THRESH
                hi = right
            elseif right - lo > hi - left
                push!(stack, (lo, right))
                lo = left
            else
                push!(stack, (left, hi))
                hi = right
            end
        end
    end
    # insertion-sort tail
    thresh = min(1 + _QSORT_MAX_THRESH, n)
    tmp = 1
    for run in 2:thresh
        lt(v[run], v[tmp]) && (tmp = run)
    end
    if tmp != 1
        v[tmp], v[1] = v[1], v[tmp]
    end
    run = 2
    while (run += 1) <= n
        tmp = run - 1
        while lt(v[run], v[tmp])
            tmp -= 1
        end
        tmp += 1
        if tmp != run
            x = v[run]
            for k in run:-1:(tmp + 1)
                v[k] = v[k - 1]
            end
            v[tmp] = x
        end
    end
    return v
end

# libmetis/gklib.c — key/value pairs as tuples (key, val)
ikvsorti!(n::Int, v) = gk_qsort!(v, n, (a, b) -> a[1] < b[1])
ikvsortii!(n::Int, v) = gk_qsort!(v, n, (a, b) -> a[1] < b[1] || (a[1] == b[1] && a[2] < b[2]))

# libmetis/bucketsort.c BucketSortKeysInc (counting sort; keys in [0, max])
function BucketSortKeysInc(n::Int, max::Int, keys, tperm, perm)
    counts = zeros(Int, max + 2)
    for i in 0:(n - 1)
        counts[keys[i + 1] + 1] += 1
    end
    for i in 1:max                     # MAKECSR over 0..max
        counts[i + 1] += counts[i]
    end
    for i in (max + 1):-1:1
        counts[i + 1] = counts[i]
    end
    counts[1] = 0
    for ii in 0:(n - 1)
        i = tperm[ii + 1]
        perm[counts[keys[i + 1] + 1] + 1] = i
        counts[keys[i + 1] + 1] += 1
    end
    return perm
end

# ---------------------------------------------------------------------------
# libmetis/struct.h graph_t (ncon = 1 fields only) — ids and offsets stay
# 0-based as in C; every array is a 1-based Julia Vector indexed [x + 1].

const NOVALUE = Int[]
const DBG = Ref(false)
const PROF = Dict{Symbol, Float64}()
const PROF_LOCK = ReentrantLock()
prof!(k, t) = (lock(() -> (PROF[k] = get(PROF, k, 0.0) + t), PROF_LOCK); nothing)

mutable struct GraphT
    nvtxs::Int
    nedges::Int
    xadj::Vector{Int}
    vwgt::Vector{Int}
    adjncy::Vector{Int}
    adjwgt::Vector{Int}
    tvwgt::Int
    invtvwgt::Float32    # real_t = float32 in the shipped binary
    label::Vector{Int}
    cmap::Vector{Int}
    # partition/refinement data
    where_::Vector{Int}
    pwgts::Vector{Int}
    nbnd::Int
    bndptr::Vector{Int}
    bndind::Vector{Int}
    id::Vector{Int}
    ed::Vector{Int}
    nrinfo::Vector{NTuple{2, Int}}     # nrinfo_t.edegrees[2]
    mincut::Int
    coarser::Union{Nothing, GraphT}
    finer::Union{Nothing, GraphT}
end

CreateGraph() = GraphT(-1, -1, NOVALUE, NOVALUE, NOVALUE, NOVALUE, 0, 0.0f0,
                       NOVALUE, NOVALUE, NOVALUE, NOVALUE, -1, NOVALUE, NOVALUE,
                       NOVALUE, NOVALUE, NTuple{2, Int}[], -1, nothing, nothing)

# libmetis/graph.c SetupGraph for OMETIS (ncon = 1, objtype = NODE)
function SetupGraph(nvtxs::Int, xadj::Vector{Int}, adjncy::Vector{Int},
                    vwgt::Union{Nothing, Vector{Int}})
    graph = CreateGraph()
    graph.nvtxs = nvtxs
    graph.nedges = xadj[nvtxs + 1]
    graph.xadj = xadj
    graph.adjncy = adjncy
    graph.vwgt = vwgt === nothing ? ones(Int, nvtxs) : vwgt
    graph.adjwgt = ones(Int, graph.nedges)
    SetupGraph_tvwgt!(graph)
    SetupGraph_label!(graph)
    return graph
end

function SetupGraph_tvwgt!(graph::GraphT)
    graph.tvwgt = sum(view(graph.vwgt, 1:graph.nvtxs); init = 0)
    graph.invtvwgt = Float32(1.0 / (graph.tvwgt > 0 ? graph.tvwgt : 1))
    return nothing
end

function SetupGraph_label!(graph::GraphT)
    graph.label = collect(0:(graph.nvtxs - 1))
    return nothing
end

function SetupSplitGraph(graph::GraphT, snvtxs::Int, snedges::Int)
    sgraph = CreateGraph()
    sgraph.nvtxs = snvtxs
    sgraph.nedges = snedges
    sgraph.xadj = Vector{Int}(undef, snvtxs + 1)
    sgraph.vwgt = Vector{Int}(undef, snvtxs)
    sgraph.adjncy = Vector{Int}(undef, snedges)
    sgraph.adjwgt = Vector{Int}(undef, snedges)
    sgraph.label = Vector{Int}(undef, snvtxs)
    return sgraph
end

# ---------------------------------------------------------------------------
# libmetis/compress.c CompressGraph — identical-adjacency vertices merge.
# Returns nothing when compression is below COMPRESSION_FRACTION.

const COMPRESSION_FRACTION = 0.85

function CompressGraph(nvtxs::Int, xadj::Vector{Int}, adjncy::Vector{Int},
                       vwgt::Union{Nothing, Vector{Int}}, cptr::Vector{Int}, cind::Vector{Int})
    mark = fill(-1, nvtxs)
    map_ = fill(-1, nvtxs)
    keys = Vector{Tuple{Int, Int}}(undef, nvtxs)
    for i in 0:(nvtxs - 1)
        k = 0
        for j in xadj[i + 1]:(xadj[i + 2] - 1)
            k += adjncy[j + 1]
        end
        keys[i + 1] = (k + i, i)
    end
    ikvsorti!(nvtxs, keys)
    l = 0
    cptr[1] = 0
    cnvtxs = 0
    for i in 0:(nvtxs - 1)
        ii = keys[i + 1][2]
        if map_[ii + 1] == -1
            mark[ii + 1] = i
            for j in xadj[ii + 1]:(xadj[ii + 2] - 1)
                mark[adjncy[j + 1] + 1] = i
            end
            map_[ii + 1] = cnvtxs
            l += 1
            cind[l] = ii
            for jj in (i + 1):(nvtxs - 1)
                iii = keys[jj + 1][2]
                if keys[i + 1][1] != keys[jj + 1][1] ||
                   xadj[ii + 2] - xadj[ii + 1] != xadj[iii + 2] - xadj[iii + 1]
                    break
                end
                if map_[iii + 1] == -1
                    ok = true
                    for p in xadj[iii + 1]:(xadj[iii + 2] - 1)
                        if mark[adjncy[p + 1] + 1] != i
                            ok = false
                            break
                        end
                    end
                    if ok
                        map_[iii + 1] = cnvtxs
                        l += 1
                        cind[l] = iii
                    end
                end
            end
            cnvtxs += 1
            cptr[cnvtxs + 1] = l
        end
    end
    cnvtxs >= COMPRESSION_FRACTION * nvtxs && return nothing
    graph = CreateGraph()
    cnedges = 0
    for i in 0:(cnvtxs - 1)
        ii = cind[cptr[i + 1] + 1]
        cnedges += xadj[ii + 2] - xadj[ii + 1]
    end
    cxadj = Vector{Int}(undef, cnvtxs + 1)
    cvwgt = zeros(Int, cnvtxs)
    cadjncy = Vector{Int}(undef, cnedges)
    fill!(mark, -1)
    l = 0
    cxadj[1] = 0
    for i in 0:(cnvtxs - 1)
        mark[i + 1] = i                  # remove diagonal in compressed graph
        for j in cptr[i + 1]:(cptr[i + 2] - 1)
            ii = cind[j + 1]
            cvwgt[i + 1] += vwgt === nothing ? 1 : vwgt[ii + 1]
            for jj in xadj[ii + 1]:(xadj[ii + 2] - 1)
                k = map_[adjncy[jj + 1] + 1]
                if mark[k + 1] != i
                    mark[k + 1] = i
                    l += 1
                    cadjncy[l] = k
                end
            end
        end
        cxadj[i + 2] = l
    end
    graph.nvtxs = cnvtxs
    graph.nedges = l
    graph.xadj = cxadj
    graph.vwgt = cvwgt
    graph.adjncy = cadjncy
    graph.adjwgt = ones(Int, l)
    SetupGraph_tvwgt!(graph)
    SetupGraph_label!(graph)
    return graph
end

# ---------------------------------------------------------------------------
# GKlib/gk_mkpqueue.h instantiated as rpq (libmetis/gklib.c:
# GK_MKPQUEUE(rpq, rpq_t, rkv_t, real_t, idx_t, ..., key_gt)): a MAX-heap
# with real_t = Float32 keys and idx_t vals. Heap layout (which of several
# equal-key items surfaces first) is observable through FM, so the sift
# logic is translated verbatim. KEY_LT(a, b) = a > b.

mutable struct RPQ
    nnodes::Int
    heap_key::Vector{Float32}
    heap_val::Vector{Int}
    locator::Vector{Int}                 # by node id (0-based), -1 = absent
end

rpqCreate(maxnodes::Int) = RPQ(0, Vector{Float32}(undef, maxnodes),
                               Vector{Int}(undef, maxnodes), fill(-1, maxnodes))
rpqLength(q::RPQ) = q.nnodes

@inline _rpq_lt(a::Float32, b::Float32) = a > b   # KEY_LT = key_gt (max-heap)

function rpqReset(q::RPQ)
    for i in q.nnodes:-1:1
        q.locator[q.heap_val[i] + 1] = -1
    end
    q.nnodes = 0
    return nothing
end

function rpqInsert(q::RPQ, node::Int, key::Float32)
    i = q.nnodes                          # 0-based heap position
    q.nnodes += 1
    while i > 0
        j = (i - 1) >> 1
        if _rpq_lt(key, q.heap_key[j + 1])
            q.heap_key[i + 1] = q.heap_key[j + 1]
            q.heap_val[i + 1] = q.heap_val[j + 1]
            q.locator[q.heap_val[i + 1] + 1] = i
            i = j
        else
            break
        end
    end
    q.heap_key[i + 1] = key
    q.heap_val[i + 1] = node
    q.locator[node + 1] = i
    return nothing
end

function rpqDelete(q::RPQ, node::Int)
    i = q.locator[node + 1]
    q.locator[node + 1] = -1
    q.nnodes -= 1
    if q.nnodes > 0 && q.heap_val[q.nnodes + 1] != node
        node = q.heap_val[q.nnodes + 1]
        newkey = q.heap_key[q.nnodes + 1]
        oldkey = q.heap_key[i + 1]
        if _rpq_lt(newkey, oldkey)        # filter up
            while i > 0
                j = (i - 1) >> 1
                if _rpq_lt(newkey, q.heap_key[j + 1])
                    q.heap_key[i + 1] = q.heap_key[j + 1]
                    q.heap_val[i + 1] = q.heap_val[j + 1]
                    q.locator[q.heap_val[i + 1] + 1] = i
                    i = j
                else
                    break
                end
            end
        else                              # filter down
            nnodes = q.nnodes
            while (j = (i << 1) + 1) < nnodes
                if _rpq_lt(q.heap_key[j + 1], newkey)
                    if j + 1 < nnodes && _rpq_lt(q.heap_key[j + 2], q.heap_key[j + 1])
                        j += 1
                    end
                    q.heap_key[i + 1] = q.heap_key[j + 1]
                    q.heap_val[i + 1] = q.heap_val[j + 1]
                    q.locator[q.heap_val[i + 1] + 1] = i
                    i = j
                elseif j + 1 < nnodes && _rpq_lt(q.heap_key[j + 2], newkey)
                    j += 1
                    q.heap_key[i + 1] = q.heap_key[j + 1]
                    q.heap_val[i + 1] = q.heap_val[j + 1]
                    q.locator[q.heap_val[i + 1] + 1] = i
                    i = j
                else
                    break
                end
            end
        end
        q.heap_key[i + 1] = newkey
        q.heap_val[i + 1] = node
        q.locator[node + 1] = i
    end
    return nothing
end

function rpqUpdate(q::RPQ, node::Int, newkey::Float32)
    i = q.locator[node + 1]
    oldkey = q.heap_key[i + 1]
    if _rpq_lt(newkey, oldkey)            # filter up
        while i > 0
            j = (i - 1) >> 1
            if _rpq_lt(newkey, q.heap_key[j + 1])
                q.heap_key[i + 1] = q.heap_key[j + 1]
                q.heap_val[i + 1] = q.heap_val[j + 1]
                q.locator[q.heap_val[i + 1] + 1] = i
                i = j
            else
                break
            end
        end
    else                                  # filter down
        nnodes = q.nnodes
        while (j = (i << 1) + 1) < nnodes
            if _rpq_lt(q.heap_key[j + 1], newkey)
                if j + 1 < nnodes && _rpq_lt(q.heap_key[j + 2], q.heap_key[j + 1])
                    j += 1
                end
                q.heap_key[i + 1] = q.heap_key[j + 1]
                q.heap_val[i + 1] = q.heap_val[j + 1]
                q.locator[q.heap_val[i + 1] + 1] = i
                i = j
            elseif j + 1 < nnodes && _rpq_lt(q.heap_key[j + 2], newkey)
                j += 1
                q.heap_key[i + 1] = q.heap_key[j + 1]
                q.heap_val[i + 1] = q.heap_val[j + 1]
                q.locator[q.heap_val[i + 1] + 1] = i
                i = j
            else
                break
            end
        end
    end
    q.heap_key[i + 1] = newkey
    q.heap_val[i + 1] = node
    q.locator[node + 1] = i
    return nothing
end

function rpqGetTop(q::RPQ)
    q.nnodes == 0 && return -1
    q.nnodes -= 1
    vtx = q.heap_val[1]
    q.locator[vtx + 1] = -1
    i = q.nnodes
    if i > 0
        key = q.heap_key[i + 1]
        node = q.heap_val[i + 1]
        i = 0
        while (j = 2 * i + 1) < q.nnodes
            if _rpq_lt(q.heap_key[j + 1], key)
                if j + 1 < q.nnodes && _rpq_lt(q.heap_key[j + 2], q.heap_key[j + 1])
                    j += 1
                end
                q.heap_key[i + 1] = q.heap_key[j + 1]
                q.heap_val[i + 1] = q.heap_val[j + 1]
                q.locator[q.heap_val[i + 1] + 1] = i
                i = j
            elseif j + 1 < q.nnodes && _rpq_lt(q.heap_key[j + 2], key)
                j += 1
                q.heap_key[i + 1] = q.heap_key[j + 1]
                q.heap_val[i + 1] = q.heap_val[j + 1]
                q.locator[q.heap_val[i + 1] + 1] = i
                i = j
            else
                break
            end
        end
        q.heap_key[i + 1] = key
        q.heap_val[i + 1] = node
        q.locator[node + 1] = i
    end
    return vtx
end


# ---------------------------------------------------------------------------
# ctrl_t subset for METIS_OP_OMETIS (libmetis/options.c SetupCtrl). All
# values that the OMETIS path reads. idx arithmetic stays in Int64: every C
# idx_t intermediate on this path fits Int32 (checked per use), so Int64
# arithmetic is value-identical.

mutable struct CtrlT
    ctype::Int          # METIS_CTYPE_SHEM = 2
    rtype::Int          # METIS_RTYPE_SEP1SIDED = 3 / SEP2SIDED = 4
    iptype::Int         # METIS_IPTYPE_EDGE = 2
    nseps::Int
    niter::Int
    ufactor::Int
    compress::Bool
    ccorder::Bool
    pfactor::Float64
    no2hop::Bool
    seed::Int
    CoarsenTo::Int
    maxvwgt::Int
    ubfactors::Float32  # real_t: I2RUBFACTOR(ufactor) + 0.0000499
    pijbm::Vector{Float32}   # real_t: Setup2WayBalMultipliers (len 2)
end

const METIS_CTYPE_RM = 0
const METIS_CTYPE_SHEM = 1
const METIS_RTYPE_FM = 0
const METIS_RTYPE_SEP2SIDED = 3
const METIS_RTYPE_SEP1SIDED = 2
const METIS_IPTYPE_EDGE = 2
const METIS_IPTYPE_NODE = 3
const UNMATCHED = -1
const COARSEN_FRACTION = 0.85
const UNMATCHEDFOR2HOP = 0.10
const HTLENGTH = (1 << 11) - 1
const MMDSWITCH = 120
const LARGENIPARTS = 7
const SMALLNIPARTS = 5
const IDX_MAX = 2147483647

# ---------------------------------------------------------------------------
# libmetis/coarsen.c

function CoarsenGraph(ctrl::CtrlT, graph::GraphT)
    eqewgts = true
    for i in 1:(graph.nedges - 1)
        if graph.adjwgt[1] != graph.adjwgt[i + 1]
            eqewgts = false
            break
        end
    end
    ctrl.maxvwgt = trunc(Int, 1.5 * graph.tvwgt / ctrl.CoarsenTo)
    while true
        DBG[] && println(lpad(graph.nvtxs, 10), lpad(graph.nedges, 11),
                         lpad(sum(view(graph.adjwgt, 1:graph.nedges); init = 0), 11),
                         " [", ctrl.CoarsenTo, "] [", lpad(ctrl.maxvwgt, 9), ":", lpad(graph.tvwgt, 8), " ]")
        isempty(graph.cmap) && (graph.cmap = Vector{Int}(undef, graph.nvtxs))
        if ctrl.ctype == METIS_CTYPE_RM || eqewgts || graph.nedges == 0
            Match_RM(ctrl, graph)
        else
            Match_SHEM(ctrl, graph)
        end
        graph = graph.coarser::GraphT
        eqewgts = false
        (graph.nvtxs > ctrl.CoarsenTo &&
         graph.nvtxs < COARSEN_FRACTION * (graph.finer::GraphT).nvtxs &&
         graph.nedges > graph.nvtxs ÷ 2) || break
    end
    DBG[] && println(lpad(graph.nvtxs, 10), lpad(graph.nedges, 11),
                     lpad(sum(view(graph.adjwgt, 1:graph.nedges); init = 0), 11),
                     " [", ctrl.CoarsenTo, "] [", lpad(ctrl.maxvwgt, 9), ":", lpad(graph.tvwgt, 8), " ]")
    return graph
end

function CoarsenGraphNlevels(ctrl::CtrlT, graph::GraphT, nlevels::Int)
    eqewgts = true
    for i in 1:(graph.nedges - 1)
        if graph.adjwgt[1] != graph.adjwgt[i + 1]
            eqewgts = false
            break
        end
    end
    ctrl.maxvwgt = trunc(Int, 1.5 * graph.tvwgt / ctrl.CoarsenTo)
    for _ in 1:nlevels
        isempty(graph.cmap) && (graph.cmap = Vector{Int}(undef, graph.nvtxs))
        if ctrl.ctype == METIS_CTYPE_RM || eqewgts || graph.nedges == 0
            Match_RM(ctrl, graph)
        else
            Match_SHEM(ctrl, graph)
        end
        graph = graph.coarser::GraphT
        eqewgts = false
        if graph.nvtxs < ctrl.CoarsenTo ||
           graph.nvtxs > COARSEN_FRACTION * (graph.finer::GraphT).nvtxs ||
           graph.nedges < graph.nvtxs ÷ 2
            break
        end
    end
    return graph
end

function Match_RM(ctrl::CtrlT, graph::GraphT)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    vwgt = graph.vwgt
    adjncy = graph.adjncy
    cmap = graph.cmap
    maxvwgt = ctrl.maxvwgt
    nunmatched = 0
    match = fill(UNMATCHED, nvtxs)
    perm = Vector{Int}(undef, nvtxs)
    irandArrayPermute!(nvtxs, perm, nvtxs ÷ 8, 1)
    cnvtxs = 0
    last_unmatched = 0                  # 0-based position
    for pi in 0:(nvtxs - 1)
        i = perm[pi + 1]
        match[i + 1] == UNMATCHED || continue
        maxidx = i
        if vwgt[i + 1] < maxvwgt
            if xadj[i + 1] == xadj[i + 2]        # island vertex
                last_unmatched = max(pi, last_unmatched) + 1
                while last_unmatched < nvtxs
                    j = perm[last_unmatched + 1]
                    if match[j + 1] == UNMATCHED
                        maxidx = j
                        break
                    end
                    last_unmatched += 1
                end
            else
                for j in xadj[i + 1]:(xadj[i + 2] - 1)
                    k = adjncy[j + 1]
                    if match[k + 1] == UNMATCHED && vwgt[i + 1] + vwgt[k + 1] <= maxvwgt
                        maxidx = k
                        break
                    end
                end
                if maxidx == i && 3 * vwgt[i + 1] < maxvwgt
                    nunmatched += 1
                    maxidx = UNMATCHED
                end
            end
        end
        if maxidx != UNMATCHED
            cmap[i + 1] = cmap[maxidx + 1] = cnvtxs
            cnvtxs += 1
            match[i + 1] = maxidx
            match[maxidx + 1] = i
        end
    end
    if !ctrl.no2hop && nunmatched > UNMATCHEDFOR2HOP * nvtxs
        cnvtxs = Match_2Hop(ctrl, graph, perm, match, cnvtxs, nunmatched)
    end
    cnvtxs = 0
    for i in 0:(nvtxs - 1)
        if match[i + 1] == UNMATCHED
            match[i + 1] = i
            cmap[i + 1] = cnvtxs
            cnvtxs += 1
        elseif i <= match[i + 1]
            cmap[i + 1] = cmap[match[i + 1] + 1] = cnvtxs
            cnvtxs += 1
        end
    end
    CreateCoarseGraph(ctrl, graph, cnvtxs, match)
    return cnvtxs
end

function Match_SHEM(ctrl::CtrlT, graph::GraphT)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    vwgt = graph.vwgt
    adjncy = graph.adjncy
    adjwgt = graph.adjwgt
    cmap = graph.cmap
    maxvwgt = ctrl.maxvwgt
    nunmatched = 0
    match = fill(UNMATCHED, nvtxs)
    perm = Vector{Int}(undef, nvtxs)
    tperm = Vector{Int}(undef, nvtxs)
    degrees = Vector{Int}(undef, nvtxs)
    irandArrayPermute!(nvtxs, tperm, nvtxs ÷ 8, 1)
    avgdegree = trunc(Int, 0.7 * (xadj[nvtxs + 1] ÷ nvtxs))
    for i in 0:(nvtxs - 1)
        d = xadj[i + 2] - xadj[i + 1]
        degrees[i + 1] = d > avgdegree ? avgdegree : d
    end
    BucketSortKeysInc(nvtxs, avgdegree, degrees, tperm, perm)
    cnvtxs = 0
    last_unmatched = 0
    for pi in 0:(nvtxs - 1)
        i = perm[pi + 1]
        match[i + 1] == UNMATCHED || continue
        maxidx = i
        maxwgt = -1
        if vwgt[i + 1] < maxvwgt
            if xadj[i + 1] == xadj[i + 2]        # island vertex
                last_unmatched = max(pi, last_unmatched) + 1
                while last_unmatched < nvtxs
                    j = perm[last_unmatched + 1]
                    if match[j + 1] == UNMATCHED
                        maxidx = j
                        break
                    end
                    last_unmatched += 1
                end
            else
                for j in xadj[i + 1]:(xadj[i + 2] - 1)
                    k = adjncy[j + 1]
                    if match[k + 1] == UNMATCHED && maxwgt < adjwgt[j + 1] &&
                       vwgt[i + 1] + vwgt[k + 1] <= maxvwgt
                        maxidx = k
                        maxwgt = adjwgt[j + 1]
                    end
                end
                if maxidx == i && 3 * vwgt[i + 1] < maxvwgt
                    nunmatched += 1
                    maxidx = UNMATCHED
                end
            end
        end
        if maxidx != UNMATCHED
            cmap[i + 1] = cmap[maxidx + 1] = cnvtxs
            cnvtxs += 1
            match[i + 1] = maxidx
            match[maxidx + 1] = i
        end
    end
    if !ctrl.no2hop && nunmatched > UNMATCHEDFOR2HOP * nvtxs
        cnvtxs = Match_2Hop(ctrl, graph, perm, match, cnvtxs, nunmatched)
    end
    cnvtxs = 0
    for i in 0:(nvtxs - 1)
        if match[i + 1] == UNMATCHED
            match[i + 1] = i
            cmap[i + 1] = cnvtxs
            cnvtxs += 1
        elseif i <= match[i + 1]
            cmap[i + 1] = cmap[match[i + 1] + 1] = cnvtxs
            cnvtxs += 1
        end
    end
    CreateCoarseGraph(ctrl, graph, cnvtxs, match)
    return cnvtxs
end

function Match_2Hop(ctrl::CtrlT, graph::GraphT, perm, match, cnvtxs::Int, nunmatched::Int)
    cnvtxs, nunmatched = Match_2HopAny(ctrl, graph, perm, match, cnvtxs, nunmatched, 2)
    cnvtxs, nunmatched = Match_2HopAll(ctrl, graph, perm, match, cnvtxs, nunmatched, 64)
    if nunmatched > 1.5 * UNMATCHEDFOR2HOP * graph.nvtxs
        cnvtxs, nunmatched = Match_2HopAny(ctrl, graph, perm, match, cnvtxs, nunmatched, 3)
    end
    if nunmatched > 2.0 * UNMATCHEDFOR2HOP * graph.nvtxs
        cnvtxs, nunmatched = Match_2HopAny(ctrl, graph, perm, match, cnvtxs, nunmatched, graph.nvtxs)
    end
    return cnvtxs
end

function Match_2HopAny(ctrl::CtrlT, graph::GraphT, perm, match, cnvtxs::Int,
                       nunmatched::Int, maxdegree::Int)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    adjncy = graph.adjncy
    cmap = graph.cmap
    colptr = zeros(Int, nvtxs + 1)
    for i in 0:(nvtxs - 1)
        if match[i + 1] == UNMATCHED && xadj[i + 2] - xadj[i + 1] < maxdegree
            for j in xadj[i + 1]:(xadj[i + 2] - 1)
                colptr[adjncy[j + 1] + 1] += 1
            end
        end
    end
    for i in 1:(nvtxs - 1)                          # MAKECSR
        colptr[i + 1] += colptr[i]
    end
    for i in nvtxs:-1:1
        colptr[i + 1] = colptr[i]
    end
    colptr[1] = 0
    rowind = Vector{Int}(undef, colptr[nvtxs + 1])
    for pi in 0:(nvtxs - 1)
        i = perm[pi + 1]
        if match[i + 1] == UNMATCHED && xadj[i + 2] - xadj[i + 1] < maxdegree
            for j in xadj[i + 1]:(xadj[i + 2] - 1)
                c = adjncy[j + 1]
                rowind[colptr[c + 1] + 1] = i
                colptr[c + 1] += 1
            end
        end
    end
    for i in nvtxs:-1:1                             # SHIFTCSR
        colptr[i + 1] = colptr[i]
    end
    colptr[1] = 0
    for pi in 0:(nvtxs - 1)
        i = perm[pi + 1]
        colptr[i + 2] - colptr[i + 1] < 2 && continue
        jj = colptr[i + 2]
        j = colptr[i + 1]
        while j < jj
            if match[rowind[j + 1] + 1] == UNMATCHED
                jj -= 1
                while jj > j
                    if match[rowind[jj + 1] + 1] == UNMATCHED
                        cmap[rowind[j + 1] + 1] = cmap[rowind[jj + 1] + 1] = cnvtxs
                        cnvtxs += 1
                        match[rowind[j + 1] + 1] = rowind[jj + 1]
                        match[rowind[jj + 1] + 1] = rowind[j + 1]
                        nunmatched -= 2
                        break
                    end
                    jj -= 1
                end
            end
            j += 1
        end
    end
    return cnvtxs, nunmatched
end

function Match_2HopAll(ctrl::CtrlT, graph::GraphT, perm, match, cnvtxs::Int,
                       nunmatched::Int, maxdegree::Int)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    adjncy = graph.adjncy
    cmap = graph.cmap
    mask = IDX_MAX ÷ maxdegree
    keys = Vector{Tuple{Int, Int}}(undef, nunmatched)
    ncand = 0
    for pi in 0:(nvtxs - 1)
        i = perm[pi + 1]
        idegree = xadj[i + 2] - xadj[i + 1]
        if match[i + 1] == UNMATCHED && idegree > 1 && idegree < maxdegree
            k = 0
            for j in xadj[i + 1]:(xadj[i + 2] - 1)
                k += adjncy[j + 1] % mask
            end
            ncand += 1
            keys[ncand] = ((k % mask) * maxdegree + idegree, i)
        end
    end
    ikvsorti!(ncand, keys)
    mark = zeros(Int, nvtxs)
    for pi in 0:(ncand - 1)
        i = keys[pi + 1][2]
        match[i + 1] == UNMATCHED || continue
        for j in xadj[i + 1]:(xadj[i + 2] - 1)
            mark[adjncy[j + 1] + 1] = i
        end
        for pk in (pi + 1):(ncand - 1)
            k = keys[pk + 1][2]
            match[k + 1] == UNMATCHED || continue
            keys[pi + 1][1] != keys[pk + 1][1] && break
            xadj[i + 2] - xadj[i + 1] != xadj[k + 2] - xadj[k + 1] && break
            allmatch = true
            for jj in xadj[k + 1]:(xadj[k + 2] - 1)
                if mark[adjncy[jj + 1] + 1] != i
                    allmatch = false
                    break
                end
            end
            if allmatch
                cmap[i + 1] = cmap[k + 1] = cnvtxs
                cnvtxs += 1
                match[i + 1] = k
                match[k + 1] = i
                nunmatched -= 2
                break
            end
        end
    end
    return cnvtxs, nunmatched
end

function SetupCoarseGraph(graph::GraphT, cnvtxs::Int)
    cgraph = CreateGraph()
    cgraph.nvtxs = cnvtxs
    cgraph.finer = graph
    graph.coarser = cgraph
    cgraph.xadj = Vector{Int}(undef, cnvtxs + 1)
    cgraph.adjncy = Vector{Int}(undef, graph.nedges)
    cgraph.adjwgt = Vector{Int}(undef, graph.nedges)
    cgraph.vwgt = Vector{Int}(undef, cnvtxs)
    return cgraph
end

function CreateCoarseGraph(ctrl::CtrlT, graph::GraphT, cnvtxs::Int, match)
    mask = HTLENGTH
    if cnvtxs < 2 * mask || graph.nedges ÷ graph.nvtxs > mask ÷ 20
        return CreateCoarseGraphNoMask(ctrl, graph, cnvtxs, match)
    end
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    for v in 0:(nvtxs - 1)
        if xadj[v + 2] - xadj[v + 1] > (mask >> 3)
            return CreateCoarseGraphNoMask(ctrl, graph, cnvtxs, match)
        end
    end
    vwgt = graph.vwgt
    adjncy = graph.adjncy
    adjwgt = graph.adjwgt
    cmap = graph.cmap
    cgraph = SetupCoarseGraph(graph, cnvtxs)
    cxadj = cgraph.xadj
    cvwgt = cgraph.vwgt
    cadjncy = cgraph.adjncy
    cadjwgt = cgraph.adjwgt
    htable = fill(-1, mask + 1)
    cxadj[1] = 0
    cnvtxs = 0
    cnedges = 0                          # base of the current coarse row
    for v in 0:(nvtxs - 1)
        u = match[v + 1]
        u < v && continue
        cvwgt[cnvtxs + 1] = vwgt[v + 1]
        nedges = 0
        for j in xadj[v + 1]:(xadj[v + 2] - 1)
            k = cmap[adjncy[j + 1] + 1]
            kk = k & mask
            m = htable[kk + 1]
            if m == -1
                cadjncy[cnedges + nedges + 1] = k
                cadjwgt[cnedges + nedges + 1] = adjwgt[j + 1]
                htable[kk + 1] = nedges
                nedges += 1
            elseif cadjncy[cnedges + m + 1] == k
                cadjwgt[cnedges + m + 1] += adjwgt[j + 1]
            else
                jj = 0
                while jj < nedges
                    if cadjncy[cnedges + jj + 1] == k
                        cadjwgt[cnedges + jj + 1] += adjwgt[j + 1]
                        break
                    end
                    jj += 1
                end
                if jj == nedges
                    cadjncy[cnedges + nedges + 1] = k
                    cadjwgt[cnedges + nedges + 1] = adjwgt[j + 1]
                    nedges += 1
                end
            end
        end
        if v != u
            cvwgt[cnvtxs + 1] += vwgt[u + 1]
            for j in xadj[u + 1]:(xadj[u + 2] - 1)
                k = cmap[adjncy[j + 1] + 1]
                kk = k & mask
                m = htable[kk + 1]
                if m == -1
                    cadjncy[cnedges + nedges + 1] = k
                    cadjwgt[cnedges + nedges + 1] = adjwgt[j + 1]
                    htable[kk + 1] = nedges
                    nedges += 1
                elseif cadjncy[cnedges + m + 1] == k
                    cadjwgt[cnedges + m + 1] += adjwgt[j + 1]
                else
                    jj = 0
                    while jj < nedges
                        if cadjncy[cnedges + jj + 1] == k
                            cadjwgt[cnedges + jj + 1] += adjwgt[j + 1]
                            break
                        end
                        jj += 1
                    end
                    if jj == nedges
                        cadjncy[cnedges + nedges + 1] = k
                        cadjwgt[cnedges + nedges + 1] = adjwgt[j + 1]
                        nedges += 1
                    end
                end
            end
            # remove the contracted adjacency weight
            jj = htable[(cnvtxs & mask) + 1]
            if jj >= 0 && cadjncy[cnedges + jj + 1] != cnvtxs
                jj = 0
                while jj < nedges
                    cadjncy[cnedges + jj + 1] == cnvtxs && break
                    jj += 1
                end
            end
            if jj >= 0 && jj < nedges && cadjncy[cnedges + jj + 1] == cnvtxs
                nedges -= 1
                cadjncy[cnedges + jj + 1] = cadjncy[cnedges + nedges + 1]
                cadjwgt[cnedges + jj + 1] = cadjwgt[cnedges + nedges + 1]
            end
        end
        for j in 0:(nedges - 1)
            htable[(cadjncy[cnedges + j + 1] & mask) + 1] = -1
        end
        htable[(cnvtxs & mask) + 1] = -1
        cnedges += nedges
        cnvtxs += 1
        cxadj[cnvtxs + 1] = cnedges
    end
    cgraph.nedges = cnedges
    SetupGraph_tvwgt!(cgraph)
    return nothing
end

function CreateCoarseGraphNoMask(ctrl::CtrlT, graph::GraphT, cnvtxs::Int, match)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    vwgt = graph.vwgt
    adjncy = graph.adjncy
    adjwgt = graph.adjwgt
    cmap = graph.cmap
    cgraph = SetupCoarseGraph(graph, cnvtxs)
    cxadj = cgraph.xadj
    cvwgt = cgraph.vwgt
    cadjncy = cgraph.adjncy
    cadjwgt = cgraph.adjwgt
    htable = fill(-1, cnvtxs)
    cxadj[1] = 0
    cnvtxs = 0
    cnedges = 0
    for v in 0:(nvtxs - 1)
        u = match[v + 1]
        u < v && continue
        cvwgt[cnvtxs + 1] = vwgt[v + 1]
        nedges = 0
        for j in xadj[v + 1]:(xadj[v + 2] - 1)
            k = cmap[adjncy[j + 1] + 1]
            m = htable[k + 1]
            if m == -1
                cadjncy[cnedges + nedges + 1] = k
                cadjwgt[cnedges + nedges + 1] = adjwgt[j + 1]
                htable[k + 1] = nedges
                nedges += 1
            else
                cadjwgt[cnedges + m + 1] += adjwgt[j + 1]
            end
        end
        if v != u
            cvwgt[cnvtxs + 1] += vwgt[u + 1]
            for j in xadj[u + 1]:(xadj[u + 2] - 1)
                k = cmap[adjncy[j + 1] + 1]
                m = htable[k + 1]
                if m == -1
                    cadjncy[cnedges + nedges + 1] = k
                    cadjwgt[cnedges + nedges + 1] = adjwgt[j + 1]
                    htable[k + 1] = nedges
                    nedges += 1
                else
                    cadjwgt[cnedges + m + 1] += adjwgt[j + 1]
                end
            end
            j = htable[cnvtxs + 1]
            if j != -1
                nedges -= 1
                cadjncy[cnedges + j + 1] = cadjncy[cnedges + nedges + 1]
                cadjwgt[cnedges + j + 1] = cadjwgt[cnedges + nedges + 1]
                htable[cnvtxs + 1] = -1
            end
        end
        for j in 0:(nedges - 1)
            htable[cadjncy[cnedges + j + 1] + 1] = -1
        end
        cnedges += nedges
        cnvtxs += 1
        cxadj[cnvtxs + 1] = cnedges
    end
    cgraph.nedges = cnedges
    SetupGraph_tvwgt!(cgraph)
    return nothing
end


rpqSeeTopVal(q::RPQ) = q.nnodes == 0 ? -1 : q.heap_val[1]

# ---------------------------------------------------------------------------
# libmetis/refine.c (2-way edge-cut params; used at the coarsest level only)

function Allocate2WayPartitionMemory!(graph::GraphT)
    graph.pwgts = Vector{Int}(undef, 2)
    graph.where_ = Vector{Int}(undef, graph.nvtxs)
    graph.bndptr = Vector{Int}(undef, graph.nvtxs)
    graph.bndind = Vector{Int}(undef, graph.nvtxs)
    graph.id = Vector{Int}(undef, graph.nvtxs)
    graph.ed = Vector{Int}(undef, graph.nvtxs)
    return nothing
end

function Compute2WayPartitionParams!(ctrl::CtrlT, graph::GraphT)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    vwgt = graph.vwgt
    adjncy = graph.adjncy
    adjwgt = graph.adjwgt
    where_ = graph.where_
    id = graph.id
    ed = graph.ed
    pwgts = graph.pwgts
    fill!(pwgts, 0)
    bndptr = graph.bndptr
    fill!(bndptr, -1)
    bndind = graph.bndind
    for i in 0:(nvtxs - 1)
        pwgts[where_[i + 1] + 1] += vwgt[i + 1]
    end
    nbnd = 0
    mincut = 0
    for i in 0:(nvtxs - 1)
        istart = xadj[i + 1]
        iend = xadj[i + 2]
        me = where_[i + 1]
        tid = 0
        ted = 0
        for j in istart:(iend - 1)
            if me == where_[adjncy[j + 1] + 1]
                tid += adjwgt[j + 1]
            else
                ted += adjwgt[j + 1]
            end
        end
        id[i + 1] = tid
        ed[i + 1] = ted
        if ted > 0 || istart == iend
            bndind[nbnd + 1] = i          # BNDInsert
            bndptr[i + 1] = nbnd
            nbnd += 1
            mincut += ted
        end
    end
    graph.mincut = mincut ÷ 2
    graph.nbnd = nbnd
    return nothing
end

# ---------------------------------------------------------------------------
# libmetis/fm.c FM_2WayRefine / FM_2WayCutRefine (ncon = 1)

FM_2WayRefine!(ctrl::CtrlT, graph::GraphT, ntpwgts, niter::Int) =
    FM_2WayCutRefine!(ctrl, graph, ntpwgts, niter)

function FM_2WayCutRefine!(ctrl::CtrlT, graph::GraphT, ntpwgts, niter::Int)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    vwgt = graph.vwgt
    adjncy = graph.adjncy
    adjwgt = graph.adjwgt
    where_ = graph.where_
    id = graph.id
    ed = graph.ed
    pwgts = graph.pwgts
    bndptr = graph.bndptr
    bndind = graph.bndind
    moved = fill(-1, nvtxs)
    swaps = Vector{Int}(undef, nvtxs)
    perm = Vector{Int}(undef, nvtxs)
    tpwgts0 = trunc(Int, graph.tvwgt * ntpwgts[1])
    tpwgts = (tpwgts0, graph.tvwgt - tpwgts0)
    limit = trunc(Int, min(max(0.01 * nvtxs, 15.0), 100.0))
    avgvwgt = min((pwgts[1] + pwgts[2]) ÷ 20, 2 * (pwgts[1] + pwgts[2]) ÷ nvtxs)
    queues = (rpqCreate(nvtxs), rpqCreate(nvtxs))
    origdiff = abs(tpwgts[1] - pwgts[1])
    for _pass in 0:(niter - 1)
        rpqReset(queues[1])
        rpqReset(queues[2])
        mincutorder = -1
        newcut = mincut = initcut = graph.mincut
        mindiff = abs(tpwgts[1] - pwgts[1])
        nbnd = graph.nbnd
        irandArrayPermute!(nbnd, perm, nbnd, 1)
        for ii in 0:(nbnd - 1)
            i = perm[ii + 1]
            b = bndind[i + 1]
            rpqInsert(queues[where_[b + 1] + 1], b, Float32(ed[b + 1] - id[b + 1]))
        end
        nswaps = 0
        while nswaps < nvtxs
            from = tpwgts[1] - pwgts[1] < tpwgts[2] - pwgts[2] ? 0 : 1
            to = (from + 1) % 2
            higain = rpqGetTop(queues[from + 1])
            higain == -1 && break
            newcut -= ed[higain + 1] - id[higain + 1]
            pwgts[to + 1] += vwgt[higain + 1]
            pwgts[from + 1] -= vwgt[higain + 1]
            if (newcut < mincut && abs(tpwgts[1] - pwgts[1]) <= origdiff + avgvwgt) ||
               (newcut == mincut && abs(tpwgts[1] - pwgts[1]) < mindiff)
                mincut = newcut
                mindiff = abs(tpwgts[1] - pwgts[1])
                mincutorder = nswaps
            elseif nswaps - mincutorder > limit
                newcut += ed[higain + 1] - id[higain + 1]
                pwgts[from + 1] += vwgt[higain + 1]
                pwgts[to + 1] -= vwgt[higain + 1]
                break
            end
            where_[higain + 1] = to
            moved[higain + 1] = nswaps
            swaps[nswaps + 1] = higain
            id[higain + 1], ed[higain + 1] = ed[higain + 1], id[higain + 1]
            if ed[higain + 1] == 0 && xadj[higain + 1] < xadj[higain + 2]
                bndind[bndptr[higain + 1] + 1] = bndind[nbnd]     # BNDDelete
                bndptr[bndind[nbnd] + 1] = bndptr[higain + 1]
                nbnd -= 1
                bndptr[higain + 1] = -1
            end
            for j in xadj[higain + 1]:(xadj[higain + 2] - 1)
                k = adjncy[j + 1]
                kwgt = to == where_[k + 1] ? adjwgt[j + 1] : -adjwgt[j + 1]
                id[k + 1] += kwgt
                ed[k + 1] -= kwgt
                if bndptr[k + 1] != -1
                    if ed[k + 1] == 0
                        bndind[bndptr[k + 1] + 1] = bndind[nbnd]
                        bndptr[bndind[nbnd] + 1] = bndptr[k + 1]
                        nbnd -= 1
                        bndptr[k + 1] = -1
                        moved[k + 1] == -1 && rpqDelete(queues[where_[k + 1] + 1], k)
                    elseif moved[k + 1] == -1
                        rpqUpdate(queues[where_[k + 1] + 1], k, Float32(ed[k + 1] - id[k + 1]))
                    end
                elseif ed[k + 1] > 0
                    bndind[nbnd + 1] = k
                    bndptr[k + 1] = nbnd
                    nbnd += 1
                    moved[k + 1] == -1 && rpqInsert(queues[where_[k + 1] + 1], k, Float32(ed[k + 1] - id[k + 1]))
                end
            end
            nswaps += 1
        end
        for i in 0:(nswaps - 1)
            moved[swaps[i + 1] + 1] = -1
        end
        nswaps -= 1
        while nswaps > mincutorder
            higain = swaps[nswaps + 1]
            to = where_[higain + 1] = (where_[higain + 1] + 1) % 2
            id[higain + 1], ed[higain + 1] = ed[higain + 1], id[higain + 1]
            if ed[higain + 1] == 0 && bndptr[higain + 1] != -1 && xadj[higain + 1] < xadj[higain + 2]
                bndind[bndptr[higain + 1] + 1] = bndind[nbnd]
                bndptr[bndind[nbnd] + 1] = bndptr[higain + 1]
                nbnd -= 1
                bndptr[higain + 1] = -1
            elseif ed[higain + 1] > 0 && bndptr[higain + 1] == -1
                bndind[nbnd + 1] = higain
                bndptr[higain + 1] = nbnd
                nbnd += 1
            end
            pwgts[to + 1] += vwgt[higain + 1]
            pwgts[((to + 1) % 2) + 1] -= vwgt[higain + 1]
            for j in xadj[higain + 1]:(xadj[higain + 2] - 1)
                k = adjncy[j + 1]
                kwgt = to == where_[k + 1] ? adjwgt[j + 1] : -adjwgt[j + 1]
                id[k + 1] += kwgt
                ed[k + 1] -= kwgt
                if bndptr[k + 1] != -1 && ed[k + 1] == 0
                    bndind[bndptr[k + 1] + 1] = bndind[nbnd]
                    bndptr[bndind[nbnd] + 1] = bndptr[k + 1]
                    nbnd -= 1
                    bndptr[k + 1] = -1
                end
                if bndptr[k + 1] == -1 && ed[k + 1] > 0
                    bndind[nbnd + 1] = k
                    bndptr[k + 1] = nbnd
                    nbnd += 1
                end
            end
            nswaps -= 1
        end
        graph.mincut = mincut
        graph.nbnd = nbnd
        (mincutorder <= 0 || mincut == initcut) && break
    end
    return nothing
end

# ---------------------------------------------------------------------------
# libmetis/balance.c (ncon = 1)

# mcutil.c ComputeLoadImbalanceDiff (ncon = 1, nparts = 2), real_t = Float32
function ComputeLoadImbalanceDiff(graph::GraphT, pijbm::Vector{Float32}, ub::Float32)
    mx = -1.0f0
    for j in 1:2
        cur = Float32(graph.pwgts[j]) * pijbm[j] - ub
        cur > mx && (mx = cur)
    end
    return mx
end

function Balance2Way!(ctrl::CtrlT, graph::GraphT, ntpwgts)
    ComputeLoadImbalanceDiff(graph, ctrl.pijbm, ctrl.ubfactors) <= 0 && return nothing
    abs(trunc(Int, ntpwgts[1] * Float32(graph.tvwgt) - graph.pwgts[1])) <
        3 * graph.tvwgt ÷ graph.nvtxs && return nothing
    if graph.nbnd > 0
        Bnd2WayBalance!(ctrl, graph, ntpwgts)
    else
        General2WayBalance!(ctrl, graph, ntpwgts)
    end
    return nothing
end

function Bnd2WayBalance!(ctrl::CtrlT, graph::GraphT, ntpwgts)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    vwgt = graph.vwgt
    adjncy = graph.adjncy
    adjwgt = graph.adjwgt
    where_ = graph.where_
    id = graph.id
    ed = graph.ed
    pwgts = graph.pwgts
    bndptr = graph.bndptr
    bndind = graph.bndind
    moved = fill(-1, nvtxs)
    perm = Vector{Int}(undef, nvtxs)
    tpwgts0 = trunc(Int, graph.tvwgt * ntpwgts[1])
    tpwgts = (tpwgts0, graph.tvwgt - tpwgts0)
    mindiff = abs(tpwgts[1] - pwgts[1])
    from = pwgts[1] < tpwgts[1] ? 1 : 0
    to = (from + 1) % 2
    queue = rpqCreate(nvtxs)
    nbnd = graph.nbnd
    irandArrayPermute!(nbnd, perm, nbnd ÷ 5, 1)
    for ii in 0:(nbnd - 1)
        i = perm[ii + 1]
        b = bndind[i + 1]
        if where_[b + 1] == from && vwgt[b + 1] <= mindiff
            rpqInsert(queue, b, Float32(ed[b + 1] - id[b + 1]))
        end
    end
    mincut = graph.mincut
    for _ in 0:(nvtxs - 1)
        higain = rpqGetTop(queue)
        higain == -1 && break
        pwgts[to + 1] + vwgt[higain + 1] > tpwgts[to + 1] && break
        mincut -= ed[higain + 1] - id[higain + 1]
        pwgts[to + 1] += vwgt[higain + 1]
        pwgts[from + 1] -= vwgt[higain + 1]
        where_[higain + 1] = to
        moved[higain + 1] = 1
        id[higain + 1], ed[higain + 1] = ed[higain + 1], id[higain + 1]
        if ed[higain + 1] == 0 && xadj[higain + 1] < xadj[higain + 2]
            bndind[bndptr[higain + 1] + 1] = bndind[nbnd]
            bndptr[bndind[nbnd] + 1] = bndptr[higain + 1]
            nbnd -= 1
            bndptr[higain + 1] = -1
        end
        for j in xadj[higain + 1]:(xadj[higain + 2] - 1)
            k = adjncy[j + 1]
            kwgt = to == where_[k + 1] ? adjwgt[j + 1] : -adjwgt[j + 1]
            id[k + 1] += kwgt
            ed[k + 1] -= kwgt
            if bndptr[k + 1] != -1
                if ed[k + 1] == 0
                    bndind[bndptr[k + 1] + 1] = bndind[nbnd]
                    bndptr[bndind[nbnd] + 1] = bndptr[k + 1]
                    nbnd -= 1
                    bndptr[k + 1] = -1
                    if moved[k + 1] == -1 && where_[k + 1] == from && vwgt[k + 1] <= mindiff
                        rpqDelete(queue, k)
                    end
                elseif moved[k + 1] == -1 && where_[k + 1] == from && vwgt[k + 1] <= mindiff
                    rpqUpdate(queue, k, Float32(ed[k + 1] - id[k + 1]))
                end
            elseif ed[k + 1] > 0
                bndind[nbnd + 1] = k
                bndptr[k + 1] = nbnd
                nbnd += 1
                if moved[k + 1] == -1 && where_[k + 1] == from && vwgt[k + 1] <= mindiff
                    rpqInsert(queue, k, Float32(ed[k + 1] - id[k + 1]))
                end
            end
        end
    end
    graph.mincut = mincut
    graph.nbnd = nbnd
    return nothing
end

function General2WayBalance!(ctrl::CtrlT, graph::GraphT, ntpwgts)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    vwgt = graph.vwgt
    adjncy = graph.adjncy
    adjwgt = graph.adjwgt
    where_ = graph.where_
    id = graph.id
    ed = graph.ed
    pwgts = graph.pwgts
    bndptr = graph.bndptr
    bndind = graph.bndind
    moved = fill(-1, nvtxs)
    perm = Vector{Int}(undef, nvtxs)
    tpwgts0 = trunc(Int, graph.tvwgt * ntpwgts[1])
    tpwgts = (tpwgts0, graph.tvwgt - tpwgts0)
    mindiff = abs(tpwgts[1] - pwgts[1])
    from = pwgts[1] < tpwgts[1] ? 1 : 0
    to = (from + 1) % 2
    queue = rpqCreate(nvtxs)
    irandArrayPermute!(nvtxs, perm, nvtxs ÷ 5, 1)
    for ii in 0:(nvtxs - 1)
        i = perm[ii + 1]
        if where_[i + 1] == from && vwgt[i + 1] <= mindiff
            rpqInsert(queue, i, Float32(ed[i + 1] - id[i + 1]))
        end
    end
    mincut = graph.mincut
    nbnd = graph.nbnd
    for _ in 0:(nvtxs - 1)
        higain = rpqGetTop(queue)
        higain == -1 && break
        pwgts[to + 1] + vwgt[higain + 1] > tpwgts[to + 1] && break
        mincut -= ed[higain + 1] - id[higain + 1]
        pwgts[to + 1] += vwgt[higain + 1]
        pwgts[from + 1] -= vwgt[higain + 1]
        where_[higain + 1] = to
        moved[higain + 1] = 1
        id[higain + 1], ed[higain + 1] = ed[higain + 1], id[higain + 1]
        if ed[higain + 1] == 0 && bndptr[higain + 1] != -1 && xadj[higain + 1] < xadj[higain + 2]
            bndind[bndptr[higain + 1] + 1] = bndind[nbnd]
            bndptr[bndind[nbnd] + 1] = bndptr[higain + 1]
            nbnd -= 1
            bndptr[higain + 1] = -1
        end
        if ed[higain + 1] > 0 && bndptr[higain + 1] == -1
            bndind[nbnd + 1] = higain
            bndptr[higain + 1] = nbnd
            nbnd += 1
        end
        for j in xadj[higain + 1]:(xadj[higain + 2] - 1)
            k = adjncy[j + 1]
            kwgt = to == where_[k + 1] ? adjwgt[j + 1] : -adjwgt[j + 1]
            id[k + 1] += kwgt
            ed[k + 1] -= kwgt
            if moved[k + 1] == -1 && where_[k + 1] == from && vwgt[k + 1] <= mindiff
                rpqUpdate(queue, k, Float32(ed[k + 1] - id[k + 1]))
            end
            if ed[k + 1] == 0 && bndptr[k + 1] != -1
                bndind[bndptr[k + 1] + 1] = bndind[nbnd]
                bndptr[bndind[nbnd] + 1] = bndptr[k + 1]
                nbnd -= 1
                bndptr[k + 1] = -1
            elseif ed[k + 1] > 0 && bndptr[k + 1] == -1
                bndind[nbnd + 1] = k
                bndptr[k + 1] = nbnd
                nbnd += 1
            end
        end
    end
    graph.mincut = mincut
    graph.nbnd = nbnd
    return nothing
end


# ---------------------------------------------------------------------------
# libmetis/initpart.c (iptype = EDGE path)

function RandomBisection!(ctrl::CtrlT, graph::GraphT, ntpwgts, niparts::Int)
    nvtxs = graph.nvtxs
    vwgt = graph.vwgt
    Allocate2WayPartitionMemory!(graph)
    where_ = graph.where_
    bestwhere = Vector{Int}(undef, nvtxs)
    perm = Vector{Int}(undef, nvtxs)
    zeromaxpwgt = trunc(Int, ctrl.ubfactors * Float32(graph.tvwgt) * ntpwgts[1])
    bestcut = 0
    for inbfs in 0:(niparts - 1)
        fill!(where_, 1)
        if inbfs > 0
            irandArrayPermute!(nvtxs, perm, nvtxs ÷ 2, 1)
            pw0 = 0
            for ii in 0:(nvtxs - 1)
                i = perm[ii + 1]
                if pw0 + vwgt[i + 1] < zeromaxpwgt
                    where_[i + 1] = 0
                    pw0 += vwgt[i + 1]
                    pw0 > zeromaxpwgt && break
                end
            end
        end
        Compute2WayPartitionParams!(ctrl, graph)
        Balance2Way!(ctrl, graph, ntpwgts)
        FM_2WayRefine!(ctrl, graph, ntpwgts, 4)
        if inbfs == 0 || bestcut > graph.mincut
            bestcut = graph.mincut
            copyto!(bestwhere, where_)
            bestcut == 0 && break
        end
    end
    graph.mincut = bestcut
    copyto!(where_, bestwhere)
    return nothing
end

function GrowBisection!(ctrl::CtrlT, graph::GraphT, ntpwgts, niparts::Int)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    vwgt = graph.vwgt
    adjncy = graph.adjncy
    Allocate2WayPartitionMemory!(graph)
    where_ = graph.where_
    bestwhere = Vector{Int}(undef, nvtxs)
    queue = Vector{Int}(undef, nvtxs)
    touched = Vector{Int}(undef, nvtxs)
    onemaxpwgt = trunc(Int, ctrl.ubfactors * Float32(graph.tvwgt) * ntpwgts[2])
    oneminpwgt = trunc(Int, (1.0 / ctrl.ubfactors) * graph.tvwgt * ntpwgts[2])
    bestcut = 0
    for inbfs in 0:(niparts - 1)
        fill!(where_, 1)
        fill!(touched, 0)
        pw0 = 0
        pw1 = graph.tvwgt
        queue[1] = irandInRange(nvtxs)
        touched[queue[1] + 1] = 1
        first = 0
        last = 1
        nleft = nvtxs - 1
        drain = false
        while true
            if first == last                       # empty: disconnected
                (nleft == 0 || drain) && break
                k = irandInRange(nleft)
                i = 0
                while i < nvtxs
                    if touched[i + 1] == 0
                        k == 0 && break
                        k -= 1
                    end
                    i += 1
                end
                queue[1] = i
                touched[i + 1] = 1
                first = 0
                last = 1
                nleft -= 1
            end
            i = queue[first + 1]
            first += 1
            if pw0 > 0 && pw1 - vwgt[i + 1] < oneminpwgt
                drain = true
                continue
            end
            where_[i + 1] = 0
            pw0 += vwgt[i + 1]
            pw1 -= vwgt[i + 1]
            pw1 <= onemaxpwgt && break
            drain = false
            for j in xadj[i + 1]:(xadj[i + 2] - 1)
                k = adjncy[j + 1]
                if touched[k + 1] == 0
                    queue[last + 1] = k
                    last += 1
                    touched[k + 1] = 1
                    nleft -= 1
                end
            end
        end
        pw1 == 0 && (where_[irandInRange(nvtxs) + 1] = 1)
        pw0 == 0 && (where_[irandInRange(nvtxs) + 1] = 0)
        Compute2WayPartitionParams!(ctrl, graph)
        Balance2Way!(ctrl, graph, ntpwgts)
        FM_2WayRefine!(ctrl, graph, ntpwgts, ctrl.niter)
        if inbfs == 0 || bestcut > graph.mincut
            bestcut = graph.mincut
            copyto!(bestwhere, where_)
            bestcut == 0 && break
        end
    end
    graph.mincut = bestcut
    copyto!(where_, bestwhere)
    return nothing
end

function InitSeparator!(ctrl::CtrlT, graph::GraphT, niparts::Int)
    ntpwgts = (0.5f0, 0.5f0)
    # Setup2WayBalMultipliers
    ctrl.pijbm = Float32[graph.invtvwgt / ntpwgts[1], graph.invtvwgt / ntpwgts[2]]
    if graph.nedges == 0
        RandomBisection!(ctrl, graph, ntpwgts, niparts)
    else
        GrowBisection!(ctrl, graph, ntpwgts, niparts)
    end
    Compute2WayPartitionParams!(ctrl, graph)
    ConstructSeparator!(ctrl, graph)
    DBG[] && println("Initial Sep: ", graph.mincut)
    return nothing
end

# ---------------------------------------------------------------------------
# libmetis/srefine.c

function Allocate2WayNodePartitionMemory!(graph::GraphT)
    graph.pwgts = Vector{Int}(undef, 3)
    graph.where_ = Vector{Int}(undef, graph.nvtxs)
    graph.bndptr = Vector{Int}(undef, graph.nvtxs)
    graph.bndind = Vector{Int}(undef, graph.nvtxs)
    graph.nrinfo = Vector{NTuple{2, Int}}(undef, graph.nvtxs)
    return nothing
end

function Compute2WayNodePartitionParams!(ctrl::CtrlT, graph::GraphT)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    vwgt = graph.vwgt
    adjncy = graph.adjncy
    where_ = graph.where_
    rinfo = graph.nrinfo
    pwgts = graph.pwgts
    fill!(pwgts, 0)
    bndind = graph.bndind
    bndptr = graph.bndptr
    fill!(bndptr, -1)
    nbnd = 0
    for i in 0:(nvtxs - 1)
        me = where_[i + 1]
        pwgts[me + 1] += vwgt[i + 1]
        if me == 2
            bndind[nbnd + 1] = i
            bndptr[i + 1] = nbnd
            nbnd += 1
            e0 = 0
            e1 = 0
            for j in xadj[i + 1]:(xadj[i + 2] - 1)
                other = where_[adjncy[j + 1] + 1]
                if other != 2
                    other == 0 ? (e0 += vwgt[adjncy[j + 1] + 1]) : (e1 += vwgt[adjncy[j + 1] + 1])
                end
            end
            rinfo[i + 1] = (e0, e1)
        end
    end
    graph.mincut = pwgts[3]
    graph.nbnd = nbnd
    return nothing
end

function Project2WayNodePartition!(ctrl::CtrlT, graph::GraphT)
    cgraph = graph.coarser::GraphT
    cwhere = cgraph.where_
    nvtxs = graph.nvtxs
    cmap = graph.cmap
    Allocate2WayNodePartitionMemory!(graph)
    where_ = graph.where_
    for i in 0:(nvtxs - 1)
        where_[i + 1] = cwhere[cmap[i + 1] + 1]
    end
    graph.coarser = nothing
    Compute2WayNodePartitionParams!(ctrl, graph)
    return nothing
end

function Refine2WayNode!(ctrl::CtrlT, orggraph::GraphT, graph::GraphT)
    if graph === orggraph
        Compute2WayNodePartitionParams!(ctrl, graph)
    else
        while true
            graph = graph.finer::GraphT
            Project2WayNodePartition!(ctrl, graph)
            FM_2WayNodeBalance!(ctrl, graph)
            if ctrl.rtype == METIS_RTYPE_SEP2SIDED
                FM_2WayNodeRefine2Sided!(ctrl, graph, ctrl.niter)
            else
                FM_2WayNodeRefine1Sided!(ctrl, graph, ctrl.niter)
            end
            graph === orggraph && break
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# libmetis/separator.c ConstructSeparator (iptype = EDGE)

function ConstructSeparator!(ctrl::CtrlT, graph::GraphT)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    nbnd = graph.nbnd
    bndind = graph.bndind
    where_ = copy(graph.where_)
    for i in 0:(nbnd - 1)
        j = bndind[i + 1]
        xadj[j + 2] - xadj[j + 1] > 0 && (where_[j + 1] = 2)
    end
    Allocate2WayNodePartitionMemory!(graph)
    copyto!(graph.where_, where_)
    Compute2WayNodePartitionParams!(ctrl, graph)
    FM_2WayNodeRefine2Sided!(ctrl, graph, 1)
    FM_2WayNodeRefine1Sided!(ctrl, graph, 4)
    return nothing
end


# ---------------------------------------------------------------------------
# libmetis/sfm.c — node-separator FM

function FM_2WayNodeRefine2Sided!(ctrl::CtrlT, graph::GraphT, niter::Int)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    adjncy = graph.adjncy
    vwgt = graph.vwgt
    bndind = graph.bndind
    bndptr = graph.bndptr
    where_ = graph.where_
    pwgts = graph.pwgts
    rinfo = graph.nrinfo
    queues = (rpqCreate(nvtxs), rpqCreate(nvtxs))
    moved = Vector{Int}(undef, nvtxs)
    swaps = Vector{Int}(undef, nvtxs)
    mptr = Vector{Int}(undef, nvtxs + 1)
    mind = Vector{Int}(undef, 2 * nvtxs)
    mult = Float32(0.5 * ctrl.ubfactors)
    badmaxpwgt = trunc(Int, mult * (pwgts[1] + pwgts[2] + pwgts[3]))
    for pass in 0:(niter - 1)
        fill!(moved, -1)
        rpqReset(queues[1])
        rpqReset(queues[2])
        mincutorder = -1
        initcut = mincut = graph.mincut
        nbnd = graph.nbnd
        irandArrayPermute!(nbnd, swaps, nbnd, 1)
        for ii in 0:(nbnd - 1)
            i = bndind[swaps[ii + 1] + 1]
            rpqInsert(queues[1], i, Float32(vwgt[i + 1] - rinfo[i + 1][2]))
            rpqInsert(queues[2], i, Float32(vwgt[i + 1] - rinfo[i + 1][1]))
        end
        limit = ctrl.compress ? min(5 * nbnd, 400) : min(2 * nbnd, 300)
        mptr[1] = 0
        nmind = 0
        mindiff = abs(pwgts[1] - pwgts[2])
        to = pwgts[1] < pwgts[2] ? 0 : 1
        nswaps = 0
        while nswaps < nvtxs
            u0 = rpqSeeTopVal(queues[1])
            u1 = rpqSeeTopVal(queues[2])
            if u0 != -1 && u1 != -1
                g0 = vwgt[u0 + 1] - rinfo[u0 + 1][2]
                g1 = vwgt[u1 + 1] - rinfo[u1 + 1][1]
                to = g0 > g1 ? 0 : (g0 < g1 ? 1 : pass % 2)
                u = (u0, u1)
                if pwgts[to + 1] + vwgt[u[to + 1] + 1] > badmaxpwgt
                    to = (to + 1) % 2
                end
            elseif u0 == -1 && u1 == -1
                break
            elseif u0 != -1 && pwgts[1] + vwgt[u0 + 1] <= badmaxpwgt
                to = 0
            elseif u1 != -1 && pwgts[2] + vwgt[u1 + 1] <= badmaxpwgt
                to = 1
            else
                break
            end
            other = (to + 1) % 2
            higain = rpqGetTop(queues[to + 1])
            moved[higain + 1] == -1 && rpqDelete(queues[other + 1], higain)
            if nmind + xadj[higain + 2] - xadj[higain + 1] >= 2 * nvtxs - 1
                break
            end
            eo = rinfo[higain + 1][other + 1]
            pwgts[3] -= vwgt[higain + 1] - eo
            newdiff = abs(pwgts[to + 1] + vwgt[higain + 1] - (pwgts[other + 1] - eo))
            if pwgts[3] < mincut || (pwgts[3] == mincut && newdiff < mindiff)
                mincut = pwgts[3]
                mincutorder = nswaps
                mindiff = newdiff
            elseif nswaps - mincutorder > 2 * limit ||
                   (nswaps - mincutorder > limit && pwgts[3] > 1.10 * mincut)
                pwgts[3] += vwgt[higain + 1] - eo
                break
            end
            bndind[bndptr[higain + 1] + 1] = bndind[nbnd]       # BNDDelete
            bndptr[bndind[nbnd] + 1] = bndptr[higain + 1]
            nbnd -= 1
            bndptr[higain + 1] = -1
            pwgts[to + 1] += vwgt[higain + 1]
            where_[higain + 1] = to
            moved[higain + 1] = nswaps
            swaps[nswaps + 1] = higain
            for j in xadj[higain + 1]:(xadj[higain + 2] - 1)
                k = adjncy[j + 1]
                if where_[k + 1] == 2
                    oldgain = vwgt[k + 1] - rinfo[k + 1][to + 1]
                    rinfo[k + 1] = to == 0 ? (rinfo[k + 1][1] + vwgt[higain + 1], rinfo[k + 1][2]) :
                                             (rinfo[k + 1][1], rinfo[k + 1][2] + vwgt[higain + 1])
                    if moved[k + 1] == -1 || moved[k + 1] == -(2 + other)
                        rpqUpdate(queues[other + 1], k, Float32(oldgain - vwgt[higain + 1]))
                    end
                elseif where_[k + 1] == other
                    bndind[nbnd + 1] = k                        # BNDInsert
                    bndptr[k + 1] = nbnd
                    nbnd += 1
                    nmind += 1
                    mind[nmind] = k
                    where_[k + 1] = 2
                    pwgts[other + 1] -= vwgt[k + 1]
                    e0 = 0
                    e1 = 0
                    for jj in xadj[k + 1]:(xadj[k + 2] - 1)
                        kk = adjncy[jj + 1]
                        if where_[kk + 1] != 2
                            where_[kk + 1] == 0 ? (e0 += vwgt[kk + 1]) : (e1 += vwgt[kk + 1])
                        else
                            oldgain = vwgt[kk + 1] - rinfo[kk + 1][other + 1]
                            rinfo[kk + 1] = other == 0 ? (rinfo[kk + 1][1] - vwgt[k + 1], rinfo[kk + 1][2]) :
                                                         (rinfo[kk + 1][1], rinfo[kk + 1][2] - vwgt[k + 1])
                            if moved[kk + 1] == -1 || moved[kk + 1] == -(2 + to)
                                rpqUpdate(queues[to + 1], kk, Float32(oldgain + vwgt[k + 1]))
                            end
                        end
                    end
                    rinfo[k + 1] = (e0, e1)
                    if moved[k + 1] == -1
                        rpqInsert(queues[to + 1], k, Float32(vwgt[k + 1] - (other == 0 ? e0 : e1)))
                        moved[k + 1] = -(2 + to)
                    end
                end
            end
            mptr[nswaps + 2] = nmind
            nswaps += 1
        end
        nswaps -= 1
        while nswaps > mincutorder
            higain = swaps[nswaps + 1]
            to = where_[higain + 1]
            other = (to + 1) % 2
            pwgts[3] += vwgt[higain + 1]
            pwgts[to + 1] -= vwgt[higain + 1]
            where_[higain + 1] = 2
            bndind[nbnd + 1] = higain
            bndptr[higain + 1] = nbnd
            nbnd += 1
            e0 = 0
            e1 = 0
            for j in xadj[higain + 1]:(xadj[higain + 2] - 1)
                k = adjncy[j + 1]
                if where_[k + 1] == 2
                    rinfo[k + 1] = to == 0 ? (rinfo[k + 1][1] - vwgt[higain + 1], rinfo[k + 1][2]) :
                                             (rinfo[k + 1][1], rinfo[k + 1][2] - vwgt[higain + 1])
                else
                    where_[k + 1] == 0 ? (e0 += vwgt[k + 1]) : (e1 += vwgt[k + 1])
                end
            end
            rinfo[higain + 1] = (e0, e1)
            for j in mptr[nswaps + 1]:(mptr[nswaps + 2] - 1)
                k = mind[j + 1]
                where_[k + 1] = other
                pwgts[other + 1] += vwgt[k + 1]
                pwgts[3] -= vwgt[k + 1]
                bndind[bndptr[k + 1] + 1] = bndind[nbnd]
                bndptr[bndind[nbnd] + 1] = bndptr[k + 1]
                nbnd -= 1
                bndptr[k + 1] = -1
                for jj in xadj[k + 1]:(xadj[k + 2] - 1)
                    kk = adjncy[jj + 1]
                    if where_[kk + 1] == 2
                        rinfo[kk + 1] = other == 0 ? (rinfo[kk + 1][1] + vwgt[k + 1], rinfo[kk + 1][2]) :
                                                     (rinfo[kk + 1][1], rinfo[kk + 1][2] + vwgt[k + 1])
                    end
                end
            end
            nswaps -= 1
        end
        graph.mincut = mincut
        graph.nbnd = nbnd
        (mincutorder == -1 || mincut >= initcut) && break
    end
    return nothing
end

function FM_2WayNodeRefine1Sided!(ctrl::CtrlT, graph::GraphT, niter::Int)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    adjncy = graph.adjncy
    vwgt = graph.vwgt
    bndind = graph.bndind
    bndptr = graph.bndptr
    where_ = graph.where_
    pwgts = graph.pwgts
    rinfo = graph.nrinfo
    queue = rpqCreate(nvtxs)
    swaps = Vector{Int}(undef, nvtxs)
    mptr = Vector{Int}(undef, nvtxs + 1)
    mind = Vector{Int}(undef, 2 * nvtxs)
    mult = Float32(0.5 * ctrl.ubfactors)
    badmaxpwgt = trunc(Int, mult * (pwgts[1] + pwgts[2] + pwgts[3]))
    to = pwgts[1] < pwgts[2] ? 1 : 0
    for pass in 0:(2 * niter - 1)
        other = to
        to = (to + 1) % 2
        rpqReset(queue)
        mincutorder = -1
        initcut = mincut = graph.mincut
        nbnd = graph.nbnd
        irandArrayPermute!(nbnd, swaps, nbnd, 1)
        for ii in 0:(nbnd - 1)
            i = bndind[swaps[ii + 1] + 1]
            rpqInsert(queue, i, Float32(vwgt[i + 1] - rinfo[i + 1][other + 1]))
        end
        limit = ctrl.compress ? min(5 * nbnd, 500) : min(3 * nbnd, 300)
        mptr[1] = 0
        nmind = 0
        mindiff = abs(pwgts[1] - pwgts[2])
        nswaps = 0
        while nswaps < nvtxs
            higain = rpqGetTop(queue)
            higain == -1 && break
            if nmind + xadj[higain + 2] - xadj[higain + 1] >= 2 * nvtxs - 1
                break
            end
            pwgts[to + 1] + vwgt[higain + 1] > badmaxpwgt && break
            eo = rinfo[higain + 1][other + 1]
            pwgts[3] -= vwgt[higain + 1] - eo
            newdiff = abs(pwgts[to + 1] + vwgt[higain + 1] - (pwgts[other + 1] - eo))
            if pwgts[3] < mincut || (pwgts[3] == mincut && newdiff < mindiff)
                mincut = pwgts[3]
                mincutorder = nswaps
                mindiff = newdiff
            elseif nswaps - mincutorder > 3 * limit ||
                   (nswaps - mincutorder > limit && pwgts[3] > 1.10 * mincut)
                pwgts[3] += vwgt[higain + 1] - eo
                break
            end
            bndind[bndptr[higain + 1] + 1] = bndind[nbnd]
            bndptr[bndind[nbnd] + 1] = bndptr[higain + 1]
            nbnd -= 1
            bndptr[higain + 1] = -1
            pwgts[to + 1] += vwgt[higain + 1]
            where_[higain + 1] = to
            swaps[nswaps + 1] = higain
            for j in xadj[higain + 1]:(xadj[higain + 2] - 1)
                k = adjncy[j + 1]
                if where_[k + 1] == 2
                    rinfo[k + 1] = to == 0 ? (rinfo[k + 1][1] + vwgt[higain + 1], rinfo[k + 1][2]) :
                                             (rinfo[k + 1][1], rinfo[k + 1][2] + vwgt[higain + 1])
                elseif where_[k + 1] == other
                    bndind[nbnd + 1] = k
                    bndptr[k + 1] = nbnd
                    nbnd += 1
                    nmind += 1
                    mind[nmind] = k
                    where_[k + 1] = 2
                    pwgts[other + 1] -= vwgt[k + 1]
                    e0 = 0
                    e1 = 0
                    for jj in xadj[k + 1]:(xadj[k + 2] - 1)
                        kk = adjncy[jj + 1]
                        if where_[kk + 1] != 2
                            where_[kk + 1] == 0 ? (e0 += vwgt[kk + 1]) : (e1 += vwgt[kk + 1])
                        else
                            rinfo[kk + 1] = other == 0 ? (rinfo[kk + 1][1] - vwgt[k + 1], rinfo[kk + 1][2]) :
                                                         (rinfo[kk + 1][1], rinfo[kk + 1][2] - vwgt[k + 1])
                            rpqUpdate(queue, kk, Float32(vwgt[kk + 1] - rinfo[kk + 1][other + 1]))
                        end
                    end
                    rinfo[k + 1] = (e0, e1)
                    rpqInsert(queue, k, Float32(vwgt[k + 1] - (other == 0 ? e0 : e1)))
                end
            end
            mptr[nswaps + 2] = nmind
            nswaps += 1
        end
        nswaps -= 1
        while nswaps > mincutorder
            higain = swaps[nswaps + 1]
            pwgts[3] += vwgt[higain + 1]
            pwgts[to + 1] -= vwgt[higain + 1]
            where_[higain + 1] = 2
            bndind[nbnd + 1] = higain
            bndptr[higain + 1] = nbnd
            nbnd += 1
            e0 = 0
            e1 = 0
            for j in xadj[higain + 1]:(xadj[higain + 2] - 1)
                k = adjncy[j + 1]
                if where_[k + 1] == 2
                    rinfo[k + 1] = to == 0 ? (rinfo[k + 1][1] - vwgt[higain + 1], rinfo[k + 1][2]) :
                                             (rinfo[k + 1][1], rinfo[k + 1][2] - vwgt[higain + 1])
                else
                    where_[k + 1] == 0 ? (e0 += vwgt[k + 1]) : (e1 += vwgt[k + 1])
                end
            end
            rinfo[higain + 1] = (e0, e1)
            for j in mptr[nswaps + 1]:(mptr[nswaps + 2] - 1)
                k = mind[j + 1]
                where_[k + 1] = other
                pwgts[other + 1] += vwgt[k + 1]
                pwgts[3] -= vwgt[k + 1]
                bndind[bndptr[k + 1] + 1] = bndind[nbnd]
                bndptr[bndind[nbnd] + 1] = bndptr[k + 1]
                nbnd -= 1
                bndptr[k + 1] = -1
                for jj in xadj[k + 1]:(xadj[k + 2] - 1)
                    kk = adjncy[jj + 1]
                    if where_[kk + 1] == 2
                        rinfo[kk + 1] = other == 0 ? (rinfo[kk + 1][1] + vwgt[k + 1], rinfo[kk + 1][2]) :
                                                     (rinfo[kk + 1][1], rinfo[kk + 1][2] + vwgt[k + 1])
                    end
                end
            end
            nswaps -= 1
        end
        graph.mincut = mincut
        graph.nbnd = nbnd
        pass % 2 == 1 && (mincutorder == -1 || mincut >= initcut) && break
    end
    return nothing
end

function FM_2WayNodeBalance!(ctrl::CtrlT, graph::GraphT)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    adjncy = graph.adjncy
    vwgt = graph.vwgt
    bndind = graph.bndind
    bndptr = graph.bndptr
    where_ = graph.where_
    pwgts = graph.pwgts
    rinfo = graph.nrinfo
    mult = Float32(0.5 * ctrl.ubfactors)
    badmaxpwgt = trunc(Int, mult * (pwgts[1] + pwgts[2]))
    max(pwgts[1], pwgts[2]) < badmaxpwgt && return nothing
    abs(pwgts[1] - pwgts[2]) < 3 * graph.tvwgt ÷ nvtxs && return nothing
    to = pwgts[1] < pwgts[2] ? 0 : 1
    other = (to + 1) % 2
    queue = rpqCreate(nvtxs)
    perm = Vector{Int}(undef, nvtxs)
    moved = fill(-1, nvtxs)
    nbnd = graph.nbnd
    irandArrayPermute!(nbnd, perm, nbnd, 1)
    for ii in 0:(nbnd - 1)
        i = bndind[perm[ii + 1] + 1]
        rpqInsert(queue, i, Float32(vwgt[i + 1] - rinfo[i + 1][other + 1]))
    end
    for _ in 0:(nvtxs - 1)
        higain = rpqGetTop(queue)
        higain == -1 && break
        moved[higain + 1] = 1
        gain = vwgt[higain + 1] - rinfo[higain + 1][other + 1]
        badmaxpwgt = trunc(Int, mult * (pwgts[1] + pwgts[2]))
        pwgts[to + 1] > pwgts[other + 1] && break
        gain < 0 && pwgts[other + 1] < badmaxpwgt && break
        pwgts[to + 1] + vwgt[higain + 1] > badmaxpwgt && continue
        pwgts[3] -= gain
        bndind[bndptr[higain + 1] + 1] = bndind[nbnd]
        bndptr[bndind[nbnd] + 1] = bndptr[higain + 1]
        nbnd -= 1
        bndptr[higain + 1] = -1
        pwgts[to + 1] += vwgt[higain + 1]
        where_[higain + 1] = to
        for j in xadj[higain + 1]:(xadj[higain + 2] - 1)
            k = adjncy[j + 1]
            if where_[k + 1] == 2
                rinfo[k + 1] = to == 0 ? (rinfo[k + 1][1] + vwgt[higain + 1], rinfo[k + 1][2]) :
                                         (rinfo[k + 1][1], rinfo[k + 1][2] + vwgt[higain + 1])
            elseif where_[k + 1] == other
                bndind[nbnd + 1] = k
                bndptr[k + 1] = nbnd
                nbnd += 1
                where_[k + 1] = 2
                pwgts[other + 1] -= vwgt[k + 1]
                e0 = 0
                e1 = 0
                for jj in xadj[k + 1]:(xadj[k + 2] - 1)
                    kk = adjncy[jj + 1]
                    if where_[kk + 1] != 2
                        where_[kk + 1] == 0 ? (e0 += vwgt[kk + 1]) : (e1 += vwgt[kk + 1])
                    else
                        oldgain = vwgt[kk + 1] - rinfo[kk + 1][other + 1]
                        rinfo[kk + 1] = other == 0 ? (rinfo[kk + 1][1] - vwgt[k + 1], rinfo[kk + 1][2]) :
                                                     (rinfo[kk + 1][1], rinfo[kk + 1][2] - vwgt[k + 1])
                        moved[kk + 1] == -1 && rpqUpdate(queue, kk, Float32(oldgain + vwgt[k + 1]))
                    end
                end
                rinfo[k + 1] = (e0, e1)
                rpqInsert(queue, k, Float32(vwgt[k + 1] - (other == 0 ? e0 : e1)))
            end
        end
    end
    graph.mincut = pwgts[3]
    graph.nbnd = nbnd
    return nothing
end


# ---------------------------------------------------------------------------
# libmetis/ometis.c — bisection drivers, split, MMD leaves, NodeND

function MlevelNodeBisectionMultiple!(ctrl::CtrlT, graph::GraphT)
    if ctrl.nseps == 1 || graph.nvtxs < (ctrl.compress ? 1000 : 2000)
        MlevelNodeBisectionL2!(ctrl, graph, LARGENIPARTS)
        return nothing
    end
    bestwhere = Vector{Int}(undef, graph.nvtxs)
    mincut = graph.tvwgt
    for i in 0:(ctrl.nseps - 1)
        MlevelNodeBisectionL2!(ctrl, graph, LARGENIPARTS)
        if i == 0 || graph.mincut < mincut
            mincut = graph.mincut
            i < ctrl.nseps - 1 && copyto!(bestwhere, graph.where_)
        end
        mincut == 0 && break
    end
    if mincut != graph.mincut
        copyto!(graph.where_, bestwhere)
        Compute2WayNodePartitionParams!(ctrl, graph)
    end
    return nothing
end

function MlevelNodeBisectionL2!(ctrl::CtrlT, graph::GraphT, niparts::Int)
    if graph.nvtxs < 5000
        MlevelNodeBisectionL1!(ctrl, graph, niparts)
        return nothing
    end
    nruns = 5
    ctrl.CoarsenTo = max(100, graph.nvtxs ÷ 30)
    t = @elapsed (cgraph = CoarsenGraphNlevels(ctrl, graph, 4))
    prof!(:coarsen, t)
    bestwhere = Vector{Int}(undef, cgraph.nvtxs)
    mincut = graph.tvwgt
    for i in 0:(nruns - 1)
        MlevelNodeBisectionL1!(ctrl, cgraph, trunc(Int, 0.7 * niparts))
        if i == 0 || cgraph.mincut < mincut
            mincut = cgraph.mincut
            i < nruns - 1 && copyto!(bestwhere, cgraph.where_)
        end
        mincut == 0 && break
    end
    if mincut != cgraph.mincut
        copyto!(cgraph.where_, bestwhere)
        Compute2WayNodePartitionParams!(ctrl, cgraph)
    end
    t = @elapsed Refine2WayNode!(ctrl, graph, cgraph)
    prof!(:refine_l2, t)
    return nothing
end

function MlevelNodeBisectionL1!(ctrl::CtrlT, graph::GraphT, niparts::Int)
    ctrl.CoarsenTo = graph.nvtxs ÷ 8
    ctrl.CoarsenTo > 100 && (ctrl.CoarsenTo = 100)
    ctrl.CoarsenTo < 40 && (ctrl.CoarsenTo = 40)
    t = @elapsed (cgraph = CoarsenGraph(ctrl, graph))
    prof!(:coarsen, t)
    niparts = max(1, cgraph.nvtxs <= ctrl.CoarsenTo ? niparts ÷ 2 : niparts)
    t = @elapsed InitSeparator!(ctrl, cgraph, niparts)
    prof!(:initsep, t)
    t = @elapsed Refine2WayNode!(ctrl, graph, cgraph)
    prof!(:refine, t)
    return nothing
end

# SplitGraphOrder: tri-section into the left/right subgraphs (separator out)
function SplitGraphOrder(ctrl::CtrlT, graph::GraphT)
    nvtxs = graph.nvtxs
    xadj = graph.xadj
    vwgt = graph.vwgt
    adjncy = graph.adjncy
    label = graph.label
    where_ = graph.where_
    bndptr = graph.bndptr
    bndind = graph.bndind
    rename = Vector{Int}(undef, nvtxs)
    snvtxs = [0, 0, 0]
    snedges = [0, 0, 0]
    for i in 0:(nvtxs - 1)
        k = where_[i + 1]
        rename[i + 1] = snvtxs[k + 1]
        snvtxs[k + 1] += 1
        snedges[k + 1] += xadj[i + 2] - xadj[i + 1]
    end
    lgraph = SetupSplitGraph(graph, snvtxs[1], snedges[1])
    rgraph = SetupSplitGraph(graph, snvtxs[2], snedges[2])
    # mark the boundary vertices of the two partitions through bndptr
    for ii in 0:(graph.nbnd - 1)
        i = bndind[ii + 1]
        for j in xadj[i + 1]:(xadj[i + 2] - 1)
            bndptr[adjncy[j + 1] + 1] = 1
        end
    end
    sn = [0, 0]
    se = [0, 0]
    lgraph.xadj[1] = 0
    rgraph.xadj[1] = 0
    sg = (lgraph, rgraph)
    for i in 0:(nvtxs - 1)
        mypart = where_[i + 1]
        mypart == 2 && continue
        g = sg[mypart + 1]
        istart = xadj[i + 1]
        iend = xadj[i + 2]
        if bndptr[i + 1] == -1                    # interior vertex
            l = se[mypart + 1]
            for j in istart:(iend - 1)
                l += 1
                g.adjncy[l] = adjncy[j + 1]
            end
            se[mypart + 1] = l
        else
            l = se[mypart + 1]
            for j in istart:(iend - 1)
                k = adjncy[j + 1]
                if where_[k + 1] == mypart
                    l += 1
                    g.adjncy[l] = k
                end
            end
            se[mypart + 1] = l
        end
        g.vwgt[sn[mypart + 1] + 1] = vwgt[i + 1]
        g.label[sn[mypart + 1] + 1] = label[i + 1]
        sn[mypart + 1] += 1
        g.xadj[sn[mypart + 1] + 1] = se[mypart + 1]
    end
    for mypart in 0:1
        g = sg[mypart + 1]
        for i in 1:se[mypart + 1]
            g.adjwgt[i] = 1
            g.adjncy[i] = rename[g.adjncy[i] + 1]
        end
        g.nvtxs = sn[mypart + 1]
        g.nedges = se[mypart + 1]
        SetupGraph_tvwgt!(g)
    end
    return lgraph, rgraph
end

# MMDOrder: multiple minimum degree on a small piece, through the exported
# libmetis__genmmd of the shipped binary (pure arrays; exact by construction)
const GENMMD_PTR = Ref{Ptr{Cvoid}}(C_NULL)   # set via init_libmetis!(path)

function init_libmetis!(path::AbstractString)
    h = Libdl.dlopen(path)
    GENMMD_PTR[] = Libdl.dlsym(h, :libmetis__genmmd)
    return nothing
end

function MMDOrder!(ctrl::CtrlT, graph::GraphT, order::Vector{Int}, lastvtx::Int)
    t0 = time()
    nvtxs = graph.nvtxs
    # 1-based CSR copies for the Fortran-style genmmd
    xadj1 = Vector{Int32}(undef, nvtxs + 1)
    for i in 1:(nvtxs + 1)
        xadj1[i] = Int32(graph.xadj[i] + 1)
    end
    adjncy1 = Vector{Int32}(undef, max(graph.nedges, 1))
    for i in 1:graph.nedges
        adjncy1[i] = Int32(graph.adjncy[i] + 1)
    end
    perm = Vector{Int32}(undef, nvtxs + 5)
    iperm = Vector{Int32}(undef, nvtxs + 5)
    head = Vector{Int32}(undef, nvtxs + 5)
    qsize = Vector{Int32}(undef, nvtxs + 5)
    list = Vector{Int32}(undef, nvtxs + 5)
    marker = Vector{Int32}(undef, nvtxs + 5)
    nofsub = Ref{Int32}(0)
    ccall(GENMMD_PTR[], Cvoid,
          (Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Int32,
           Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Int32, Ref{Int32}),
          Int32(nvtxs), xadj1, adjncy1, iperm, perm, Int32(1),
          head, qsize, list, marker, Int32(IDX_MAX), nofsub)
    label = graph.label
    firstvtx = lastvtx - nvtxs
    for i in 0:(nvtxs - 1)
        order[label[i + 1] + 1] = firstvtx + Int(iperm[i + 1]) - 1
    end
    prof!(:mmd, time() - t0)
    return nothing
end

function MlevelNestedDissection!(ctrl::CtrlT, graph::GraphT, order::Vector{Int}, lastvtx::Int)
    MlevelNodeBisectionMultiple!(ctrl, graph)
    DBG[] && println("Nvtxs: ", lpad(graph.nvtxs, 6), ", [", lpad(graph.pwgts[1], 6),
                     lpad(graph.pwgts[2], 7), lpad(graph.pwgts[3], 7), "]")
    nbnd = graph.nbnd
    bndind = graph.bndind
    label = graph.label
    for i in 0:(nbnd - 1)
        lastvtx -= 1
        order[label[bndind[i + 1] + 1] + 1] = lastvtx
    end
    t = @elapsed ((lgraph, rgraph) = SplitGraphOrder(ctrl, graph))
    prof!(:split, t)
    if lgraph.nvtxs > MMDSWITCH && lgraph.nedges > 0
        MlevelNestedDissection!(ctrl, lgraph, order, lastvtx - rgraph.nvtxs)
    else
        MMDOrder!(ctrl, lgraph, order, lastvtx - rgraph.nvtxs)
    end
    if rgraph.nvtxs > MMDSWITCH && rgraph.nedges > 0
        MlevelNestedDissection!(ctrl, rgraph, order, lastvtx)
    else
        MMDOrder!(ctrl, rgraph, order, lastvtx)
    end
    return nothing
end

"""
    NodeND(nvtxs, xadj, adjncy; vwgt = nothing, seed = -1) -> (perm, iperm)

libmetis METIS_NodeND with OMETIS defaults, 0-based CSR in, 0-based perm and
iperm out. ccorder and pfactor are not translated (defaults are off).
"""
function NodeND(nvtxs::Int, xadj::Vector{Int}, adjncy::Vector{Int};
                vwgt::Union{Nothing, Vector{Int}} = nothing, seed::Int = -1)
    ctrl = CtrlT(METIS_CTYPE_SHEM, METIS_RTYPE_SEP1SIDED, METIS_IPTYPE_EDGE,
                 1, 10, 200, true, false, 0.0, false, seed, 100, 0,
                 Float32(Float64(Float32(1.0 + 0.001 * 200)) + 0.0000499), Float32[])
    InitRandom(ctrl.seed)
    nnvtxs = 0
    graph = nothing
    cptr = Int[]
    cind = Int[]
    if ctrl.compress
        cptr = Vector{Int}(undef, nvtxs + 1)
        cind = Vector{Int}(undef, nvtxs)
        graph = CompressGraph(nvtxs, xadj, adjncy, vwgt, cptr, cind)
        if graph === nothing
            ctrl.compress = false
        else
            nnvtxs = graph.nvtxs
            cfactor = 1.0 * nvtxs / nnvtxs
            cfactor > 1.5 && ctrl.nseps == 1 && (ctrl.nseps = 2)
        end
    end
    graph === nothing && (graph = SetupGraph(nvtxs, xadj, adjncy, vwgt))
    iperm = Vector{Int}(undef, nvtxs)
    perm = Vector{Int}(undef, nvtxs)
    MlevelNestedDissection!(ctrl, graph, iperm, graph.nvtxs)
    if ctrl.compress                      # uncompress the ordering
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
    for i in 0:(nvtxs - 1)
        perm[iperm[i + 1] + 1] = i
    end
    return perm, iperm
end

"""
    nodend_exact(A::SparseMatrixCSC; seed = -1) -> (perm, iperm) 1-based

The `Metis.permutation(A)` replacement: symmetric pattern of `A`, diagonal
dropped, vwgt = NULL, all-default options.
"""
function nodend_exact(A; seed::Int = -1)
    n = size(A, 1)
    xadj = Vector{Int}(undef, n + 1)
    adjncy = Int[]
    rows = SparseArrays.rowvals(A)
    xadj[1] = 0
    for j in 1:n
        for p in SparseArrays.nzrange(A, j)
            i = rows[p]
            i != j && push!(adjncy, i - 1)
        end
        xadj[j + 1] = length(adjncy)
    end
    perm, iperm = NodeND(n, xadj, adjncy; seed)
    return perm .+ 1, iperm .+ 1
end

end # module

# Device construction of the symbolic maps (PLAN §2.3 step 6 hot spots).
# The host analysis stays the reference; a GPU solver routes three
# data-parallel pieces through KernelAbstractions kernels with element-exact
# results (the tests assert equality against the host originals):
#
#   * `device_symmetric_pattern` — the both-triangle adjacency of
#     `SymmetricPattern` by packed-key sort and deduplication;
#   * `device_assembly_map`     — `assembly_map` for the symmetric
#     structures, one thread per CSR row, binary search per entry;
#   * `device_group_amap`       — `_group_amap` (unsigned) as a packed-key
#     sort reproducing the host `(abs, neg, p)` order exactly.
#
# The packed 64-bit keys need `n`, `nnz` < 2^32 and factor offsets < 2^31;
# [`device_maps_supported`](@ref) checks the bounds and the callers fall back
# to the host route when it returns `false` (a larger-than-2^31-entry factor
# is a > 16 GiB panel buffer, beyond a single-GPU analysis anyway).

"""
    device_maps_supported(backend, structure, view, n, nnz, factor_len) -> Bool

Whether the device symbolic-map routes apply: a GPU backend, a symmetric
structure (`"SPD"`, `"HPD"`, `"S"`, `"H"`), a triangular stored view, and
sizes that fit the packed 64-bit sort keys.
"""
function device_maps_supported(backend, structure, view, n::Integer, nnz::Integer,
                               factor_len::Integer)
    backend isa KernelAbstractions.GPU || return false
    s = _structure(structure)
    (s == STRUCTURE_SPD || s == STRUCTURE_HPD || s == STRUCTURE_SYMMETRIC ||
     s == STRUCTURE_HERMITIAN) || return false
    _matrix_view(view) == VIEW_FULL && return false
    return n < 2^32 && nnz < 2^32 && factor_len < 2^31
end

# device vector of `x` as `I`
function _device_vec(backend, ::Type{I}, x) where {I <: Integer}
    y = KernelAbstractions.allocate(backend, I, length(x))
    isempty(x) || copyto!(y, I.(x))
    return y
end

# ---------------------------------------------------------------------------
# symmetric pattern

@kernel function _pattern_keys_kernel!(keys, @Const(rowptr), @Const(colval), base::Int32,
                                        upper::Bool)
    r = @index(Global)
    @inbounds for p in (rowptr[r] + (one(eltype(rowptr)) - base)):(rowptr[r + 1] - base)
        c = Int64(colval[p]) + (1 - Int64(base))
        q = Int64(p)
        # the host `_entry_filter`: diagonal dropped by the pattern, entries
        # outside the declared triangle dropped by the filter
        if c == r || (upper ? c < r : c > r)
            keys[2q - 1] = typemax(UInt64)
            keys[2q] = typemax(UInt64)
        else
            keys[2q - 1] = (UInt64(r) << 32) | UInt64(c)
            keys[2q] = (UInt64(c) << 32) | UInt64(r)
        end
    end
end

"""
    device_symmetric_pattern(backend, rowptr, colval, n, view, index) -> (colptr, rowval)

The `SymmetricPattern` arrays (1-based, sorted, deduplicated, no diagonal)
of the stored entries, built on `backend`. `rowptr`/`colval` are host or
device vectors with base `index`.
"""
function device_symmetric_pattern(backend, rowptr, colval, n::Integer, view, index)
    m = length(colval)
    base = _index_base(index) == INDEX_ONE ? Int32(1) : Int32(0)
    upper = _matrix_view(view) == VIEW_UPPER
    I = m + 1 <= typemax(Int32) && n + 1 <= typemax(Int32) ? Int32 : Int64
    drp = _device_vec(backend, I, Array(rowptr))
    dcv = _device_vec(backend, I, Array(colval))
    keys = KernelAbstractions.allocate(backend, UInt64, 2m)
    n > 0 && _pattern_keys_kernel!(backend)(keys, drp, dcv, base, upper; ndrange = Int(n))
    KernelAbstractions.synchronize(backend)
    isempty(keys) || sort!(keys)   # empty: device sorts reject length 0
    hk = Array(keys)
    nvalid = searchsortedfirst(hk, typemax(UInt64)) - 1
    colptr = zeros(Int, n + 1)
    rowval = Vector{Int}(undef, nvalid)
    colptr[1] = 1
    nv = 0
    prev = typemax(UInt64)
    @inbounds for i in 1:nvalid
        k = hk[i]
        k == prev && continue                          # duplicate of a 'F' view
        prev = k
        nv += 1
        rowval[nv] = Int(k & 0xffffffff)
        colptr[Int(k >> 32) + 1] += 1
    end
    resize!(rowval, nv)
    for j in 1:n
        colptr[j + 1] += colptr[j]
    end
    return colptr, rowval
end

# ---------------------------------------------------------------------------
# assembly map

@kernel function _amap_kernel!(amap, @Const(rowptr), @Const(colval), @Const(iperm),
                               @Const(col2sn), @Const(snrowptr), @Const(snrowval),
                               @Const(superptr), @Const(panelptr), base::Int32,
                               upper::Bool, neg::Bool)
    r = @index(Global)
    T = eltype(amap)
    @inbounds for p in (rowptr[r] + (one(eltype(rowptr)) - base)):(rowptr[r + 1] - base)
        c = Int(colval[p]) + (1 - Int(base))
        if upper ? r <= c : r >= c
            i = Int(iperm[r])
            j = Int(iperm[c])
            flip = i < j
            if flip
                i, j = j, i
            end
            s = Int(col2sn[j])
            lo = Int(snrowptr[s])
            hi = Int(snrowptr[s + 1]) - 1
            f = hi - lo + 1
            a = lo                                     # binary search for i
            b = hi
            while a < b
                mid = (a + b) >> 1
                if Int(snrowval[mid]) < i
                    a = mid + 1
                else
                    b = mid
                end
            end
            off = Int(panelptr[s]) + (j - Int(superptr[s])) * f + (a - lo + 1) - 1
            amap[p] = T(flip && neg ? -off : off)
        else
            amap[p] = zero(T)
        end
    end
end

"""
    device_assembly_map(backend, sp, layout, rowptr, colval, n, structure; view, index)
        -> amap::Vector{Int}

[`assembly_map`](@ref) on `backend` for the symmetric structures with a
triangular stored view; identical output. The caller checks
[`device_maps_supported`](@ref).
"""
function device_assembly_map(backend, sp::SupernodePartition, layout::Layout,
                             rowptr, colval, n::Integer, structure;
                             view = VIEW_FULL, index = INDEX_ONE)
    s = _structure(structure)
    v = _matrix_view(view)
    base = _index_base(index) == INDEX_ONE ? Int32(1) : Int32(0)
    I = Int32                                          # bounds checked by the caller
    drp = _device_vec(backend, I, Array(rowptr))
    dcv = _device_vec(backend, I, Array(colval))
    amap = KernelAbstractions.allocate(backend, I, length(colval))
    n > 0 && _amap_kernel!(backend)(amap, drp, dcv, _device_vec(backend, I, sp.iperm),
                                    _device_vec(backend, I, sp.col2sn),
                                    _device_vec(backend, I, sp.rowptr),
                                    _device_vec(backend, I, sp.rowval),
                                    _device_vec(backend, I, sp.super_ptr),
                                    _device_vec(backend, I, layout.panel_ptr),
                                    base, v == VIEW_UPPER,
                                    _is_hermitian(s); ndrange = Int(n))
    KernelAbstractions.synchronize(backend)
    return Vector{Int}(Array(amap))
end

# ---------------------------------------------------------------------------
# amap grouping

@kernel function _group_keys_kernel!(keys, @Const(amap))
    p = @index(Global)
    @inbounds begin
        a = Int(amap[p])
        if a == 0
            keys[p] = typemax(UInt64)                  # dropped entries sort last
        else
            off = UInt64(a < 0 ? -a : a)
            # the host order: (abs(amap[p]), amap[p] < 0, p)
            keys[p] = (off << 33) | (UInt64(a < 0) << 32) | UInt64(p)
        end
    end
end

@kernel function _group_count_kernel!(cnt, @Const(keys), @Const(panelptr), nvalid::Int64)
    gi = @index(Global)
    @inbounds if gi <= nvalid
        off = Int((keys[gi] >> 33) & 0x7fffffff)
        a = 1                                          # searchsortedlast(panel_ptr, off)
        b = length(panelptr)
        while a < b
            mid = (a + b + 1) >> 1
            if Int(panelptr[mid]) <= off
                a = mid
            else
                b = mid - 1
            end
        end
        KernelAbstractions.@atomic cnt[a + 1] += one(eltype(cnt))
    end
end

"""
    device_group_amap(backend, amap, layout, ns) -> (amap_ptr, order)

`_group_amap` (unsigned) on `backend`: identical order and per-owner
counts. The signed (`"G"`) grouping stays on the host.
"""
function device_group_amap(backend, amap::Vector{Int}, layout::Layout, ns::Int)
    m = length(amap)
    damap = _device_vec(backend, Int32, amap)
    keys = KernelAbstractions.allocate(backend, UInt64, m)
    m > 0 && _group_keys_kernel!(backend)(keys, damap; ndrange = m)
    KernelAbstractions.synchronize(backend)
    isempty(keys) || sort!(keys)   # empty: device sorts reject length 0
    hk = Array(keys)
    nvalid = searchsortedfirst(hk, typemax(UInt64)) - 1
    order = Vector{Int}(undef, nvalid)
    @inbounds for i in 1:nvalid
        order[i] = Int(hk[i] & 0xffffffff)
    end
    cnt = KernelAbstractions.allocate(backend, Int32, ns + 1)
    fill!(cnt, Int32(0))
    m > 0 && _group_count_kernel!(backend)(cnt, keys, _device_vec(backend, Int32, layout.panel_ptr),
                                           Int64(nvalid); ndrange = m)
    KernelAbstractions.synchronize(backend)
    hc = Array(cnt)
    amap_ptr = Vector{Int}(undef, ns + 1)
    amap_ptr[1] = 1
    for s in 1:ns
        amap_ptr[s + 1] = amap_ptr[s] + Int(hc[s + 1])
    end
    return amap_ptr, order
end

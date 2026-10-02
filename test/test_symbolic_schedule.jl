# T07: schedule (levels, regimes, bins, chunks), static layout, device maps.

# Serial multifrontal stack of a regime-A subtree processed in `nodes` order (bytes):
# children's blocks are on top of the stack when their parent is assembled; fronts
# and contribution blocks are packed lower triangles (T11).
function simulated_subtree_peak(sc::SDS.Schedule, sp::SDS.SupernodePartition, nodes)
    stack = Tuple{Int, Int}[]               # (node, contribution-block entries)
    peak = 0
    for v in nodes
        kids = count(t -> sp.snparent[t[1]] == v, stack)
        live = sum(last, stack; init = 0)
        peak = max(peak, live + SDS.packed_length(sc.rows[v]))
        all(t -> sp.snparent[t[1]] == v, stack[(end - kids + 1):end]) || return -1   # not a stack order
        resize!(stack, length(stack) - kids)
        push!(stack, (v, SDS.packed_length(sc.rows[v] - sc.width[v])))
    end
    return peak * sc.elsize
end

function check_schedule(S::SDS.Symbolic)
    sp, sc, L = S.partition, S.schedule, S.layout
    ns = SDS.nsupernodes(sp)
    # levels: strictly increasing along every edge, for the tree height and the schedule level
    @test all(s -> sp.snparent[s] == 0 || sc.level[sp.snparent[s]] > sc.level[s], 1:ns)
    @test all(s -> sp.snparent[s] == 0 || sc.regime[s] == SDS.REGIME_A ||
                   sc.slevel[sp.snparent[s]] > sc.slevel[s], 1:ns)
    @test all(s -> sp.snparent[s] == 0 || sc.regime[sp.snparent[s]] == SDS.REGIME_A ||
                   sc.step[sp.snparent[s]] > sc.step[s], 1:ns)
    # regime A is closed under descendants, and its subtrees are maximal
    @test all(s -> sp.snparent[s] == 0 || sc.regime[s] == SDS.REGIME_A || sc.regime[sp.snparent[s]] != SDS.REGIME_A,
              1:ns)
    # every supernode appears exactly once: in a subtree of an A group or in a B/C group
    seen = zeros(Int, ns)
    ok = true
    for g in sc.groups
        ids = sc.group_nodes[g.first:g.last]
        if g.regime == SDS.REGIME_A
            for t in ids
                ok &= sc.subtree_class[t] == g.class
                for v in sc.subtree_nodes[sc.subtree_ptr[t]:(sc.subtree_ptr[t + 1] - 1)]
                    seen[v] += 1
                end
            end
        else
            for s in ids
                seen[s] += 1
                ok &= sc.regime[s] == g.regime && sc.step[s] == g.step && sc.slevel[s] == g.level
                ok &= g.regime != SDS.REGIME_B || sc.bin[s] == g.class
            end
        end
    end
    @test ok
    @test all(==(1), seen)
    # bins and the regime-C rule
    ok = true
    for s in 1:ns
        w, f = sc.width[s], sc.rows[s]
        big = w > sc.regime_c_width || f > sc.regime_c_rows
        big && (ok &= sc.regime[s] == SDS.REGIME_C)
        if sc.regime[s] == SDS.REGIME_B
            wc = sc.wclasses[(sc.bin[s] - 1) ÷ length(sc.fclasses) + 1]
            fc = sc.fclasses[(sc.bin[s] - 1) % length(sc.fclasses) + 1]
            ok &= w <= wc && (wc == 8 || w > wc ÷ 2)
            ok &= f <= fc && (fc == 64 || f > fc ÷ 2)
        else
            ok &= sc.bin[s] == 0
        end
    end
    @test ok
    # subtrees: peak within the budget class (recomputed by simulating the stack), postorder
    ok = true
    for t in 1:SDS.nsubtrees(sc)
        nodes = sc.subtree_nodes[sc.subtree_ptr[t]:(sc.subtree_ptr[t + 1] - 1)]
        ok &= nodes[end] == sc.subtree_root[t] && all(v -> sc.subtree[v] == t, nodes)
        ok &= simulated_subtree_peak(sc, sp, nodes) == sc.subtree_peak[t]
        cap(c) = SDS.subtree_capacity(sc.budgets[c], sc.elsize) * sc.elsize
        ok &= sc.subtree_peak[t] <= cap(sc.subtree_class[t]) <= sc.budgets[sc.subtree_class[t]]
        ok &= sc.subtree_class[t] == 1 || sc.subtree_peak[t] > cap(sc.subtree_class[t] - 1)
        # local layout: every front inside the peak, contribution blocks below their parent's front
        ok &= L.local_len[t] * sc.elsize == sc.subtree_peak[t]
        for v in nodes
            ok &= 1 <= L.local_front[v] && L.local_front[v] + SDS.packed_length(sc.rows[v]) - 1 <= L.local_len[t]
            p = sp.snparent[v]
            if v == sc.subtree_root[t]
                ok &= L.local_cb[v] == 0
            else
                ok &= L.local_cb[v] + SDS.packed_length(sc.rows[v] - sc.width[v]) <= L.local_front[p]
                ok &= L.local_cb[v] <= L.local_front[v]
            end
        end
    end
    @test ok
    @test count(==(SDS.REGIME_A), sc.regime) == length(sc.subtree_nodes)
    # chunks: produced update-stack bytes within the budget unless a single front
    if sc.memory_budget >= 0
        ok = true
        for t in 1:sc.nsteps
            fronts = [s for s in 1:ns if sc.step[s] == t]
            bytes = sum(s -> SDS.packed_length(sc.rows[s] - sc.width[s]) * sc.elsize, fronts; init = 0)
            ok &= bytes <= sc.memory_budget || length(fronts) == 1
        end
        @test ok
    end
    # layout: panels contiguous f×w
    @test L.panel_ptr[1] == 1
    @test all(s -> L.panel_ptr[s + 1] - L.panel_ptr[s] == sc.rows[s] * sc.width[s], 1:ns)
    @test L.factor_len == L.panel_ptr[end] - 1
    @test L.d_len == 2 * sp.n
    # update stack: blocks live at the same step never overlap; the high-water mark is the maximum over steps
    tops = zeros(Int, sc.nsteps + 1)
    disjoint = true
    for t in 0:sc.nsteps
        live = [s for s in 1:ns if L.cb_len[s] > 0 && L.cb_first[s] <= t <= L.cb_last[s]]
        iv = sort!([(L.cb_ptr[s], L.cb_ptr[s] + L.cb_len[s] - 1) for s in live])
        disjoint &= all(k -> iv[k][2] < iv[k + 1][1], 1:(length(iv) - 1))
        tops[t + 1] = maximum(last, iv; init = 0)
    end
    @test disjoint
    @test tops == L.step_top
    @test L.stack_len == maximum(tops; init = 0)
    # blocks that must go through the stack do: B/C fronts and subtree roots with a parent and m > 0
    ok = true
    for s in 1:ns
        p = sp.snparent[s]
        m = sc.rows[s] - sc.width[s]
        needs = p != 0 && m > 0 && (sc.regime[s] != SDS.REGIME_A || sc.subtree_root[sc.subtree[s]] == s)
        ok &= (L.cb_len[s] > 0) == needs && (L.cb_ptr[s] > 0) == needs
        needs && (ok &= L.cb_len[s] == m * (m + 1) ÷ 2 && L.cb_first[s] == sc.step[s] && L.cb_last[s] == sc.step[p])
    end
    @test ok
    @test S.cb_ptr == L.cb_ptr
    # relind: parent_rows[relind] == child_rows[w+1:end]
    ok = true
    for c in 1:ns
        p = sp.snparent[c]
        ri = S.relind[S.relind_ptr[c]:(S.relind_ptr[c + 1] - 1)]
        ok &= p == 0 ? isempty(ri) : SDS.snrows(sp, p)[ri] == SDS.snrows(sp, c)[(SDS.snwidth(sp, c) + 1):end]
    end
    @test ok
    # children lists, in increasing order
    kids = [Int[] for _ in 1:ns]
    foreach(c -> sp.snparent[c] == 0 || push!(kids[sp.snparent[c]], c), 1:ns)
    @test all(s -> S.child_list[S.child_ptr[s]:(S.child_ptr[s + 1] - 1)] == kids[s], 1:ns)
    # memory estimates cover what the layout allocates
    for T in ELTYPES, INT in INTTYPES
        est = SDS.memory_estimates(S, T, INT)
        @test length(est) == 16 && eltype(est) == Int64
        actual = (L.factor_len + L.d_len + L.stack_len) * sizeof(T) + SDS.device_map_bytes(S, INT)
        @test est[2] >= actual
        @test est[1] + est[9] == est[2]
        @test est[7] == L.factor_len * sizeof(T) && est[9] == L.stack_len * sizeof(T)
        @test all(==(0), est[13:16])
    end
    return nothing
end

# Scatter nzval through amap into host panels and read the panels back as a lower-triangular matrix.
function reconstruct_lower(S::SDS.Symbolic, nzval::AbstractVector{T}) where {T}
    sp, L = S.partition, S.layout
    buf = zeros(T, L.factor_len)
    for p in eachindex(nzval)
        off = S.amap[p]
        off == 0 && continue
        buf[abs(off)] += off < 0 ? conj(nzval[p]) : nzval[p]
    end
    I = Int[]; J = Int[]; V = T[]
    for s in 1:SDS.nsupernodes(sp)
        rows = SDS.snrows(sp, s)
        f = length(rows)
        for (k, j) in enumerate(SDS.sncols(sp, s)), pos in k:f
            push!(I, rows[pos]); push!(J, j); push!(V, buf[L.panel_ptr[s] + (k - 1) * f + pos - 1])
        end
    end
    return dropzeros!(sparse(I, J, V, sp.n, sp.n))
end

# The same through the owner-pull grouping (amap_ptr / amap_src), front by front.
function reconstruct_pull(S::SDS.Symbolic, nzval::AbstractVector{T}) where {T}
    buf = zeros(T, S.layout.factor_len)
    owned = true
    for s in 1:SDS.nsupernodes(S), k in S.amap_ptr[s]:(S.amap_ptr[s + 1] - 1)
        p = S.amap_src[k]
        off = S.amap[p]
        owned &= S.layout.panel_ptr[s] <= abs(off) < S.layout.panel_ptr[s + 1]
        buf[abs(off)] += off < 0 ? conj(nzval[p]) : nzval[p]
    end
    @test owned
    return buf
end

triangle(A, view) = view == 'L' ? tril(A) : view == 'U' ? triu(A) : A

@testset "options: schedule tuning knobs" begin
    opts = Options(regime_c_width = 128, regime_c_rows = 1024, subtree_budgets = (48 * 1024, 16 * 1024),
                   memory_budget = 1 << 20)
    @test opts.regime_c_width == 128 && opts.regime_c_rows == 1024
    @test opts.subtree_budgets == [16 * 1024, 48 * 1024]
    @test opts.memory_budget == 1 << 20
    @test Options().subtree_budgets == [16 * 1024, 32 * 1024, 48 * 1024]
    @test Options().memory_budget == -1
    @test copy(opts).subtree_budgets == opts.subtree_budgets
    @test thrown(() -> Options(regime_c_width = 0)) isa InvalidValueError
    @test thrown(() -> Options(subtree_budgets = [0])) isa InvalidValueError
    @test thrown(() -> Options(memory_budget = 1.5)) isa InvalidValueError
    @test thrown(() -> setparam!(Options(), "memory_budget", 1)) isa ArgumentError
end

@testset "small example" begin
    # arrow matrix: four leaves 1..4 coupled to the 2×2 root block 5..6
    A = sparse(Float64[4 0 0 0 1 1; 0 4 0 0 1 1; 0 0 4 0 1 1; 0 0 0 4 1 1; 1 1 1 1 4 1; 1 1 1 1 1 4])
    opts = Options(reordering_alg = "algo5", use_superpanels = 0)
    S = SDS.symbolic_analysis(SDS.CSR(A), "SPD", 'L'; opts)
    sc = S.schedule
    @test SDS.nsupernodes(S) == 5
    @test sc.level == [1, 1, 1, 1, 2]
    @test all(==(SDS.REGIME_A), sc.regime)                       # everything fits 16 KiB
    @test SDS.nsubtrees(sc) == 1 && sc.subtree_root == [5]
    @test sc.subtree_peak == [(3 + 3 + 3 + 6) * 8]                # three packed 2×2 blocks waiting + packed 3×3 front
    @test SDS.nlaunches(sc) == 1
    @test S.layout.stack_len == 0
    check_schedule(S)
    # without regime A: one level of four B fronts (same bin), then the root
    S0 = SDS.symbolic_analysis(SDS.CSR(A), "SPD", 'L'; opts = Options(reordering_alg = "algo5", use_superpanels = 0,
                                                                      subtree_budgets = Int[]))
    sc0 = S0.schedule
    @test all(==(SDS.REGIME_B), sc0.regime)
    @test sc0.slevel == [1, 1, 1, 1, 2]
    @test SDS.nlaunches(sc0) == 2
    @test S0.relind == [1, 2, 1, 2, 1, 2, 1, 2]
    @test S0.layout.cb_len == [3, 3, 3, 3, 0]                    # packed 2×2 lower triangles
    @test S0.layout.stack_len == 12
    check_schedule(S0)
    # vendor everywhere: five C fronts (four with a contribution block)
    S2 = SDS.symbolic_analysis(SDS.CSR(A), "SPD", 'L'; opts = Options(reordering_alg = "algo5", use_superpanels = 0,
                                                                      subtree_budgets = Int[], factorization_alg = "algo2"))
    @test all(==(SDS.REGIME_C), S2.schedule.regime)
    @test SDS.nlaunches(S2.schedule) == (1 + 4 * 4) + (1 + 1)   # potrf, trsm, syrk, pack_add!
    check_schedule(S2)
    # "G" maps are not implemented yet
    @test thrown(() -> SDS.symbolic_analysis(SDS.CSR(A), "G", 'F')) isa NotSupportedError
end

@testset "reconstruction through amap: $T $structure '$view' index '$index'" for T in ELTYPES,
        structure in (T <: Real ? ("S", "SPD") : ("S", "H")), view in ('L', 'U', 'F'), index in ('O', 'Z')
    n = 60
    A = structure == "SPD" ? random_spd(T, n, 0.08) :
        random_symindef(T, n, 0.08; hermitian = structure == "H")
    @test A == (structure == "S" ? transpose(A) : A')
    for opts in (Options(), Options(use_superpanels = 0, subtree_budgets = Int[]),
                 Options(regime_c_width = 8, regime_c_rows = 16, memory_budget = 64))
        C = SDS.CSR(triangle(A, view); index)
        S = SDS.symbolic_analysis(C, structure, view; opts)
        perm = S.partition.perm
        ref = dropzeros!(tril(A[perm, perm]))
        nzval = Array(C.nzval)
        @test reconstruct_lower(S, nzval) == ref
        @test reconstruct_pull(S, nzval) == let buf = zeros(T, S.layout.factor_len)
            for p in eachindex(nzval)
                S.amap[p] == 0 || (buf[abs(S.amap[p])] += S.amap[p] < 0 ? conj(nzval[p]) : nzval[p])
            end
            buf
        end
        @test count(!=(0), S.amap) == (view == 'F' ? nnz(tril(A)) : length(nzval))
        structure == "S" && @test all(>=(0), S.amap)
        check_schedule(S)
    end
end

@testset "duplicated entries are summed" begin
    # lower triangle of a 3×3 matrix with entry (3, 1) stored twice (CSR, row by row)
    rowptr = [1, 2, 4, 7]
    colval = [1, 1, 2, 1, 1, 3]
    nzval = [4.0, 1.0, 4.0, 0.5, 0.25, 4.0]
    S = SDS.Symbolic(let P = SDS.SymmetricPattern(rowptr, colval, 3, "SPD"; view = 'L')
                         sp = SDS.supernode_partition(P, 1:3, Options(use_superpanels = 0))
                         sc = SDS.build_schedule(sp)
                         (sp, sc, SDS.build_layout(sp, sc))
                     end..., rowptr, colval, 3, "SPD"; view = 'L')
    @test S.amap[4] == S.amap[5] != 0
    L = reconstruct_lower(S, nzval)
    perm = S.partition.perm
    A = sparse([1, 2, 2, 3, 3], [1, 1, 2, 1, 3], [4.0, 1.0, 4.0, 0.75, 4.0], 3, 3)
    A = A + tril(A, -1)'
    @test L == tril(A[perm, perm])
end

@testset "model problems: $name" for (name, A) in (("laplacian2d(100, 100)", laplacian2d(100, 100)),
                                                    ("laplacian3d(12, 12, 12)", laplacian3d(12, 12, 12)),
                                                    ("kkt_matrix(600, 200)", kkt_matrix(600, 200, 1.0e-8)))
    for alg in ("algo3", "default"), T in (Float32, ComplexF64)
        S = SDS.symbolic_analysis(SDS.CSR(SparseMatrixCSC{T}(A)), "S", 'L'; opts = Options(reordering_alg = alg))
        check_schedule(S)
        @test reconstruct_lower(S, Array(SDS.CSR(SparseMatrixCSC{T}(A)).nzval)) ==
              dropzeros!(tril(SparseMatrixCSC{T}(A)[S.partition.perm, S.partition.perm]))
    end
    # level chunking under a memory budget: more steps, same fronts, each chunk within the budget
    S = SDS.symbolic_analysis(SDS.CSR(A), "S", 'L'; opts = Options(reordering_alg = "algo3", subtree_budgets = Int[]))
    Sc = SDS.symbolic_analysis(SDS.CSR(A), "S", 'L';
                               opts = Options(reordering_alg = "algo3", subtree_budgets = Int[], memory_budget = 4096))
    @test Sc.schedule.nslevels == S.schedule.nslevels
    @test Sc.schedule.nsteps > S.schedule.nsteps
    @test SDS.nlaunches(Sc.schedule) >= SDS.nlaunches(S.schedule)
    check_schedule(Sc)
end

@testset "nlaunches on laplacian2d(100, 100) with AMD" begin
    A = laplacian2d(100, 100)
    S = SDS.symbolic_analysis(SDS.CSR(A), "SPD", 'L'; opts = Options(reordering_alg = "algo3"))
    S0 = SDS.symbolic_analysis(SDS.CSR(A), "SPD", 'L'; opts = Options(reordering_alg = "algo3", subtree_budgets = Int[]))
    nA, n0 = SDS.nlaunches(S.schedule), SDS.nlaunches(S0.schedule)
    println("  nlaunches laplacian2d(100, 100) AMD: $nA with regime A ($(SDS.nsubtrees(S.schedule)) subtrees), ",
            "$n0 without; ", S.schedule)
    @test nA < n0
    # the top separator (about 100 columns) is one regime-C front since wide
    # fundamental supernodes stay whole (issue #48): its vendor calls add launches
    sc0 = S0.schedule
    extra = sum(SDS._c_front_launches(sc0.rows[s], sc0.width[s]) for s in 1:length(sc0.regime) if sc0.regime[s] == SDS.REGIME_C; init = 0)
    @test n0 == length(sc0.groups) + extra
end

@testset "adapt: $(backend_name(backend)) $INT" for backend in BACKENDS, INT in INTTYPES
    A = laplacian2d(30, 30)
    S = SDS.symbolic_analysis(SDS.CSR(A), "SPD", 'U'; opts = Options(reordering_alg = "algo3"))
    Sd = SDS.adapt(backend, S, INT)
    @test Sd isa SDS.Symbolic{INT}
    for f in SDS.DEVICE_MAPS
        x = getfield(Sd, f)
        @test eltype(x) == INT
        @test KernelAbstractions.get_backend(x) == backend
        @test Array(x) == getfield(S, f)
    end
    if CUDA_LOADED && backend isa CUDABackend
        @test Sd.amap isa CuVector{INT}
    end
    @test Sd.partition === S.partition && Sd.schedule === S.schedule && Sd.layout === S.layout
    # round trip to the host
    Sh = SDS.adapt(CPU(), Sd, Int)
    @test all(f -> getfield(Sh, f) == getfield(S, f), SDS.DEVICE_MAPS)
    @test SDS.memory_estimates(Sd, Float64)[10] == SDS.device_map_bytes(S, INT)
    # offsets that do not fit the index type
    @test thrown(() -> SDS.adapt(backend, S, Int8)) isa InvalidValueError
end

@testset "update stack vs factor (issue #48)" begin
    # Regression guard for the update-stack high-water mark relative to the
    # factor. Keeping wide fundamental supernodes whole (no chain of max_width
    # panels) brought the KKT ratio from 7.8 to 5.6 (T07 Report); placing the
    # contribution blocks offline over their known lifetimes (best of first fit,
    # largest first, largest size × lifetime first) removed the fragmentation of
    # the step-by-step first fit: 6.2 -> 4.5 (KKT) and 6.1 -> 4.5 (random SPD);
    # packed lower-triangular contribution blocks (T11, which closes issue #48)
    # halve what remains: measured 3.0 (KKT), 2.9 (random SPD) and 0.24 (2-D
    # Laplacian, also fewer blocks on the stack since the larger regime-A
    # subtrees keep theirs in local memory) with the first fit alone. Bounds sit
    # above the measured values so a regression shows up; print the values for
    # the Report. The random generators depend on the RNG state left by the
    # preceding testsets, which differs with the backend list, so reseed here.
    # The sprand stream for a given seed also differs between Julia versions
    # (on Julia 1.10 the full-block ratios were 5.37 and 5.05 and stack_len
    # 1.09 and 1.19 times the live bound), so the bounds of the random matrices
    # carry that margin. The Laplacian is RNG-free and keeps the tight bounds.
    Random.seed!(666)
    for (name, A, bound, slack) in
        (("kkt_matrix(3000, 1000)", kkt_matrix(3000, 1000, 1.0e-8), 3.5, 1.25),
         ("random_spd(2000, 0.002)", random_spd(2000, 0.002), 3.5, 1.25),
         ("laplacian2d(100, 100)", laplacian2d(100, 100), 0.5, 1.05))
        S = SDS.symbolic_analysis(SDS.CSR(A), "S", 'L'; opts = Options(reordering_alg = "algo3"))
        L = S.layout
        ratio = L.stack_len / L.factor_len
        # entries live in the fullest step: no placement can go below it
        live = zeros(Int, S.schedule.nsteps + 1)
        for s in eachindex(L.cb_len), t in L.cb_first[s]:L.cb_last[s]
            L.cb_len[s] > 0 && (live[t + 1] += L.cb_len[s])
        end
        println("  update stack / factor on $name (AMD): $(round(ratio; digits = 2)), ",
                "live bound $(round(maximum(live) / L.factor_len; digits = 2))")
        @test ratio <= bound
        println("  stack_len / live bound on $name: ",
                "$(round(L.stack_len / maximum(live); digits = 3))")
        @test maximum(live) <= L.stack_len <= slack * maximum(live)
        check_schedule(S)
    end
end

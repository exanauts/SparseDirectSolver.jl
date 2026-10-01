# T06: fundamental supernodes, GPU-tuned amalgamation, supernodal symbolic factorization (host only).

const NO_AMALGAMATION = Options(use_superpanels = 0)

# Structural checks every partition must pass; returns nothing, records @test results.
function check_partition(sp::SDS.SupernodePartition, P::SDS.SymmetricPattern)
    n = sp.n
    ns = SDS.nsupernodes(sp)
    @test SDS.nsuperpanels(sp) == ns
    # ranges contiguous, nonempty and covering 1:n
    @test sp.super_ptr[1] == 1
    @test sp.super_ptr[end] == n + 1
    @test all(s -> sp.super_ptr[s] < sp.super_ptr[s + 1], 1:ns)
    @test all(j -> j in SDS.sncols(sp, sp.col2sn[j]), 1:n)
    # the final permutation, and the column etree relabelled consistently with it
    @test isperm(sp.perm)
    @test sp.iperm == invperm(sp.perm)
    @test sp.parent == SDS.etree(P, sp.perm)
    @test sp.counts == SDS.colcounts(P, sp.perm, sp.parent, SDS.postorder(sp.parent))
    @test sp.nnz_L == SDS.nnz_L(sp.counts)
    # snparent: a forest consistent with the column etree
    @test all(s -> sp.snparent[s] == 0 || sp.snparent[s] > s, 1:ns)
    ok_edges = true
    for s in 1:ns, j in SDS.sncols(sp, s)
        p = sp.parent[j]
        if j == last(SDS.sncols(sp, s))
            ok_edges &= (p == 0 ? sp.snparent[s] == 0 : sp.col2sn[p] == sp.snparent[s])
        else
            ok_edges &= p != 0 && sp.col2sn[p] == s     # the etree stays inside the supernode
        end
    end
    @test ok_edges
    # snpost: a postorder of the supernodal tree
    @test isperm(sp.snpost)
    pos = invperm(sp.snpost)
    @test all(s -> sp.snparent[s] == 0 || pos[s] < pos[sp.snparent[s]], 1:ns)
    # rows: sorted, own columns first, below-rows of a child inside its parent's rows
    ok_rows = true
    nnz_stored = 0
    flops = 0.0
    for s in 1:ns
        r = SDS.snrows(sp, s)
        w = SDS.snwidth(sp, s)
        f = length(r)
        ok_rows &= issorted(r) && allunique(r) && r[1:w] == SDS.sncols(sp, s) && all(<=(n), r)
        p = sp.snparent[s]
        p != 0 && (ok_rows &= issubset(r[(w + 1):end], SDS.snrows(sp, p)))
        p == 0 && (ok_rows &= f == w)
        nnz_stored += w * f - w * (w - 1) ÷ 2
        flops += sum(k -> Float64(f - k)^2, 0:(w - 1))
    end
    @test ok_rows
    @test sp.nnz_stored == nnz_stored
    @test sp.flops ≈ flops
    @test sp.nnz_stored >= sp.nnz_L
    return nothing
end

# Every structural nonzero L[i, j] (i ≥ j) of the filled pattern lies in the stored panel of j.
function covers_fill(sp::SDS.SupernodePartition, F::AbstractMatrix{Bool})
    for j in 1:sp.n
        r = SDS.snrows(sp, sp.col2sn[j])
        for i in j:sp.n
            F[i, j] && !(i in r) && return false
        end
    end
    return true
end

@testset "fundamental supernodes: small example" begin
    # arrow matrix: columns 1..4 independent leaves coupled only to 5..6 (dense block)
    A = sparse(Float64[4 0 0 0 1 1; 0 4 0 0 1 1; 0 0 4 0 1 1; 0 0 0 4 1 1; 1 1 1 1 4 1; 1 1 1 1 1 4])
    P = SDS.SymmetricPattern(SDS.CSR(A), "SPD"; view = 'F')
    perm = collect(1:6)
    parent = SDS.etree(P, perm)
    post = SDS.postorder(parent)
    counts = SDS.colcounts(P, perm, parent, post)
    @test parent == [5, 5, 5, 5, 6, 0]
    cp = SDS.fundamental_supernodes(parent, post, counts)
    @test cp.order == post
    @test cp.super_ptr == [1, 2, 3, 4, 5, 7]
    @test cp.snparent == [5, 5, 5, 5, 0]
    @test SDS.nsupernodes(cp) == 5
    sp = SDS.supernode_partition(P, perm, NO_AMALGAMATION)
    @test sp.nnz_stored == sp.nnz_L == 3 * 4 + 3
    @test !sp.amalgamated
    @test SDS.snrows(sp, 1) == [1, 5, 6]
    @test SDS.snrows(sp, 5) == [5, 6]
    check_partition(sp, P)
    # with amalgamation the four leaves (3 rows each) merge into the root: 6×6 dense, 6 explicit zeros
    spa = SDS.supernode_partition(P, perm, Options(amalgamation = (max_width = 8, zero_fraction = 0.5, min_width = 8)))
    @test spa.amalgamated
    @test SDS.nsupernodes(spa) == 1
    @test spa.nnz_stored == 21
    check_partition(spa, P)
    # max_width = 4 caps the panel width; zero_fraction = 0 forbids explicit zeros
    @test all(s -> SDS.snwidth(SDS.supernode_partition(P, perm, Options(amalgamation = (max_width = 4, min_width = 2))), s) <= 4,
              1:3)
    sp0 = SDS.supernode_partition(P, perm, Options(amalgamation = (zero_fraction = 0.0,)))
    @test sp0.nnz_stored == sp0.nnz_L
    # amalgamate needs exact supernodes
    padded = SDS.ColumnPartition(collect(1:6), [1, 7], [0])
    @test thrown(() -> SDS.amalgamate(padded, parent, counts, Options().amalgamation)) isa InvalidValueError
    @test thrown(() -> SDS.fundamental_supernodes(parent, [6, 1, 2, 3, 4, 5], counts)) isa InvalidValueError
    # empty pattern
    P0 = SDS.SymmetricPattern(0, [1], Int[])
    for opts in (Options(), NO_AMALGAMATION)
        sp = SDS.supernode_partition(P0, Int[], opts)
        @test SDS.nsupernodes(sp) == 0
        @test sp.nnz_stored == sp.nnz_L == 0
    end
end

@testset "wide supernodes are split to max_width" begin
    A = sparse(ones(100, 100) + 100I)
    P = SDS.SymmetricPattern(SDS.CSR(A), "SPD"; view = 'F')
    sp = SDS.supernode_partition(P, 1:100, NO_AMALGAMATION)
    @test SDS.nsupernodes(sp) == 1
    spa = SDS.supernode_partition(P, 1:100, Options())
    @test SDS.nsupernodes(spa) == 4
    @test all(s -> SDS.snwidth(spa, s) == 25, 1:4)
    @test spa.nnz_stored == spa.nnz_L == sp.nnz_L
    @test spa.flops == sp.flops
    check_partition(spa, P)
end

@testset "structure vs brute force" begin
    params = ((max_width = 32, zero_fraction = 0.25, min_width = 8),
              (max_width = 4, zero_fraction = 0.5, min_width = 2),
              (max_width = 8, zero_fraction = 1.0, min_width = 8),
              (max_width = 3, zero_fraction = 0.0, min_width = 1))
    for trial in 1:200
        n = rand(5:60)
        density = rand((0.02, 0.05, 0.1, 0.2, 0.4))
        A = random_symindef(n, density)
        P = SDS.SymmetricPattern(SDS.CSR(A), "S"; view = 'F')
        perm = isodd(trial) ? randperm(n) : collect(1:n)
        parent = SDS.etree(P, perm)
        counts = SDS.colcounts(P, perm, parent, SDS.postorder(parent))
        # without amalgamation: exact structure
        sp = SDS.supernode_partition(P, perm, NO_AMALGAMATION)
        _, ref_counts, F = brute_force_symbolic(A, sp.perm)
        @test sp.counts == ref_counts
        @test covers_fill(sp, F)
        @test sp.nnz_stored == sp.nnz_L == SDS.nnz_L(counts)
        @test all(s -> length(SDS.snrows(sp, s)) == sp.counts[first(SDS.sncols(sp, s))], 1:SDS.nsupernodes(sp))
        @test sp.flops == SDS.cholesky_flops(counts)
        check_partition(sp, P)
        # with amalgamation: fill covered, bounds respected
        prm = params[mod1(trial, length(params))]
        spa = SDS.supernode_partition(P, perm, Options(amalgamation = prm))
        _, _, Fa = brute_force_symbolic(A, spa.perm)
        @test covers_fill(spa, Fa)
        @test spa.nnz_L == sp.nnz_L
        @test spa.nnz_stored <= (1 + prm.zero_fraction) * spa.nnz_L
        @test all(s -> SDS.snwidth(spa, s) <= prm.max_width, 1:SDS.nsupernodes(spa))
        @test spa.flops >= SDS.cholesky_flops(counts)
        @test SDS.nsupernodes(spa) <= SDS.nsupernodes(sp) || prm.max_width < maximum(diff(sp.super_ptr))
        check_partition(spa, P)
    end
end

@testset "amalgamation on model problems" begin
    opts = Options()
    zf = opts.amalgamation.zero_fraction
    for (name, A) in (("laplacian2d(100, 100)", laplacian2d(100, 100)),
                      ("laplacian3d(12, 12, 12)", laplacian3d(12, 12, 12)),
                      ("kkt_matrix(600, 200)", kkt_matrix(600, 200, 1.0e-8)),
                      ("random_spd(2000, 0.002)", random_spd(2000, 0.002)))
        P = SDS.SymmetricPattern(SDS.CSR(A), "S"; view = 'F')
        for alg in (:natural, :amd, :nd)
            ord = SDS.compute_ordering(P, opts; alg)
            @testset "$name $alg" begin
                sp = SDS.supernode_partition(P, ord.perm, NO_AMALGAMATION)
                spa = SDS.supernode_partition(P, ord.perm, opts)
                check_partition(sp, P)
                check_partition(spa, P)
                @test sp.nnz_stored == sp.nnz_L == ord.stats.nnz_L
                @test sp.flops == ord.stats.flops
                @test spa.nnz_L == sp.nnz_L
                @test spa.nnz_stored <= (1 + zf) * spa.nnz_L
                @test all(s -> SDS.snwidth(spa, s) <= opts.amalgamation.max_width, 1:SDS.nsupernodes(spa))
                @test spa.flops >= sp.flops
                @test SDS.nsupernodes(spa) < SDS.nsupernodes(sp)
                # nnz(L) of the composed permutation agrees with CHOLMOD
                if name == "laplacian2d(100, 100)" || name == "random_spd(2000, 0.002)"
                    L = dropzeros!(sparse(cholesky(A; perm = spa.perm).L))
                    @test spa.nnz_L == nnz(L)
                end
            end
        end
    end
    # the T06 target: at least 2× fewer supernodes on the 2D Laplacian with AMD
    A = laplacian2d(100, 100)
    P = SDS.SymmetricPattern(SDS.CSR(A), "SPD"; view = 'F')
    perm = SDS.compute_ordering(P, opts; alg = :amd).perm
    ns0 = SDS.nsupernodes(SDS.supernode_partition(P, perm, NO_AMALGAMATION))
    ns1 = SDS.nsupernodes(SDS.supernode_partition(P, perm, opts))
    @test 2 * ns1 <= ns0
end

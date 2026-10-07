# T08: CPU reference multifrontal Cholesky (the oracle), host only.

triangle_of(A, view) = view == 'L' ? tril(A) : view == 'U' ? triu(A) : A

# analysis + numeric storage + factorization of A given through `view`
function reference_cholesky(A::SparseMatrixCSC{T}; view = 'L', index = 'O', opts = Options()) where {T}
    C = SDS.CSR(triangle_of(A, view); index)
    S = SDS.symbolic_analysis(C, spd_structure(T), view; opts)
    N = SDS.allocate_numeric(S, T, CPU())
    info = SDS.ref_factorize!(N, S, C.nzval)
    return S, N, info, C
end

# ‖P A Pᵀ − L Lᴴ‖_F / ‖A‖_F
function factor_error(A, S, N)
    p = S.partition.perm
    L = SDS.extract_L(S, N)
    return norm(A[p, p] - L * L') / norm(A)
end

factor_tol(::Type{T}) where {T} = real(T) == Float64 ? 1.0e-12 : 1.0e-5

reference_matrices(::Type{T}) where {T} =
    T <: Complex ? (("laplacian2d(40,40)", laplacian2d(T, 40, 40)), ("random_spd(500,0.01)", random_spd(T, 500, 0.01)),
                    ("random_hpd(400,0.02)", random_hpd(T, 400, 0.02))) :
                   (("laplacian2d(40,40)", laplacian2d(T, 40, 40)), ("random_spd(500,0.01)", random_spd(T, 500, 0.01)))

@testset "allocation from the layout" begin
    A = laplacian2d(Float64, 10, 10)
    S, _, _, _ = reference_cholesky(A)
    for T in ELTYPES
        N = SDS.allocate_numeric(S, T, CPU())
        @test N isa SDS.Numeric{T, Vector{T}, Vector{Int64}}
        @test eltype(N) == T
        @test length(N.factor) == S.layout.factor_len && length(N.d) == S.layout.d_len &&
              length(N.stack) == S.layout.stack_len
        @test length(N.stats) == SDS.FRONT_STATS_FIELDS * SDS.nsupernodes(S)
        @test all(iszero, N.factor) && all(iszero, N.stats)
    end
    # Cholesky needs a positive-definite structure, and "HPD" for complex values
    C = SDS.CSR(tril(A))
    S2 = SDS.symbolic_analysis(C, "S", 'L')
    @test thrown(() -> SDS.ref_factorize!(SDS.allocate_numeric(S2, Float64), S2, C.nzval)) isa InvalidValueError
    @test thrown(() -> SDS.ref_factorize!(SDS.allocate_numeric(S, ComplexF64), S, ComplexF64.(C.nzval))) isa
          InvalidValueError
    @test thrown(() -> SDS.ref_factorize!(SDS.allocate_numeric(S, Float64), S, C.nzval[1:(end - 1)])) isa
          InvalidValueError
end

@testset "P A Pᵀ = L Lᴴ and solves: $T" for T in ELTYPES
    Random.seed!(666)
    for (name, A) in reference_matrices(T)
        @testset "$name" begin
            S, N, info, _ = reference_cholesky(A)
            @test info == 0
            @test factor_error(A, S, N) <= factor_tol(T)
            L = SDS.extract_L(S, N)
            @test istril(L) && nnz(L) == S.partition.nnz_stored
            @test all(s -> N.stats[(s - 1) * SDS.FRONT_STATS_FIELDS + 1] == SDS.snwidth(S.partition, s),
                      1:SDS.nsupernodes(S))
            for nrhs in (1, 5)
                b = nrhs == 1 ? rand(T, size(A, 1)) : rand(T, size(A, 1), nrhs)
                x = similar(b)
                @test SDS.ref_solve!(x, S, N, b) === x
                @test relres(A, x, b) <= tol(T)
                if real(T) == Float64
                    xc = cholesky(Hermitian(A, :L)) \ b
                    @test norm(x - xc) <= 1.0e-8 * norm(xc)
                end
                # in place
                y = copy(b)
                SDS.ref_solve!(y, S, N, y)
                @test y == x
            end
        end
    end
end

@testset "views, index bases and refactorization: $T" for T in ELTYPES
    Random.seed!(666)
    A = random_spd(T, 300, 0.02)
    S, N, info, _ = reference_cholesky(A; view = 'L')
    @test info == 0
    L0 = SDS.extract_L(S, N)
    for view in ('U', 'F'), index in ('O', 'Z')
        Sv, Nv, infov, _ = reference_cholesky(A; view, index)
        @test infov == 0
        @test Sv.partition.perm == S.partition.perm
        @test SDS.extract_L(Sv, Nv) == L0
    end
    # new values on the same pattern and the same Symbolic
    B = copy(A)
    nonzeros(B) .= rand(T, nnz(B))
    B = (B + B') / 2 + 2 * size(B, 1) * I
    @test SparseMatrixCSC(B .!= 0) == SparseMatrixCSC(A .!= 0)
    for view in ('L', 'U', 'F')
        Sv, Nv, _, _ = reference_cholesky(A; view)
        Cb = SDS.CSR(triangle_of(B, view))
        @test SDS.ref_factorize!(Nv, Sv, Cb.nzval) == 0
        @test factor_error(B, Sv, Nv) <= factor_tol(T)
        b = rand(T, size(B, 1), 2)
        @test relres(B, SDS.ref_solve!(similar(b), Sv, Nv, b), b) <= tol(T)
        @test SDS.ref_factorize!(Nv, Sv, SDS.CSR(triangle_of(A, view)).nzval) == 0     # and back
        @test SDS.extract_L(Sv, Nv) == L0
    end
end

@testset "info: first non-positive pivot: $T" for T in ELTYPES
    Random.seed!(666)
    n, j = 60, 23
    for opts in (Options(), Options(reordering_alg = "algo5"), Options(use_superpanels = 0))
        A = singular_block_matrix(T, n, j; stored_zero = true)
        A[j, j] = 3
        S, N, info, _ = reference_cholesky(A; opts)
        @test info == 0
        @test factor_error(A, S, N) <= factor_tol(T)
        A[j, j] = -3                                    # not positive definite at column j
        S, N, info, _ = reference_cholesky(A; opts)
        @test info == j
        s = S.partition.col2sn[S.partition.iperm[j]]
        @test N.stats[s * SDS.FRONT_STATS_FIELDS] == S.partition.iperm[j] - S.partition.super_ptr[s] + 1
        A[j, j] = 0                                     # zero pivot
        @test reference_cholesky(A; opts)[3] == j
    end
    # a non-positive pivot that only appears after elimination: [1 2; 2 1] block at columns 3, 4
    A = sparse(T[4 0 0 0; 0 4 0 0; 0 0 1 2; 0 0 2 1])
    _, _, info, _ = reference_cholesky(A; opts = Options(reordering_alg = "algo5"))
    @test info == 4
end

@testset "amalgamation on and off: $T" for T in ELTYPES
    Random.seed!(666)
    for A in (laplacian2d(T, 30, 30), random_spd(T, 400, 0.01), laplacian3d(T, 8, 8, 8))
        b = rand(T, size(A, 1), 3)
        S1, N1, i1, _ = reference_cholesky(A)
        S0, N0, i0, _ = reference_cholesky(A; opts = Options(use_superpanels = 0))
        @test i1 == i0 == 0
        @test S1.partition.amalgamated && !S0.partition.amalgamated
        x1 = SDS.ref_solve!(similar(b), S1, N1, b)
        x0 = SDS.ref_solve!(similar(b), S0, N0, b)
        @test norm(x1 - x0) <= tol(T) * norm(x0)
        @test relres(A, x1, b) <= tol(T) && relres(A, x0, b) <= tol(T)
        # every regime mix of the schedule gives the same factor (the reference ignores regimes)
        S2, N2, _, _ = reference_cholesky(A; opts = Options(subtree_budgets = Int[], regime_c_width = 8))
        @test S2.partition.perm == S1.partition.perm
        @test SDS.extract_L(S2, N2) == SDS.extract_L(S1, N1)
    end
end

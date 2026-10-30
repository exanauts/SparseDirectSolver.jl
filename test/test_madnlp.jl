# The MadNLP linear-solver extension (ext/SparseDirectSolverMadNLPExt.jl, issue #162): `SDSSolver` built from the
# lower triangle of a KKT matrix as MadNLP hands it over, driven through MadNLP's linear-solver interface
# (`factorize!`, `solve!`, `inertia`, `is_inertia`, `introduce`, `improve!`, `last_info`).
#
# Real element types only: MadNLP's interior-point method is real, and the extension declares `is_supported` for
# Float32 and Float64 alone. The default `sds_pivoting = :none` (`:cholesky` and `:ldl`) runs the fused pipeline,
# which the KernelAbstractions CPU backend cannot run (see test_fused.jl), so its testsets run on the GPU backends
# with a vendor CSC type (`vendor_sparse_types`: CUDA, ROCm, oneAPI; Metal has none); `:bk` (a DirectSolver) runs
# on every backend, from a SparseMatrixCSC on the CPU backend as MadNLP's CPU KKT systems pass it (#175).

using MadNLP

const SDSSolver = SDS.SDSSolver
const SDSSolverOptions = SDS.SDSSolverOptions

# the backends of the fused pipeline (`sds_pivoting = :none`): GPU backends with a vendor CSC type
const MADNLP_FUSED_BACKENDS = filter(b -> vendor_sparse_types(b) !== nothing, BACKENDS)
# the backends of the `:bk` path: every one of them, the CPU backend from a SparseMatrixCSC
const MADNLP_BK_BACKENDS = filter(b -> b isa CPU || vendor_sparse_types(b) !== nothing, BACKENDS)

# the lower triangle of A as MadNLP passes it: a vendor CSC on a GPU `backend`, a SparseMatrixCSC on the CPU
function madnlp_csc(backend, A::SparseMatrixCSC{T}) where {T}
    _, SpCSC, DVec = vendor_sparse_types(backend)
    L = SparseMatrixCSC{T, Int32}(tril(A))
    return SpCSC{T, Int32}(DVec{Int32}(L.colptr), DVec{Int32}(L.rowval), DVec{T}(L.nzval), size(L))
end
madnlp_csc(::CPU, A::SparseMatrixCSC{T}) where {T} = SparseMatrixCSC{T, Int32}(tril(A))
csc_nzval(csc::SparseMatrixCSC) = csc.nzval
csc_nzval(csc) = csc.nzVal

# overwrite the values of the solver's matrix with those of A2 (same pattern), as MadNLP does between iterations
function set_values!(M, A2::SparseMatrixCSC{T}) where {T}
    copyto!(csc_nzval(M.tril), SparseMatrixCSC{T, Int32}(tril(A2)).nzval)
    return M
end

# MadNLP.solve! on a random right-hand side; the relative residual against the full matrix A
function madnlp_relres(backend, M, A::SparseMatrixCSC{T}) where {T}
    b = rand(T, size(A, 1))
    xb = to_device(backend, b)
    MadNLP.solve!(M, xb)
    return relres(A, to_host(xb), b)
end

@testset "MadNLP interface" begin
    @test MadNLP.input_type(SDSSolver) === :csc
    @test MadNLP.default_options(SDSSolver) isa SDSSolverOptions
    @test MadNLP.is_supported(SDSSolver, Float32)
    @test MadNLP.is_supported(SDSSolver, Float64)
    o = SDSSolverOptions()
    @test o.sds_factorization === :cholesky
    @test o.sds_pivoting === :none
    @test o.sds_reordering == "algo4"
end

@testset "option validation on $(backend_name(backend))" for backend in MADNLP_BK_BACKENDS
    A = laplacian2d(Float64, 6, 6)
    @test_throws ArgumentError SDSSolver(madnlp_csc(backend, A); opt = SDSSolverOptions(; sds_pivoting = :diag))
    # :bk needs the LDLᵀ factorization
    @test_throws ArgumentError SDSSolver(madnlp_csc(backend, A); opt = SDSSolverOptions(; sds_pivoting = :bk))
    if backend isa CPU
        # the fused pipeline needs a GPU: `:none` is refused on the CPU backend, pointing at `:bk` (#175)
        @test_throws SDS.NotSupportedError SDSSolver(madnlp_csc(backend, A))
        @test_throws SDS.NotSupportedError SDSSolver(madnlp_csc(backend, A);
                                                     opt = SDSSolverOptions(; sds_factorization = :ldl))
    end
end

@testset "SDSSolver (fused) on $(backend_name(backend))" for backend in MADNLP_FUSED_BACKENDS
    @testset ":cholesky, laplacian2d 30×30: $T" for T in eltypes_among((Float32, Float64))
        A = laplacian2d(T, 30, 30)
        n = size(A, 1)
        M = SDSSolver(madnlp_csc(backend, A))
        @test M isa SDSSolver{T}
        @test M.inner isa SDS.FusedSolver
        @test MadNLP.introduce(M) isa String
        @test MadNLP.improve!(M) == false
        @test MadNLP.is_inertia(M)
        MadNLP.factorize!(M)
        @test M.last_info == 0
        @test MadNLP.inertia(M) == (n, 0, 0)
        @test madnlp_relres(backend, M, A) <= tol(T)
    end

    # the KKT matrix of #141: under nested dissection (algo4) the root front is 80 wide with pivots of both signs
    @testset ":ldl, :none, grid_kkt_matrix 40×40, $reordering: $T" for T in eltypes_among((Float32, Float64)),
                                                                        reordering in ("algo4", "algo3")
        m = 40 * 40
        A = grid_kkt_matrix(T, 40, 40, 1.0)
        opt = SDSSolverOptions(; sds_factorization = :ldl, sds_pivoting = :none, sds_reordering = reordering)
        M = SDSSolver(madnlp_csc(backend, A); opt)
        @test M.inner isa SDS.FusedSolver
        @test MadNLP.introduce(M) isa String
        MadNLP.factorize!(M)
        @test M.last_info == 0
        @test MadNLP.inertia(M) == (m, 0, m)
        @test madnlp_relres(backend, M, A) <= tol(T)
        # the next interior-point iteration: new values in place, refactorization
        A2 = grid_kkt_matrix(T, 40, 40, 0.5)
        MadNLP.factorize!(set_values!(M, A2))
        @test M.last_info == 0
        @test MadNLP.inertia(M) == (m, 0, m)
        @test madnlp_relres(backend, M, A2) <= tol(T)
    end

    # a failed factorization: `last_info != 0` and the inertia (n - 2, 1, 1) that makes MadNLP regularize and
    # factorize again; the regularized matrix then factors
    @testset ":cholesky on an indefinite matrix: $T" for T in eltypes_among((Float32, Float64))
        n, j = 40, 17
        A = singular_block_matrix(T, n, j; stored_zero = true)
        A[j, j] = -1
        M = SDSSolver(madnlp_csc(backend, A))
        MadNLP.factorize!(M)
        @test M.last_info != 0
        @test MadNLP.is_inertia(M)
        @test MadNLP.inertia(M) == (n - 2, 1, 1)
        A[j, j] = 1
        MadNLP.factorize!(set_values!(M, A))
        @test M.last_info == 0
        @test MadNLP.inertia(M) == (n, 0, 0)
        @test madnlp_relres(backend, M, A) <= tol(T)
    end
end

@testset "SDSSolver (:bk) on $(backend_name(backend))" for backend in MADNLP_BK_BACKENDS
    # (A, A2): the matrix and the values of the next interior-point iteration on the same pattern. In both
    # the inertia is the sign count of the diagonal: random_symindef is strictly diagonally dominant, grid_kkt_matrix
    # has inertia (nx·ny, 0, nx·ny) with a positive primal and a negative dual diagonal.
    @testset ":ldl, :bk, $label: $T" for T in eltypes_among((Float32, Float64)),
                                         (label, gen) in (
                                             ("grid_kkt_matrix 40×40",
                                              () -> (grid_kkt_matrix(T, 40, 40, 1.0), grid_kkt_matrix(T, 40, 40, 0.5))),
                                             ("random_symindef 400",
                                              () -> (A = random_symindef(T, 400, 0.01); (A, 2A))))
        A, A2 = gen()
        npos, nneg = count(>(0), real(diag(A))), count(<(0), real(diag(A)))
        opt = SDSSolverOptions(; sds_factorization = :ldl, sds_pivoting = :bk)
        M = SDSSolver(madnlp_csc(backend, A); opt)
        @test M.inner isa SDS.DirectSolver
        @test MadNLP.introduce(M) isa String
        @test MadNLP.improve!(M) == false
        MadNLP.factorize!(M)
        @test M.last_info == 0
        @test MadNLP.inertia(M) == (npos, 0, nneg)
        @test madnlp_relres(backend, M, A) <= tol(T)
        MadNLP.factorize!(set_values!(M, A2))
        @test M.last_info == 0
        @test MadNLP.inertia(M) == (npos, 0, nneg)
        @test madnlp_relres(backend, M, A2) <= tol(T)
    end
end

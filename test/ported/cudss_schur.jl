# Port of CUDSS.jl's `test/test_schur_cudss.jl` (T20): `cudss_schur_lu`,
# `cudss_schur_ldlt` and `cudss_schur_cholesky` on the three 5×5 examples of
# `../CUDSS.jl/docs/src/schur_complement.md` (`schur_example_*` in
# test/matrices.jl).
#
# Changes, besides the names: `cudss_set(solver, "schur_matrix", CudssMatrix(S))`
# + `cudss_get(solver, "schur_matrix")` is `setparam!(solver, "schur_matrix", S)`
# + `getparam(solver, "schur_matrix")`, where a sparse destination is a CSR (a
# `CSR` with the requested index base, or with one-based indices the vendor CSR
# matrix of `api_csr`) and its view is passed as `(S, view)`. The dense solves of
# `S` (cuSOLVER `getrf`/`sytrf`/`potrf` and `getrs`/`sytrs`/`potrs`) and the
# sparse ones (`lu`/`ldlt`/`cholesky` of `S`) are done on the host with
# LinearAlgebra on the exported `S`. Comparisons `≈` use `rtol = tol(T)`.

# the Schur complement destination of a test case: dense, or a CSR with `nnz` entries and index base `index`
function schur_destination(backend, ::Type{T}, ::Type{INT}, ns, nnz, dense, index, k) where {T, INT}
    dense && return to_device(backend, zeros(T, ns, ns))
    if index == 'O' && isodd(k)       # the vendor CSR matrix (CSR on the CPU backend)
        return api_csr(backend, ones(INT, ns + 1), ones(INT, nnz), zeros(T, nnz), ns)
    end
    return CSR(to_device(backend, zeros(INT, ns + 1)), to_device(backend, zeros(INT, nnz)),
               to_device(backend, zeros(T, nnz)), ns, ns; index)
end

schur_host(S::AbstractMatrix) = to_host(S)
schur_host(S::CSR) = Matrix(SparseMatrixCSC(S))
schur_host(S) = Matrix(SparseMatrixCSC(CSR(S)))

function ported_schur_api_matrix(backend, A::SparseMatrixCSC{T}, ::Type{INT}, index) where {T, INT}
    rowptr, colval, nzval = host_csr_arrays(A, INT)
    if index == 'Z'
        rowptr .-= one(INT)
        colval .-= one(INT)
    end
    return api_csr(backend, rowptr, colval, nzval, size(A, 1))
end

function ported_schur_lu(backend, ::Type{T}, ::Type{INT}, index, dense_schur, k) where {T, INT}
    ex = schur_example_lu(T)
    A_cpu = ex.A
    A11 = Matrix{T}(A_cpu[1:2, 1:2])
    A12 = Matrix{T}(A_cpu[1:2, 3:5])
    A21 = Matrix{T}(A_cpu[3:5, 1:2])
    A22 = Matrix{T}(A_cpu[3:5, 3:5])
    S_cpu = A22 - A21 * (A11 \ A12)
    b_cpu = ex.b

    A_gpu = ported_schur_api_matrix(backend, A_cpu, INT, index)
    x_gpu = to_device(backend, zeros(T, 5))
    b_gpu = to_device(backend, b_cpu)
    solver = DirectSolver(A_gpu, "G", 'F'; index)

    setparam!(solver, "schur_mode", 1)
    setparam!(solver, "user_schur_indices", INT[0, 0, 1, 1, 1])

    execute!("analysis", solver, x_gpu, b_gpu)
    execute!("factorization", solver, x_gpu, b_gpu; asynchronous = false)

    (nrows_S, ncols_S, nnz_S) = getparam(solver, "schur_shape")
    @test (nrows_S, ncols_S) == (3, 3)
    S_gpu = schur_destination(backend, T, INT, nrows_S, nnz_S, dense_schur, index, k)
    setparam!(solver, "schur_matrix", S_gpu)
    @test getparam(solver, "schur_matrix") === S_gpu
    S = schur_host(S_gpu)
    @test S ≈ S_cpu rtol = tol(T)

    execute!("solve_fwd_schur", solver, x_gpu, b_gpu; asynchronous = false)
    bs_gpu = to_host(x_gpu)[3:5]
    bs_cpu = b_cpu[3:5] - A21 * (A11 \ b_cpu[1:2])
    @test bs_gpu ≈ bs_cpu rtol = tol(T)

    x2_gpu = lu(S) \ bs_gpu
    x2_cpu = lu(S_cpu) \ bs_cpu
    @test x2_gpu ≈ x2_cpu rtol = tol(T)

    x_host = to_host(x_gpu)
    x_host[3:5] .= x2_gpu
    copyto!(x_gpu, x_host)
    execute!("solve_bwd_schur", solver, b_gpu, x_gpu; asynchronous = false)
    x1_gpu = to_host(b_gpu)[1:2]
    x1_cpu = A11 \ (b_cpu[1:2] - A12 * x2_cpu)
    @test x1_gpu ≈ x1_cpu rtol = tol(T)
    @test to_host(b_gpu) ≈ ex.x rtol = tol(T)
    return nothing
end

function ported_schur_sym(backend, ::Type{T}, ::Type{INT}, index, dense_schur, uplo, op, cholesky_case,
                          k) where {T, INT}
    ex = cholesky_case ? schur_example_cholesky(T) : schur_example_ldlt(T)
    A_cpu = ex.A
    A11 = Matrix{T}(A_cpu[1:2, 1:2])
    A12 = Matrix{T}(A_cpu[1:2, 3:5])
    A21 = Matrix{T}(A_cpu[3:5, 1:2])
    A22 = Matrix{T}(A_cpu[3:5, 3:5])
    S_cpu = A11 - A12 * (A22 \ A21)
    b_cpu = ex.b

    A_gpu = ported_schur_api_matrix(backend, op(A_cpu), INT, index)
    x_gpu = to_device(backend, zeros(T, 5))
    b_gpu = to_device(backend, b_cpu)
    structure = cholesky_case ? spd_structure(T) : sym_structure(T)
    solver = DirectSolver(A_gpu, structure, uplo; index)

    setparam!(solver, "schur_mode", 1)
    setparam!(solver, "user_schur_indices", INT[1, 1, 0, 0, 0])

    execute!("analysis", solver, x_gpu, b_gpu)
    execute!("factorization", solver, x_gpu, b_gpu; asynchronous = false)

    (nrows_S, ncols_S, nnz_S) = getparam(solver, "schur_shape")
    @test (nrows_S, ncols_S) == (2, 2)
    if dense_schur
        S_gpu = schur_destination(backend, T, INT, nrows_S, nnz_S, true, index, k)
        setparam!(solver, "schur_matrix", S_gpu)
        getparam(solver, "schur_matrix")
        @test schur_host(S_gpu) ≈ S_cpu rtol = tol(T)
    else
        # Maximum number of nonzeros in one triangle of the Schur complement
        nnz_S = min(nnz_S, nrows_S * (nrows_S + 1) ÷ 2)
        S_gpu = schur_destination(backend, T, INT, nrows_S, nnz_S, false, index, k)
        setparam!(solver, "schur_matrix", (S_gpu, uplo))
        getparam(solver, "schur_matrix")
        @test schur_host(S_gpu) ≈ op(S_cpu) rtol = tol(T)
    end

    execute!("solve_fwd_schur", solver, x_gpu, b_gpu; asynchronous = false)
    cholesky_case || execute!("solve_diag", solver, x_gpu, x_gpu; asynchronous = false)
    bs_gpu = to_host(x_gpu)[4:5]
    bs_cpu = b_cpu[1:2] - A12 * (A22 \ b_cpu[3:5])
    @test bs_gpu ≈ bs_cpu rtol = tol(T)

    S = cholesky_case ? Hermitian(S_cpu) : (T <: Real ? Symmetric(S_cpu) : Hermitian(S_cpu))
    x1_gpu = (cholesky_case ? cholesky(S) : bunchkaufman(S)) \ bs_gpu
    x1_cpu = (cholesky_case ? cholesky(S) : bunchkaufman(S)) \ bs_cpu
    @test x1_gpu ≈ x1_cpu rtol = tol(T)

    x_host = to_host(x_gpu)
    x_host[4:5] .= x1_gpu
    copyto!(x_gpu, x_host)
    execute!("solve_bwd_schur", solver, b_gpu, x_gpu; asynchronous = false)
    x2_gpu = to_host(b_gpu)[3:5]
    x2_cpu = A22 \ (b_cpu[3:5] - A21 * x1_cpu)
    @test x2_gpu ≈ x2_cpu rtol = tol(T)
    @test to_host(b_gpu) ≈ ex.x rtol = tol(T)
    return nothing
end

@testset "Schur complement -- LU ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES,
                                                                          INT in INTTYPES
    k = 0
    @testset "indexing = $index" for index in ('Z', 'O')
        @testset "Dense Schur complement = $dense_schur" for dense_schur in (false, true)
            ported_schur_lu(backend, T, INT, index, dense_schur, k += 1)
        end
    end
end

for (name, cholesky_case) in (("LDLᵀ and LDLᴴ", false), ("LLᵀ and LLᴴ", true))
    @testset "Schur complement -- $name ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES,
                                                                                  INT in INTTYPES
        k = 0
        @testset "indexing = $index" for index in ('Z', 'O')
            @testset "Dense Schur complement = $dense_schur" for dense_schur in (false, true)
                @testset "Triangle of the matrix: $uplo" for (uplo, op) in (('L', tril), ('U', triu), ('F', identity))
                    (!dense_schur && uplo == 'F') && continue
                    ported_schur_sym(backend, T, INT, index, dense_schur, uplo, op, cholesky_case, k += 1)
                end
            end
        end
    end
end

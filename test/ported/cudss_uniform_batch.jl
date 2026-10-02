# Port of CUDSS.jl test/test_uniform_batch_cudss.jl (T17): `uniform_batch_ldlt()`
# and `uniform_batch_cholesky()`, each with the cuDSS API (`DirectSolver` +
# `"ubatch_size"` + `MatrixDescriptor(T, n[, nrhs]; nbatch)`) and the generic
# API (`ldlt`/`cholesky` on a matrix whose values are longer than its pattern),
# strided and non-strided storage, views 'L', 'U', 'F'.
#
# Changes: `uniform_batch_lu()` is skipped (structure "G" is T19). The data are
# CUDSS.jl's, written once as `nnz × nbatch` matrices (the strided storage is
# their `vec`, exactly CUDSS.jl's strided vectors). Matrices come from
# `api_csr` (a `CSR` on the CPU backend); CUDSS.jl's `As_gpu.nzVal = …` is a new
# matrix on the same pattern. Residuals are checked per member with
# `relres ≤ tol(T)` (CUDSS.jl: `norm` of the absolute residuals `≤ √eps(R)`).

# pattern and values (nnz × nbatch, then the refactorization values) of the 5×5 examples
function ported_ubatch_ldlt_data(::Type{T}, uplo) where {T}
    real_t = T <: Real
    if uplo == 'L'
        rowptr, colval = [1, 2, 3, 6, 7, 9], [1, 2, 1, 2, 3, 4, 3, 5]
        nz = real_t ? [4 2; 3 3; 1 1; 2 1; 5 6; 1 4; 1 2; 2 8] :
             [4 2; 3 3; 1+im 1-im; 2-im 1+im; 5 6; 1 4; 1+im 2-im; 2 8]
        new = real_t ? [-4 -2; -3 -3; -1 -1; -2 -1; -5 -6; -1 -4; -1 -2; -2 -8] :
              [-4 -2; -3 -3; -1-im -1+im; -2+im -1-im; -5 -6; -1 -4; -1-im -2+im; -2 -8]
    elseif uplo == 'U'
        rowptr, colval = [1, 3, 5, 7, 8, 9], [1, 3, 2, 3, 3, 5, 4, 5]
        nz = real_t ? [4 2; 1 1; 3 3; 2 1; 5 6; 1 2; 1 4; 2 8] :
             [4 2; 1-im 1+im; 3 3; 2+im 1-im; 5 6; 1-im 2+im; 1 4; 2 8]
        new = real_t ? [-4 -2; -1 -1; -3 -3; -2 -1; -5 -6; -1 -2; -1 -4; -2 -8] :
              [-4 -2; -1+im -1-im; -3 -3; -2-im -1+im; -5 -6; -1+im -2-im; -1 -4; -2 -8]
    else
        rowptr, colval = [1, 3, 5, 9, 10, 12], [1, 3, 2, 3, 1, 2, 3, 5, 4, 3, 5]
        nz = real_t ? [4 2; 1 1; 3 3; 2 1; 1 1; 2 1; 5 6; 1 2; 1 4; 1 2; 2 8] :
             [4 2; 1-im 1+im; 3 3; 2+im 1-im; 1+im 1-im; 2-im 1+im; 5 6; 1-im 2+im; 1 4; 1+im 2-im; 2 8]
        new = real_t ? [-4 -2; -1 -1; -3 -3; -2 -1; -1 -1; -2 -1; -5 -6; -1 -2; -1 -4; -1 -2; -2 -8] :
              [-4 -2; -1+im -1-im; -3 -3; -2-im -1+im; -1-im -1+im; -2+im -1-im; -5 -6; -1+im -2-im; -1 -4;
               -1-im -2+im; -2 -8]
    end
    return rowptr, colval, T.(nz), T.(new)
end

function ported_ubatch_cholesky_data(::Type{T}, uplo) where {T}
    if uplo == 'L'
        rowptr, colval = [1, 2, 3, 6, 7, 9], [1, 2, 1, 2, 3, 4, 3, 5]
        nz = [4 2; 3 3; 1 1; 2 1; 5 6; 1 2; 1 4; 2 8]
        new = [8 6; 6 9; 2 3; 4 3; 10 18; 2 12; 2 6; 4 24]
    elseif uplo == 'U'
        rowptr, colval = [1, 3, 5, 7, 8, 9], [1, 3, 2, 3, 3, 5, 4, 5]
        nz = [4 2; 1 1; 3 3; 2 1; 5 6; 1 2; 1 4; 2 8]
        new = [8 6; 2 3; 6 9; 4 3; 10 18; 2 6; 2 12; 4 24]
    else
        rowptr, colval = [1, 3, 5, 9, 10, 12], [1, 3, 2, 3, 1, 2, 3, 5, 4, 3, 5]
        nz = [4 2; 1 1; 3 3; 2 1; 1 1; 2 1; 5 6; 1 2; 1 4; 1 2; 2 8]
        new = [8 6; 2 3; 6 9; 4 3; 2 3; 4 3; 10 18; 2 6; 2 12; 2 6; 4 24]
    end
    return rowptr, colval, T.(nz), T.(new)
end

# the full Hermitian members of a batch on the host (CUDSS.jl: A + Aᴴ - Diagonal(A) for a triangle)
function ported_ubatch_members(rowptr, colval, nz::AbstractMatrix, uplo)
    n = length(rowptr) - 1
    return map(1:size(nz, 2)) do k
        A = host_csc(rowptr, colval, nz[:, k], n, n)
        uplo == 'F' ? A : A + A' - Diagonal(A)
    end
end

function ported_ubatch(backend, ::Type{T}, ::Type{INT}, uplo, generic::Bool, strided::Bool,
                       cholesky_case::Bool) where {T, INT}
    n, nbatch = 5, 2
    nrhs = cholesky_case ? 1 : 2
    rowptr, colval, nz, new_nz = cholesky_case ? ported_ubatch_cholesky_data(T, uplo) : ported_ubatch_ldlt_data(T, uplo)
    rowptr, colval = Vector{INT}(rowptr), Vector{INT}(colval)
    structure = cholesky_case ? "HPD" : "H"
    if cholesky_case
        B = T.(reshape([7, 12, 25, 4, 13, 13, 15, 29, 8, 14], n, nrhs, nbatch))
        new_B = B
    elseif T <: Real
        B = T.(cat([7 -7; 12 -12; 25 -25; 4 -4; 13 -13], [13 -13; 15 -15; 29 -29; 8 -8; 14 -14]; dims = 3))
        new_B = B[:, :, [2, 1]]
    else
        B = T.(cat([7+im -7+im; 12+im -12+im; 25+im -25+im; 4+im -4+im; 13+im -13+im],
                   [13-im -13-im; 15-im -15-im; 29-im -29-im; 8-im -8-im; 14-im -14-im]; dims = 3))
        new_B = B[:, :, [2, 1]]
    end
    # strided storage: vectors; otherwise an nnz × nbatch matrix of values and n × nbatch / n × nrhs × nbatch arrays
    shape_x = strided ? (n * nrhs * nbatch,) : cholesky_case ? (n, nbatch) : (n, nrhs, nbatch)
    nz_dev(x) = to_device(backend, strided ? vec(x) : x)
    Bs_gpu = to_device(backend, reshape(B, shape_x))
    Xs_gpu = to_device(backend, zeros(T, shape_x))

    if generic
        As_gpu = api_csr(backend, rowptr, colval, vec(nz), n)
        solver = cholesky_case ? cholesky(As_gpu; view = uplo) : ldlt(As_gpu; view = uplo)
        @test solver.nbatch == nbatch
        ldiv!(Xs_gpu, solver, Bs_gpu)
    else
        solver = DirectSolver(to_device(backend, rowptr), to_device(backend, colval), nz_dev(nz), structure, uplo)
        setparam!(solver, "ubatch_size", nbatch)
        Bs_desc = cholesky_case ? MatrixDescriptor(T, n; nbatch) : MatrixDescriptor(T, n, nrhs; nbatch)
        Xs_desc = cholesky_case ? MatrixDescriptor(T, n; nbatch) : MatrixDescriptor(T, n, nrhs; nbatch)
        update!(Bs_desc, Bs_gpu)
        update!(Xs_desc, Xs_gpu)
        execute!("analysis", solver, Xs_desc, Bs_desc)
        execute!("factorization", solver, Xs_desc, Bs_desc; asynchronous = false)
        execute!("solve", solver, Xs_desc, Bs_desc; asynchronous = false)
    end
    @test maximum(batch_relres(ported_ubatch_members(rowptr, colval, nz, uplo), to_host(Xs_gpu), B)) <= tol(T)

    new_Bs_gpu = to_device(backend, reshape(new_B, shape_x))
    if generic
        As_new = api_csr(backend, rowptr, colval, vec(new_nz), n)
        cholesky_case ? cholesky!(solver, As_new) : ldlt!(solver, As_new)
        Xs_gpu .= new_Bs_gpu
        ldiv!(solver, Xs_gpu)
    else
        update!(solver, to_device(backend, rowptr), to_device(backend, colval), nz_dev(new_nz))
        execute!("refactorization", solver, Xs_desc, Bs_desc; asynchronous = false)
        update!(Bs_desc, new_Bs_gpu)
        execute!("solve", solver, Xs_desc, Bs_desc; asynchronous = false)
    end
    @test maximum(batch_relres(ported_ubatch_members(rowptr, colval, new_nz, uplo), to_host(Xs_gpu), new_B)) <=
          tol(T)
    return nothing
end

for (name, cholesky_case) in (("LDLᵀ and LDLᴴ", false), ("Cholesky", true))
    @testset "uniform batch $name ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES,
                                                                            INT in INTTYPES
        for generic in (false, true), strided in (false, true), uplo in ('L', 'U', 'F')
            @testset "$(generic ? "Generic" : "cuDSS") API, strided = $strided, view = $uplo" begin
                ported_ubatch(backend, T, INT, uplo, generic, strided, cholesky_case)
            end
        end
    end
end

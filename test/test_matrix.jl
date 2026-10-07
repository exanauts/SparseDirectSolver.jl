# T02: CSR container, backend adapters, matrix descriptors.

# The stored CSR matrix with its `transposed` flag applied.
logical_matrix(B) = B.transposed ? copy(transpose(SparseMatrixCSC(B))) : SparseMatrixCSC(B)

@testset "CSR host round trip ($T, $INT)" for T in ELTYPES, INT in INTTYPES
    A = SparseMatrixCSC{T, INT}(random_general(T, 30, 0.15))
    R = SparseMatrixCSC{T, INT}(sprand(T, 17, 9, 0.3))   # rectangular
    for M in (A, R), index in ('O', 'Z', SDS.INDEX_ZERO)
        B = CSR(M; index)
        @test B isa CSR{T, INT}
        @test B.index == convert(SDS.IndexBase, index isa Char ? index : convert(Char, index))
        @test !B.transposed
        @test size(B) == size(M)
        @test nnz(B) == nnz(M)
        @test nbatch(B) == 1
        @test eltype(B) == T
        @test get_backend(B) == CPU()
        C = SparseMatrixCSC(B)
        @test C isa SparseMatrixCSC{T, INT}
        @test C == M
        base = B.index == SDS.INDEX_ZERO ? 0 : 1
        @test B.rowptr[1] == base
        @test B.rowptr[end] == nnz(M) + base
    end
    # raw arrays, zero-based, square size deduced from rowptr
    Bz = CSR(A; index = 'Z')
    Braw = CSR(Bz.rowptr, Bz.colval, Bz.nzval; index = 'Z')
    @test size(Braw) == size(A)
    @test SparseMatrixCSC(Braw) == A
    @test occursin("zero-based", sprint(show, Braw))
end

@testset "csr_of_transpose ($T, $INT)" for T in ELTYPES, INT in INTTYPES
    A = SparseMatrixCSC{T, INT}(sprand(T, 12, 7, 0.4))
    B = csr_of_transpose(A)
    @test B.transposed
    @test B.index == SDS.INDEX_ONE
    @test size(B) == (7, 12)
    @test pointer(B.rowptr) == pointer(A.colptr)
    @test pointer(B.colval) == pointer(A.rowval)
    @test pointer(B.nzval) == pointer(A.nzval)
    @test SparseMatrixCSC(B) == transpose(A)   # plain transpose, also for complex T
    @test logical_matrix(B) == A
    @test occursin("transposed", sprint(show, B))
    # spare capacity in rowval/nzval (allowed by SparseMatrixCSC) is refused, not read as entries (#31)
    # (the constructor checks the lengths, so the spare entries are pushed afterwards)
    Ar, Av = copy(A), copy(A)
    push!(Ar.rowval, INT(1))
    push!(Av.nzval, zero(T))
    @test thrown(() -> csr_of_transpose(Ar)) isa InvalidValueError
    @test thrown(() -> csr_of_transpose(Av)) isa InvalidValueError
end

@testset "nbatch ($T)" for T in ELTYPES
    A = random_general(T, 10, 0.3)
    B = CSR(A)
    nz = nnz(B)
    @test nbatch(B) == 1
    vals = rand(T, nz, 3)
    Bv = CSR(B.rowptr, B.colval, vec(vals), 10, 10)
    Bm = CSR(B.rowptr, B.colval, vals, 10, 10)
    @test nbatch(Bv) == 3
    @test nbatch(Bm) == 3
    @test nnz(Bv) == nnz(Bm) == nz
    for k in 1:3
        @test SparseMatrixCSC(Bv, k).nzval == SparseMatrixCSC(Bm, k).nzval
        @test SparseMatrixCSC(Bm, k) == SparseMatrixCSC(CSR(B.rowptr, B.colval, vals[:, k], 10, 10))
    end
    @test thrown(() -> SparseMatrixCSC(Bm, 4)) isa InvalidValueError
    # empty matrix
    E = CSR(spzeros(T, 3, 3))
    @test nnz(E) == 0
    @test nbatch(E) == 1
    @test SparseMatrixCSC(E) == spzeros(T, 3, 3)
end

@testset "CSR validation" begin
    @test thrown(() -> CSR([1, 2], [1], [1.0], 3, 3)) isa InvalidValueError          # rowptr length
    @test thrown(() -> CSR([1, 2, 3], [1, 2], [1.0, 2.0, 3.0])) isa InvalidValueError  # 3 % 2 ≠ 0
    @test thrown(() -> CSR([1, 2, 3], [1, 2], zeros(3, 2))) isa InvalidValueError     # matrix rows ≠ nnz
    @test thrown(() -> CSR(Int32[1, 2, 3], [1, 2], [1.0, 2.0])) isa InvalidValueError # mixed index types
    @test thrown(() -> CSR([1, 2, 3], [1, 2], [1.0, 2.0]; index = 'X')) isa InvalidValueError
    @test thrown(() -> CSR(sparse([1.0 0; 0 1]); index = "O")) isa InvalidValueError
    @test thrown(() -> CSR(ones(Int8, 201), Int8[], Float64[], 200, 200)) isa InvalidValueError  # sizes overflow INT
end

@testset "to_backend ($(backend_name(backend)), $T, $INT)" for backend in BACKENDS, T in ELTYPES, INT in INTTYPES
    A = SparseMatrixCSC{T, INT}(random_general(T, 25, 0.2))
    for index in ('O', 'Z')
        B = to_backend(A, backend; index)
        @test B isa CSR{T, INT}
        @test get_backend(B) == backend
        @test get_backend(B.rowptr) == backend
        @test get_backend(B.colval) == backend
        @test SparseMatrixCSC(B) == A
        @test to_backend(B, backend).nzval === B.nzval   # already there: no copy
    end
    Bt = to_backend(csr_of_transpose(A), backend)
    @test Bt.transposed
    @test logical_matrix(Bt) == A
end

@testset "CUDA adapters ($T, $INT)" for backend in BACKENDS, T in ELTYPES, INT in INTTYPES
    backend_name(backend) == "CUDA" || continue
    A = SparseMatrixCSC{T, INT}(random_general(T, 20, 0.2))
    # Built from the host CSR arrays: `to_device(::CUDABackend, A, Int64)` yields Int32
    # indices because cuSPARSE ignores the index type parameter (see the T02 Report).
    Bh = CSR(A)
    dA = CuSparseMatrixCSR{T, INT}(CuVector{INT}(Bh.rowptr), CuVector{INT}(Bh.colval), CuVector{T}(Bh.nzval), size(A))
    B = CSR(dA)
    @test B isa CSR{T, INT}
    @test !B.transposed
    @test B.index == SDS.INDEX_ONE
    @test pointer(B.rowptr) == pointer(dA.rowPtr)
    @test pointer(B.colval) == pointer(dA.colVal)
    @test pointer(B.nzval) == pointer(dA.nzVal)
    @test get_backend(B) == backend
    @test SparseMatrixCSC(B) == A
    dC = CuSparseMatrixCSC{T, INT}(CuVector{INT}(A.colptr), CuVector{INT}(A.rowval), CuVector{T}(A.nzval), size(A))
    Bt = CSR(dC)
    @test Bt.transposed
    @test size(Bt) == reverse(size(A))
    @test pointer(Bt.rowptr) == pointer(dC.colPtr)
    @test pointer(Bt.nzval) == pointer(dC.nzVal)
    @test logical_matrix(Bt) == A
    dA2 = CuSparseMatrixCSR(B)
    @test dA2 isa CuSparseMatrixCSR{T, INT}
    @test pointer(dA2.nzVal) == pointer(dA.nzVal)
    @test SparseMatrixCSC(CSR(dA2)) == A
    Bz = to_backend(A, backend; index = 'Z')
    dZ = CuSparseMatrixCSR(Bz)
    @test dZ isa CuSparseMatrixCSR{T, INT}
    @test pointer(dZ.rowPtr) != pointer(Bz.rowptr)   # rebased copy
    @test SparseMatrixCSC(CSR(dZ)) == A
    Bd = to_backend(A, backend)
    @test Bd.rowptr isa CuVector{INT}
    @test SparseMatrixCSC(Bd) == A
end

@testset "MatrixDescriptor ($(backend_name(backend)), $T)" for backend in BACKENDS, T in ELTYPES
    n, p, nb = 6, 3, 4
    d = MatrixDescriptor(T, n; nbatch = nb)
    @test size(d) == (n, 1)
    @test nbatch(d) == nb
    @test d.data === nothing
    x = to_device(backend, rand(T, n * nb))
    @test update!(d, x) === d
    @test d.data === x
    @test get_backend(d) == backend
    @test thrown(() -> update!(d, to_device(backend, rand(T, n * nb + 1)))) isa InvalidValueError
    @test thrown(() -> update!(d, to_device(backend, rand(T, n, nb + 1)))) isa InvalidValueError
    @test thrown(() -> update!(d, to_device(backend, rand(T === Float32 ? Float64 : Float32, n * nb)))) isa
          InvalidValueError
    @test d.data === x                      # failed updates leave the descriptor alone

    X3 = to_device(backend, rand(T, n, p, nb))
    d3 = MatrixDescriptor(X3)
    @test size(d3) == (n, p)
    @test nbatch(d3) == nb
    @test !d3.transposed
    Y3 = to_device(backend, rand(T, n, p, nb))
    @test update!(d3, Y3).data === Y3
    @test thrown(() -> update!(d3, to_device(backend, rand(T, p, n, nb)))) isa InvalidValueError
    @test thrown(() -> update!(d3, to_device(backend, rand(T, n, p, nb - 1)))) isa InvalidValueError
    @test thrown(() -> update!(d3, vec(Y3))) isa InvalidValueError   # typed descriptor: same array type only
    @test update!(d, vec(to_device(backend, rand(T, n, nb)))).data isa AbstractVector

    dm = MatrixDescriptor(T, n, p)
    @test size(dm) == (n, p) && nbatch(dm) == 1
    @test update!(dm, to_device(backend, rand(T, n, p))).data isa AbstractMatrix
    @test thrown(() -> update!(dm, to_device(backend, rand(T, p, n)))) isa InvalidValueError

    dt = MatrixDescriptor(T, n, p; transposed = true)   # CUDSS.jl convention: logical p × n, row-major
    @test size(dt) == (p, n) && dt.transposed
    @test update!(dt, to_device(backend, rand(T, n, p))).data isa AbstractMatrix
    @test thrown(() -> update!(dt, to_device(backend, rand(T, p, n)))) isa InvalidValueError

    M = to_device(backend, rand(T, n, p))
    dmt = MatrixDescriptor(M; transposed = true)
    @test size(dmt) == (p, n)
    v = to_device(backend, rand(T, n))
    dv = MatrixDescriptor(v)
    @test size(dv) == (n, 1) && nbatch(dv) == 1
    @test thrown(() -> MatrixDescriptor(v; transposed = true)) isa InvalidValueError
    @test thrown(() -> MatrixDescriptor(T, -1)) isa InvalidValueError
    @test thrown(() -> MatrixDescriptor(T, n; nbatch = 0)) isa InvalidValueError
end

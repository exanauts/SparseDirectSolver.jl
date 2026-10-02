# Backends under test (TASKS.md "Shared test conventions").
#
# The KernelAbstractions CPU backend is tested unless SDS_TEST_CPU == "0" (the
# GPU CI jobs set it: the CPU-only jobs already cover the CPU backend). A GPU
# backend is added when SDS_TEST_GPU != "0", its package is installed in the test
# environment (CI adds CUDA/AMDGPU to test/Project.toml itself; locally run
# `julia --project=test -e 'using Pkg; Pkg.add("CUDA")'` and do not commit it),
# and the package reports a functional device. An empty backend list is an error.

using KernelAbstractions
using SparseArrays

const TEST_CPU = get(ENV, "SDS_TEST_CPU", "1") != "0"
const TEST_GPU = get(ENV, "SDS_TEST_GPU", "1") != "0"

is_package_installed(name::String) = Base.find_package(name) !== nothing

const CUDA_LOADED = TEST_GPU && is_package_installed("CUDA")
if CUDA_LOADED
    using CUDA
    using CUDA.cuSPARSE
    CUDA.allowscalar(false)
end
const CUDA_FUNCTIONAL = CUDA_LOADED && CUDA.functional()

const AMDGPU_LOADED = TEST_GPU && is_package_installed("AMDGPU")
if AMDGPU_LOADED
    using AMDGPU
    using AMDGPU.rocSPARSE
    AMDGPU.allowscalar(false)
end
const AMDGPU_FUNCTIONAL = AMDGPU_LOADED && AMDGPU.functional()

"""
    BACKENDS

Backends every numeric test loops over: `CPU()` first (unless `SDS_TEST_CPU=0`),
then each functional GPU.
"""
const BACKENDS = Any[]
TEST_CPU && push!(BACKENDS, CPU())
CUDA_FUNCTIONAL && push!(BACKENDS, CUDABackend())
AMDGPU_FUNCTIONAL && push!(BACKENDS, ROCBackend())
isempty(BACKENDS) &&
    error("no backend to test: SDS_TEST_CPU=0 and no functional GPU backend (SDS_TEST_GPU=$(get(ENV, "SDS_TEST_GPU", "1")), ",
          "CUDA loaded: $CUDA_LOADED, AMDGPU loaded: $AMDGPU_LOADED)")

"""
    backend_name(backend) -> String

Short name for testset titles: `"CPU"`, `"CUDA"`, `"ROCm"`.
"""
backend_name(::CPU) = "CPU"

"""
    to_device(backend, x)
    to_device(backend, A::SparseMatrixCSC, INT)

Copy a host `Array` or `SparseMatrixCSC` to `backend`: `Array`/`SparseMatrixCSC`
on the CPU backend, `CuArray`/`CuSparseMatrixCSR` on CUDA, `ROCArray`/
`ROCSparseMatrixCSR` on ROCm. The three-argument form also sets the index type
of the sparse matrix (`Int32` or `Int64`).
"""
to_device(::CPU, x::AbstractArray) = copy(x)
to_device(::CPU, A::SparseMatrixCSC{T}, ::Type{INT}) where {T, INT} = SparseMatrixCSC{T, INT}(A)

"""
    to_host(x)

Inverse of [`to_device`](@ref): `Array` for dense arrays, `SparseMatrixCSC` for
sparse matrices.
"""
to_host(x::AbstractArray) = Array(x)
# strided device views (panels, padded batches): copy the parent, index on the host
to_host(x::SubArray) = Array(parent(x))[x.indices...]
to_host(A::SparseMatrixCSC) = copy(A)

"""
    index_eltype(A)

Index type of a host or device sparse matrix (`Int32` or `Int64`).
"""
index_eltype(::SparseMatrixCSC{<:Any, INT}) where {INT} = INT

# The vendor host-to-device sparse constructors do not honour a requested index
# type: cuSPARSE's `CuSparseMatrixCSR{T, INT}(::SparseMatrixCSC)` and AMDGPU's
# `ROCSparseMatrixCSR` both upload `Cint` indices (issue #30), and cuSPARSE's
# `SparseMatrixCSC(::CuSparseMatrixCSR{T, Int64})` fails. So the CSR arrays are
# built on the host (the CSC of the transpose is the CSR of `A`) and uploaded
# with the requested type, and the inverse goes the same way.
host_csr_arrays(A::SparseMatrixCSC{T}, ::Type{INT}) where {T, INT} =
    (At = SparseMatrixCSC{T, INT}(sparse(transpose(A))); (At.colptr, At.rowval, At.nzval))
function host_csc(rowptr, colval, nzval::AbstractVector{T}, m, n) where {T}
    At = SparseMatrixCSC{T, Int}(n, m, Array{Int}(rowptr), Array{Int}(colval), Array{T}(nzval))
    return SparseMatrixCSC{T, Int}(sparse(transpose(At)))
end

"""
    api_matrix(backend, A, INT = index type of A)

The sparse matrix a user of the public API passes on `backend` (T13): a host
[`CSR`](@ref) on the CPU backend (`cholesky(::SparseMatrixCSC)` belongs to
CHOLMOD), the vendor CSR matrix of [`to_device`](@ref) on a GPU.
"""
api_matrix(backend, A::SparseMatrixCSC{<:Any, INT}) where {INT} = api_matrix(backend, A, INT)
api_matrix(backend, A::SparseMatrixCSC, ::Type{INT}) where {INT} = to_device(backend, A, INT)
api_matrix(::CPU, A::SparseMatrixCSC{T}, ::Type{INT}) where {T, INT} = CSR(SparseMatrixCSC{T, INT}(A))

"""
    device_allocated(backend, f) -> Union{Int, Missing}

Bytes of device memory allocated by `f()` (T13): `@allocated` on the CPU
backend (device memory is host memory), `CUDA.@allocated` on CUDA; `missing`
when the backend offers no counter.
"""
device_allocated(::CPU, f) = @allocated f()
device_allocated(backend, f) = missing

if CUDA_LOADED
    if isdefined(CUDA, Symbol("@allocated"))
        @eval device_allocated(::CUDABackend, f) = CUDA.@allocated f()
    end
    backend_name(::CUDABackend) = "CUDA"
    to_device(::CUDABackend, x::Array) = CuArray(x)
    to_device(backend::CUDABackend, A::SparseMatrixCSC{T, INT}) where {T, INT} = to_device(backend, A, INT)
    function to_device(::CUDABackend, A::SparseMatrixCSC{T}, ::Type{INT}) where {T, INT}
        rowptr, colval, nzval = host_csr_arrays(A, INT)
        return CuSparseMatrixCSR{T, INT}(CuVector{INT}(rowptr), CuVector{INT}(colval), CuVector{T}(nzval), size(A))
    end
    to_host(A::CuSparseMatrixCSR) = host_csc(A.rowPtr, A.colVal, A.nzVal, size(A)...)
    index_eltype(A::CuSparseMatrixCSR) = eltype(A.rowPtr)
end

if AMDGPU_LOADED
    backend_name(::ROCBackend) = "ROCm"
    to_device(::ROCBackend, x::Array) = ROCArray(x)
    to_device(backend::ROCBackend, A::SparseMatrixCSC{T, INT}) where {T, INT} = to_device(backend, A, INT)
    function to_device(::ROCBackend, A::SparseMatrixCSC{T}, ::Type{INT}) where {T, INT}
        rowptr, colval, nzval = host_csr_arrays(A, INT)
        return ROCSparseMatrixCSR{T, INT}(ROCVector{INT}(rowptr), ROCVector{INT}(colval), ROCVector{T}(nzval), size(A))
    end
    to_host(A::ROCSparseMatrixCSR) = host_csc(A.rowPtr, A.colVal, A.nzVal, size(A)...)
    index_eltype(A::ROCSparseMatrixCSR) = eltype(A.rowPtr)
end

let gpus = String[]
    CUDA_FUNCTIONAL && push!(gpus, "CUDA ($(CUDA.name(CUDA.device())))")
    AMDGPU_FUNCTIONAL && push!(gpus, "ROCm ($(AMDGPU.device()))")
    CUDA_LOADED && !CUDA_FUNCTIONAL && push!(gpus, "CUDA installed but not functional: skipped")
    AMDGPU_LOADED && !AMDGPU_FUNCTIONAL && push!(gpus, "AMDGPU installed but not functional: skipped")
    TEST_GPU || push!(gpus, "GPU backends disabled by SDS_TEST_GPU=0")
    TEST_CPU || push!(gpus, "CPU backend disabled by SDS_TEST_CPU=0")
    println("Backends under test: ", join(backend_name.(BACKENDS), ", "),
            isempty(gpus) ? "" : "  [" * join(gpus, "; ") * "]")
end

"""
    to_panel(backend, X; lead = 3, trail = 2)

Copy the host matrix `X` (`f × w`) into a flat device buffer with `lead`
entries before and `trail` after it, and return the panel view
`reshape(view(buf, lead+1:lead+f*w), f, w)`, the way fronts are laid out in the
factor buffer.
"""
function to_panel(backend, X::Matrix{T}; lead::Integer = 3, trail::Integer = 2) where {T}
    buf = to_device(backend, vcat(zeros(T, lead), vec(X), zeros(T, trail)))
    return reshape(view(buf, (lead + 1):(lead + length(X))), size(X)...)
end

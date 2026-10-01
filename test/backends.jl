# Backends under test (TASKS.md "Shared test conventions").
#
# The KernelAbstractions CPU backend is always tested. A GPU backend is added
# when SDS_TEST_GPU != "0", its package is installed in the test environment
# (CI adds CUDA/AMDGPU to test/Project.toml itself; locally run
# `julia --project=test -e 'using Pkg; Pkg.add("CUDA")'` and do not commit it),
# and the package reports a functional device.

using KernelAbstractions
using SparseArrays

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

Backends every numeric test loops over: `CPU()` first, then each functional GPU.
"""
const BACKENDS = Any[CPU()]
CUDA_FUNCTIONAL && push!(BACKENDS, CUDABackend())
AMDGPU_FUNCTIONAL && push!(BACKENDS, ROCBackend())

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
to_host(A::SparseMatrixCSC) = copy(A)

if CUDA_LOADED
    backend_name(::CUDABackend) = "CUDA"
    to_device(::CUDABackend, x::Array) = CuArray(x)
    to_device(::CUDABackend, A::SparseMatrixCSC) = CuSparseMatrixCSR(A)
    to_device(::CUDABackend, A::SparseMatrixCSC{T}, ::Type{INT}) where {T, INT} = CuSparseMatrixCSR{T, INT}(A)
    to_host(A::CuSparseMatrixCSR) = SparseMatrixCSC(A)
end

if AMDGPU_LOADED
    backend_name(::ROCBackend) = "ROCm"
    to_device(::ROCBackend, x::Array) = ROCArray(x)
    # AMDGPU.jl converts host indices to Cint eagerly and has no
    # `ROCSparseMatrixCSR{T, INT}(::SparseMatrixCSC)`, so the CSR arrays are
    # built on the host (CSC of the transpose) and uploaded with the requested
    # index type; the inverse goes the same way, without rocSPARSE.
    to_device(backend::ROCBackend, A::SparseMatrixCSC{T, INT}) where {T, INT} = to_device(backend, A, INT)
    function to_device(::ROCBackend, A::SparseMatrixCSC{T}, ::Type{INT}) where {T, INT}
        At = SparseMatrixCSC{T, INT}(sparse(transpose(A)))
        return ROCSparseMatrixCSR{T, INT}(ROCVector{INT}(At.colptr), ROCVector{INT}(At.rowval),
                                          ROCVector{T}(At.nzval), size(A))
    end
    function to_host(A::ROCSparseMatrixCSR{T}) where {T}
        m, n = size(A)
        At = SparseMatrixCSC{T, Int}(n, m, Array{Int}(A.rowPtr), Array{Int}(A.colVal), Array{T}(A.nzVal))
        return SparseMatrixCSC{T, Int}(sparse(transpose(At)))
    end
end

let gpus = String[]
    CUDA_FUNCTIONAL && push!(gpus, "CUDA ($(CUDA.name(CUDA.device())))")
    AMDGPU_FUNCTIONAL && push!(gpus, "ROCm ($(AMDGPU.device()))")
    CUDA_LOADED && !CUDA_FUNCTIONAL && push!(gpus, "CUDA installed but not functional: skipped")
    AMDGPU_LOADED && !AMDGPU_FUNCTIONAL && push!(gpus, "AMDGPU installed but not functional: skipped")
    TEST_GPU || push!(gpus, "GPU backends disabled by SDS_TEST_GPU=0")
    println("Backends under test: ", join(backend_name.(BACKENDS), ", "),
            isempty(gpus) ? "" : "  [" * join(gpus, "; ") * "]")
end

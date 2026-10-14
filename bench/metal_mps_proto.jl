# Low-level MPS prototype on offset/strided panel storage. This intentionally
# does not modify the solver; it gates direct MPS integration.

using LinearAlgebra
using Metal
using Metal.MPS
using Metal.MTL
using SparseDirectSolver
using Statistics

function mps_panel(A::MtlVector{T}, p0, rows, cols, ld) where {T}
    desc = MPSMatrixDescriptor(cols, rows, sizeof(T) * ld, T)
    return MPSMatrix(A, desc, (p0 - 1) * sizeof(T))
end

function mps_potrf!(A::MtlVector{Float32}, p0, n, ld)
    dev = Metal.device()
    status = MtlArray{MPS.MPSMatrixDecompositionStatus, 0, SharedStorage}(undef)
    mat = mps_panel(A, p0, n, n, ld)
    cmdbuf = MPSCommandBuffer(global_queue(dev))
    encode!(cmdbuf, MPSMatrixDecompositionCholesky(dev, false, n), mat, mat, status)
    commit!(cmdbuf)
    return status
end

function mps_potrf_plan(A::MtlVector{Float32}, p0, n, ld)
    dev = Metal.device()
    status = MtlArray{MPS.MPSMatrixDecompositionStatus, 0, SharedStorage}(undef)
    mat = mps_panel(A, p0, n, n, ld)
    kernel = MPSMatrixDecompositionCholesky(dev, false, n)
    return (; dev, status, mat, kernel)
end

function mps_potrf!(plan)
    cmdbuf = MPSCommandBuffer(global_queue(plan.dev))
    encode!(cmdbuf, plan.kernel, plan.mat, plan.mat, plan.status)
    commit!(cmdbuf)
    return plan.status
end

function panel_buffer(A, f, w; lead = 7)
    buf = zeros(Float32, lead + f * w + 5)
    buf[(lead + 1):(lead + f * w)] .= vec(A)
    return MtlArray(buf), lead + 1
end

function run_potrf(n; f = n + 13, samples = 20)
    G = randn(Float32, n, n)
    S = G * G' + n * I
    P = zeros(Float32, f, n)
    P[1:n, :] .= S
    buf, p0 = panel_buffer(P, f, n)
    ref = cholesky(Symmetric(S)).L

    mps_potrf!(buf, p0, n, f)
    Metal.synchronize()
    H = reshape(Array(buf)[p0:(p0 + f * n - 1)], f, n)
    err = norm(LowerTriangular(H[1:n, :]) - ref) / norm(ref)

    plan = mps_potrf_plan(buf, p0, n, f)
    times = Float64[]
    for _ in 1:samples
        copyto!(view(buf, p0:(p0 + f * n - 1)), MtlArray(vec(P)))
        push!(times, @elapsed begin
            mps_potrf!(plan)
            Metal.synchronize()
        end)
    end
    info = MtlVector{Int32}(undef, 1)
    F11 = view(reshape(view(buf, p0:(p0 + f * n - 1)), f, n), 1:n, 1:n)
    ka = Float64[]
    for _ in 1:samples
        copyto!(view(buf, p0:(p0 + f * n - 1)), MtlArray(vec(P)))
        push!(ka, @elapsed begin
            SparseDirectSolver.ka_potrf!('L', F11, info)
            Metal.synchronize()
        end)
    end
    return median(times), minimum(times), median(ka), minimum(ka), err
end

println("n,mps_median_s,mps_min_s,ka_median_s,ka_min_s,error")
for n in (16, 32, 64, 96, 128, 192, 256)
    println(join((n, run_potrf(n)...), ','))
end

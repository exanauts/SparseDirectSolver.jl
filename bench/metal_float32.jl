# Float32 CPU/Metal comparison on generated SPD matrices.
#
# Run from an environment containing SparseDirectSolver and Metal:
#   julia +1.13 --project=<env> bench/metal_float32.jl

using LinearAlgebra
using KernelAbstractions
using Metal
using Random
using SparseArrays
using SparseDirectSolver
using Statistics

function laplacian2d(n::Int)
    N = n * n
    I = Int[]
    J = Int[]
    V = Float32[]
    for y in 1:n, x in 1:n
        k = x + (y - 1) * n
        push!(I, k); push!(J, k); push!(V, 4.0f0)
        for (xx, yy) in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
            if 1 <= xx <= n && 1 <= yy <= n
                push!(I, k); push!(J, xx + (yy - 1) * n); push!(V, -1.0f0)
            end
        end
    end
    return sparse(I, J, V, N, N)
end

function run_case(label, backend, A; samples = 10)
    C = to_backend(tril(A), backend)
    solver = DirectSolver(C, "SPD", 'L')
    b = rand(Float32, size(A, 1))
    bd = KernelAbstractions.allocate(backend, Float32, length(b))
    xd = similar(bd)
    copyto!(bd, b)
    phase(p) = execute!(p, solver, xd, bd; asynchronous = false)

    ta = @elapsed phase("analysis")
    phase("factorization")
    phase("solve")
    tf = median([@elapsed phase("refactorization") for _ in 1:samples])
    ts = median([@elapsed phase("solve") for _ in 1:samples])
    x = Array(xd)
    residual = norm(b - A * x) / norm(b)
    residual <= sqrt(eps(Float32)) || error("$label residual $residual")
    println(join((label, size(A, 1), nnz(A), ta, tf, ts, residual), ','))
end

Random.seed!(666)
println("backend,n,nnz,analysis_s,refactorization_s,solve_s,relative_residual")
for grid in (40, 80, 120)
    A = laplacian2d(grid)
    run_case("CPU-$grid", KernelAbstractions.CPU(), A)
    run_case("Metal-$grid", MetalBackend(), A)
end

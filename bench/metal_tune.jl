# Metal scheduling sweep for Float32 Cholesky. Correctness gates every result.

include("metal_float32.jl")

function tune_case(A, label, opts; samples = 7)
    solver = DirectSolver(to_backend(tril(A), MetalBackend()), "SPD", 'L')
    solver.options = opts
    b = rand(Float32, size(A, 1))
    bd, xd = MtlArray(b), MtlArray(similar(b))
    execute!("analysis", solver, nothing, nothing)
    execute!("factorization", solver, nothing, nothing; asynchronous = false)
    execute!("solve", solver, xd, bd; asynchronous = false)
    residual = norm(b - A * Array(xd)) / norm(b)
    residual <= sqrt(eps(Float32)) || error("$label residual $residual")
    tf = median([@elapsed execute!("refactorization", solver, nothing, nothing; asynchronous = false)
                 for _ in 1:samples])
    ts = median([@elapsed execute!("solve", solver, xd, bd; asynchronous = false)
                 for _ in 1:samples])
    sc = solver.host_symbolic.schedule
    regimes = count.((==(SparseDirectSolver.REGIME_A), ==(SparseDirectSolver.REGIME_B),
                       ==(SparseDirectSolver.REGIME_C)), Ref(sc.regime))
    println(join((label, regimes..., SparseDirectSolver.nsubtrees(sc), SparseDirectSolver.nlaunches(sc),
                  tf, ts, residual), ','))
end

println("label,A,B,C,subtrees,launches,refactorization_s,solve_s,residual")
A = laplacian2d(120)
for p in (512, 1024, 2048, 4096, 8192, 16384), mf in (0, 16, 32, 64)
    tune_case(A, "p$(p)-f$(mf)", Options(subtree_parallelism = p, subtree_max_fronts = mf))
end
for (cw, cr) in ((32, 128), (32, 256), (64, 256), (64, 512), (64, 1024), (128, 1024))
    tune_case(A, "c$(cw)x$(cr)", Options(regime_c_width = cw, regime_c_rows = cr))
end

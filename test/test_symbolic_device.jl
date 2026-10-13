# Device construction of the symbolic maps (src/symbolic/device.jl): the
# kernels must produce ELEMENT-IDENTICAL results to the host originals, on
# every backend (the CPU backend runs the same kernels through KA).

random_pattern_csr(n, density, view) = begin
    A = random_spd(Float64, n, density)
    T = view == 'U' ? triu(A) : tril(A)
    C = SDS.CSR(SparseMatrixCSC{Float64, Int}(T))
    (Vector{Int}(C.rowptr), Vector{Int}(C.colval))
end

@testset "device symbolic maps ($(backend_name(backend)))" for backend in BACKENDS
    for (n, density) in ((0, 0.0), (1, 0.0), (40, 0.15), (300, 0.02), (1500, 0.004)),
        view in ('L', 'U'), structure in ("SPD", "S")

        rowptr, colval = random_pattern_csr(n, density, view)
        # pattern: host route vs device route
        Ph = SDS.SymmetricPattern(rowptr, colval, n, structure; view, index = 'O')
        Pd = SDS.SymmetricPattern(rowptr, colval, n, structure; view, index = 'O', device = backend)
        @test Ph == Pd                                # GPU: device route; CPU: gated to host
        n == 0 && continue
        # maps: build the partition once, then both map routes on it
        opts = Options()
        ord = SDS.compute_ordering(Ph, opts; alg = :amd)
        sp = SDS.supernode_partition(SDS.factor_pattern(Ph, ord), ord.perm, opts)
        sc = SDS.build_schedule(sp, opts, Float64;
                                reserve = SDS.subtree_local_reserve(structure),
                                elsize = SDS.schedule_elsize(structure, Float64))
        layout = SDS.build_layout(sp, sc; ldlt = structure == "S")
        amap_h = SDS.assembly_map(sp, layout, rowptr, colval, n, structure; view, index = 'O')
        ptr_h, src_h = SDS._group_amap(amap_h, layout, SDS.nsupernodes(sp))
        # the kernels run on every backend here (the solver gate excludes the
        # CPU backend only as a performance choice; correctness is universal)
        amap_d = SDS.device_assembly_map(backend, sp, layout, rowptr, colval, n, structure;
                                         view, index = 'O')
        ptr_d, src_d = SDS.device_group_amap(backend, amap_h, layout, SDS.nsupernodes(sp))
        cp_d, rv_d = SDS.device_symmetric_pattern(backend, rowptr, colval, n, view, 'O')
        @test length(amap_h) > 0                      # the comparisons are not vacuous
        @test amap_d == amap_h
        @test ptr_d == ptr_h
        @test src_d == src_h
        @test cp_d == Ph.colptr
        @test rv_d == Ph.rowval
    end
    # end-to-end: an analyzed solver's stored maps equal the host originals
    # recomputed from its own state (on GPU backends the stored maps came
    # through the device route; on the CPU backend this is a host identity)
    for T in eltypes_among((Float64,))
        A = random_spd(T, 200, 0.03)
        solver = DirectSolver(api_matrix(backend, tril(A), Int32), spd_structure(T), 'L')
        execute!("analysis", solver, nothing, nothing)
        Sh = solver.host_symbolic
        amap_ref = SDS.assembly_map(Sh.partition, Sh.layout, solver.host_rowptr, solver.analysis_colval,
                                    200, spd_structure(T); view = 'L', index = solver.A.index)
        @test length(amap_ref) > 0
        @test Sh.amap == amap_ref
        ptr_ref, src_ref = SDS._group_amap(amap_ref, Sh.layout, SDS.nsupernodes(Sh.partition))
        @test Sh.amap_ptr == ptr_ref
        @test Sh.amap_src == src_ref
        # the factorization still works through the device-built maps
        execute!("factorization", solver, nothing, nothing)
        b = rand(T, 200)
        bd = to_device(backend, b)
        xd = similar(bd)
        execute!("solve", solver, xd, bd; asynchronous = false)
        @test relres(A, to_host(xd), b) <= tol(T)
    end
end

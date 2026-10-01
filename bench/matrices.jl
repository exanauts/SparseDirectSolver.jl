# Benchmark matrices: generated Laplacians, SuiteSparse matrices through
# MatrixDepot.jl (optional, needs the network on first use) and MadNLP KKT dumps
# under `bench/data/` (see `dump_madnlp_kkt.jl`).
#
# Everything lives in the module `BenchMatrices` so that the test suite can
# `include` this file next to its own generators without name clashes.

module BenchMatrices

using LinearAlgebra
using SparseArrays

export BenchMatrix, generated_matrices, suitesparse_matrices, dump_matrices, bench_matrices,
       read_mtx, write_mtx, symmetrize_triangle, parse_dump_name, dump_name

const DATA_DIR = joinpath(@__DIR__, "data")

# MatrixDepot is optional: the generated matrices and the dump loader work without it.
const HAS_MATRIXDEPOT = Base.find_package("MatrixDepot") !== nothing
if HAS_MATRIXDEPOT
    import MatrixDepot
end

"""
    BenchMatrix(name, source, structures, A)

One benchmark matrix. `name` is unique and used in the CSV (`lap2d_300`,
`HB/bcsstk17`, `kkt_<case>_<kind>_<iter>`), `source ∈ (:generated, :suitesparse,
:dump)`, `structures` lists the cuDSS structure strings the baseline runs
(`"SPD"`, `"S"` for symmetric matrices, `"G"` for unsymmetric ones), `A` is the
full (both triangles) `SparseMatrixCSC{Float64,Int}`.
"""
struct BenchMatrix
    name::String
    source::Symbol
    structures::Vector{String}
    A::SparseMatrixCSC{Float64,Int}
end

Base.size(M::BenchMatrix) = size(M.A)
SparseArrays.nnz(M::BenchMatrix) = nnz(M.A)

function Base.show(io::IO, M::BenchMatrix)
    print(io, "BenchMatrix(\"", M.name, "\", :", M.source, ", ", M.structures, ", n = ",
          size(M.A, 1), ", nnz = ", nnz(M.A), ")")
end

# ---------------------------------------------------------------------------
# Generated matrices

_lap1d(n) = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.0, n), 1 => fill(-1.0, n - 1))
_eye(n) = sparse(1.0I, n, n)

"""
    laplacian2d(nx, ny)

5-point Dirichlet Laplacian on an `nx × ny` grid (`n = nx·ny`, SPD, Float64).
"""
laplacian2d(nx::Integer, ny::Integer) = kron(_eye(ny), _lap1d(nx)) + kron(_lap1d(ny), _eye(nx))

"""
    laplacian3d(nx, ny, nz)

7-point Dirichlet Laplacian on an `nx × ny × nz` grid (`n = nx·ny·nz`, SPD, Float64).
"""
function laplacian3d(nx::Integer, ny::Integer, nz::Integer)
    Ix, Iy, Iz = _eye(nx), _eye(ny), _eye(nz)
    Lx, Ly, Lz = _lap1d(nx), _lap1d(ny), _lap1d(nz)
    return kron(Iz, kron(Iy, Lx)) + kron(Iz, kron(Ly, Ix)) + kron(Lz, kron(Iy, Ix))
end

"""
    GENERATED

Documented generated matrices: `(name, n, nnz)`.

* `lap2d_300`: 2-D Laplacian on a 300 × 300 grid, `n = 90_000`, `nnz = 448_800`.
* `lap3d_40`: 3-D Laplacian on a 40 × 40 × 40 grid, `n = 64_000`, `nnz = 438_400`.
"""
const GENERATED = ((name = "lap2d_300", n = 90_000, nnz = 448_800),
                   (name = "lap3d_40", n = 64_000, nnz = 438_400))

"""
    generated_matrices() -> Vector{BenchMatrix}

The generated Laplacians of [`GENERATED`](@ref), structures `["SPD", "S"]`.
No network, no optional package.
"""
function generated_matrices()
    return [BenchMatrix("lap2d_300", :generated, ["SPD", "S"], laplacian2d(300, 300)),
            BenchMatrix("lap3d_40", :generated, ["SPD", "S"], laplacian3d(40, 40, 40))]
end

# ---------------------------------------------------------------------------
# SuiteSparse through MatrixDepot

"""
    SUITESPARSE

SuiteSparse matrices of the baseline: `(name, n, structures)`. `n` is the
documented size from the SuiteSparse collection, checked when loading.
"""
const SUITESPARSE = ((name = "HB/bcsstk17", n = 10_974, structures = ["SPD", "S"]),
                     (name = "Boeing/bcsstk38", n = 8_032, structures = ["SPD", "S"]),
                     (name = "GHS_psdef/apache2", n = 715_176, structures = ["SPD", "S"]),
                     (name = "Rajat/rajat21", n = 411_676, structures = ["G"]),
                     (name = "TSOPF/TSOPF_RS_b39_c7", n = 14_098, structures = ["G"]))

_full(A) = SparseMatrixCSC{Float64,Int}(sparse(A))

"""
    suitesparse_matrices(; names = [s.name for s in SUITESPARSE]) -> Vector{BenchMatrix}

Downloads (first use) and loads the named SuiteSparse matrices through
MatrixDepot.jl. Returns an empty vector with a warning when MatrixDepot is not
installed; a matrix that fails to load is skipped with a warning.
"""
function suitesparse_matrices(; names = [s.name for s in SUITESPARSE])
    if !HAS_MATRIXDEPOT
        @warn "MatrixDepot.jl is not installed in this environment; skipping SuiteSparse matrices"
        return BenchMatrix[]
    end
    get!(ENV, "DATADEPS_ALWAYS_ACCEPT", "true")
    out = BenchMatrix[]
    for name in names
        entry = findfirst(s -> s.name == name, SUITESPARSE)
        try
            A = _full(MatrixDepot.mdopen(name).A)
            structures = if entry === nothing
                issymmetric(A) ? ["S"] : ["G"]
            else
                SUITESPARSE[entry].n == size(A, 1) ||
                    @warn "$name: expected n = $(SUITESPARSE[entry].n), got $(size(A, 1))"
                SUITESPARSE[entry].structures
            end
            push!(out, BenchMatrix(name, :suitesparse, structures, A))
        catch err
            @warn "could not load $name through MatrixDepot" exception = (err, catch_backtrace())
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# MatrixMarket I/O (no dependency; coordinate format only)

"""
    read_mtx(path) -> SparseMatrixCSC{Float64,Int}

Reads a MatrixMarket coordinate file (`real`, `double`, `integer` or `pattern`;
`general`, `symmetric` or `skew-symmetric`) and returns the full matrix.
"""
function read_mtx(path::AbstractString)
    open(path) do io
        header = split(lowercase(readline(io)))
        (length(header) == 5 && header[1] == "%%matrixmarket" && header[2] == "matrix" &&
         header[3] == "coordinate") || error("$path: not a MatrixMarket coordinate file")
        field, symmetry = header[4], header[5]
        field in ("real", "double", "integer", "pattern") ||
            error("$path: unsupported MatrixMarket field \"$field\"")
        symmetry in ("general", "symmetric", "skew-symmetric") ||
            error("$path: unsupported MatrixMarket symmetry \"$symmetry\"")
        line = readline(io)
        while startswith(line, '%') || isempty(strip(line))
            line = readline(io)
        end
        m, n, nz = parse.(Int, split(line))
        I = Vector{Int}(undef, nz)
        J = Vector{Int}(undef, nz)
        V = Vector{Float64}(undef, nz)
        k = 0
        while k < nz
            line = readline(io)
            (startswith(line, '%') || isempty(strip(line))) && continue
            k += 1
            parts = split(line)
            I[k] = parse(Int, parts[1])
            J[k] = parse(Int, parts[2])
            V[k] = field == "pattern" ? 1.0 : parse(Float64, parts[3])
        end
        if symmetry != "general"
            off = findall(I .!= J)
            s = symmetry == "symmetric" ? 1.0 : -1.0
            I, J, V = vcat(I, J[off]), vcat(J, I[off]), vcat(V, s .* V[off])
        end
        return sparse(I, J, V, m, n)
    end
end

"""
    write_mtx(path, A; symmetric = false)

Writes `A` as a MatrixMarket coordinate `real` file. With `symmetric = true`
only the lower triangle is written (`A` must be symmetric).
"""
function write_mtx(path::AbstractString, A::SparseMatrixCSC; symmetric::Bool = false)
    B = symmetric ? tril(A) : A
    rows, vals = rowvals(B), nonzeros(B)
    open(path, "w") do io
        println(io, "%%MatrixMarket matrix coordinate real ", symmetric ? "symmetric" : "general")
        println(io, size(B, 1), " ", size(B, 2), " ", nnz(B))
        for j in axes(B, 2), k in nzrange(B, j)
            println(io, rows[k], " ", j, " ", repr(Float64(real(vals[k]))))
        end
    end
    return path
end

# ---------------------------------------------------------------------------
# MadNLP KKT dumps: bench/data/kkt_<case>_<kind>_<iter>.mtx (or .jld2)

const DUMP_KINDS = ("k2", "condensed")
const DUMP_REGEX = r"^kkt_(.+)_(k2|condensed)_(\d+)\.(mtx|jld2)$"

"""
    dump_name(case, kind, iter; ext = "mtx")

File name of a KKT dump: `kkt_<case>_<kind>_<iter>.<ext>`, `kind ∈ ("k2", "condensed")`.
"""
function dump_name(case::AbstractString, kind::AbstractString, iter::Integer; ext::AbstractString = "mtx")
    kind in DUMP_KINDS || throw(ArgumentError("kind must be one of $DUMP_KINDS, got \"$kind\""))
    return "kkt_$(case)_$(kind)_$(iter).$(ext)"
end

"""
    parse_dump_name(filename) -> NamedTuple or nothing

Parses `kkt_<case>_<kind>_<iter>.<mtx|jld2>` into `(case, kind, iter, ext)`;
`nothing` for names that do not follow the convention.
"""
function parse_dump_name(filename::AbstractString)
    m = match(DUMP_REGEX, basename(filename))
    m === nothing && return nothing
    return (case = String(m[1]), kind = String(m[2]), iter = parse(Int, m[3]), ext = String(m[4]))
end

"""
    symmetrize_triangle(T) -> SparseMatrixCSC{Float64,Int}

Full symmetric matrix from its stored lower or upper triangle `T`, keeping
explicitly stored zeros (the pattern of a KKT matrix must not depend on the
values of one iteration).
"""
function symmetrize_triangle(T::SparseMatrixCSC)
    I, J, V = findnz(T)
    off = I .!= J
    return sparse(vcat(I, J[off]), vcat(J, I[off]), Float64.(vcat(V, V[off])), size(T)...)
end

# K2 systems are symmetric indefinite; condensed systems are SPD by construction
# (MadNLP still often factorizes them with LDLᵀ, so both are benchmarked).
_dump_structures(kind) = kind == "condensed" ? ["SPD", "S"] : ["S"]

function _load_jld2(path)
    Base.find_package("JLD2") === nothing &&
        error("$path: loading .jld2 dumps needs JLD2.jl in the bench environment")
    JLD2 = Base.require(Base.PkgId(Base.UUID("033835bb-8acc-5ee8-8aae-3f567f8a3819"), "JLD2"))
    return Base.invokelatest(JLD2.load, path, "A")
end

"""
    dump_matrices(dir = bench/data) -> Vector{BenchMatrix}

Loads every file in `dir` that follows the naming convention of
[`parse_dump_name`](@ref), sorted by name. `.mtx` files are read with
[`read_mtx`](@ref); `.jld2` files need JLD2.jl and must hold the matrix under
the key `"A"`. A stored triangle is expanded to the full symmetric matrix.
"""
function dump_matrices(dir::AbstractString = DATA_DIR)
    isdir(dir) || return BenchMatrix[]
    out = BenchMatrix[]
    for file in sort(readdir(dir))
        meta = parse_dump_name(file)
        meta === nothing && continue
        path = joinpath(dir, file)
        A = meta.ext == "mtx" ? read_mtx(path) : _full(_load_jld2(path))
        if istril(A) || istriu(A)  # a stored triangle of a symmetric matrix
            A = symmetrize_triangle(A)
        end
        push!(out, BenchMatrix(first(splitext(file)), :dump, _dump_structures(meta.kind), _full(A)))
    end
    return out
end

"""
    bench_matrices(; generated = true, suitesparse = true, dumps = true) -> Vector{BenchMatrix}

All benchmark matrices available in this environment.
"""
function bench_matrices(; generated::Bool = true, suitesparse::Bool = true, dumps::Bool = true)
    out = BenchMatrix[]
    generated && append!(out, generated_matrices())
    suitesparse && append!(out, suitesparse_matrices())
    dumps && append!(out, dump_matrices())
    return out
end

end # module BenchMatrices

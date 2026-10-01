# Prints baseline CSV files as aligned tables.
#
#   julia --project=bench bench/report.jl [file.csv ...]
#
# Without arguments, every CSV under bench/results/ is printed.

using DelimitedFiles

"""
    print_report(io, path)

Prints the CSV at `path` (written by `cudss_baseline.jl`) as a table, timings
in milliseconds.
"""
function print_report(io::IO, path::AbstractString)
    data, header = readdlm(path, ',', String; header = true)
    header = vec(header)
    cols = ["matrix", "structure", "n", "nnz", "analysis_s", "factorization_s",
            "refactorization_s", "solve_s", "lu_nnz", "flops", "nsuperpanels", "relres", "status"]
    idx = [findfirst(==(c), header) for c in cols]
    any(isnothing, idx) && error("$path: missing columns $(cols[isnothing.(idx)])")
    titles = [replace(c, "_s" => " [ms]") for c in cols]
    cells = Matrix{String}(undef, size(data, 1), length(cols))
    for i in axes(data, 1), (j, c) in enumerate(cols)
        v = data[i, idx[j]]
        cells[i, j] = if endswith(c, "_s") && !isempty(v)
            string(round(parse(Float64, v) * 1e3; sigdigits = 4))
        elseif c in ("flops", "relres") && !isempty(v)
            string(round(parse(Float64, v); sigdigits = 3))
        else
            v
        end
    end
    widths = [max(length(titles[j]), maximum(length, cells[:, j]; init = 0)) for j in eachindex(cols)]
    solver = size(data, 1) > 0 ? data[1, findfirst(==("solver"), header)] : "?"
    println(io, "## ", basename(path), " (solver: ", solver, ")")
    println(io, "| ", join((rpad(t, w) for (t, w) in zip(titles, widths)), " | "), " |")
    println(io, "|", join(("-"^(w + 2) for w in widths), "|"), "|")
    for i in axes(cells, 1)
        println(io, "| ", join((rpad(cells[i, j], widths[j]) for j in eachindex(cols)), " | "), " |")
    end
    println(io)
end

function main(args = ARGS)
    dir = joinpath(@__DIR__, "results")
    files = !isempty(args) ? args :
            isdir(dir) ? sort(filter(f -> endswith(f, ".csv"), readdir(dir; join = true))) : String[]
    isempty(files) && println("no CSV files; run bench/cudss_baseline.jl first")
    foreach(f -> print_report(stdout, f), files)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

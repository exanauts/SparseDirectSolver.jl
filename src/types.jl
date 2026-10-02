# Enumerations behind the CUDSS.jl strings and characters (PLAN §1.1–§1.4, §1.7).
#
# Integer values follow the cuDSS 0.8 C enums wherever cuDSS has a counterpart,
# so a future C shim can pass them through unchanged. Every enum has a table of
# spellings (`enum_spellings`) that drives both conversion directions:
#
#     convert(Structure, "SPD") === STRUCTURE_SPD      convert(String, STRUCTURE_SPD) == "SPD"
#     convert(MatrixView, 'L')  === VIEW_LOWER         convert(Char, VIEW_LOWER) == 'L'
#
# Unknown spellings raise `InvalidValueError`.

"""
    Structure

Matrix structure, `"G"`, `"S"`, `"H"`, `"SPD"` or `"HPD"` (cuDSS `cudssMatrixType_t`).
"""
@enum Structure::Int32 begin
    STRUCTURE_GENERAL = 0
    STRUCTURE_SYMMETRIC = 1
    STRUCTURE_HERMITIAN = 2
    STRUCTURE_SPD = 3
    STRUCTURE_HPD = 4
end

"""
    MatrixView

Which part of the matrix is read: `'F'` full, `'L'` lower, `'U'` upper triangle
(cuDSS `cudssMatrixViewType_t`).
"""
@enum MatrixView::Int32 begin
    VIEW_FULL = 0
    VIEW_LOWER = 1
    VIEW_UPPER = 2
end

"""
    IndexBase

Index base of `rowPtr`/`colVal`: `'Z'` zero-based, `'O'` one-based (cuDSS `cudssIndexBase_t`).
"""
@enum IndexBase::Int32 begin
    INDEX_ZERO = 0
    INDEX_ONE = 1
end

"""
    Phase

Execution phases with the cuDSS bit values (`cudssPhase_t`). Composite phases
are the bitwise OR of their parts: `"analysis"` = reordering | symbolic
factorization, `"solve"` = all six solve sub-phases, and the CUDSS.jl
shorthands `"solve_fwd_schur"` = fwd_perm | fwd and `"solve_bwd_schur"` =
bwd | bwd_perm. See [`phase_includes`](@ref).
"""
@enum Phase::Int32 begin
    PHASE_REORDERING = 1
    PHASE_SYMBOLIC_FACTORIZATION = 2
    PHASE_ANALYSIS = 3
    PHASE_FACTORIZATION = 4
    PHASE_REFACTORIZATION = 8
    PHASE_SOLVE_FWD_PERM = 16
    PHASE_SOLVE_FWD = 32
    PHASE_SOLVE_FWD_SCHUR = 48
    PHASE_SOLVE_DIAG = 64
    PHASE_SOLVE_BWD = 128
    PHASE_SOLVE_BWD_SCHUR = 384
    PHASE_SOLVE_BWD_PERM = 256
    PHASE_SOLVE_REFINEMENT = 512
    PHASE_SOLVE = 1008
end

"""
    phase_includes(phase::Phase, part::Phase) -> Bool

Whether the bits of `part` are all set in `phase`, e.g.
`phase_includes(PHASE_SOLVE, PHASE_SOLVE_DIAG) == true`.
"""
phase_includes(phase::Phase, part::Phase) = (Int32(phase) & Int32(part)) == Int32(part)

"""
    PivotType

`pivot_type` values (cuDSS `cudssPivotType_t`): `'A'` auto, `'N'` none, `'C'`/`'R'`
global column/row pivoting (not supported, PLAN §3.3), `'D'` diagonal, `'L'`
local block, `'B'` Bunch–Kaufman 1×1/2×2.
"""
@enum PivotType::Int32 begin
    PIVOT_AUTO = 0
    PIVOT_NONE = 1
    PIVOT_GLOBAL_COL = 2
    PIVOT_GLOBAL_ROW = 3
    PIVOT_DIAGONAL = 4
    PIVOT_LOCAL_BLOCK = 5
    PIVOT_BUNCH_KAUFMAN = 6
end

"""
    ReorderingAlg

`reordering_alg` values (cuDSS `cudssReorderingAlg_t`): `"default"` automatic
AMD/ND choice, `"algo1"` BTF+COLAMD and `"algo2"` COLAMD (both fall back to the
symmetric-pattern path), `"algo3"` AMD, `"algo4"` nested dissection, `"algo5"`
natural ordering.
"""
@enum ReorderingAlg::Int32 begin
    REORDERING_DEFAULT = 0
    REORDERING_BTF_COLAMD = 1
    REORDERING_COLAMD = 2
    REORDERING_AMD = 3
    REORDERING_ND = 4
    REORDERING_NATURAL = 5
end

"""
    FactorizationAlg

`factorization_alg` values (cuDSS `cudssFactorizationAlg_t`, reinterpreted,
PLAN §1.3): `"default"` automatic, `"algo1"` very-sparse-factor path (regimes
A/B only, no vendor calls), `"algo2"` vendor dense calls above the subtree regime.
"""
@enum FactorizationAlg::Int32 begin
    FACTORIZATION_DEFAULT = 0
    FACTORIZATION_VERY_SPARSE = 1
    FACTORIZATION_VENDOR = 2
end

"""
    SolveAlg

`solve_alg` values (cuDSS `cudssSolveAlg_t`, reinterpreted, PLAN §1.3):
`"default"` level-batched sweeps, `"algo1"` partitioned-inverse diagonal blocks.
"""
@enum SolveAlg::Int32 begin
    SOLVE_DEFAULT = 0
    SOLVE_PARTITIONED_INVERSE = 1
end

"""
    MatchingAlg

`matching_alg` values (cuDSS `cudssMatchingAlg_t`): `"default"` no matching,
`"algo1"`–`"algo5"` MC64 jobs 1–5, `"algo6"` automatic (job 5).
"""
@enum MatchingAlg::Int32 begin
    MATCHING_NONE = 0
    MATCHING_MAX_DIAG_COUNT = 1
    MATCHING_MAX_MIN_DIAG = 2
    MATCHING_MAX_MIN_DIAG_ALT = 3
    MATCHING_MAX_DIAG_SUM = 4
    MATCHING_MAX_DIAG_PRODUCT = 5
    MATCHING_AUTO = 6
end

"""
    PivotEpsilonAlg

`pivot_epsilon_alg` values (cuDSS `cudssPivotEpsilonAlg_t`): `"default"`,
`"algo1"` scaled perturbation, `"algo2"` static perturbation.
"""
@enum PivotEpsilonAlg::Int32 begin
    PIVOT_EPSILON_DEFAULT = 0
    PIVOT_EPSILON_SCALED = 1
    PIVOT_EPSILON_STATIC = 2
end

"""
    ScheduleKind

`schedule` values (PLAN §1.7): `"auto"`, `"subtree+level"` (portable baseline),
`"syncfree"` (CUDA/ROCm only).
"""
@enum ScheduleKind::Int32 begin
    SCHEDULE_AUTO = 0
    SCHEDULE_SUBTREE_LEVEL = 1
    SCHEDULE_SYNCFREE = 2
end

"""
    IRMode

`ir_mode` values (PLAN §1.7): `"ir"` plain iterative refinement, `"fgmres"`
FGMRES with the factorization as preconditioner.
"""
@enum IRMode::Int32 begin
    IR_PLAIN = 0
    IR_FGMRES = 1
end

"""
    PivotPairsMode

`pivot_pairs` values (beyond cuDSS), for the analysis of `"S"`/`"H"` matrices:
`"default"` (2×2 pivot pairs for the candidates whose pivot is structurally zero
in the ordering, see [`zero_pivot_pairs!`](@ref)), `"all"` (every candidate paired, see
[`pivot_pairs`](@ref)) and `"none"`.
"""
@enum PivotPairsMode::Int32 begin
    PIVOT_PAIRS_DEFAULT = 0
    PIVOT_PAIRS_NONE = 1
    PIVOT_PAIRS_ALL = 2
end

"""
    enum_spellings(E) -> Tuple{Vararg{Pair}}

The CUDSS.jl spellings of every instance of the enum type `E`, as
`spelling => instance` pairs. Spellings are `String`s, except for
[`MatrixView`](@ref), [`IndexBase`](@ref) and [`PivotType`](@ref), which use `Char`s.
"""
function enum_spellings end

enum_spellings(::Type{Structure}) = (
    "G" => STRUCTURE_GENERAL,
    "S" => STRUCTURE_SYMMETRIC,
    "H" => STRUCTURE_HERMITIAN,
    "SPD" => STRUCTURE_SPD,
    "HPD" => STRUCTURE_HPD,
)

enum_spellings(::Type{MatrixView}) = ('F' => VIEW_FULL, 'L' => VIEW_LOWER, 'U' => VIEW_UPPER)

enum_spellings(::Type{IndexBase}) = ('Z' => INDEX_ZERO, 'O' => INDEX_ONE)

enum_spellings(::Type{Phase}) = (
    "reordering" => PHASE_REORDERING,
    "symbolic_factorization" => PHASE_SYMBOLIC_FACTORIZATION,
    "analysis" => PHASE_ANALYSIS,
    "factorization" => PHASE_FACTORIZATION,
    "refactorization" => PHASE_REFACTORIZATION,
    "solve_fwd_perm" => PHASE_SOLVE_FWD_PERM,
    "solve_fwd" => PHASE_SOLVE_FWD,
    "solve_fwd_schur" => PHASE_SOLVE_FWD_SCHUR,
    "solve_diag" => PHASE_SOLVE_DIAG,
    "solve_bwd_schur" => PHASE_SOLVE_BWD_SCHUR,
    "solve_bwd" => PHASE_SOLVE_BWD,
    "solve_bwd_perm" => PHASE_SOLVE_BWD_PERM,
    "solve_refinement" => PHASE_SOLVE_REFINEMENT,
    "solve" => PHASE_SOLVE,
)

enum_spellings(::Type{PivotType}) = (
    'A' => PIVOT_AUTO,
    'N' => PIVOT_NONE,
    'C' => PIVOT_GLOBAL_COL,
    'R' => PIVOT_GLOBAL_ROW,
    'D' => PIVOT_DIAGONAL,
    'L' => PIVOT_LOCAL_BLOCK,
    'B' => PIVOT_BUNCH_KAUFMAN,
)

enum_spellings(::Type{ReorderingAlg}) = (
    "default" => REORDERING_DEFAULT,
    "algo1" => REORDERING_BTF_COLAMD,
    "algo2" => REORDERING_COLAMD,
    "algo3" => REORDERING_AMD,
    "algo4" => REORDERING_ND,
    "algo5" => REORDERING_NATURAL,
)

enum_spellings(::Type{FactorizationAlg}) = (
    "default" => FACTORIZATION_DEFAULT,
    "algo1" => FACTORIZATION_VERY_SPARSE,
    "algo2" => FACTORIZATION_VENDOR,
)

enum_spellings(::Type{SolveAlg}) = ("default" => SOLVE_DEFAULT, "algo1" => SOLVE_PARTITIONED_INVERSE)

enum_spellings(::Type{MatchingAlg}) = (
    "default" => MATCHING_NONE,
    "algo1" => MATCHING_MAX_DIAG_COUNT,
    "algo2" => MATCHING_MAX_MIN_DIAG,
    "algo3" => MATCHING_MAX_MIN_DIAG_ALT,
    "algo4" => MATCHING_MAX_DIAG_SUM,
    "algo5" => MATCHING_MAX_DIAG_PRODUCT,
    "algo6" => MATCHING_AUTO,
)

enum_spellings(::Type{PivotEpsilonAlg}) = (
    "default" => PIVOT_EPSILON_DEFAULT,
    "algo1" => PIVOT_EPSILON_SCALED,
    "algo2" => PIVOT_EPSILON_STATIC,
)

enum_spellings(::Type{ScheduleKind}) = (
    "auto" => SCHEDULE_AUTO,
    "subtree+level" => SCHEDULE_SUBTREE_LEVEL,
    "syncfree" => SCHEDULE_SYNCFREE,
)

enum_spellings(::Type{IRMode}) = ("ir" => IR_PLAIN, "fgmres" => IR_FGMRES)

enum_spellings(::Type{PivotPairsMode}) =
    ("default" => PIVOT_PAIRS_DEFAULT, "none" => PIVOT_PAIRS_NONE, "all" => PIVOT_PAIRS_ALL)

"""
    StringEnum

Union of the enums spelled with strings (`convert(E, ::AbstractString)`, `convert(String, x)`).
"""
const StringEnum = Union{
    Structure, Phase, ReorderingAlg, FactorizationAlg, SolveAlg, MatchingAlg,
    PivotEpsilonAlg, ScheduleKind, IRMode, PivotPairsMode,
}

"""
    CharEnum

Union of the enums spelled with characters (`convert(E, ::AbstractChar)`, `convert(Char, x)`).
"""
const CharEnum = Union{MatrixView, IndexBase, PivotType}

"""
    AlgorithmEnum

Union of the enums set through `"default"`/`"algoN"` strings (or the integer `N`).
"""
const AlgorithmEnum = Union{ReorderingAlg, FactorizationAlg, SolveAlg, MatchingAlg, PivotEpsilonAlg}

# Human-readable names used in error messages.
enum_description(::Type{Structure}) = "structure"
enum_description(::Type{MatrixView}) = "view"
enum_description(::Type{IndexBase}) = "index base"
enum_description(::Type{Phase}) = "phase"
enum_description(::Type{PivotType}) = "pivot type"
enum_description(::Type{ReorderingAlg}) = "reordering algorithm"
enum_description(::Type{FactorizationAlg}) = "factorization algorithm"
enum_description(::Type{SolveAlg}) = "solve algorithm"
enum_description(::Type{MatchingAlg}) = "matching algorithm"
enum_description(::Type{PivotEpsilonAlg}) = "pivot epsilon algorithm"
enum_description(::Type{ScheduleKind}) = "schedule"
enum_description(::Type{IRMode}) = "refinement mode"
enum_description(::Type{PivotPairsMode}) = "pivot pairs mode"

function _unknown_spelling(::Type{E}, x) where {E}
    expected = join((repr(first(p)) for p in enum_spellings(E)), ", ")
    return InvalidValueError("unknown $(enum_description(E)) $(repr(x)); expected one of $expected")
end

function _parse_spelling(::Type{E}, x) where {E}
    for (spelling, instance) in enum_spellings(E)
        spelling == x && return instance
    end
    throw(_unknown_spelling(E, x))
end

function _spelling(x::E) where {E}
    for (spelling, instance) in enum_spellings(E)
        instance == x && return spelling
    end
    throw(ArgumentError("$x has no spelling"))  # unreachable: every instance is listed
end

Base.convert(::Type{E}, x::AbstractString) where {E <: StringEnum} = _parse_spelling(E, x)
Base.convert(::Type{E}, x::AbstractChar) where {E <: CharEnum} = _parse_spelling(E, x)
Base.convert(::Type{String}, x::StringEnum) = _spelling(x)
Base.convert(::Type{Char}, x::CharEnum) = _spelling(x)

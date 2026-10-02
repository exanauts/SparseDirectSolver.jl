# Logging and the host interrupt flag (PLAN §1.4 `user_host_interrupt`).
#
# Messages go through Julia's logging: phase summaries and refinement details
# are emitted with `@info` when the environment variable `SDS_LOG_LEVEL` (read
# when the package is loaded, or set later with `set_log_level!`) asks for
# them, and with `@debug` otherwise (visible with `JULIA_DEBUG=SparseDirectSolver`).
# Messages are built only when they are emitted.

const LOG_NONE = 0
const LOG_INFO = 1
const LOG_DEBUG = 2

const LOG_LEVEL = Ref(LOG_NONE)

function _parse_log_level(value)
    s = lowercase(strip(string(value)))
    s in ("", "0", "none", "off", "warn") && return LOG_NONE
    s in ("1", "info") && return LOG_INFO
    s in ("2", "debug") && return LOG_DEBUG
    throw(InvalidValueError("invalid log level $(repr(value)); expected 0/\"none\", 1/\"info\" or 2/\"debug\""))
end

"""
    SparseDirectSolver.set_log_level!(level) -> Int

Set the verbosity of the solver's log messages: `0`/`"none"` (default; the
messages are `@debug` records), `1`/`"info"` (phase summaries: analysis,
factorization, refinement steps performed, as `@info`), `2`/`"debug"` (also the
relative residual of every refinement step). The initial value comes from the
environment variable `SDS_LOG_LEVEL`. Returns the previous level.
"""
function set_log_level!(level)
    old = LOG_LEVEL[]
    LOG_LEVEL[] = level isa Integer ? _parse_log_level(string(level)) : _parse_log_level(level)
    return old
end

function _init_log_level()
    try
        LOG_LEVEL[] = _parse_log_level(get(ENV, "SDS_LOG_LEVEL", ""))
    catch err
        err isa InvalidValueError || rethrow()
        @warn "SDS_LOG_LEVEL: $(err.msg); logging stays off"
        LOG_LEVEL[] = LOG_NONE
    end
    return nothing
end

# emit `msg()` at `level` (LOG_INFO or LOG_DEBUG)
function _log(level::Int, msg)
    if LOG_LEVEL[] >= level
        @info msg()
    else
        @debug msg()
    end
    return nothing
end

# raise InterruptedError when the user's flag is set (a host read, no device synchronization)
@inline function _poll_interrupt(flag::Union{Nothing, Threads.Atomic{Bool}})
    flag !== nothing && flag[] && throw(InterruptedError())
    return nothing
end

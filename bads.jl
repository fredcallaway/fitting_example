using JSON3

Optional{T} = Union{Nothing,T}

struct BadsConfig
    x0::Vector{Float64}
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    plausible_lower_bounds::Vector{Float64}
    plausible_upper_bounds::Vector{Float64}
    options::Dict{Any,Any}
end

fill_missing(a, b) = ismissing(a) ? (b, b) : ismissing(b) ? (a, a) : (a, b)

function BadsConfig(;x0=missing, lower_bounds=missing, upper_bounds=missing,
                     plausible_lower_bounds=missing, plausible_upper_bounds=missing, options...)
    lower_bounds, plausible_lower_bounds = fill_missing(lower_bounds, plausible_lower_bounds)
    upper_bounds, plausible_upper_bounds = fill_missing(upper_bounds, plausible_upper_bounds)
    if ismissing(x0)
        x0 = (plausible_lower_bounds .+ plausible_upper_bounds) ./ 2
    end
    BadsConfig(x0, lower_bounds, upper_bounds, plausible_lower_bounds, plausible_upper_bounds, Dict(options))
end

mutable struct BadsProcess
    stdin::Pipe
    stdout::Pipe
    proc::Base.Process
    lines::Vector{String}
    status::Symbol
end

function BadsProcess(conf::BadsConfig)
    stdin = Pipe()
    stdout = Pipe()
    conf = JSON3.write(conf; allow_inf=true)
    proc = run(pipeline(`python bads.py $conf`; stdin, stdout, stderr), wait = false)
    process_running(proc) || error("There was a problem.")
    BadsProcess(stdin, stdout, proc, String[], :active)
end

function get_next_request!(bp::BadsProcess)
    bp.status == :active || return nothing
    while process_running(bp.proc)
        line = readline(bp.stdout)
        push!(bp.lines, line)
        if startswith(line, "FINAL_RESULT")
            bp.status = :done
            return nothing
        elseif startswith(line, "REQUEST_EVALUATION")
            return float.(collect(JSON3.read(bp.lines[end][20:end])))
        elseif startswith(line, "EXCEPTION")
            bp.status = :error
            error("Exception raised in pybads. See error message above.")
        else
            println(line)
        end
    end
    bp.status = :error
    return nothing
end

struct BADS
    config::BadsConfig
    process::BadsProcess
    xs::Vector{Vector{Float64}}
    ys::Vector{Float64}
    σs::Vector{Float64}
end

BADS(config::BadsConfig) = BADS(config, BadsProcess(config), [], [], [])
BADS(;config...) = BADS(BadsConfig(;config...))

status(bads::BADS) = bads.process.status
active(bads::BADS) = status(bads) == :active

function ask(bads::BADS)
    if length(bads.xs) == length(bads.ys)
        req = get_next_request!(bads.process)
        if isnothing(req)
            return nothing
        else
            push!(bads.xs, req)
        end
    end
    return bads.xs[end]
end

function tell(bads::BADS, y)
    if length(bads.xs) == length(bads.ys)
        error("No active request")
    end
    @assert length(bads.xs) == length(bads.ys) + 1
    if length(y) == 2
        y, σ = y
        push!(bads.ys, y)
        push!(bads.σs, σ)
        write(bads.process.stdin, string([y, σ], "\n"))
    else
        push!(bads.ys, y)
        write(bads.process.stdin, string(y, "\n"))
    end
    ask(bads)
end

function parse_result(line::String)
    @assert line[1:12] == "FINAL_RESULT"
    JSON3.read(line[14:end])
end

function get_result(bads::BADS)
    status(bads) == :done || error("No result available: status is $(status(bads))")
    parse_result(bads.process.lines[end])
end


function optimize_bads(f::Function, bads::BADS)
    while active(bads)
        x = ask(bads)
        tell(bads, f(x))
    end
    bads
end
optimize_bads(f::Function, config::BadsConfig) = optimize_bads(f, BADS(config))
optimize_bads(f::Function; config...) = optimize_bads(f, BadsConfig(;config...))

#= EXAMPLE

optimize_bads(;x0 = zeros(2), plausible_lower_bounds=-10ones(2), plausible_upper_bounds=10ones(2)) do x
    sum((x .- [.2,3]) .^2) + 2randn()
end

config = BadsConfig(;
    x0 = zeros(2), plausible_lower_bounds=-10ones(2), plausible_upper_bounds=10ones(2),
    specify_target_noise=true, max_fun_evals = 200
)
optimize_bads(config) do x
    σ = 1.2 + sin(sum(x))
    y = sum((x .- [.2,3]) .^2) + σ * randn()
    y, σ
end

=#


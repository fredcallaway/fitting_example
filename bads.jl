using JSON3


struct BadsOptions
    x0::Vector{Float64}
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    plausible_lower_bounds::Vector{Float64}
    plausible_upper_bounds::Vector{Float64}
end


mutable struct BadsProcess
    stdin::Pipe
    stdout::Pipe
    proc::Base.Process
    lines::Vector{String}
    status::Symbol
end

function BadsProcess(conf::BadsOptions)
    stdin = Pipe()
    stdout = Pipe()
    conf = JSON3.write(conf; allow_inf=true)
    proc = run(pipeline(`python bads.py $conf`; stdin, stdout, stderr), wait = false)
    process_running(proc) || error("There was a problem.")
    BadsProcess(stdin, stdout, proc, String[], :active)
end

function get_next_request!(bp::BadsProcess)
    bp.status == :done && return nothing
    while process_running(bp.proc)
        line = readline(bp.stdout)
        push!(bp.lines, line)
        if startswith(line, "FINAL_RESULT")
            bp.status = :done
            return nothing
        elseif startswith(line, "REQUEST_EVALUATION")
            return float.(collect(JSON3.read(bp.lines[end][20:end])))
        end
    end
    bp.status = :error
    return nothing
end


struct BADS
    options::BadsOptions
    process::BadsProcess
    xs::Vector{Vector{Float64}}
    ys::Vector{Float64}
end

BADS(options::BadsOptions) = BADS(options, BadsProcess(options), [], [])
function BADS(;x0, lower_bounds=fill(-Inf, length(x0)), upper_bounds=fill(Inf, length(x0)),
              plausible_lower_bounds, plausible_upper_bounds)
    BADS(BadsOptions(x0, lower_bounds, upper_bounds, plausible_lower_bounds, plausible_upper_bounds))
end

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
    push!(bads.ys, y)
    write(bads.process.stdin, string(y, "\n"))
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


function optimize_bads(f::Function, opt::BadsOptions)
    bads = BADS(opt)
    while active(bads)
        x = ask(bads)
        tell(bads, f(x))
    end
    get_result(bads)
end

function optimize_bads(f::Function; x0, lower_bounds=fill(-Inf, length(x0)), upper_bounds=fill(Inf, length(x0)),
                       plausible_lower_bounds, plausible_upper_bounds)
    optimize_bads(f, BadsOptions(x0, lower_bounds, upper_bounds, plausible_lower_bounds, plausible_upper_bounds))
end

#= EXAMPLE

optimize_bads(;x0 = zeros(2), plausible_lower_bounds=-10ones(2), plausible_upper_bounds=10ones(2)) do x
    sum((x .- [.2,3]) .^2) + 2randn()
end

=#
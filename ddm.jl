using Random
using Distributions
using Base: @kwdef

@kwdef struct DDM
    d::Float64 = .0002
    σ::Float64 = .02
    threshold::Float64 = 1
end

function check_termination(dv, threshold)
    # hardcoding the termination criterion for small N makes it a lot faster
    if length(dv) == 2
        a, b = dv
        a - b > threshold && return 1
        b - a > threshold && return 2
        return 0
    elseif length(dv) == 3
        a, b, c = dv
        a - max(b, c) > threshold && return 1
        b - max(a, c) > threshold && return 2
        c - max(a, b) > threshold && return 3
        return 0
    else
        best, next = partialsortperm(dv, 1:2, rev=true)
        dv[best] - dv[next] > model.threshold && return best
        return 0
    end
end

function simulate(model::DDM, v::Vector{Float64}; maxt=100000, logger=(dv, t) -> nothing)
    N = length(v)
    noise = Normal(0, model.σ)
    drift = model.d * v
    dv = zeros(N)  # total accumulated evidence
    choice = 0
    for t in 1:maxt
        logger(dv, t)
        for i in 1:N
            dv[i] += drift[i] + rand(noise)
        end
        best, next = partialsortperm(dv, 1:2, rev=true)
        choice = check_termination(dv, model.threshold)
        if choice != 0
            return (choice, t)
        end
    end
    (0, -1)
end


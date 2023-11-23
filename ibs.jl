using SpecialFunctions: trigamma

mutable struct IBSEstimate{F}
    sample_hit::F
    k::Int
    logp::Float64
end
IBSEstimate(f::Function) = IBSEstimate(f, 1, 0.)

Distributions.var(est::IBSEstimate) = trigamma(1) - trigamma(est.k)
Distributions.mean(est::IBSEstimate) = est.logp


function sample_hit!(est::IBSEstimate)
    if est.sample_hit()
        true
    else
        est.logp -= 1 / (est.k)
        est.k += 1
        false
    end
end

function ibs(hit_samplers::Vector{<:Function}; repeats=1, min_logp=-Inf)
    total_logp = 0.
    total_var = 0.
    n_call = 0
    for i in 1:repeats
        unconverged = Set(IBSEstimate(f) for f in hit_samplers)
        converged_logp = 0.
        converged_var = 0.
        while !isempty(unconverged)
            unconverged_logp = 0.
            for est in unconverged
                n_call += 1
                if sample_hit!(est)
                    converged_logp += mean(est)
                    converged_var += var(est)
                    delete!(unconverged, est)
                else
                    unconverged_logp += mean(est)
                end
            end
            if converged_logp + unconverged_logp < min_logp
                return (logp=min_logp, std=missing, converged=false, n_call)
            end
        end
        total_logp += converged_logp
        total_var += converged_var
    end

    return (logp=total_logp / repeats, std=√total_var / repeats, converged=true, n_call)
end

function ibs(sample_hit::Function, data::AbstractArray; kws...)
    hit_samplers = map(data) do d
        () -> sample_hit(d)
    end
    ibs(hit_samplers; kws...)
end


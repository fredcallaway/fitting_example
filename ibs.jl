using SpecialFunctions: digamma, trigamma
import Statistics: mean, var


ibs_loglike(k) = digamma(1) - digamma(k)
ibs_var(k) = trigamma(1) - trigamma(k)

mutable struct IBSEstimate{F}
    sample_hit::F
    k::Int
end
IBSEstimate(f::Function) = IBSEstimate(f, 1)
mean(est::IBSEstimate) = ibs_loglike(est.k)
var(est::IBSEstimate) = ibs_var(est.k)


function sample_hit!(est::IBSEstimate)
    if est.sample_hit()
        true
    else
        est.k += 1
        false
    end
end

function ibs(hit_samplers::Vector{<:Function}; repeats=1, min_logp=-Inf)
    total_logp = 0.
    total_var = 0.
    n_call = 0
    for i in 1:repeats
        unconverged = Set(IBSEstimate.(hit_samplers))
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

using SharedArrays

function try_many(sample_hit::Function, n::Int)
    for i in 1:n
        if sample_hit()
            return (i, true)
        end
    end
    return (n, false)
end

function parallel_ibs(hit_samplers::Vector{<:Function}; repeats=1, min_logp=-Inf)
    total_logp = 0.
    total_var = 0.
    n_call = 0
    N = length(hit_samplers)
    for i in 1:repeats
        k = SharedArray{Int}(N)
        converged = SharedArray{Bool}(N)
        cur_logp = 0.
        cur_var = 0.
        unconverged = collect(1:N)
        while !isempty(unconverged)
            @sync @distributed for i in unconverged
                n, success = try_many(hit_samplers[i], 100)
                k[i] += n
                converged[i] = success
            end
            filter!(unconverged) do i
                !converged[i]
            end
            cur_logp = sum(ibs_loglike, k)
            cur_var = sum(ibs_var, k)
            if cur_logp < min_logp  #  + 2√cur_var
                return (logp=min_logp, std=missing, converged=false, n_call)
            end
        end
        n_call += sum(k)
        total_logp += cur_logp
        total_var += cur_var
    end

    return (logp=total_logp / repeats, std=√total_var / repeats, converged=true, n_call)
end

function ibs(sample_hit::Function, data::AbstractArray; parallel=false, kws...)
    hit_samplers = map(data) do d
        () -> sample_hit(d)
    end
    if parallel
        parallel_ibs(hit_samplers; kws...)
    else
        ibs(hit_samplers; kws...)
    end
end


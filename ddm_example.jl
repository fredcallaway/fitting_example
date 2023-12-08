using Distributed
@time @everywhere begin
    include("ddm.jl")
    include("box.jl")
    include("ibs.jl")

    using Sobol
    using ProgressMeter
    using Serialization


    struct Trial
        v::Vector{Float64}
        choice::Int
        rt::Int
    end

    function log_likelihood(model, trials::Vector{Trial}; kws...)
        mapreduce(+, trials) do trial
            log_likelihood(model, trial; kws...)
        end
    end

    struct LapseModel
        N::Int
        max_rt::Int
    end

    function LapseModel(trials)
        N = only(unique(length(t.v for t in trials)))
        max_rt = maximum(t.rt for t in trials)
        LapseModel(N, max_rt)
    end

    simulate(model::LapseModel) = (rand(1:model.N), rand(1:model.max_rt))

    function log_likelihood(model::LapseModel, t::Trial; rt_tol=1)
        n_rt_hits = length(max(1, t.rt - rt_tol):min(model.max_rt, t.rt + rt_tol))
        log(1 / model.N) + log(n_rt_hits / model.max_rt)
    end

    function log_likelihood(model::DDM, trials::Vector{Trial}; parallel=false, ε=.01, rt_tol=1, kws...)
        lapse = LapseModel(trials)
        min_logp = log_likelihood(lapse, trials; rt_tol)
        ibs(trials; parallel, min_logp, kws...) do t
            if rand() < ε
                choice, rt = simulate(lapse)
            else
                choice, rt = simulate(model, t.v; maxt=t.rt+rt_tol+1)  # stop simulating when we know it's a miss
            end
            choice == t.choice && abs(rt - t.rt) ≤ rt_tol
        end
    end
end

# %% --------
# generate some trials

Random.seed!(1)
model = DDM(d=.03, σ=.2,)
trials = map(1:500) do i
    v = randn(3)
    choice, rt = simulate(model, v)
    Trial(v, choice, rt)
end;

# %% --------
# define search space

box = Box(
    :d => (.01, .1),
    :σ => (.05, .5)
)

# %% --------
# grid search

params = grid(5, box)

like_grid = @showprogress pmap(params) do prm
    log_likelihood(DDM(;prm...), trials; rt_tol=1, repeats=3)
end

# serialize("results/like_grid", like_grid)
# like_grid = deserialize("results/like_grid")

using StatsPlots
L = getfield.(like_grid, :logp)
heatmap(L)
savefig("like_grid.png")
run(`open "like_grid.png"`)

mle = params[argmax(like_grid)]
model

# %% --------
# Bayesian Optimization

include("gp_min.jl")

# initialize with values drawn from sobol sequence (covers space better than grid)
initX = sobol(20, box)
init = @showprogress map(initX) do x
    model = DDM(;box(x)...)
    log_likelihood(model, trials; repeats=5)
end

# IBS gives a variance estimate, so we tell this to the Gaussian Process
lognoise = log(maximum(getfield.(init, :std)))
noisebounds = [lognoise, lognoise]

# use the seed values as the GP initialization points
y = getfield.(init, :logp)
init_Xy = (reduce(hcat, initX), -y)

result_gp = gp_minimize(length(box); iterations=180, verbose=true, init_Xy, noisebounds) do x
    model = DDM(;box(x)...)
    -log_likelihood(model, trials; repeats=5).logp
end

serialize("results/gp", result_gp)

mle = DDM(;box(result_gp.model_optimizer)...)
log_likelihood(mle, trials; repeats=50)
log_likelihood(model, trials; repeats=50)

# %% --------
# BADS
include("bads.jl")
model
@everywhere using Random
@everywhere Random.seed!(myid())
@time log_likelihood(DDM(d=.05, σ=.15), trials, parallel=true)
@time log_likelihood(DDM(d=.05, σ=.15), trials, parallel=false)

Random.seed!(1)
log_likelihood(DDM(d=.05, σ=.15), trials)

bads = optimize_bads(lower_bounds=lower(box), upper_bounds=upper(box), specify_target_noise=true, tol_fun=1, max_fun_evals=1000) do (d, σ)
    model = DDM(;d, σ)
    (;logp, std) = log_likelihood(model, trials; repeats=1)
    if ismissing(std)
        std = 1.
    end
    -logp, std
end

d, σ = get_result(bads)[:x]
DDM(;d, σ)


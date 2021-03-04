using Distributions
using SplitApplyCombine  # for invert and combinedims, very handy package
using Optim
using StatsFuns  # for logistic
using BlackBoxOptim  # for differential evolution
using Random

include("box.jl")
include("gp_min.jl")

# %% ==================== Define types ====================

struct Trial
    stimulus::Float64
    response::Float64
end

Base.@kwdef struct Model   # Base.@kwdef allows settitng values by key words, very useful
    intercept::Float64
    slope::Float64
    σ::Float64
end

# %% ==================== Simulate a dataset ====================

function simulate(model::Model, stimulus=randn())
    response = model.intercept + stimulus * model.slope + model.σ * randn()
    Trial(stimulus, response)
end

true_model = Model(intercept=1., slope=3., σ=2.)

trials = map(1:1000) do i
    stimulus = randn()
    simulate(true_model, stimulus)
end

#= 
Note regarding the do syntax: The above is equivalent to:

function _anonymous(i)
    stimulus = randn()
    simulate(true_model, stimulus)
end
map(_anonymous, 1:1000)

This is a super convenient syntax. Use it! Love it!
=#

# %% ==================== Define likelihood ====================

function log_likelihood(model::Model, trial::Trial)
    yhat = model.intercept + model.slope * trial.stimulus
    predictive = Normal(yhat, model.σ)
    logpdf(predictive, trial.response)
end

function log_likelihood(model::Model, trials::Vector{Trial})
    mapreduce(+, trials) do trial  # add up all the log likelihoods
        log_likelihood(model, trial)
    end
end

true_logp = log_likelihood(true_model, trials)

# %% ==================== Define search space ====================

space = Box(
    intercept = (-5, 5),
    slope = (-5, 5),
    σ = (.1, 10, :log)  # use log scaling when the parameter spans multiple orders of magnitude
)

space(zeros(3))  # lower bounds
space(ones(3))   # upper bounds

# %% ==================== Sanity check: Nelder Mead ====================

x0 = randn(3)
result_nm = optimize(x0) do x
    # this is the definition of the loss function
    squashed = logistic.(x)  # put in [0, 1]
    prm = space(squashed)  # rescale to be in our search space
    model = Model(;prm...)  # ;prm... means we pass the key/values in prm as keyword arguments
    -log_likelihood(model, trials)  # don't forget the negative because we are minimizing!
end

recovered_nm = Model(;space(logistic.(result_nm.minimizer))...)
@assert log_likelihood(recovered_nm, trials) ≥ true_logp

# %% ==================== Differential evolution ====================

# This method works well if the likelihood is not very noisy
# and you can afford a larger number of evaluations


result_de = bboptimize(; SearchRange = (0., 1.), NumDimensions = length(space), MaxFuncEvals=400) do x
    model = Model(;space(x)...)
    # NOTE: we need to scale the loss function so that a good value is around 1
    -log_likelihood(model, trials)
end
recovered_de = Model(;space(best_candidate(result_de))...)
logp_de = log_likelihood(recovered_de, trials)
logp_de - true_logp

# %% --------

# Try again with a noisy likelihood; it sometimes does well but sometimes doesn't
result_de_noisy = bboptimize(; SearchRange = (0., 1.), NumDimensions = length(space), MaxFuncEvals=400) do x
    model = Model(;space(x)...)
    -log_likelihood(model, trials) + 20randn()
end
recovered_de_noisy = Model(;space(best_candidate(result_de_noisy))...)
logp_de_noisy = log_likelihood(recovered_de_noisy, trials)
logp_de_noisy - true_logp

# %% ==================== Gaussian Process Optimization ====================

# Finally, using a GP. This method is in theory the best way to handle
# noisy objecive functions. The problem is that it's finicky. Sometimes
# the GP gets confused and you'll basically be doing random search (or worse).
# This can sometimes be diagnosed by looking at the GP after the fact (see below).

Random.seed!(1)

result_gp = gp_minimize(length(space); iterations=300, verbose=false) do x
    # we don't need to squash because the GP search is bounded already
    model = Model(;space(x)...)
    # NOTE: we need to scale the loss function so that a good value is around 1
    (-log_likelihood(model, trials) + 20randn()) / 1000
end

# There are two candidate parameter values: the one that empirically produced
# the best likelihood and the one that the GP model thinks is best. It's
# generally a good idea to check which one actually performs better using
# multiple runs to estimate the likelihood if it's stochastic.
function choose_optimum(result; repeats=10)
    loss(x) = mean(result.func(x) for i in 1:repeats)
    if loss(result.observed_optimizer) < loss(result.model_optimizer)
        println("Using observed optimum")
        return result.observed_optimizer
    else
        println("Using model optimum")
        return result.model_optimizer
    end
end

recovered_gp = Model(;space(choose_optimum(result_gp))...)
log_likelihood(recovered_gp, trials) - true_logp

# That... does not look great...


# %% ==================== Diagnostics ====================
using Plots
gr(lw=2, label="")  # set defaults

# What's going on? A first thing to check is convergence:
gp = result_gp.model
nll_trace = -gp.y * 1000
# WARNING: we have to negate gp.y again here because the package assumes
# maximization by default so it trains the GP to predict the negative of what 
# you give it.

plot(nll_trace, ylabel="Negative Log Likelihood", xlabel="Iteration")
hline!([-true_logp], color=:red)
# Notice that the likelihood is terrible for the first 75 iterations.
# This is because we start with pseudo-random (sobol) search. I use 1/4 of the total
# iterations for random search which is more than the standard recommendation
# but I've found that it makes the results more robust because you make sure to
# at least roughly search the entire space.

plot(76:300, nll_trace[76:end], ylabel="Negative Log Likelihood", xlabel="Iteration")
hline!([-true_logp], color=:red)
# That looks more reasonable.

# %% --------


#=
When things inevitably don't work, it's useful to look at what the GP learned.
The simplest thing to do is to compare the model optimizer to the observed optimizer.
The code below shows the predicted likelihood as a function of each parameter,
setting the other parameters to the model-inferred optimum.
=#

function plot_gp_result(space, result)
    xx = 0:.01:1
    x0 = copy(result.model_optimizer)
    plots = map(enumerate(pairs(space.dims))) do (i, (name, d))
        x = copy(x0)
        y, ystd = map(xx) do x_target
            x[i] = x_target
            f, fv = predict_f(result.model, reshape(x, result.model.dim, 1))
            -f[1], √fv[1]
        end |> invert
        maybelog = :log in d ? (:log,) : ()
        plot(rescale(d, xx), y, xaxis=(string(name), maybelog...), ribbon=ystd)

        # plot the true values; of course this wouldn't be possible
        # in the real setting
        vline!([getfield(true_model, name)], color=:red, ls=:dash)
        y_true = map(xx) do x_target
            x[i] = x_target
            -log_likelihood(Model(;space(x)...), trials) / 1000
        end
        plot!(rescale(d, xx), y_true, color=:red, ls=:dash)

    end
    plot(plots..., size=(600, 600))
end

plot_gp_result(space, result_gp)

#=
Assuming the random seed worked, you'll see that the GP captured the true
likelihood (plotted in red) pretty well, but that it still has a lot of
uncertainty about the parameters other than σ, which it has correctly learned
as having a massive effect on likelihood.

The problem here is that we specified too large of a search range for σ. For
one thing, time is being wasted trying bad parameters. But more importantly,
the extreme likelihood values can confuse the GP. 

So, if you see a result like this (just the blue of course), you would
want to raise the minimum value of σ. Be careful though; you should
check that you aren't fitting the search space to a single participant. In
general, you should only eliminate areas that have a terrible loglikelihood
(maybe around 10x the best one you find). If the areas that are good for
one participant are terrible for another... you're going to have a bad time.
=#
# %% --------

better_space = Box(
    intercept = (-5, 5),
    slope = (-5, 5),
    σ = (10^-0.5, 10, :log)
)
better_result = gp_minimize(length(better_space); iterations=300, verbose=false) do x
    model = Model(;better_space(x)...)
    (-log_likelihood(model, trials) + 20randn()) / 1000
end

recovered_better = Model(;better_space(choose_optimum(better_result))...)
log_likelihood(recovered_better, trials) - true_logp
# Much better! 

plot_gp_result(better_space, better_result)
# This is exactly what we want our GP to look like. Low uncertainty near
# the optimum and reasonable and similar scales on all the y axes.

# Another sign that things are going well is that the model and observed
# opimizers are very close. If the model optimizer is far from the observed
# optimizer AND the observed optimizer is actually better (when we recompute
# the likelihood) that often means that the GP is confused.
better_result.model_optimizer .- better_result.observed_optimizer

# %% ==================== Continuing an optimization ====================

# It is a good idea to save your result so that you can reuse those
# evaluations later if you want. In particular you can continue optimizing if
# you think you didn't use enough iterations the first time. Use `serialize`
# for this.
using Serialization
serialize("my_result", better_result)
result = deserialize("my_result")

init_Xy = (result.model.x, -result.model.y)  # don't forget the stupid negative
result_cont = gp_minimize(result.func, length(space);
    iterations=100, verbose=false, init_Xy) # same as init_Xy=init_Xy
# Note that you don't have to use result.func here, you could redefine the loss
# function and get the same result. This is just easier in this context.

recovered_cont = Model(;better_space(choose_optimum(result_cont))...)
log_likelihood(recovered_cont, trials) - true_logp
# Even better! Although at this point, it really doesn't matter.

# %% --------

# Bonus! We can combine the previous two ideas to try reducing the search space
# without throwing away all the work we've already done.

# First, we have to transform the previous x values into the new space
bad_result = result_gp  # just to be explicit
transformed_X = map(eachcol(bad_result.model.x)) do x
    new_x = better_space(space(x))
end |> combinedims
# This looks like magic because we're (ab)using function overloading. This
# should give you a clue what's happening:
@assert space(space([.1, .2, .3])) ≈ [.1, .2, .3]

# Second, we throw out the evaluations that are outside of the new space.
keep = map(eachcol(transformed_X)) do x
    all(0 .<= x .<= 1)
end
init_Xy = (transformed_X[:, keep], -bad_result.model.y[keep])

# Run 100 more iterations...
result_improved = gp_minimize(length(space); iterations=100, verbose=false, init_Xy) do x
    # We do have to redefine the loss function this time because we changed it
    # by changing the space!
    model = Model(;better_space(x)...)
    (-log_likelihood(model, trials) + 20randn()) / 1000
end

recovered_improved = Model(;better_space(choose_optimum(result_improved))...)
log_likelihood(recovered_improved, trials) - true_logp
log_likelihood(recovered_gp, trials) - true_logp  # for comparison

# Looking better! Might be great depending on random noise. If not, we could
# keep trying this until we got a satisfactory result. Note, however, that
# this is really only useful at the prototyping/exploration stage. Your final
# results should always be generated in one run so that they are reproducible.


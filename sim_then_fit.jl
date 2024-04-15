using Distributions
using StatsFuns: logistic, logsumexp
using Optim
include("box.jl")

struct DriftingHungerModel{D<:Distribution,T}
    drift::D
    frugality::T
    β::T
end

# trial/stimulus
struct Snack
    tastiness::Float64
    price::Float64
end

function diffusion(d::Distribution; init=0., n_step=10)
    z = zeros(n_step)
    z[1] = init
    for i in 2:n_step
        z[i] = z[i-1] + rand(d)
    end
    z
end

function prob_eat(hunger, frugality, β, snack)
    value = hunger + snack.tastiness - frugality * snack.price
    logistic(β * value)
end

function simulate(model::DriftingHungerModel, snacks::Vector{Snack})
    (;drift, frugality, β) = model
    hungers = diffusion(drift; n_step=length(snacks))
    map(hungers, snacks) do hunger, snack
        p = prob_eat(hunger, frugality, β, snack)
        rand(Bernoulli(p))
    end
end

snacks = map(1:200) do i
    Snack(rand(Normal(1, 2)), rand(Uniform(1,2)))
end

model = DriftingHungerModel(Normal(0.0, 0.01), 1.5, .5)
choices = simulate(model, snacks)

# %% ==================== fit the model ====================

function logmeanexp(x)
    logsumexp(x) + log(1/length(x))
end

function choice_likelihood(hunger, frugality, β, snack, choice)
    p = prob_eat(hunger, frugality, β, snack)
    pdf(Bernoulli(p), choice)
end

# let's assume we know the drift; then we can sample possible
# realizations of the latent hunger variable
known_drift = model.drift
n_sample = 1000
hunger_samples = map(1:1000) do i
    diffusion(known_drift; n_step=length(snacks))
end

# search boundaries
space = Box(
    frugality = (0., 3.),
    β = (0., 2.,)
)

x0 = zeros(n_free(space))  # initialize in center of space
res = optimize(x0) do x
    (;frugality, β) = space(logistic.(x))
    # we compute the log likelihood for each sampled instantiation of hunger
    logp_samples = map(hunger_samples) do hungers
        # compute the likelihood of the data given this particular instantation
        # and the parameters we're optimizing (frugality, β)
        mapreduce(+, hungers, snacks, choices) do hunger, snacks, choice
            log(choice_likelihood(hunger, frugality, β, snacks, choice))
        end
    end
    # we want to compute the expected log likelihood marginalizing over hunger
    # we estimate the expectation by monte carlo (averaging over samples)
    # for numerical stability, we do this in log space, hence logmeanexp
    -logmeanexp(logp_samples)
end
mle_prm = space(logistic.(res.minimizer))
println("true: ", (;model.frugality, model.β))
println("fitted: ", mle_prm)


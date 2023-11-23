using Distributions
include("ibs.jl")
# %% --------
N = 1000
res = map(Iterators.product(ps, ps)) do (data_p, model_p)
    n_true = Int(round(N * data_p))
    data = [trues(n_true); falses(N - n_true)]
    est = ibs(data; repeats=1) do d
        rand(Bernoulli(model_p)) == d
    end
    true_logp = loglikelihood(Bernoulli(model_p), data)
    (;est.logp, est.std, true_logp, data_p, model_p)
end

for x in res
    @assert abs(x.logp - x.true_logp) < 4x.std
end

@assert mean(res) do x
    (x.logp - x.true_logp) / x.true_logp
end < .01

# # %% --------
# N = 1000
# true_p = 0.5
# truth = Bernoulli(true_p)
# data = [rand(truth) for i in 1:N]
# p = 0.5

# ests = map(1:1000) do i
#     ibs(data; repeats=4) do d
#         rand(Bernoulli(p)) == d
#     end
# end

# using SplitApplyCombine
# lp, v = invert(ests)

# var(lp)
# mean(v)
# ============================================================================
# model_comparison.jl
#
# Likelihood-generic WAIC / LPML / PSIS-LOO for the real-data analyses.
#
# The package's `compute_waic` and `compute_lpml` are hard-coded to a Poisson
# observation model (they take a matrix of rate parameters λ, not log-densities),
# so they cannot score the Negative Binomial or SkewNormal fits. The functions
# here take a log-likelihood matrix instead and are therefore model-agnostic:
#
#   ll[s, i] = log p(y_i | θ^(s))     (n_draws × n_obs)
#
# built by each model's `loglik_matrix` method (see `negbin_population_model.jl`
# and `skewnormal_model.jl`). PSIS-LOO already accepts this form and is reused
# from the package as `compute_psis_loo`.
# ============================================================================

using DistanceDependentCRP, Distributions, Statistics, Printf, Random
import DistanceDependentCRP: posterior_predictive

"""
    waic_from_loglik(ll) -> NamedTuple

Watanabe–Akaike Information Criterion from a `(n_draws × n_obs)` log-likelihood
matrix. Lower is better.

    lppd   = Σ_i log mean_s p(y_i | θ^(s))
    p_WAIC = Σ_i Var_s log p(y_i | θ^(s))
    WAIC   = -2 (lppd - p_WAIC)

Same definition as the package's Poisson-only `compute_waic`, but driven by
log-densities so any likelihood can be scored.
"""
function waic_from_loglik(ll::AbstractMatrix{<:Real})
    n = size(ll, 2)
    lppd = 0.0
    p_waic = 0.0
    waic_i = Vector{Float64}(undef, n)
    for i in 1:n
        ll_i = @view ll[:, i]
        m = maximum(ll_i)
        lppd_i = m + log(mean(exp.(ll_i .- m)))
        pw_i = var(ll_i)
        waic_i[i] = -2.0 * (lppd_i - pw_i)
        lppd += lppd_i
        p_waic += pw_i
    end
    return (waic = -2.0 * (lppd - p_waic), lppd = lppd, p_waic = p_waic, waic_i = waic_i)
end

"""
    lpml_from_loglik(ll) -> Float64

Log Pseudo-Marginal Likelihood via the CPO harmonic-mean estimator. Higher is
better. Computed with log-sum-exp for stability.
"""
function lpml_from_loglik(ll::AbstractMatrix{<:Real})
    return sum(lcpo_from_loglik(ll))
end

"""
    lcpo_from_loglik(ll) -> Vector{Float64}

Per-observation log CPO, whose sum is the LPML.
"""
function lcpo_from_loglik(ll::AbstractMatrix{<:Real})
    lcpo = Vector{Float64}(undef, size(ll, 2))
    for i in 1:size(ll, 2)
        neg = -(@view ll[:, i])
        m = maximum(neg)
        lcpo[i] = -(m + log(mean(exp.(neg .- m))))
    end
    return lcpo
end

"""
    score_fit(label, ll) -> NamedTuple

WAIC, LPML and PSIS-LOO for one fit, plus the count of unreliable PSIS
diagnostics (`k̂ > 0.7`) — a large count means the LOO estimate should not be
trusted and WAIC/LPML should be read instead.
"""
function score_fit(label::AbstractString, ll::AbstractMatrix{<:Real})
    return first(score_fit_pointwise(label, ll))
end

"""
    score_fit_pointwise(label, ll) -> (scores, pointwise)

`score_fit` plus the per-observation contributions, from one pass over `ll`.
`pointwise` holds `elpd_waic_i` (= -waic_i/2), `elpd_loo_i`, `lcpo_i` and
`k_hat`, each of length n_obs. Comparing two fits needs these: the standard
error of an elpd difference is `sqrt(n * var(a_i - b_i))`, which the totals
alone cannot give.
"""
function score_fit_pointwise(label::AbstractString, ll::AbstractMatrix{<:Real})
    w = waic_from_loglik(ll)
    loo = compute_psis_loo(Matrix{Float64}(ll))
    lcpo = lcpo_from_loglik(ll)
    scores = (
        label = label,
        waic = w.waic,
        lppd = w.lppd,
        p_waic = w.p_waic,
        lpml = sum(lcpo),
        elpd_loo = loo.elpd_loo,
        n_bad_k = count(>(0.7), loo.k_hat),
    )
    pointwise = (elpd_waic_i = -w.waic_i ./ 2, elpd_loo_i = loo.loo_i,
                 lcpo_i = lcpo, k_hat = loo.k_hat)
    return scores, pointwise
end

"""
    elpd_difference(a, b) -> (diff, se)

Difference in total elpd between two fits (positive favours `a`) and its
standard error, from matched pointwise contributions.
"""
function elpd_difference(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    d = a .- b
    return (diff = sum(d), se = sqrt(length(d) * var(d)))
end

"""
    print_comparison(scores)

Print a comparison table, sorted best-first by WAIC, with ΔWAIC relative to the
best fit.
"""
function print_comparison(scores::AbstractVector)
    sorted = sort(collect(scores); by = s -> s.waic)
    best = sorted[1].waic
    @printf("%-38s %12s %10s %12s %12s %8s\n",
            "model", "WAIC", "ΔWAIC", "LPML", "elpd_loo", "k̂>0.7")
    println("-"^96)
    for s in sorted
        @printf("%-38s %12.1f %10.1f %12.1f %12.1f %8d\n",
                s.label, s.waic, s.waic - best, s.lpml, s.elpd_loo, s.n_bad_k)
    end
end

"""
    loglik_matrix(::PoissonPopulationRates, samples, data) -> Matrix

Log-likelihood matrix for the package's non-marginalised Poisson offset model,
so it can be scored on the same footing as the Negative Binomial fit.

Note the *marginalised* `PoissonPopulationRatesMarg` cannot be scored this way —
it stores no per-observation rate, λ having been integrated out. It is used as
the clustering benchmark (agreement in ARI), not in the likelihood comparison.
"""
function loglik_matrix(::PoissonPopulationRates,
        samples::PoissonPopulationRatesSamples, data)
    y = observations(data)
    P = population(data)
    S, n = size(samples.ρ)
    ll = Matrix{Float64}(undef, S, n)
    for i in 1:n
        E = P isa Real ? float(P) : float(P[i])
        for s in 1:S
            λ = E * samples.ρ[s, i]
            ll[s, i] = λ > 0 ? logpdf(Poisson(λ), y[i]) : -Inf
        end
    end
    return ll
end

"""
    posterior_predictive(::PoissonPopulationRates, samples, data; n_rep=nothing)

Replicate datasets from the non-marginalised Poisson offset model, so the same
posterior predictive checks can be run on it as on the Negative Binomial fit.
The package only ships a `posterior_predictive` for the *marginalised* Poisson
model, and that one imputes missing observations rather than generating full
replicates.
"""
function posterior_predictive(::PoissonPopulationRates,
        samples::PoissonPopulationRatesSamples, data; n_rep::Union{Nothing,Int} = nothing)
    P = population(data)
    S, n = size(samples.ρ)
    draws = isnothing(n_rep) ? (1:S) : round.(Int, range(1, S; length = min(n_rep, S)))
    y_rep = zeros(Int, length(draws), n)
    for (s_idx, s) in enumerate(draws), i in 1:n
        E = P isa Real ? float(P) : float(P[i])
        λ = E * samples.ρ[s, i]
        y_rep[s_idx, i] = λ > 0 ? rand(Poisson(λ)) : 0
    end
    return y_rep
end

"""
    ppc_pvalue(T_obs, T_rep) -> Float64

Posterior predictive p-value: the fraction of replicate datasets whose
discrepancy statistic is at least the observed one. Values near 0 or 1 flag a
statistic the model fails to reproduce.
"""
ppc_pvalue(T_obs::Real, T_rep::AbstractVector{<:Real}) = mean(T_rep .>= T_obs)

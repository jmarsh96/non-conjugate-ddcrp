# WAIC, LPML and PSIS-LOO from a log-likelihood matrix ll[s, i] = log p(y_i | θ^(s)).

using DistanceDependentCRP, Distributions, Statistics, Printf, Random
import DistanceDependentCRP: posterior_predictive

# WAIC = -2 (lppd - p_WAIC); lower is better.
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

# Per-observation log CPO (harmonic-mean estimator); the sum is the LPML.
function lcpo_from_loglik(ll::AbstractMatrix{<:Real})
    lcpo = Vector{Float64}(undef, size(ll, 2))
    for i in 1:size(ll, 2)
        neg = -(@view ll[:, i])
        m = maximum(neg)
        lcpo[i] = -(m + log(mean(exp.(neg .- m))))
    end
    return lcpo
end

# Scores for one fit plus the per-observation contributions needed for elpd differences.
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

# Difference in total elpd (positive favours `a`) and its standard error.
function elpd_difference(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    d = a .- b
    return (diff = sum(d), se = sqrt(length(d) * var(d)))
end

# Comparison table sorted by WAIC.
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

# Log-likelihood matrix for the non-marginalised Poisson offset model.
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

# Replicate datasets from the non-marginalised Poisson offset model.
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

# Posterior predictive p-value: fraction of replicates with statistic >= the observed one.
ppc_pvalue(T_obs::Real, T_rep::AbstractVector{<:Real}) = mean(T_rep .>= T_obs)

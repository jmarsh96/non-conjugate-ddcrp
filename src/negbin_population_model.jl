# ============================================================================
# negbin_population_model.jl
#
# Negative Binomial cluster model with population offsets for the ddCRP RJMCMC
# framework — a 2-parameter, FULLY non-conjugate count likelihood.
#
#   y_i | ρ_k, r_k, E_i ~ NegativeBinomial(mean = E_i ρ_k, size = r_k)
#                                                   for obs i in cluster k
#     E_i = P_i                       exposure/offset supplied by the data
#     ρ_k ~ Gamma(ρ_a, ρ_b)           cluster incidence rate, support (0, ∞)
#     r_k ~ Gamma(r_a, r_b)           cluster dispersion,     support (0, ∞)
#
# Var(y_i) = μ_i + μ_i²/r_k, so r_k → ∞ recovers the Poisson model. With r_k
# free, ρ_k loses its Gamma conjugacy, so BOTH cluster parameters are sampled
# explicitly: birth/death via RJMCMC, within-cluster via Metropolis–Hastings.
# This is the count-data analogue of `skewnormal_model.jl`, and the honest
# non-conjugate counterpart of the package's `PoissonPopulationRates(Marg)`.
#
# This file is PROTOTYPED IN THE ANALYSIS REPO (per the user's instruction) by
# adding methods to DistanceDependentCRP's exported generic functions via
# multiple dispatch; it is intended to be upstreamed into the package later.
# It follows the interface exercised by `poisson_population_rates.jl` (exposure
# offsets) and `skewnormal_model.jl` (multi-parameter non-conjugate clusters via
# the NamedTuple-of-dicts / MixedProposal machinery).
# ============================================================================

using DistanceDependentCRP
using Distributions, Statistics, StatsBase, Random

import DistanceDependentCRP: cluster_param_dicts, table_contribution, posterior,
    update_params!, sample_birth_params, birth_params_logpdf,
    sample_birth_param, birth_param_logpdf,
    initialise_state, allocate_samples, extract_samples!,
    requires_population, posterior_predictive

const DD = DistanceDependentCRP

# ============================================================================
# Types
# ============================================================================

"""
    NegBinPopulationRates <: LikelihoodModel

Negative Binomial model with exposure offsets and explicit cluster-specific
incidence (ρ) and dispersion (r) parameters. Fully non-conjugate — no baseline
Gibbs sampler exists. Requires `CountDataWithPopulation`.
"""
struct NegBinPopulationRates <: LikelihoodModel end

requires_population(::NegBinPopulationRates) = true

"""
    NegBinPopulationRatesState{T} <: AbstractMCMCState{T}

Fields: assignments `c`, and one `Dict{Vector{Int},T}` (sorted table → scalar)
per cluster parameter (`ρ_dict`, `r_dict`).
"""
mutable struct NegBinPopulationRatesState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    ρ_dict::Dict{Vector{Int}, T}
    r_dict::Dict{Vector{Int}, T}
end

"""
    NegBinPopulationRatesPriors{T} <: AbstractPriors

- `ρ_a, ρ_b`: Gamma(shape, rate) prior on the incidence rate ρ
- `r_a, r_b`: Gamma(shape, rate) prior on the dispersion r
"""
struct NegBinPopulationRatesPriors{T<:Real} <: AbstractPriors
    ρ_a::T; ρ_b::T
    r_a::T; r_b::T
end

function NegBinPopulationRatesPriors(;
    ρ_a::Real = 2.0, ρ_b::Real = 0.002,
    r_a::Real = 2.0, r_b::Real = 0.5,
)
    T = promote_type(typeof(ρ_a), typeof(ρ_b), typeof(r_a), typeof(r_b))
    NegBinPopulationRatesPriors{T}(T(ρ_a), T(ρ_b), T(r_a), T(r_b))
end

# Gamma(shape a, rate b) — Distributions parameterises by scale, hence 1/b.
ρ_prior(p::NegBinPopulationRatesPriors) = Gamma(p.ρ_a, 1 / p.ρ_b)
r_prior(p::NegBinPopulationRatesPriors) = Gamma(p.r_a, 1 / p.r_b)
ρ_prior_mean(p::NegBinPopulationRatesPriors) = p.ρ_a / p.ρ_b
r_prior_mean(p::NegBinPopulationRatesPriors) = p.r_a / p.r_b

"""
    NegBinPopulationRatesSamples{T} <: AbstractMCMCSamples

Per-observation matrices `ρ`, `r` (each obs stores its cluster's value), plus
`c`, `logpost`, `α_ddcrp`, `s_ddcrp`. The ρ/r matrices are auto-discovered by
`get_parameter_fields` so diagnostics (ESS, etc.) work with no extra wiring.
"""
struct NegBinPopulationRatesSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    ρ::Matrix{T}
    r::Matrix{T}
    logpost::Vector{T}
    α_ddcrp::Vector{T}
    s_ddcrp::Vector{T}
end

# ============================================================================
# Method-of-moments NegBin fit (used by moment-matching proposals + init)
# ============================================================================

"""
    nb_dist(μ, r) -> NegativeBinomial

Mean/dispersion parameterisation: `NegativeBinomial(r, p)` with `p = r/(r+μ)`
has mean `μ` and variance `μ + μ²/r`.
"""
nb_dist(μ::Real, r::Real) = NegativeBinomial(r, r / (r + μ))

"""
    fit_negbin_moments(ys, Es; min_size=3) -> (ρ, r) or nothing

Method-of-moments Negative Binomial estimate for a set of counts `ys` with
exposures `Es`:

  ρ̂ = Σy / ΣE,  μ̂_i = E_i ρ̂
  φ̂ = quasi-Poisson dispersion = (1/(m-1)) Σ (y_i - μ̂_i)² / μ̂_i
  r̂ = mean(μ̂) / (φ̂ - 1)          since Var = μ(1 + μ/r) ⇒ φ ≈ 1 + μ/r

Returns `nothing` when the set is too small, carries no exposure, or is
**underdispersed** (φ̂ ≤ 1) — in which case r̂ would diverge. Callers then fall
back to the prior, as recommended in the paper for small moving sets.
`r̂` is clamped to `[0.05, 500]`: beyond the upper end the likelihood is
numerically indistinguishable from Poisson.
"""
function fit_negbin_moments(ys, Es; min_size::Int = 3)
    m = length(ys)
    m < min_size && return nothing
    ΣE = sum(Es)
    (ΣE <= 0 || !isfinite(ΣE)) && return nothing

    ρ̂ = sum(ys) / ΣE
    (ρ̂ <= 0 || !isfinite(ρ̂)) && return nothing

    μ̂ = [E * ρ̂ for E in Es]
    any(<=(0), μ̂) && return nothing

    φ̂ = sum((y - μ)^2 / μ for (y, μ) in zip(ys, μ̂)) / (m - 1)
    (!isfinite(φ̂) || φ̂ <= 1) && return nothing        # underdispersed ⇒ no NB fit

    r̂ = mean(μ̂) / (φ̂ - 1)
    (!isfinite(r̂) || r̂ <= 0) && return nothing
    return (ρ̂, clamp(r̂, 0.05, 500.0))
end

# Fit restricted to a moving set / table, pulling counts and exposures from data.
function _fit_S(S_i, data, min_size::Int)
    y = observations(data)
    P = population(data)
    Es = P isa Real ? fill(float(P), length(S_i)) : [float(P[i]) for i in S_i]
    return fit_negbin_moments([float(y[i]) for i in S_i], Es; min_size = min_size)
end

# Exposure for a single observation, tolerating a scalar or per-observation P.
_exposure(data, i) = (P = population(data); P isa Real ? float(P) : float(P[i]))

# ============================================================================
# RJMCMC interface — parameter registry
# ============================================================================

cluster_param_dicts(state::NegBinPopulationRatesState) =
    (ρ = state.ρ_dict, r = state.r_dict)

# ============================================================================
# Per-parameter birth proposals (singular Val{:name} dispatch).
# These power MixedProposal (birth) and Resample / MixedFixedDim (fixed-dim).
# Both ρ and r are positive, so all four proposal families apply to both.
# ============================================================================

# ---- PriorProposal ---------------------------------------------------------
function sample_birth_param(::NegBinPopulationRates, ::Val{:ρ}, ::PriorProposal,
        S_i, state::NegBinPopulationRatesState, data, priors::NegBinPopulationRatesPriors)
    Q = ρ_prior(priors); v = rand(Q); return v, logpdf(Q, v)
end
birth_param_logpdf(::NegBinPopulationRates, ::Val{:ρ}, ::PriorProposal, v, S_i,
        state::NegBinPopulationRatesState, data, priors::NegBinPopulationRatesPriors) =
    (v <= 0 ? -Inf : logpdf(ρ_prior(priors), v))

function sample_birth_param(::NegBinPopulationRates, ::Val{:r}, ::PriorProposal,
        S_i, state::NegBinPopulationRatesState, data, priors::NegBinPopulationRatesPriors)
    Q = r_prior(priors); v = rand(Q); return v, logpdf(Q, v)
end
birth_param_logpdf(::NegBinPopulationRates, ::Val{:r}, ::PriorProposal, v, S_i,
        state::NegBinPopulationRatesState, data, priors::NegBinPopulationRatesPriors) =
    (v <= 0 ? -Inf : logpdf(r_prior(priors), v))

# ---- Moment-matched centres ------------------------------------------------
# Each family targets the same MoM estimate; they differ only in proposal shape.
# Falling back to the prior mean keeps the proposal proper when the fit fails.
_mm_centre(::Val{:ρ}, est, priors) = isnothing(est) ? ρ_prior_mean(priors) : est[1]
_mm_centre(::Val{:r}, est, priors) = isnothing(est) ? r_prior_mean(priors) : est[2]

_mm_est(prop, S_i, data) = _fit_S(S_i, data, prop.min_size)
# NormalMomentMatch carries no min_size field; use the package default of 3.
_mm_est(::NormalMomentMatch, S_i, data) = _fit_S(S_i, data, 3)

# ---- NormalMomentMatch (truncated to (0, ∞) for positivity) ----------------
for P in (:ρ, :r)
    @eval function sample_birth_param(::NegBinPopulationRates, v::Val{$(QuoteNode(P))},
            prop::NormalMomentMatch, S_i, state::NegBinPopulationRatesState, data,
            priors::NegBinPopulationRatesPriors)
        centre = max(_mm_centre(v, _mm_est(prop, S_i, data), priors), 1e-6)
        Q = truncated(Normal(centre, prop.σ[1]), 0.0, Inf)
        x = rand(Q); return x, logpdf(Q, x)
    end
    @eval function birth_param_logpdf(::NegBinPopulationRates, v::Val{$(QuoteNode(P))},
            prop::NormalMomentMatch, x, S_i, state::NegBinPopulationRatesState, data,
            priors::NegBinPopulationRatesPriors)
        x <= 0 && return -Inf
        centre = max(_mm_centre(v, _mm_est(prop, S_i, data), priors), 1e-6)
        return logpdf(truncated(Normal(centre, prop.σ[1]), 0.0, Inf), x)
    end
end

# ---- LogNormalMomentMatch --------------------------------------------------
# log(θ') ~ Normal(log θ̂, σ); Jacobian −log(θ') on the natural scale.
for P in (:ρ, :r)
    @eval function sample_birth_param(::NegBinPopulationRates, v::Val{$(QuoteNode(P))},
            prop::LogNormalMomentMatch, S_i, state::NegBinPopulationRatesState, data,
            priors::NegBinPopulationRatesPriors)
        μ = log(max(_mm_centre(v, _mm_est(prop, S_i, data), priors), 1e-6))
        logθ = rand(Normal(μ, prop.σ[1])); θ = exp(logθ)
        return θ, logpdf(Normal(μ, prop.σ[1]), logθ) - logθ
    end
    @eval function birth_param_logpdf(::NegBinPopulationRates, v::Val{$(QuoteNode(P))},
            prop::LogNormalMomentMatch, θ, S_i, state::NegBinPopulationRatesState, data,
            priors::NegBinPopulationRatesPriors)
        θ <= 0 && return -Inf
        μ = log(max(_mm_centre(v, _mm_est(prop, S_i, data), priors), 1e-6))
        return logpdf(Normal(μ, prop.σ[1]), log(θ)) - log(θ)
    end
end

# ---- InverseGammaMomentMatch -----------------------------------------------
# InverseGamma(a, θ̂(a-1)) has mean θ̂ (paper §2.5.1.2 — same target as the
# log-normal variant, different proposal family). `a` sets the spread
# (CV = 1/√(a-2)); the birth/death MH ratio corrects any mismatch exactly.
const _IG_SHAPE = 5.0
for P in (:ρ, :r)
    @eval function _ig_dist(v::Val{$(QuoteNode(P))}, prop::InverseGammaMomentMatch, S_i,
            data, priors::NegBinPopulationRatesPriors)
        est = _mm_est(prop, S_i, data)
        θ̂ = _mm_centre(v, est, priors)
        (!isfinite(θ̂) || θ̂ <= 0) && (θ̂ = _mm_centre(v, nothing, priors))
        return InverseGamma(_IG_SHAPE, θ̂ * (_IG_SHAPE - 1))
    end
    @eval function sample_birth_param(::NegBinPopulationRates, v::Val{$(QuoteNode(P))},
            prop::InverseGammaMomentMatch, S_i, state::NegBinPopulationRatesState, data,
            priors::NegBinPopulationRatesPriors)
        Q = _ig_dist(v, prop, S_i, data, priors); x = rand(Q); return x, logpdf(Q, x)
    end
    @eval function birth_param_logpdf(::NegBinPopulationRates, v::Val{$(QuoteNode(P))},
            prop::InverseGammaMomentMatch, x, S_i, state::NegBinPopulationRatesState, data,
            priors::NegBinPopulationRatesPriors)
        x <= 0 && return -Inf
        return logpdf(_ig_dist(v, prop, S_i, data, priors), x)
    end
end

# ============================================================================
# Whole-vector birth proposal for PriorProposal (MixedProposal is auto-composed
# from the singular methods above by the @generated dispatch in the package).
# ============================================================================

function sample_birth_params(m::NegBinPopulationRates, ::PriorProposal, S_i,
        state::NegBinPopulationRatesState, data, priors::NegBinPopulationRatesPriors)
    ρv, lqρ = sample_birth_param(m, Val(:ρ), PriorProposal(), S_i, state, data, priors)
    rv, lqr = sample_birth_param(m, Val(:r), PriorProposal(), S_i, state, data, priors)
    return (ρ = ρv, r = rv), lqρ + lqr
end
function birth_params_logpdf(m::NegBinPopulationRates, ::PriorProposal, params_old,
        S_i, state::NegBinPopulationRatesState, data, priors::NegBinPopulationRatesPriors)
    return birth_param_logpdf(m, Val(:ρ), PriorProposal(), params_old.ρ, S_i, state, data, priors) +
           birth_param_logpdf(m, Val(:r), PriorProposal(), params_old.r, S_i, state, data, priors)
end

# ============================================================================
# Table contribution and posterior
# ============================================================================

function table_contribution(::NegBinPopulationRates, table::AbstractVector{Int},
        state::NegBinPopulationRatesState, data, priors::NegBinPopulationRatesPriors)
    y = observations(data)
    ρ = state.ρ_dict[table]; r = state.r_dict[table]
    (ρ <= 0 || r <= 0 || !isfinite(ρ) || !isfinite(r)) && return -Inf

    log_lik = 0.0
    for i in table
        μ = _exposure(data, i) * ρ
        μ <= 0 && return -Inf
        log_lik += logpdf(nb_dist(μ, r), y[i])
    end
    log_prior = logpdf(ρ_prior(priors), ρ) + logpdf(r_prior(priors), r)
    return log_lik + log_prior
end

function posterior(model::NegBinPopulationRates, data, state::NegBinPopulationRatesState,
        priors::NegBinPopulationRatesPriors, log_DDCRP::AbstractMatrix)
    return sum(table_contribution(model, table, state, data, priors)
               for table in keys(state.ρ_dict)) +
           DD.ddcrp_contribution(state.c, log_DDCRP)
end

# ============================================================================
# Within-cluster Metropolis–Hastings updates (fixed partition)
# ============================================================================

# Both parameters are positive, so both use a random walk on the log scale with
# Jacobian log(θ') − log(θ).
function _update_positive!(model, dict, table, state, data, priors, prop_sd)
    old = dict[table]
    log_old = log(old)
    log_new = log_old + prop_sd * randn()
    lp_old = table_contribution(model, table, state, data, priors)
    dict[table] = exp(log_new)
    lp_new = table_contribution(model, table, state, data, priors)
    log(rand()) >= lp_new - lp_old + (log_new - log_old) && (dict[table] = old)
end

function update_params!(model::NegBinPopulationRates, state::NegBinPopulationRatesState,
        data, priors::NegBinPopulationRatesPriors, tables::Vector{Vector{Int}},
        log_DDCRP::AbstractMatrix, opts::MCMCOptions)
    for table in collect(keys(state.ρ_dict))
        should_infer(opts, :ρ) &&
            _update_positive!(model, state.ρ_dict, table, state, data, priors, get_prop_sd(opts, :ρ))
        should_infer(opts, :r) &&
            _update_positive!(model, state.r_dict, table, state, data, priors, get_prop_sd(opts, :r))
    end
end

# ============================================================================
# Initialisation and sample extraction
# ============================================================================

function initialise_state(::NegBinPopulationRates, data, ddcrp_params::DDCRPParams,
        priors::NegBinPopulationRatesPriors)
    y = observations(data)
    D = distance_matrix(data)
    c = simulate_ddcrp(D; α = ddcrp_params.α, scale = ddcrp_params.scale,
                       decay_fn = ddcrp_params.decay_fn)
    tables = table_vector(c)

    ρ_dict = Dict{Vector{Int}, Float64}()
    r_dict = Dict{Vector{Int}, Float64}()
    for table in tables
        key = sort(table)
        est = _fit_S(key, data, 3)
        if isnothing(est)
            # No usable NB fit: match the mean exactly, start mildly overdispersed.
            ΣE = sum(_exposure(data, i) for i in key)
            ρv = ΣE > 0 ? max(sum(y[i] for i in key) / ΣE, 1e-3) : ρ_prior_mean(priors)
            rv = r_prior_mean(priors)
        else
            ρv, rv = est
        end
        ρ_dict[key] = ρv
        r_dict[key] = rv
    end
    return NegBinPopulationRatesState(c, ρ_dict, r_dict)
end

function allocate_samples(::NegBinPopulationRates, n_samples::Int, n::Int)
    NegBinPopulationRatesSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # ρ (per observation)
        zeros(n_samples, n),        # r
        zeros(n_samples),           # logpost
        zeros(n_samples),           # α_ddcrp
        zeros(n_samples),           # s_ddcrp
    )
end

function extract_samples!(::NegBinPopulationRates, state::NegBinPopulationRatesState,
        samples::NegBinPopulationRatesSamples, iter::Int)
    samples.c[iter, :] = state.c
    for (table, ρv) in state.ρ_dict
        rv = state.r_dict[table]
        for i in table
            samples.ρ[iter, i] = ρv
            samples.r[iter, i] = rv
        end
    end
end

# ============================================================================
# Model comparison and posterior predictive (see `model_comparison.jl`)
# ============================================================================

"""
    loglik_matrix(::NegBinPopulationRates, samples, data) -> Matrix

`(n_draws × n_obs)` matrix of `log p(y_i | ρ^(s)_i, r^(s)_i)`, the input WAIC,
LPML and PSIS-LOO are computed from. Pass an already-thinned samples object.
"""
function loglik_matrix(::NegBinPopulationRates,
        samples::NegBinPopulationRatesSamples, data)
    y = observations(data)
    S, n = size(samples.ρ)
    ll = Matrix{Float64}(undef, S, n)
    for i in 1:n
        E = _exposure(data, i)
        for s in 1:S
            ρ = samples.ρ[s, i]; r = samples.r[s, i]
            ll[s, i] = (ρ > 0 && r > 0) ? logpdf(nb_dist(E * ρ, r), y[i]) : -Inf
        end
    end
    return ll
end

"""
    posterior_predictive(::NegBinPopulationRates, samples, data; n_rep=nothing)

Draw one replicate dataset `y_rep` per retained posterior draw, using that
draw's per-observation (ρ, r) and the observed exposures. Returns an
`(n_draws × n_obs)` integer matrix.
"""
function posterior_predictive(::NegBinPopulationRates,
        samples::NegBinPopulationRatesSamples, data; n_rep::Union{Nothing,Int} = nothing)
    S, n = size(samples.ρ)
    draws = isnothing(n_rep) ? (1:S) : round.(Int, range(1, S; length = min(n_rep, S)))
    y_rep = zeros(Int, length(draws), n)
    for (s_idx, s) in enumerate(draws), i in 1:n
        μ = _exposure(data, i) * samples.ρ[s, i]
        y_rep[s_idx, i] = rand(nb_dist(μ, samples.r[s, i]))
    end
    return y_rep
end

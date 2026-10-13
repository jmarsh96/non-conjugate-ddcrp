# Negative Binomial cluster model with population offsets for the ddCRP RJMCMC framework:
#   y_i | ρ_k, r_k ~ NegativeBinomial(mean = E_i ρ_k, size = r_k)
#   ρ_k ~ Gamma(ρ_a, ρ_b),  r_k ~ Gamma(r_a, r_b)
# Both cluster parameters are non-conjugate. Adds methods to DistanceDependentCRP's generic functions.

using DistanceDependentCRP
using Distributions, Statistics, StatsBase, Random

import DistanceDependentCRP: cluster_param_dicts, table_contribution, posterior,
    update_params!, sample_birth_params, birth_params_logpdf,
    sample_birth_param, birth_param_logpdf,
    initialise_state, allocate_samples, extract_samples!,
    requires_population, posterior_predictive

const DD = DistanceDependentCRP

# Types

struct NegBinPopulationRates <: LikelihoodModel end

requires_population(::NegBinPopulationRates) = true

mutable struct NegBinPopulationRatesState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    ρ_dict::Dict{Vector{Int}, T}
    r_dict::Dict{Vector{Int}, T}
end

# Gamma(shape, rate) priors on ρ and r.
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

# Distributions parameterises the Gamma by scale, hence 1/b.
ρ_prior(p::NegBinPopulationRatesPriors) = Gamma(p.ρ_a, 1 / p.ρ_b)
r_prior(p::NegBinPopulationRatesPriors) = Gamma(p.r_a, 1 / p.r_b)
ρ_prior_mean(p::NegBinPopulationRatesPriors) = p.ρ_a / p.ρ_b
r_prior_mean(p::NegBinPopulationRatesPriors) = p.r_a / p.r_b

# Per-observation matrices `ρ`, `r`, plus `c`, `logpost`, `α_ddcrp`, `s_ddcrp`.
struct NegBinPopulationRatesSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    ρ::Matrix{T}
    r::Matrix{T}
    logpost::Vector{T}
    α_ddcrp::Vector{T}
    s_ddcrp::Vector{T}
end

# Method-of-moments NegBin fit (moment-matching proposals and initialisation)

# NegativeBinomial with mean μ and variance μ + μ²/r.
nb_dist(μ::Real, r::Real) = NegativeBinomial(r, r / (r + μ))

# Method-of-moments (ρ, r) for counts `ys` with exposures `Es`: ρ̂ = Σy / ΣE and
# r̂ = mean(μ̂) / (φ̂ - 1) with φ̂ the quasi-Poisson dispersion. Returns `nothing` when the set
# is too small or underdispersed, and callers fall back to the prior.
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

function _fit_S(S_i, data, min_size::Int)
    y = observations(data)
    P = population(data)
    Es = P isa Real ? fill(float(P), length(S_i)) : [float(P[i]) for i in S_i]
    return fit_negbin_moments([float(y[i]) for i in S_i], Es; min_size = min_size)
end

_exposure(data, i) = (P = population(data); P isa Real ? float(P) : float(P[i]))

# RJMCMC interface: parameter registry

cluster_param_dicts(state::NegBinPopulationRatesState) =
    (ρ = state.ρ_dict, r = state.r_dict)

# Per-parameter birth proposals (used by MixedProposal, Resample and MixedFixedDim)

# PriorProposal
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

# Moment-matched centres; the prior mean is the fallback when the fit fails.
_mm_centre(::Val{:ρ}, est, priors) = isnothing(est) ? ρ_prior_mean(priors) : est[1]
_mm_centre(::Val{:r}, est, priors) = isnothing(est) ? r_prior_mean(priors) : est[2]

_mm_est(prop, S_i, data) = _fit_S(S_i, data, prop.min_size)
_mm_est(::NormalMomentMatch, S_i, data) = _fit_S(S_i, data, 3)

# NormalMomentMatch (truncated to (0, ∞))
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

# LogNormalMomentMatch: log(θ') ~ Normal(log θ̂, σ)
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

# InverseGammaMomentMatch: InverseGamma(a, θ̂(a-1)) has mean θ̂.
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

# Whole-vector birth proposal for PriorProposal

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

# Table contribution and posterior

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

# Within-cluster Metropolis-Hastings updates (random walk on the log scale)

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

# Initialisation and sample extraction

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
            # No usable NB fit: match the mean, start mildly overdispersed.
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

# Model comparison and posterior predictive

# (n_draws × n_obs) matrix of log p(y_i | ρ_i, r_i).
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

# One replicate dataset per retained posterior draw.
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

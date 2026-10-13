# Skew-normal cluster model for the ddCRP RJMCMC framework:
#   y_i | ξ_k, ω_k, α_k ~ SkewNormal(ξ_k, ω_k, α_k)
#   ξ_k ~ Normal(ξ_m, ξ_s),  ω_k ~ InverseGamma(ω_a, ω_b),  α_k ~ Normal(α_m, α_s)
# Adds methods to DistanceDependentCRP's generic functions.

using DistanceDependentCRP
using Distributions, Statistics, StatsBase, Random

import DistanceDependentCRP: cluster_param_dicts, table_contribution, posterior,
    update_params!, sample_birth_params, birth_params_logpdf,
    sample_birth_param, birth_param_logpdf,
    initialise_state, allocate_samples, extract_samples!

const DD = DistanceDependentCRP

# Types

struct SkewNormalClusters <: LikelihoodModel end

mutable struct SkewNormalClustersState{T<:Real} <: AbstractMCMCState{T}
    c::Vector{Int}
    ξ_dict::Dict{Vector{Int}, T}
    ω_dict::Dict{Vector{Int}, T}
    α_dict::Dict{Vector{Int}, T}
end

# Normal priors on ξ and α, InverseGamma(shape, scale) on ω.
struct SkewNormalClustersPriors{T<:Real} <: AbstractPriors
    ξ_m::T; ξ_s::T
    ω_a::T; ω_b::T
    α_m::T; α_s::T
end

function SkewNormalClustersPriors(;
    ξ_m::Real = 0.0, ξ_s::Real = 10.0,
    ω_a::Real = 3.0, ω_b::Real = 4.0,
    α_m::Real = 0.0, α_s::Real = 5.0,
)
    T = promote_type(typeof(ξ_m), typeof(ξ_s), typeof(ω_a), typeof(ω_b),
                     typeof(α_m), typeof(α_s))
    SkewNormalClustersPriors{T}(T(ξ_m), T(ξ_s), T(ω_a), T(ω_b), T(α_m), T(α_s))
end

# Per-observation matrices `ξ`, `ω`, `α`, plus `c`, `logpost`, `α_ddcrp`, `s_ddcrp`.
struct SkewNormalClustersSamples{T<:Real} <: AbstractMCMCSamples
    c::Matrix{Int}
    ξ::Matrix{T}
    ω::Matrix{T}
    α::Matrix{T}
    logpost::Vector{T}
    α_ddcrp::Vector{T}
    s_ddcrp::Vector{T}
end

# Method-of-moments skew-normal fit (moment-matching proposals and initialisation)

# Method-of-moments (ξ, ω, α) from the sample mean, variance and skewness; `nothing` when
# there are too few points or the variance is non-positive.
function fit_skewnormal_moments(ys; min_size::Int = 4)
    n = length(ys)
    n < min_size && return nothing
    m = mean(ys)
    s2 = var(ys; corrected = false)
    (s2 <= 0 || !isfinite(s2)) && return nothing
    s = sqrt(s2)
    g = mean(((yi - m) / s)^3 for yi in ys)          # sample skewness
    g_max = 0.9952                                    # SkewNormal skewness bound
    g_abs = min(abs(g), g_max)
    sgn = g < 0 ? -1.0 : 1.0
    c2 = g_abs^(2/3)
    δ2 = (π / 2) * c2 / (c2 + ((4 - π) / 2)^(2/3))
    δ2 = min(δ2, 0.9999)
    δ = sgn * sqrt(δ2)
    α = δ / sqrt(1 - δ^2)
    ω = sqrt(s2 / (1 - 2 * δ^2 / π))
    ξ = m - ω * δ * sqrt(2 / π)
    (isfinite(ξ) && isfinite(ω) && ω > 0 && isfinite(α)) || return nothing
    return (ξ, ω, clamp(α, -30.0, 30.0))
end

ω_prior_mean(p::SkewNormalClustersPriors) = p.ω_a > 1 ? p.ω_b / (p.ω_a - 1) : p.ω_b
_view_S(data, S_i) = view(observations(data), S_i)

# RJMCMC interface: parameter registry

cluster_param_dicts(state::SkewNormalClustersState) =
    (ξ = state.ξ_dict, ω = state.ω_dict, α = state.α_dict)

# Per-parameter birth proposals (used by MixedProposal, Resample and MixedFixedDim)

# PriorProposal
function sample_birth_param(::SkewNormalClusters, ::Val{:ξ}, ::PriorProposal,
        S_i, state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors)
    Q = Normal(priors.ξ_m, priors.ξ_s); v = rand(Q); return v, logpdf(Q, v)
end
birth_param_logpdf(::SkewNormalClusters, ::Val{:ξ}, ::PriorProposal, v, S_i,
        state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors) =
    logpdf(Normal(priors.ξ_m, priors.ξ_s), v)

function sample_birth_param(::SkewNormalClusters, ::Val{:ω}, ::PriorProposal,
        S_i, state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors)
    Q = InverseGamma(priors.ω_a, priors.ω_b); v = rand(Q); return v, logpdf(Q, v)
end
birth_param_logpdf(::SkewNormalClusters, ::Val{:ω}, ::PriorProposal, v, S_i,
        state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors) =
    (v <= 0 ? -Inf : logpdf(InverseGamma(priors.ω_a, priors.ω_b), v))

function sample_birth_param(::SkewNormalClusters, ::Val{:α}, ::PriorProposal,
        S_i, state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors)
    Q = Normal(priors.α_m, priors.α_s); v = rand(Q); return v, logpdf(Q, v)
end
birth_param_logpdf(::SkewNormalClusters, ::Val{:α}, ::PriorProposal, v, S_i,
        state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors) =
    logpdf(Normal(priors.α_m, priors.α_s), v)

# Moment-matched proposals. Below MM_MIN_SIZE observations (or when the fit fails) ξ is
# centred on the moving-set mean and ω, α are drawn from their priors.
const MM_MIN_SIZE = 3
_mm_est(S_i, data) = length(S_i) >= MM_MIN_SIZE ?
    fit_skewnormal_moments(_view_S(data, S_i); min_size = MM_MIN_SIZE) : nothing

_prior_dist(::Val{:ω}, priors) = InverseGamma(priors.ω_a, priors.ω_b)
_prior_dist(::Val{:α}, priors) = Normal(priors.α_m, priors.α_s)

# NormalMomentMatch for ξ
function _nmm_xi_dist(prop, S_i, data)
    est = _mm_est(S_i, data)
    centre = isnothing(est) ? mean(_view_S(data, S_i)) : est[1]
    return Normal(centre, prop.σ[1])
end
function sample_birth_param(::SkewNormalClusters, ::Val{:ξ}, prop::NormalMomentMatch,
        S_i, state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors)
    Q = _nmm_xi_dist(prop, S_i, data); x = rand(Q); return x, logpdf(Q, x)
end
birth_param_logpdf(::SkewNormalClusters, ::Val{:ξ}, prop::NormalMomentMatch, x, S_i,
        state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors) =
    logpdf(_nmm_xi_dist(prop, S_i, data), x)

# NormalMomentMatch for α
function _nmm_alpha_dist(prop, S_i, data, priors)
    est = _mm_est(S_i, data)
    return isnothing(est) ? _prior_dist(Val(:α), priors) : Normal(est[3], prop.σ[1])
end
function sample_birth_param(::SkewNormalClusters, ::Val{:α}, prop::NormalMomentMatch,
        S_i, state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors)
    Q = _nmm_alpha_dist(prop, S_i, data, priors); x = rand(Q); return x, logpdf(Q, x)
end
birth_param_logpdf(::SkewNormalClusters, ::Val{:α}, prop::NormalMomentMatch, x, S_i,
        state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors) =
    logpdf(_nmm_alpha_dist(prop, S_i, data, priors), x)

# NormalMomentMatch for ω, truncated to (0, ∞)
function _nmm_omega_dist(prop, S_i, data, priors)
    est = _mm_est(S_i, data)
    isnothing(est) && return _prior_dist(Val(:ω), priors)
    return truncated(Normal(max(est[2], 1e-3), prop.σ[1]), 0.0, Inf)
end
function sample_birth_param(::SkewNormalClusters, ::Val{:ω}, prop::NormalMomentMatch,
        S_i, state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors)
    Q = _nmm_omega_dist(prop, S_i, data, priors); v = rand(Q); return v, logpdf(Q, v)
end
function birth_param_logpdf(::SkewNormalClusters, ::Val{:ω}, prop::NormalMomentMatch,
        v, S_i, state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors)
    v <= 0 && return -Inf
    return logpdf(_nmm_omega_dist(prop, S_i, data, priors), v)
end

# LogNormalMomentMatch for ω: log(ω') ~ Normal(log ω̂, σ)
function sample_birth_param(::SkewNormalClusters, ::Val{:ω}, prop::LogNormalMomentMatch,
        S_i, state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors)
    est = _mm_est(S_i, data)
    if isnothing(est)
        Q = _prior_dist(Val(:ω), priors); v = rand(Q); return v, logpdf(Q, v)
    end
    μ = log(max(est[2], 1e-3))
    logω = rand(Normal(μ, prop.σ[1])); ω = exp(logω)
    return ω, logpdf(Normal(μ, prop.σ[1]), logω) - logω
end
function birth_param_logpdf(::SkewNormalClusters, ::Val{:ω}, prop::LogNormalMomentMatch,
        ω, S_i, state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors)
    ω <= 0 && return -Inf
    est = _mm_est(S_i, data)
    isnothing(est) && return logpdf(_prior_dist(Val(:ω), priors), ω)
    μ = log(max(est[2], 1e-3))
    return logpdf(Normal(μ, prop.σ[1]), log(ω)) - log(ω)
end

# InverseGammaMomentMatch for ω: InverseGamma(a, ω̂(a-1)) has mean ω̂
const _IG_OMEGA_SHAPE = 5.0
function _ig_omega_dist(prop::InverseGammaMomentMatch, S_i, data, priors::SkewNormalClustersPriors)
    est = _mm_est(S_i, data)
    (isnothing(est) || !(est[2] > 0)) && return _prior_dist(Val(:ω), priors)
    return InverseGamma(_IG_OMEGA_SHAPE, est[2] * (_IG_OMEGA_SHAPE - 1))
end
function sample_birth_param(::SkewNormalClusters, ::Val{:ω}, prop::InverseGammaMomentMatch,
        S_i, state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors)
    Q = _ig_omega_dist(prop, S_i, data, priors); v = rand(Q); return v, logpdf(Q, v)
end
function birth_param_logpdf(::SkewNormalClusters, ::Val{:ω}, prop::InverseGammaMomentMatch,
        v, S_i, state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors)
    v <= 0 && return -Inf
    return logpdf(_ig_omega_dist(prop, S_i, data, priors), v)
end

# Whole-vector birth proposal for PriorProposal

function sample_birth_params(m::SkewNormalClusters, ::PriorProposal, S_i,
        state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors)
    ξv, lqξ = sample_birth_param(m, Val(:ξ), PriorProposal(), S_i, state, data, priors)
    ωv, lqω = sample_birth_param(m, Val(:ω), PriorProposal(), S_i, state, data, priors)
    αv, lqα = sample_birth_param(m, Val(:α), PriorProposal(), S_i, state, data, priors)
    return (ξ = ξv, ω = ωv, α = αv), lqξ + lqω + lqα
end
function birth_params_logpdf(m::SkewNormalClusters, ::PriorProposal, params_old,
        S_i, state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors)
    return birth_param_logpdf(m, Val(:ξ), PriorProposal(), params_old.ξ, S_i, state, data, priors) +
           birth_param_logpdf(m, Val(:ω), PriorProposal(), params_old.ω, S_i, state, data, priors) +
           birth_param_logpdf(m, Val(:α), PriorProposal(), params_old.α, S_i, state, data, priors)
end

# Table contribution and posterior

# Switched off by geweke_skewnormal.jl so that the target is the prior.
const LIKELIHOOD_ON = Ref(true)

function table_contribution(::SkewNormalClusters, table::AbstractVector{Int},
        state::SkewNormalClustersState, data, priors::SkewNormalClustersPriors)
    y = observations(data)
    ξ = state.ξ_dict[table]; ω = state.ω_dict[table]; α = state.α_dict[table]
    (ω <= 0 || !isfinite(ω)) && return -Inf
    d = SkewNormal(ξ, ω, α)
    log_lik = LIKELIHOOD_ON[] ? sum(logpdf(d, y[i]) for i in table) : 0.0
    log_prior = logpdf(Normal(priors.ξ_m, priors.ξ_s), ξ) +
                logpdf(InverseGamma(priors.ω_a, priors.ω_b), ω) +
                logpdf(Normal(priors.α_m, priors.α_s), α)
    return log_lik + log_prior
end

function posterior(model::SkewNormalClusters, data, state::SkewNormalClustersState,
        priors::SkewNormalClustersPriors, log_DDCRP::AbstractMatrix)
    return sum(table_contribution(model, table, state, data, priors)
               for table in keys(state.ξ_dict)) +
           DD.ddcrp_contribution(state.c, log_DDCRP)
end

# Within-cluster Metropolis-Hastings updates

# ξ: random walk on the natural scale.
function _update_location!(model, table, state, data, priors, prop_sd)
    old = state.ξ_dict[table]
    lp_old = table_contribution(model, table, state, data, priors)
    state.ξ_dict[table] = old + prop_sd * randn()
    lp_new = table_contribution(model, table, state, data, priors)
    log(rand()) >= lp_new - lp_old && (state.ξ_dict[table] = old)
end

# ω: random walk on the log scale.
function _update_scale!(model, table, state, data, priors, prop_sd)
    old = state.ω_dict[table]
    log_old = log(old)
    log_new = log_old + prop_sd * randn()
    lp_old = table_contribution(model, table, state, data, priors)
    state.ω_dict[table] = exp(log_new)
    lp_new = table_contribution(model, table, state, data, priors)
    log(rand()) >= lp_new - lp_old + (log_new - log_old) && (state.ω_dict[table] = old)
end

# α: random walk on the natural scale.
function _update_skew!(model, table, state, data, priors, prop_sd)
    old = state.α_dict[table]
    lp_old = table_contribution(model, table, state, data, priors)
    state.α_dict[table] = old + prop_sd * randn()
    lp_new = table_contribution(model, table, state, data, priors)
    log(rand()) >= lp_new - lp_old && (state.α_dict[table] = old)
end

function update_params!(model::SkewNormalClusters, state::SkewNormalClustersState,
        data, priors::SkewNormalClustersPriors, tables::Vector{Vector{Int}},
        log_DDCRP::AbstractMatrix, opts::MCMCOptions)
    for table in collect(keys(state.ξ_dict))
        should_infer(opts, :ξ) && _update_location!(model, table, state, data, priors, get_prop_sd(opts, :ξ))
        should_infer(opts, :ω) && _update_scale!(model, table, state, data, priors, get_prop_sd(opts, :ω))
        should_infer(opts, :α) && _update_skew!(model, table, state, data, priors, get_prop_sd(opts, :α))
    end
end

# Initialisation and sample extraction

function initialise_state(::SkewNormalClusters, data, ddcrp_params::DDCRPParams,
        priors::SkewNormalClustersPriors)
    y = observations(data)
    D = distance_matrix(data)
    c = simulate_ddcrp(D; α = ddcrp_params.α, scale = ddcrp_params.scale,
                       decay_fn = ddcrp_params.decay_fn)
    tables = table_vector(c)

    ξ_dict = Dict{Vector{Int}, Float64}()
    ω_dict = Dict{Vector{Int}, Float64}()
    α_dict = Dict{Vector{Int}, Float64}()
    for table in tables
        key = sort(table)
        yk = view(y, key)
        est = fit_skewnormal_moments(yk; min_size = 4)
        if isnothing(est)
            ξv = mean(yk)
            ωv = length(key) > 1 ? max(std(yk; corrected = false), 1.0) : 1.0
            αv = 0.0
        else
            ξv, ωv, αv = est
        end
        ξ_dict[key] = ξv
        ω_dict[key] = max(ωv, 1e-2)
        α_dict[key] = clamp(αv, -30.0, 30.0)
    end
    return SkewNormalClustersState(c, ξ_dict, ω_dict, α_dict)
end

function allocate_samples(::SkewNormalClusters, n_samples::Int, n::Int)
    SkewNormalClustersSamples(
        zeros(Int, n_samples, n),   # c
        zeros(n_samples, n),        # ξ (per observation)
        zeros(n_samples, n),        # ω
        zeros(n_samples, n),        # α
        zeros(n_samples),           # logpost
        zeros(n_samples),           # α_ddcrp
        zeros(n_samples),           # s_ddcrp
    )
end

function extract_samples!(::SkewNormalClusters, state::SkewNormalClustersState,
        samples::SkewNormalClustersSamples, iter::Int)
    samples.c[iter, :] = state.c
    for (table, ξv) in state.ξ_dict
        ωv = state.ω_dict[table]; αv = state.α_dict[table]
        for i in table
            samples.ξ[iter, i] = ξv
            samples.ω[iter, i] = ωv
            samples.α[iter, i] = αv
        end
    end
end

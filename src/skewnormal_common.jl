# Skew-normal study setup shared by run_skewnormal.jl and run_gibbs_comparison.jl.

const N_OBS = 150
const K_TRUE = 3

# Simulated data

function make_cluster_x(n::Int, centers::Vector{Float64}, σ::Float64; seed::Int=0)
    rng = MersenneTwister(seed)
    K = length(centers)
    sizes = fill(n ÷ K, K)
    sizes[end] += n - sum(sizes)
    vcat([centers[k] .+ σ * randn(rng, sizes[k]) for k in 1:K]...)
end

const C_FIXED = vcat([fill(1 + (k - 1) * (N_OBS ÷ K_TRUE), N_OBS ÷ K_TRUE) for k in 1:K_TRUE]...)

function simulate_skewnormal(c, pars; seed=1)
    rng = MersenneTwister(seed)
    labels = compute_table_assignments(c)
    [rand(rng, SkewNormal(pars[labels[i]]...)) for i in eachindex(c)]
end

function build_scenario(x_centers, x_sd, pars; x_seed=0, y_seed=2)
    x = make_cluster_x(N_OBS, x_centers, x_sd; seed=x_seed)
    D = pairwise(Euclidean(), reshape(x, 1, :))
    y = simulate_skewnormal(C_FIXED, pars; seed=y_seed)
    return (y=y, D=D, c=C_FIXED, x=x, pars=pars)
end

const SIM_OVERLAPPING = build_scenario(
    [-3.0, 0.0, 3.0], 1.5, [(0.0, 2.0, 5.0), (4.0, 2.0, 0.0), (8.0, 2.0, -5.0)]; x_seed=0, y_seed=2)
const SIM_SEPARATED = build_scenario(
    [-6.0, 0.0, 6.0], 1.0, [(0.0, 1.5, 5.0), (8.0, 1.5, 0.0), (16.0, 1.5, -5.0)]; x_seed=1, y_seed=3)
const SCENARIOS = [("overlapping", SIM_OVERLAPPING), ("separated", SIM_SEPARATED)]

# Model settings

const DDCRP_PARAMS = DDCRPParams(1.0, 0.5, DistanceDependentCRP.exp_decay, 1.0, 0.01, 1.0, 0.01)
# Diffuse priors.
const PRIORS = SkewNormalClustersPriors(ξ_m=0.0, ξ_s=50.0, ω_a=2.0, ω_b=2.0, α_m=0.0, α_s=10.0)
const PROP_SDS = (ξ = 0.6, ω = 0.15, α = 1.2)   # within-cluster MH

# Per-parameter proposal-family builders

function _birth_family(fam::Symbol, σ)
    fam === :prior && return PriorProposal()
    fam === :nmm   && return NormalMomentMatch(σ)
    fam === :lnmm  && return LogNormalMomentMatch(σ)
    fam === :ig    && return InverseGammaMomentMatch()
    error("unknown family $fam")
end

# σ is a NamedTuple (ξ, ω, α) of birth standard deviations.
make_birth(fξ, fω, fα, σ) = MixedProposal((
    ξ = _birth_family(fξ, σ.ξ), ω = _birth_family(fω, σ.ω), α = _birth_family(fα, σ.α)))

make_fdim(fξ, fω, fα, σr; resample::Bool) =
    resample ? MixedFixedDim((
        ξ = Resample(_birth_family(fξ, σr)),
        ω = Resample(_birth_family(fω, σr)),
        α = Resample(_birth_family(fα, σr)))) : NoUpdate()

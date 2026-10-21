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

# Sampler-study scenarios (run_sampler_study.jl), replicate `rep` ∈ 1:3. Replicate 1 of
# overlapping and separated is the paper's data set.
#   aba: groups A1 | B | A2 along x, A1 and A2 sharing parameters, so moving between the
#        merged state and {A1 ∪ A2, B} needs several links to change together.
#   k5:  n = 300 in five groups with mixed overlap.
const STUDY_SCENARIOS = ["overlapping", "separated", "aba", "k5"]

function _labels_to_c(labels)
    first_of = Dict{Int,Int}()
    return [get!(first_of, l, i) for (i, l) in enumerate(labels)]
end

function make_scenario(name::AbstractString, rep::Int)
    xs, ys = 100 * rep, 100 * rep + 2
    if name == "overlapping"
        rep == 1 && return SIM_OVERLAPPING
        return build_scenario([-3.0, 0.0, 3.0], 1.5, SIM_OVERLAPPING.pars; x_seed = xs, y_seed = ys)
    elseif name == "separated"
        rep == 1 && return SIM_SEPARATED
        return build_scenario([-6.0, 0.0, 6.0], 1.0, SIM_SEPARATED.pars; x_seed = xs + 1, y_seed = ys + 1)
    elseif name == "aba"
        pars = [(0.0, 2.0, 5.0), (4.0, 2.0, -5.0), (0.0, 2.0, 5.0)]      # A1, B, A2
        x = make_cluster_x(N_OBS, [-5.0, 0.0, 5.0], 1.0; seed = xs + 2)
        seg = vcat([fill(k, N_OBS ÷ 3) for k in 1:3]...)
        rng = MersenneTwister(ys + 2)
        y = [rand(rng, SkewNormal(pars[k]...)) for k in seg]
        c = _labels_to_c([k == 3 ? 1 : k for k in seg])
        return (y = y, D = pairwise(Euclidean(), reshape(x, 1, :)), c = c, x = x, pars = pars)
    elseif name == "k5"
        n, K = 300, 5
        pars = [(0.0, 2.0, 4.0), (4.0, 2.0, -4.0), (8.0, 2.0, 4.0), (12.0, 2.0, -4.0), (16.0, 2.0, 0.0)]
        x = make_cluster_x(n, [-8.0, -4.0, 0.0, 4.0, 8.0], 1.2; seed = xs + 3)
        seg = vcat([fill(k, n ÷ K) for k in 1:K]...)
        rng = MersenneTwister(ys + 3)
        y = [rand(rng, SkewNormal(pars[k]...)) for k in seg]
        return (y = y, D = pairwise(Euclidean(), reshape(x, 1, :)), c = _labels_to_c(seg), x = x, pars = pars)
    end
    error("unknown scenario $name")
end

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

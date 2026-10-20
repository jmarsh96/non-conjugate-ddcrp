# Poisson study setup shared by run_poisson.jl and run_gibbs_comparison.jl.

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

const X_SIM = make_cluster_x(N_OBS, [-3.0, 0.0, 3.0], 1.5; seed=0)

# Pointer representation: each observation points to the first member of its cluster.
const C_FIXED = vcat([fill(1 + (k - 1) * (N_OBS ÷ K_TRUE), N_OBS ÷ K_TRUE) for k in 1:K_TRUE]...)
const D_SIM = pairwise(Euclidean(), reshape(X_SIM, 1, :))

function simulate_poisson(c, λs; seed=1)
    rng = MersenneTwister(seed)
    labels = compute_table_assignments(c)   # convert pointer repr → labels 1…K
    [rand(rng, Poisson(λs[labels[i]])) for i in eachindex(c)]
end

const Y_OVERLAPPING = simulate_poisson(C_FIXED, [1.0, 4.0, 7.0]; seed=2)
const SIM_OVERLAPPING = (y=Y_OVERLAPPING, D=D_SIM, c=C_FIXED, x=X_SIM, λs=[1.0, 4.0, 7.0])
const SCENARIOS = [("overlapping", SIM_OVERLAPPING)]

# Model settings

const DDCRP_PARAMS = DDCRPParams(1.0, 0.5, DistanceDependentCRP.exp_decay, 1.0, 0.01, 1.0, 0.01)
const PRIORS = PoissonClusterRatesPriors(1.0, 0.1)
const PRIORS_MARG = PoissonClusterRatesMargPriors(1.0, 0.1)

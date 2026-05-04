using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using DistanceDependentCRP, DataFrames, CSV, Random, Distances, StatsPlots, JLD2

is_test = "--test" in ARGS

const N_SAMPLES = is_test ? 2500 : 250_000
const BURNIN = is_test ? 500 : 50_000
const THIN_FACTOR = 20
const OUTDIR = joinpath(@__DIR__, "..", "results", "old_faithful")
mkpath(OUTDIR)

const MODEL = GammaClusterShapeMarg()

α_init = 1.0
s_init = 0.2
decay_fn = DDCRP.exp_decay
α_prior_a = 1.0
α_prior_b = 0.01
s_prior_a = 1.0
s_prior_b = 0.01
const DDCRP_PARAMS = DDCRPParams(
    α_init, s_init, decay_fn, α_prior_a, α_prior_b, s_prior_a, s_prior_b
)

const PRIORS = GammaClusterShapeMargPriors(α_a=2.0, α_b=0.5, β_a=2.0, β_b=0.5)
const BIRTH_PROP = InverseGammaMomentMatch()
const FIXED_PROP = Resample(InverseGammaMomentMatch())

const OPTS = MCMCOptions(
    n_samples = N_SAMPLES,
    verbose = true,
    infer_params = Dict(
        :α => true, 
        :c => true,
        :α_ddcrp => true, 
        :s_ddcrp => false
    ),
    prop_sds = Dict(:α => 0.5, :s_ddcrp => 0.3),
    track_diagnostics = true
)

# load and clean data
println("Loading Old Faithful data...")
faithful = DataFrame(CSV.File(joinpath("data", "faithful.csv")))
y = Float64.(faithful.eruptions)   # response: eruption duration (min)
w = Float64.(faithful.waiting)     # covariate: waiting time (min)
n = length(y)
D = pairwise(Euclidean(), w)
println("  n = $n observations")
println("  Eruption duration: [$(round(minimum(y), digits=2)), $(round(maximum(y), digits=2))] min")
println("  Waiting time:      [$(minimum(w)), $(maximum(w))] min")


# run inference

println("\nRunning RJMCMC (n=$n, $N_SAMPLES iterations, $BURNIN burn-in)...")
Random.seed!(1)
total_time = @elapsed begin
    samples, diagnostics = mcmc(
        MODEL, ContinuousData(y, D), DDCRP_PARAMS, PRIORS,
        BIRTH_PROP;
        fixed_dim_proposal = FIXED_PROP,
        opts = OPTS
    )
end
println("  Done. Total time: $(round(diagnostics.total_time, digits=1)) s")

raw_chain_file = joinpath(OUTDIR, "mcmc_samples_raw.jld2")
jldsave(raw_chain_file, samples=samples, diagnostics=diagnostics)

idx = (BURNIN+1):THIN_FACTOR:N_SAMPLES
thinned = GammaClusterShapeMargSamples(
    samples.c[idx, :],
    samples.α[idx, :],
    samples.logpost[idx],
    samples.α_ddcrp[idx],
    samples.s_ddcrp[idx],
)

chain_file = joinpath(OUTDIR, "mcmc_samples.jld2")
jldsave(chain_file, samples=thinned, diagnostics=diagnostics)
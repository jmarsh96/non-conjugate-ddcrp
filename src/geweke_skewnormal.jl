# Prior-recovery check (Geweke 2004) for the skew-normal samplers: with the likelihood off
# (LIKELIHOOD_ON[] = false) the target is the prior. P(K) is compared with 200k draws of the
# ddCRP prior, and each parameter's prior PIT with Uniform(0, 1), as z-scores using the chain's ESS.
# Run after run_skewnormal.jl. --rerun repeats the Resample configs with first-pass |z| > 2
# at 4x length from two further seeds.
#
#   julia --project=. src/geweke_skewnormal.jl [--test] [--slurm] [--rerun]

using DistanceDependentCRP, CSV, DataFrames, Statistics, StatsBase, Random, JLD2, Printf, Distributions
using Distributed

const USE_SLURM = "--slurm" in ARGS
const IS_TEST = "--test" in ARGS
const RERUN = "--rerun" in ARGS

if USE_SLURM
    using SlurmClusterManager
    addprocs(SlurmManager(launch_timeout = 300.0))
else
    addprocs(8)
end

@everywhere begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using DistanceDependentCRP, Statistics, StatsBase, Distributions, Random
    include(joinpath(@__DIR__, "skewnormal_model.jl"))
    LIKELIHOOD_ON[] = false
end

const SKEW_DIR = joinpath(@__DIR__, "..", "results", "simulation_study",
                          IS_TEST ? "skewnormal_test" : "skewnormal")
const OUTDIR = joinpath(SKEW_DIR, "geweke"); mkpath(OUTDIR)

# A rerun is four times as long and thinned four times as hard.
const LEN_MULT = RERUN ? 4 : 1
const N_SAMPLES = LEN_MULT * (IS_TEST ? 2_000 : 100_000)
const BURN = LEN_MULT * (IS_TEST ? 200 : 10_000)
const THIN = LEN_MULT * 5
const N_REF = IS_TEST ? 5_000 : 200_000
const S_DECAY = 0.5
const ALPHA_DDCRP = 1.0

@everywhere const PROP_SDS = (ξ = 0.6, ω = 0.15, α = 1.2)
const SIGMA_RESAMPLE = 0.25
const PRIORS_G = SkewNormalClustersPriors(ξ_m = 5.0, ξ_s = 5.0, ω_a = 3.0, ω_b = 4.0, α_m = 0.0, α_s = 3.0)

sim = load(joinpath(SKEW_DIR, "sim_data.jld2"))["sim_overlapping"]
tuned = DataFrame(CSV.File(joinpath(SKEW_DIR, "tuned_sigma.csv")))
const SIGMA_BIRTH = (ξ = tuned.σ[tuned.param .== "ξ"][1], ω = tuned.σ[tuned.param .== "ω"][1],
                     α = tuned.σ[tuned.param .== "α"][1])
println("Birth σ from tuning: $SIGMA_BIRTH")

@everywhere function _birth_family(fam::Symbol, σ)
    fam === :prior && return PriorProposal()
    fam === :nmm   && return NormalMomentMatch(σ)
    fam === :lnmm  && return LogNormalMomentMatch(σ)
    fam === :ig    && return InverseGammaMomentMatch()
    error("unknown family $fam")
end

@everywhere function run_prior_chain(cfg)
    Random.seed!(cfg.seed)
    opts = MCMCOptions(
        n_samples = cfg.n_samples, verbose = false, track_diagnostics = true,
        infer_params = Dict{Symbol,Bool}(:α_ddcrp => false, :s_ddcrp => false),
        prop_sds = Dict{Symbol,Float64}(:ξ => PROP_SDS.ξ, :ω => PROP_SDS.ω, :α => PROP_SDS.α),
    )
    σ = cfg.σ_birth
    birth = MixedProposal((ξ = _birth_family(cfg.fam[1], σ.ξ), ω = _birth_family(cfg.fam[2], σ.ω),
                           α = _birth_family(cfg.fam[3], σ.α)))
    fdim = cfg.resample ? MixedFixedDim((
        ξ = Resample(_birth_family(cfg.fam[1], cfg.σ_resample)),
        ω = Resample(_birth_family(cfg.fam[2], cfg.σ_resample)),
        α = Resample(_birth_family(cfg.fam[3], cfg.σ_resample)))) : NoUpdate()
    samples, diag = mcmc(SkewNormalClusters(), ContinuousData(cfg.y, cfg.D), cfg.ddcrp_params,
                         cfg.priors, birth; fixed_dim_proposal = fdim, opts = opts)
    keep = (cfg.burn + 1):cfg.thin:cfg.n_samples
    return (k = Float64.(calculate_n_clusters(samples.c[keep, :])),
            ξ = samples.ξ[keep, 1], ω = samples.ω[keep, 1], α = samples.α[keep, 1],
            acc = acceptance_rates(diag))
end

# Exact reference: P(K) under the ddCRP prior at (α, s, D)
Random.seed!(1)
k_ref = [length(unique(compute_table_assignments(
             simulate_ddcrp(sim.D; α = ALPHA_DDCRP, scale = S_DECAY,
                            decay_fn = DistanceDependentCRP.exp_decay)))) for _ in 1:N_REF]
@printf("Reference: %d prior draws, E[K] = %.3f, sd(K) = %.3f\n", N_REF, mean(k_ref), std(k_ref))

# Every birth family × fixed-dim strategy of the study
const BIRTH_SPECS = [
    (:nmm, :lnmm, :nmm), (:prior, :lnmm, :nmm), (:nmm, :prior, :nmm), (:nmm, :nmm, :nmm),
    (:nmm, :ig, :nmm), (:nmm, :lnmm, :prior), (:nmm, :prior, :prior), (:prior, :prior, :prior),
]
ddcrp_params = DDCRPParams(ALPHA_DDCRP, S_DECAY, DistanceDependentCRP.exp_decay, 1.0, 0.01, 1.0, 0.01)
# The Resample configurations with a first-pass |z| > 2, and the seeds of their reruns.
const RERUN_SPECS = [(:nmm, :lnmm, :nmm), (:prior, :lnmm, :nmm), (:nmm, :nmm, :nmm)]
const RERUN_SEEDS = [101, 202]
_cfg(f, rs, seed) = (fam = f, resample = rs, σ_birth = SIGMA_BIRTH, σ_resample = SIGMA_RESAMPLE,
                     priors = PRIORS_G, y = sim.y, D = sim.D, ddcrp_params = ddcrp_params,
                     n_samples = N_SAMPLES, burn = BURN, thin = THIN, seed = seed)
configs = RERUN ? [_cfg(f, true, sd) for f in RERUN_SPECS for sd in RERUN_SEEDS] :
                  [_cfg(f, rs, 42) for f in BIRTH_SPECS for rs in (false, true)]

println("Running $(length(configs)) prior-only chains of $(N_SAMPLES) iterations...")
results = pmap(run_prior_chain, configs)

# z-score of a chain mean against a known mean and variance (ESS floored at 10).
zscore(x, μ, v) = (mean(x) - μ) / sqrt(v / max(effective_sample_size(x), 10.0))

prior_cdf = (ξ = Normal(PRIORS_G.ξ_m, PRIORS_G.ξ_s), ω = InverseGamma(PRIORS_G.ω_a, PRIORS_G.ω_b),
             α = Normal(PRIORS_G.α_m, PRIORS_G.α_s))
ks = 1:maximum(k_ref)
p_ref = [mean(k_ref .== k) for k in ks]

rows = NamedTuple[]
for (cfg, r) in zip(configs, results)
    p_chain = [mean(r.k .== k) for k in ks]
    # The reference variance of K plus the Monte Carlo variance of the reference mean.
    z_K = (mean(r.k) - mean(k_ref)) /
          sqrt(var(k_ref) / max(effective_sample_size(r.k), 10.0) + var(k_ref) / N_REF)
    pit(p) = cdf.(getproperty(prior_cdf, p), getproperty(r, p))
    push!(rows, (
        birth = join(String.(cfg.fam), "/"), fdim = cfg.resample ? "resample" : "none",
        seed = cfg.seed,
        mean_K = mean(r.k), mean_K_ref = mean(k_ref),
        tv_K = 0.5 * sum(abs.(p_chain .- p_ref)),
        z_K = z_K,
        z_ξ = zscore(pit(:ξ), 0.5, 1 / 12), z_ω = zscore(pit(:ω), 0.5, 1 / 12),
        z_α = zscore(pit(:α), 0.5, 1 / 12),
        ess_K = effective_sample_size(r.k),
        acc_birth = r.acc.birth, acc_death = r.acc.death, acc_fixed = r.acc.fixed,
    ))
end
df = DataFrame(rows)
df.max_abs_z = [maximum(abs, (r.z_K, r.z_ξ, r.z_ω, r.z_α)) for r in eachrow(df)]
const SUFFIX = RERUN ? "_rerun" : ""
CSV.write(joinpath(OUTDIR, "summary_geweke$(SUFFIX).csv"), df)

println()
@printf("%-22s %-9s %7s %7s %7s %7s %7s %7s %7s\n",
        "ξ / ω / α", "fixed", "E[K]", "TV(K)", "z_K", "z_ξ", "z_ω", "z_α", "ESS(K)")
for r in eachrow(df)
    @printf("%-22s %-9s %7.3f %7.4f %7.2f %7.2f %7.2f %7.2f %7.0f%s\n", r.birth, r.fdim, r.mean_K,
            r.tv_K, r.z_K, r.z_ξ, r.z_ω, r.z_α, r.ess_K, r.max_abs_z > 3.5 ? "  <-- FAILS" : "")
end
@printf("Reference E[K] = %.3f\n", mean(k_ref))

if !RERUN
    open(joinpath(OUTDIR, "geweke_reference.txt"), "w") do io
        @printf(io, "reference_mean_K %.4f\nn_ref %d\nn_samples %d\n", mean(k_ref), N_REF, N_SAMPLES)
    end
end
println("Wrote $(joinpath(OUTDIR, "summary_geweke$(SUFFIX).csv"))")

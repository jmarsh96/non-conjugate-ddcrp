# ============================================================================
# run_skewnormal.jl
#
# Simulation study for SkewNormal clustering — a genuinely 3-parameter, FULLY
# non-conjugate likelihood (location ξ ∈ ℝ, scale ω > 0, skew α ∈ ℝ). There is
# NO marginalised Gibbs baseline; correctness is judged by agreement across
# independent chains (split-R̂ over SEEDS) and across the RJMCMC variants, and
# by the prior-recovery check in `geweke_skewnormal.jl`.
#
# GOAL: compare every applicable birth-proposal family FOR EACH cluster parameter
# and decide which is best per parameter.
#   ξ (ℝ)  ∈ {Prior, NormalMM}
#   ω (>0) ∈ {Prior, NormalMM (truncated), LogNormalMM, InverseGammaMM}
#   α (ℝ)  ∈ {Prior, NormalMM}
# We use a coordinate design: start from the all-moment-match baseline
# (ξ:NMM, ω:LNMM, α:NMM) and swap each parameter's family across its options
# while holding the other two at baseline, plus an all-Prior reference and the
# combination the coordinate comparison recommends (ξ:NMM, ω:Prior, α:Prior).
# Every birth config is run under both fixed-dim strategies {NoUpdate, Resample}.
#
# Parts:
#   1. Tuning — the birth σ of each moment-matched parameter, one parameter at a
#      time in the order ξ, ω, α, on the overlapping scenario (baseline birth,
#      NoUpdate), selected by ESJD(K) averaged over two short chains. Writes
#      summary_tune.csv and tuned_sigma.csv (read by geweke_skewnormal.jl).
#   2. Comparison — every config × scenario × seed in SEEDS at full length.
#      Only the first seed's chain is saved (thinned); the rest are reduced to
#      their summaries. Writes summary_final_seeds.csv (one row per chain) and
#      summary_final.csv (one row per config, pooled over seeds with split-R̂).
#
# Fixed proposal parameters, from short runs:
#   within-cluster MH steps : prop_sds = (ξ=0.6, ω=0.15, α=1.2)   [α was the bottleneck]
#   resample σ (NMM/LNMM)   : 0.25 for every parameter
# (InverseGammaMM and Prior are parameter-free.)
#
# Two scenarios: overlapping (harder) and separated (clean recovery).
# ============================================================================

using DistanceDependentCRP, CSV, DataFrames, Distances, Statistics, StatsBase, Random, JLD2, Printf, Distributions
using Distributed

const USE_SLURM = "--slurm" in ARGS
const IS_TEST = "--test" in ARGS
const INFER_S = "--infer_s" in ARGS

if USE_SLURM
    using SlurmClusterManager
    # launch_timeout defaults to 60s, which is not enough for a wide allocation
    # loading packages off /rds; the workers then die with "launch_timeout
    # exceeded" before they can call home.
    addprocs(SlurmManager(launch_timeout = 300.0))
else
    addprocs(8)
end

@everywhere begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using DistanceDependentCRP, Statistics, StatsBase, Distributions, JLD2, Random
    include(joinpath(@__DIR__, "skewnormal_model.jl"))
end

# split_rhat lives with the dengue helpers; the file holds pure functions only.
include(joinpath(@__DIR__, "dengue_common.jl"))

# ============================================================================
# Proposal parameters
# ============================================================================

@everywhere const PROP_SDS = (ξ = 0.6, ω = 0.15, α = 1.2)   # within-cluster MH
const SIGMA_RESAMPLE = 0.25                                  # NMM / LNMM resample σ
const SIGMA_START = (ξ = 1.0, ω = 1.0, α = 1.0)              # before tuning

# Birth σ grids, tuned in this order. ω's σ is on the log scale for the
# log-normal baseline; the truncated-normal ω proposal uses the same value on
# the natural scale.
const SIGMA_GRID = IS_TEST ?
    (ξ = [1.0, 2.0], ω = [0.5, 1.0], α = [1.0, 5.0]) :
    (ξ = [0.5, 1.0, 2.0, 4.0], ω = [0.25, 0.5, 1.0], α = [1.0, 2.5, 5.0, 10.0])

# ============================================================================
# Per-parameter proposal-family builders
# ============================================================================

@everywhere function _birth_family(fam::Symbol, σ)
    fam === :prior && return PriorProposal()
    fam === :nmm   && return NormalMomentMatch(σ)
    fam === :lnmm  && return LogNormalMomentMatch(σ)
    fam === :ig    && return InverseGammaMomentMatch()
    error("unknown family $fam")
end

# σ is a NamedTuple (ξ, ω, α) of birth standard deviations.
@everywhere make_birth(fξ, fω, fα, σ) = MixedProposal((
    ξ = _birth_family(fξ, σ.ξ), ω = _birth_family(fω, σ.ω), α = _birth_family(fα, σ.α)))

@everywhere make_fdim(fξ, fω, fα, σr; resample::Bool) =
    resample ? MixedFixedDim((
        ξ = Resample(_birth_family(fξ, σr)),
        ω = Resample(_birth_family(fω, σr)),
        α = Resample(_birth_family(fα, σr)))) : NoUpdate()

# ============================================================================
# Unit of work
# ============================================================================

const PK_MAX = 12   # P(K = k) is recorded for k = 1..PK_MAX

@everywhere function run_and_save(cfg)
    Random.seed!(cfg.seed)

    opts = MCMCOptions(
        n_samples = cfg.n_samples,
        verbose = false,
        track_diagnostics = true,
        infer_params = Dict{Symbol,Bool}(:α_ddcrp => true, :s_ddcrp => cfg.infer_s),
        prop_sds = Dict{Symbol,Float64}(
            :ξ => PROP_SDS.ξ, :ω => PROP_SDS.ω, :α => PROP_SDS.α, :s_ddcrp => 0.3),
    )

    birth = make_birth(cfg.fam_ξ, cfg.fam_ω, cfg.fam_α, cfg.σ_birth)
    fdim = make_fdim(cfg.fam_ξ, cfg.fam_ω, cfg.fam_α, cfg.σ_resample; resample = cfg.resample)

    total_time = @elapsed begin
        samples, diag = mcmc(
            SkewNormalClusters(), ContinuousData(cfg.y, cfg.D), cfg.ddcrp_params, cfg.priors, birth;
            fixed_dim_proposal = fdim, opts = opts,
        )
    end

    keep = (cfg.burn + 1):cfg.thin:cfg.n_samples
    post_c = samples.c[keep, :]
    post_lp = samples.logpost[keep]

    k_trace = calculate_n_clusters(post_c)
    ari_tr = compute_ari_trace(post_c, cfg.true_c)
    vi_tr = compute_vi_trace(post_c, cfg.true_c)
    acc = acceptance_rates(diag)
    ess_K = effective_sample_size(Float64.(k_trace))
    ess_logpost = effective_sample_size(post_lp)
    esjd_K = mean(diff(Float64.(k_trace)).^2)

    # Per-parameter mixing (median ESS across observations, post burn-in/thinning).
    med_ess(M) = median([effective_sample_size(M[keep, i]) for i in 1:size(M, 2)])

    if cfg.save_chain
        # Save the thinned post-burn-in chain only, as burn = 0 and thin = 1,
        # so the analysis script reads it exactly as it would a raw chain.
        thinned = SkewNormalClustersSamples(post_c, samples.ξ[keep, :], samples.ω[keep, :],
            samples.α[keep, :], post_lp, samples.α_ddcrp[keep], samples.s_ddcrp[keep])
        chain_dir = joinpath(cfg.outdir, cfg.scenario, "chains")
        mkpath(chain_dir)
        jldsave(joinpath(chain_dir, cfg.chain_filename); samples=thinned, burn=0, thin=1)
    end

    return (
        scenario = cfg.scenario,
        label = cfg.label,
        seed = cfg.seed,
        fam_ξ = String(cfg.fam_ξ),
        fam_ω = String(cfg.fam_ω),
        fam_α = String(cfg.fam_α),
        birth = "$(cfg.fam_ξ)/$(cfg.fam_ω)/$(cfg.fam_α)",
        fdim = cfg.resample ? "resample" : "none",
        varied = cfg.varied,
        σ_ξ = cfg.σ_birth.ξ, σ_ω = cfg.σ_birth.ω, σ_α = cfg.σ_birth.α,
        n_post = length(k_trace),
        mean_K = mean(k_trace),
        mode_K = Int(mode(k_trace)),
        std_K = std(Float64.(k_trace)),
        prob_K_true = mean(k_trace .== cfg.k_true),
        acc_birth = acc.birth,
        acc_death = acc.death,
        acc_fixed = acc.fixed,
        acc_overall = acc.overall,
        mean_ari = mean(ari_tr),
        final_ari = ari_tr[end],
        mean_vi = mean(vi_tr),
        ess_K = ess_K,
        ess_logpost = ess_logpost,
        med_ess_ξ = med_ess(samples.ξ),
        med_ess_ω = med_ess(samples.ω),
        med_ess_α = med_ess(samples.α),
        esjd_K = esjd_K,
        ess_per_sec = ess_K / total_time,
        total_time = total_time,
        # Traces for split-R̂ across seeds; dropped before the per-chain CSV.
        k_trace = Float64.(k_trace),
        lp_trace = post_lp,
    )
end

# ============================================================================
# Constants
# ============================================================================

const N_SAMPLES = IS_TEST ? 1_500 : 110_000
const BURN = IS_TEST ? 300 : 10_000
const N_TUNE = IS_TEST ? 1_000 : 40_000
const BURN_TUNE = IS_TEST ? 200 : 5_000
const THIN = 5
const SEEDS = IS_TEST ? [42, 202] : [42, 202, 7, 13]
const TUNE_SEEDS = [42, 202]
const K_TRUE = 3
const N_OBS = 150

const SKEW_DIR = IS_TEST ? joinpath(@__DIR__, "..", "results", "simulation_study", "skewnormal_test") :
    INFER_S ? joinpath(@__DIR__, "..", "results", "simulation_study", "skewnormal_infer_s") :
    joinpath(@__DIR__, "..", "results", "simulation_study", "skewnormal")
mkpath(SKEW_DIR)

# ============================================================================
# Simulate data
# ============================================================================

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

Random.seed!(42)
const SIM_OVERLAPPING = build_scenario(
    [-3.0, 0.0, 3.0], 1.5, [(0.0, 2.0, 5.0), (4.0, 2.0, 0.0), (8.0, 2.0, -5.0)]; x_seed=0, y_seed=2)
const SIM_SEPARATED = build_scenario(
    [-6.0, 0.0, 6.0], 1.0, [(0.0, 1.5, 5.0), (8.0, 1.5, 0.0), (16.0, 1.5, -5.0)]; x_seed=1, y_seed=3)
const SCENARIOS = [("overlapping", SIM_OVERLAPPING), ("separated", SIM_SEPARATED)]

jldsave(joinpath(SKEW_DIR, "sim_data.jld2");
        sim_overlapping=SIM_OVERLAPPING, sim_separated=SIM_SEPARATED)
println("Saved sim_data.jld2")
for (sname, sim) in SCENARIOS
    @printf("%-12s y: min=%.2f max=%.2f  (true (ξ,ω,α) = %s)\n",
            sname, minimum(sim.y), maximum(sim.y), string(sim.pars))
end

# ============================================================================
# Shared model settings
# ============================================================================

const DDCRP_PARAMS = DDCRPParams(1.0, 0.5, DistanceDependentCRP.exp_decay, 1.0, 0.01, 1.0, 0.01)
# Deliberately diffuse (uninformative) priors: wide location, heavy-tailed scale
# (InverseGamma shape 2 ⇒ infinite variance), wide skew.
const PRIORS = SkewNormalClustersPriors(ξ_m=0.0, ξ_s=50.0, ω_a=2.0, ω_b=2.0, α_m=0.0, α_s=10.0)

# ============================================================================
# Config builder
# ============================================================================

fmtlabel(s) = replace(lowercase(s), r"[^a-z0-9]+" => "_")

function make_cfg(scenario_name, sim, fam_ξ, fam_ω, fam_α, resample, varied;
                  σ_birth, seed, n_samples = N_SAMPLES, burn = BURN,
                  save_chain = false, suffix = "")
    label = "Skew ξ:$(fam_ξ) ω:$(fam_ω) α:$(fam_α) + $(resample ? "Resample" : "NoUpdate")"
    (
        label = label,
        scenario = scenario_name,
        fam_ξ = fam_ξ, fam_ω = fam_ω, fam_α = fam_α,
        resample = resample,
        varied = varied,
        σ_birth = σ_birth, σ_resample = SIGMA_RESAMPLE,
        priors = PRIORS,
        y = sim.y, D = sim.D, true_c = sim.c, k_true = K_TRUE,
        ddcrp_params = DDCRP_PARAMS,
        n_samples = n_samples, burn = burn, thin = THIN,
        outdir = SKEW_DIR,
        chain_filename = fmtlabel(label) * "_" * scenario_name * suffix * ".jld2",
        seed = seed, infer_s = INFER_S, save_chain = save_chain,
    )
end

# The per-chain CSVs hold scalars only.
strip_traces(r) = Base.structdiff(r, NamedTuple{(:k_trace, :lp_trace)})

# ============================================================================
# Part 1: tune the birth σ of each moment-matched parameter
# ============================================================================

const BASE = (:nmm, :lnmm, :nmm)

println("\n$(repeat('=', 80))")
println("Part 1: birth σ tuning (overlapping, baseline birth, NoUpdate, " *
        "$(N_TUNE) iterations × $(length(TUNE_SEEDS)) seeds, by ESJD(K))")
println("$(repeat('=', 80))")

σ_cur = SIGMA_START
tune_rows = NamedTuple[]
for p in (:ξ, :ω, :α)
    global σ_cur
    grid = getproperty(SIGMA_GRID, p)
    cfgs = [make_cfg("overlapping", SIM_OVERLAPPING, BASE..., false, "tune:$p";
                     σ_birth = merge(σ_cur, NamedTuple{(p,)}((v,))), seed = s,
                     n_samples = N_TUNE, burn = BURN_TUNE)
            for v in grid for s in TUNE_SEEDS]
    rows = [merge(strip_traces(r), (tuned = String(p),)) for r in pmap(run_and_save, cfgs)]
    append!(tune_rows, rows)
    df_p = DataFrame(rows)
    by_σ = combine(groupby(df_p, Symbol("σ_$p")), :esjd_K => mean => :esjd_K, :ess_K => mean => :ess_K)
    best = by_σ[argmax(by_σ.esjd_K), Symbol("σ_$p")]
    for r in eachrow(by_σ)
        @printf("  σ_%s = %-5.2f  ESJD(K) = %.4f  ESS(K) = %6.1f%s\n", p, r[Symbol("σ_$p")],
                r.esjd_K, r.ess_K, r[Symbol("σ_$p")] == best ? "  <- selected" : "")
    end
    σ_cur = merge(σ_cur, NamedTuple{(p,)}((best,)))
end
const SIGMA_BIRTH = σ_cur
println("Tuned birth σ: $(SIGMA_BIRTH)")

CSV.write(joinpath(SKEW_DIR, "summary_tune.csv"), DataFrame(tune_rows))
CSV.write(joinpath(SKEW_DIR, "tuned_sigma.csv"),
          DataFrame(param = ["ξ", "ω", "α"], σ = [SIGMA_BIRTH.ξ, SIGMA_BIRTH.ω, SIGMA_BIRTH.α]))

# ============================================================================
# Part 2: the comparison, every config × scenario × seed
# ============================================================================

# (fam_ξ, fam_ω, fam_α, "varied" descriptor)
birth_specs = Tuple{Symbol,Symbol,Symbol,String}[
    (BASE...,            "baseline"),
    (:prior, :lnmm, :nmm, "ξ:prior"),
    (:nmm,  :prior, :nmm, "ω:prior"),
    (:nmm,  :nmm,   :nmm, "ω:nmm"),
    (:nmm,  :ig,    :nmm, "ω:ig"),
    (:nmm,  :lnmm, :prior, "α:prior"),
    (:nmm,  :prior, :prior, "ω,α:prior"),
    (:prior, :prior, :prior, "all-prior"),
]

configs = Any[]
for (sname, sim) in SCENARIOS, (fξ, fω, fα, varied) in birth_specs, resample in (false, true), s in SEEDS
    push!(configs, make_cfg(sname, sim, fξ, fω, fα, resample, varied;
                            σ_birth = SIGMA_BIRTH, seed = s, save_chain = s == first(SEEDS)))
end

println("\n$(repeat('=', 80))")
println("Part 2: proposal comparison: $(length(configs)) chains " *
        "($(length(birth_specs)) birth families × {NoUpdate,Resample} × " *
        "$(length(SCENARIOS)) scenarios × $(length(SEEDS)) seeds)")
println("  σ_birth=$(SIGMA_BIRTH)  σ_resample=$(SIGMA_RESAMPLE)  prop_sds=$(PROP_SDS)")
println("  $(N_SAMPLES) iterations each")
println("$(repeat('=', 80))\n")

results = pmap(run_and_save, configs)

# One row per chain.
df_seeds = DataFrame([strip_traces(r) for r in results])
for k in 1:PK_MAX
    df_seeds[!, Symbol("pk_$k")] = [mean(r.k_trace .== k) for r in results]
end
CSV.write(joinpath(SKEW_DIR, "summary_final_seeds.csv"), df_seeds)

# One row per config, pooled over seeds. P(K) is pooled over all draws; its
# Monte Carlo standard error is the spread of the per-seed estimates, and
# split-R̂ compares the seeds' chains. ESS columns are means per chain.
groups = Dict{Tuple{String,String}, Vector{Any}}()
for r in results
    push!(get!(groups, (r.scenario, r.label), Any[]), r)
end
agg_rows = NamedTuple[]
for ((sname, label), rs) in groups
    r1 = rs[1]
    pvals = [r.prob_K_true for r in rs]
    push!(agg_rows, (
        scenario = sname, label = label,
        fam_ξ = r1.fam_ξ, fam_ω = r1.fam_ω, fam_α = r1.fam_α,
        birth = r1.birth, fdim = r1.fdim, varied = r1.varied,
        σ_ξ = r1.σ_ξ, σ_ω = r1.σ_ω, σ_α = r1.σ_α,
        n_seeds = length(rs),
        mean_K = mean(r.mean_K for r in rs),
        prob_K_true = mean(pvals),
        prob_K_true_se = std(pvals) / sqrt(length(rs)),
        prob_K_true_min = minimum(pvals),
        prob_K_true_max = maximum(pvals),
        rhat_K = split_rhat([r.k_trace for r in rs]),
        rhat_logpost = split_rhat([r.lp_trace for r in rs]),
        esjd_K = mean(r.esjd_K for r in rs),
        ess_K = mean(r.ess_K for r in rs),
        ess_logpost = mean(r.ess_logpost for r in rs),
        med_ess_ξ = mean(r.med_ess_ξ for r in rs),
        med_ess_ω = mean(r.med_ess_ω for r in rs),
        med_ess_α = mean(r.med_ess_α for r in rs),
        acc_birth = mean(r.acc_birth for r in rs),
        acc_death = mean(r.acc_death for r in rs),
        acc_fixed = mean(r.acc_fixed for r in rs),
        mean_ari = mean(r.mean_ari for r in rs),
        total_time = mean(r.total_time for r in rs),
        pk = [mean(mean(r.k_trace .== k) for r in rs) for k in 1:PK_MAX],
    ))
end
df = DataFrame(agg_rows)
for k in 1:PK_MAX
    df[!, Symbol("pk_$k")] = [v[k] for v in df.pk]
end
select!(df, Not(:pk))
sort!(df, [:scenario, :fdim, :varied])

function print_summary_table(df, title)
    W = 150
    println(); println("=" ^ W)
    println(title * " (pooled over $(length(SEEDS)) seeds)"); println("=" ^ W)
    println(rpad("ξ / ω / α", 20) * rpad("Fixed-dim", 11) *
            "Mean K  P(K=3)  ±SE     R̂(K)   MeanARI  Acc.B%  Acc.F%  ESS(K)  ESS_ξ  ESS_ω  ESS_α  ESJD(K)")
    println("-" ^ W)
    for r in eachrow(df)
        @printf("%-20s%-11s%6.2f  %6.3f  %5.3f  %6.3f  %7.3f  %6.3f  %6.3f  %6.0f %6.0f %6.0f %6.0f  %7.4f\n",
            r.birth, r.fdim, r.mean_K, r.prob_K_true, r.prob_K_true_se, r.rhat_K, r.mean_ari,
            r.acc_birth*100, r.acc_fixed*100, r.ess_K,
            r.med_ess_ξ, r.med_ess_ω, r.med_ess_α, r.esjd_K)
    end
    println("=" ^ W)
end

for (sname, _) in SCENARIOS
    sub = filter(r -> r.scenario == sname, df)
    print_summary_table(sub, "SkewNormal proposal comparison [$sname]")
end

CSV.write(joinpath(SKEW_DIR, "summary_final.csv"), df)
println("\nComparison summary → $(joinpath(SKEW_DIR, "summary_final.csv"))")

for (sname, _) in SCENARIOS
    scenario_dir = joinpath(SKEW_DIR, sname); mkpath(scenario_dir)
    CSV.write(joinpath(scenario_dir, "summary_final_$(sname).csv"), filter(r -> r.scenario == sname, df))
    CSV.write(joinpath(scenario_dir, "summary_final_seeds_$(sname).csv"),
              filter(r -> r.scenario == sname, df_seeds))
end

println("\nAll chains complete. Run src/geweke_skewnormal.jl for the prior-recovery check " *
        "and src/analyse_skewnormal.jl for figures and tables.")

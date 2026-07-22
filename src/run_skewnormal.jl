# ============================================================================
# run_skewnormal.jl
#
# Simulation study for SkewNormal clustering — a genuinely 3-parameter, FULLY
# non-conjugate likelihood (location ξ ∈ ℝ, scale ω > 0, skew α ∈ ℝ). There is
# NO marginalised Gibbs baseline; correctness is judged by mutual agreement
# across the RJMCMC variants (as in the Old Faithful analysis).
#
# GOAL: compare every applicable birth-proposal family FOR EACH cluster parameter
# and decide which is best per parameter.
#   ξ (ℝ)  ∈ {Prior, NormalMM}
#   ω (>0) ∈ {Prior, NormalMM (truncated), LogNormalMM, InverseGammaMM}
#   α (ℝ)  ∈ {Prior, NormalMM}
# We use a coordinate design: start from the all-moment-match baseline
# (ξ:NMM, ω:LNMM, α:NMM) and swap each parameter's family across its options
# while holding the other two at baseline, plus an all-Prior reference. Every
# birth config is run under both fixed-dim strategies {NoUpdate, Resample}.
#
# PROPOSAL PARAMETERS ARE MANUALLY TUNED from short runs (see scratch tuning):
#   within-cluster MH steps : prop_sds = (ξ=0.6, ω=0.15, α=1.2)   [α was the bottleneck]
#   birth σ (NMM/LNMM)      : 1.0   [near-max ESJD(K), ~1% birth acceptance]
#   resample σ (NMM/LNMM)   : 0.25  [fixed-dim acceptance ≈ 0.29, best ESJD balance]
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
    addprocs(SlurmManager())
else
    addprocs(8)
end

@everywhere begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using DistanceDependentCRP, Statistics, StatsBase, Distributions, JLD2, Random
    include(joinpath(@__DIR__, "skewnormal_model.jl"))
end

# ============================================================================
# Manually-tuned proposal parameters (from short runs)
# ============================================================================

@everywhere const PROP_SDS = (ξ = 0.6, ω = 0.15, α = 1.2)   # within-cluster MH
const SIGMA_BIRTH = 1.0                                      # NMM / LNMM birth σ
const SIGMA_RESAMPLE = 0.25                                  # NMM / LNMM resample σ

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

@everywhere make_birth(fξ, fω, fα, σ) = MixedProposal((
    ξ = _birth_family(fξ, σ), ω = _birth_family(fω, σ), α = _birth_family(fα, σ)))

@everywhere make_fdim(fξ, fω, fα, σr; resample::Bool) =
    resample ? MixedFixedDim((
        ξ = Resample(_birth_family(fξ, σr)),
        ω = Resample(_birth_family(fω, σr)),
        α = Resample(_birth_family(fα, σr)))) : NoUpdate()

# ============================================================================
# Unit of work
# ============================================================================

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

    post_c = samples.c[(cfg.burn + 1):cfg.thin:end, :]
    post_lp = samples.logpost[(cfg.burn + 1):cfg.thin:end]

    k_trace = calculate_n_clusters(post_c)
    ari_tr = compute_ari_trace(post_c, cfg.true_c)
    vi_tr = compute_vi_trace(post_c, cfg.true_c)
    acc = acceptance_rates(diag)
    ess_K = effective_sample_size(Float64.(k_trace))
    ess_logpost = effective_sample_size(post_lp)
    esjd_K = mean(diff(Float64.(k_trace)).^2)

    # Per-parameter mixing (median ESS across observations, post burn-in/thinning).
    keep = (cfg.burn + 1):cfg.thin:cfg.n_samples
    med_ess(M) = median([effective_sample_size(M[keep, i]) for i in 1:size(M, 2)])

    chain_dir = joinpath(cfg.outdir, cfg.scenario, "chains")
    mkpath(chain_dir)
    jldsave(joinpath(chain_dir, cfg.chain_filename); samples=samples, burn=cfg.burn, thin=cfg.thin)

    return (
        scenario = cfg.scenario,
        label = cfg.label,
        fam_ξ = String(cfg.fam_ξ),
        fam_ω = String(cfg.fam_ω),
        fam_α = String(cfg.fam_α),
        birth = "$(cfg.fam_ξ)/$(cfg.fam_ω)/$(cfg.fam_α)",
        fdim = cfg.resample ? "resample" : "none",
        varied = cfg.varied,
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
    )
end

# ============================================================================
# Constants
# ============================================================================

const N_SAMPLES = IS_TEST ? 1_500 : 110_000
const BURN = IS_TEST ? 300 : 10_000
const THIN = 5
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
const PRIORS = SkewNormalClustersPriors(ξ_m=0.0, ξ_s=20.0, ω_a=3.0, ω_b=4.0, α_m=0.0, α_s=5.0)

# ============================================================================
# Config builder
# ============================================================================

fmtlabel(s) = replace(lowercase(s), r"[^a-z0-9]+" => "_")

function make_cfg(scenario_name, sim, fam_ξ, fam_ω, fam_α, resample, varied)
    label = "Skew ξ:$(fam_ξ) ω:$(fam_ω) α:$(fam_α) + $(resample ? "Resample" : "NoUpdate")"
    (
        label = label,
        scenario = scenario_name,
        fam_ξ = fam_ξ, fam_ω = fam_ω, fam_α = fam_α,
        resample = resample,
        varied = varied,
        σ_birth = SIGMA_BIRTH, σ_resample = SIGMA_RESAMPLE,
        priors = PRIORS,
        y = sim.y, D = sim.D, true_c = sim.c, k_true = K_TRUE,
        ddcrp_params = DDCRP_PARAMS,
        n_samples = N_SAMPLES, burn = BURN, thin = THIN,
        outdir = SKEW_DIR,
        chain_filename = fmtlabel(label) * "_" * scenario_name * ".jld2",
        seed = 42, infer_s = INFER_S,
    )
end

# ============================================================================
# Coordinate enumeration: swap each parameter's family, others at baseline.
# Baseline = (ξ:nmm, ω:lnmm, α:nmm).
# ============================================================================

# (fam_ξ, fam_ω, fam_α, "varied" descriptor)
const BASE = (:nmm, :lnmm, :nmm)
birth_specs = Tuple{Symbol,Symbol,Symbol,String}[
    (BASE...,            "baseline"),
    (:prior, :lnmm, :nmm, "ξ:prior"),
    (:nmm,  :prior, :nmm, "ω:prior"),
    (:nmm,  :nmm,   :nmm, "ω:nmm"),
    (:nmm,  :ig,    :nmm, "ω:ig"),
    (:nmm,  :lnmm, :prior, "α:prior"),
    (:prior, :prior, :prior, "all-prior"),
]

configs = Any[]
for (sname, sim) in SCENARIOS
    for (fξ, fω, fα, varied) in birth_specs
        for resample in (false, true)
            push!(configs, make_cfg(sname, sim, fξ, fω, fα, resample, varied))
        end
    end
end

# ============================================================================
# Summary printer
# ============================================================================

function print_summary_table(df, title)
    W = 152
    println(); println("=" ^ W)
    println(title * " (post burn-in, thinned by $THIN)"); println("=" ^ W)
    println(rpad("ξ / ω / α", 20) * rpad("Fixed-dim", 11) *
            "Mean K  Mode K  P(K=3)  MeanARI  Acc.B%  Acc.D%  Acc.F%  " *
            "ESS(K)  ESS_ξ  ESS_ω  ESS_α  ESJD(K)  Time(s)")
    println("-" ^ W)
    for r in eachrow(df)
        @printf("%-20s%-11s%6.2f  %6d  %6.3f  %7.3f  %6.3f  %6.3f  %6.3f  %6.0f %6.0f %6.0f %6.0f  %7.4f  %7.1f\n",
            r.birth, r.fdim, r.mean_K, r.mode_K, r.prob_K_true, r.mean_ari,
            r.acc_birth*100, r.acc_death*100, r.acc_fixed*100, r.ess_K,
            r.med_ess_ξ, r.med_ess_ω, r.med_ess_α, r.esjd_K, r.total_time)
    end
    println("=" ^ W)
end

# ============================================================================
# Run the comparison
# ============================================================================

println("\n$(repeat('=', 80))")
println("Proposal comparison: $(length(configs)) configs " *
        "($(length(birth_specs)) birth families × {NoUpdate,Resample} × $(length(SCENARIOS)) scenarios)")
println("  σ_birth=$(SIGMA_BIRTH)  σ_resample=$(SIGMA_RESAMPLE)  prop_sds=$(PROP_SDS)")
println("  $(N_SAMPLES) iterations each")
println("$(repeat('=', 80))\n")

rows = pmap(run_and_save, configs)
df = DataFrame(collect(rows))

for (sname, _) in SCENARIOS
    sub = filter(r -> r.scenario == sname, df)
    print_summary_table(sub, "SkewNormal proposal comparison [$sname]")
end

CSV.write(joinpath(SKEW_DIR, "summary_final.csv"), df)
println("\nComparison summary → $(joinpath(SKEW_DIR, "summary_final.csv"))")

for (sname, _) in SCENARIOS
    sub = filter(r -> r.scenario == sname, df)
    scenario_dir = joinpath(SKEW_DIR, sname); mkpath(scenario_dir)
    CSV.write(joinpath(scenario_dir, "summary_final_$(sname).csv"), sub)
end

# ── "Best family per parameter" summary (by median parameter ESS, NoUpdate rows) ──
println("\nBest birth family per parameter (by median param ESS, NoUpdate rows):")
for (sname, _) in SCENARIOS
    println("  [$sname]")
    sub = filter(r -> r.scenario == sname && r.fdim == "none", df)
    for (pname, esscol, famcol, options) in [
            ("ξ", :med_ess_ξ, :fam_ξ, ["prior","nmm"]),
            ("ω", :med_ess_ω, :fam_ω, ["prior","nmm","lnmm","ig"]),
            ("α", :med_ess_α, :fam_α, ["prior","nmm"])]
        # rows where this parameter is the one varied (others at baseline) OR baseline
        cand = filter(r -> getproperty(r, famcol) in options &&
                           (r.varied == "$pname:$(getproperty(r,famcol))" ||
                            r.varied == "baseline" || r.varied == "all-prior"), sub)
        isempty(cand) && continue
        best = cand[argmax([getproperty(r, esscol) for r in eachrow(cand)]), :]
        @printf("    %s: best = %-6s  (median ESS=%.1f, P(K=3)=%.3f, ARI=%.3f)\n",
                pname, getproperty(best, famcol), getproperty(best, esscol),
                best.prob_K_true, best.mean_ari)
    end
end

println("\nAll chains complete. Run src/analyse_skewnormal.jl to generate figures and tables.")

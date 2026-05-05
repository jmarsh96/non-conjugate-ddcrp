# ============================================================================
# run_poisson.jl
#
# Simulation study for Poisson clustering (no population offset).
#
# Scenario: overlapping — cluster rates (1, 4, 7); harder separation
#
# Part 1 — Proposal-type sweep (parallel): 7 configs × 110,000 iterations
#   Compares:
#     - PoissonClusterRatesMarg  – Gibbs (λ_k marginalised)
#     - PoissonClusterRates      × birth ∈ {Prior, NMM σ=0.5, LNMM σ=0.5}
#                                × fixed-dim ∈ {NoUpdate, Resample}
#
# Part 2 — Tuning-parameter sweep (parallel): shorter runs to select σ
#   NMM and LNMM:
#     NoUpdate configs : 1D sweep over σ_birth
#     Resample configs : 2D grid σ_birth × σ_resample
#   Selects best σ per (proposal, fixed-dim) group by ESJD(K).
#
# Part 3 — Final comparison (parallel): full-length chains for 4 best-tuned
#   RJMCMC variants alongside Gibbs and Prior baselines from Part 1.
#
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
end

@everywhere function run_and_save(cfg)
    Random.seed!(cfg.seed)

    opts = MCMCOptions(
        n_samples = cfg.n_samples,
        verbose = false,
        track_diagnostics = true,
        infer_params = Dict{Symbol,Bool}(
            :α_ddcrp => true,
            :s_ddcrp => cfg.infer_s,
        ),
    )

    total_time = @elapsed begin
        samples, diag = mcmc(
            cfg.model, cfg.y, cfg.D, cfg.ddcrp_params, cfg.priors, cfg.birth;
            fixed_dim_proposal = cfg.fdim,
            opts = opts,
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
    esjd_logpost = mean(diff(post_lp).^2)

    chain_dir = joinpath(cfg.outdir, cfg.scenario, "chains")
    mkpath(chain_dir)
    chain_path = joinpath(chain_dir, cfg.chain_filename)
    jldsave(chain_path; samples=samples, burn=cfg.burn, thin=cfg.thin)

    return (
        scenario = cfg.scenario,
        label = cfg.label,
        birth = cfg.birth_label,
        fdim = cfg.fdim_label,
        tune_param = cfg.tune_param,
        tune_param_r = cfg.tune_param_r,
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
        esjd_K = esjd_K,
        esjd_logpost = esjd_logpost,
        ess_per_sec = ess_K / total_time,
        total_time = total_time,
    )
end

# ============================================================================
# Constants
# ============================================================================

const N_SAMPLES_SWEEP = IS_TEST ? 1_000 : 110_000
const BURN_SWEEP = IS_TEST ? 200 : 10_000
const N_SAMPLES_TUNE = IS_TEST ? 1_000 : 60_000
const BURN_TUNE = IS_TEST ? 200 : 10_000
const THIN = 5
const K_TRUE = 3
const N_OBS = 150

const POISSON_DIR = IS_TEST ? joinpath(@__DIR__, "..", "results", "simulation_study", "poisson_test") :
    INFER_S ? joinpath(@__DIR__, "..", "results", "simulation_study", "poisson_infer_s") :
    joinpath(@__DIR__, "..", "results", "simulation_study", "poisson")
mkpath(POISSON_DIR)

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

Random.seed!(42)
x_centers = [-3.0, 0.0, 3.0]
x = make_cluster_x(N_OBS, x_centers, 1.5; seed=0)

# Pointer representation: each obs points to the first member of its cluster.
# Cluster 1: obs 1–50  → all point to 1
# Cluster 2: obs 51–100 → all point to 51
# Cluster 3: obs 101–150 → all point to 101
const C_FIXED = vcat([fill(1 + (k - 1) * (N_OBS ÷ K_TRUE), N_OBS ÷ K_TRUE) for k in 1:K_TRUE]...)
const D_SIM = pairwise(Euclidean(), reshape(x, 1, :))

function simulate_poisson(c, λs; seed=1)
    rng = MersenneTwister(seed)
    labels = compute_table_assignments(c)   # convert pointer repr → labels 1…K
    [rand(rng, Poisson(λs[labels[i]])) for i in eachindex(c)]
end

const Y_OVERLAPPING = simulate_poisson(C_FIXED, [1.0, 4.0, 7.0]; seed=2)

const SIM_OVERLAPPING = (y=Y_OVERLAPPING, D=D_SIM, c=C_FIXED, x=x, λs=[1.0, 4.0, 7.0])

jldsave(joinpath(POISSON_DIR, "sim_data.jld2"); sim_overlapping=SIM_OVERLAPPING)
println("Saved sim_data.jld2")
@printf(
    "Overlapping y: min=%d max=%d  (true λ = %s)\n",
    minimum(Y_OVERLAPPING), maximum(Y_OVERLAPPING), join(SIM_OVERLAPPING.λs, ", ")
)

const SCENARIOS = [("overlapping", SIM_OVERLAPPING)]

# ============================================================================
# Shared model settings
# ============================================================================

const DDCRP_PARAMS = DDCRPParams(1.0, 0.5, DistanceDependentCRP.exp_decay, 1.0, 0.01, 1.0, 0.01)
const PRIORS = PoissonClusterRatesPriors(1.0, 0.1)
const PRIORS_MARG = PoissonClusterRatesMargPriors(1.0, 0.1)

const σ_NMM = 0.5
const σ_LNMM = 0.5

# ============================================================================
# Config builder
# ============================================================================

fmtlabel(s) = replace(lowercase(s), r"[^a-z0-9]+" => "_")

function make_cfg(label, scenario_name, sim, model, priors, birth, fdim,
                  birth_label, fdim_label;
                  seed=42, n_samples=N_SAMPLES_SWEEP, burn=BURN_SWEEP,
                  tune_param=NaN, tune_param_r=NaN, filename_suffix="")
    (
        label = label,
        scenario = scenario_name,
        model = model,
        priors = priors,
        birth = birth,
        fdim = fdim,
        birth_label = birth_label,
        fdim_label = fdim_label,
        y = sim.y,
        D = sim.D,
        true_c = sim.c,
        k_true = K_TRUE,
        ddcrp_params = DDCRP_PARAMS,
        n_samples = n_samples,
        burn = burn,
        thin = THIN,
        outdir = POISSON_DIR,
        chain_filename = fmtlabel(label) * "_" * scenario_name * filename_suffix * ".jld2",
        seed = seed,
        tune_param = tune_param,
        tune_param_r = tune_param_r,
        infer_s = INFER_S,
    )
end

# ============================================================================
# Summary printer
# ============================================================================

function print_summary_table(df, title, thin_factor)
    W = 135
    println()
    println("=" ^ W)
    println(title * " (post burn-in, thinned by $thin_factor)")
    println("=" ^ W)
    header = rpad("Label", 48) *
             "  Birth       Fixed-dim      σ_b   σ_r    Mean K  Mode K   Std K  " *
             "MeanARI  Acc.B%  Acc.O%    ESS(K)  ESS/sec   Time(s)"
    println(header)
    println("-" ^ W)
    for r in eachrow(df)
        b_pct = isnan(r.acc_birth) ? "   —  " : @sprintf("%6.3f", r.acc_birth * 100)
        o_pct = isnan(r.acc_overall) ? "   —  " : @sprintf("%6.3f", r.acc_overall * 100)
        σb_str = isnan(r.tune_param) ? "  —  " : @sprintf("%5.2f", r.tune_param)
        σr_str = isnan(r.tune_param_r) ? "  —  " : @sprintf("%5.2f", r.tune_param_r)
        println(
            rpad(r.label, 48) * "  " *
            rpad(r.birth, 10) * "  " *
            rpad(r.fdim,  10) * "  " *
            @sprintf(
                "%s  %s  %6.2f  %6d  %6.3f  %6.3f  %s  %s  %8.1f  %8.3f  %7.1f",
                σb_str, σr_str,
                r.mean_K, r.mode_K, r.std_K, r.mean_ari,
                b_pct, o_pct, r.ess_K, r.ess_per_sec, r.total_time
            )
        )
    end
    println("=" ^ W)
end

# ============================================================================
# Part 1: Parallel proposal sweep
# ============================================================================

sweep_configs = Any[]
for (sname, sim) in SCENARIOS
    append!(sweep_configs, [
        make_cfg(
            "Pois Marg (Gibbs)", sname, sim,
            PoissonClusterRatesMarg(), PRIORS_MARG,
            ConjugateProposal(), NoUpdate(),
            "conjugate", "none",
        ),
        make_cfg(
            "Pois Prior + NoUpdate", sname, sim,
            PoissonClusterRates(), PRIORS,
            PriorProposal(), NoUpdate(),
            "prior", "none",
        ),
        make_cfg(
            "Pois Prior + Resample", sname, sim,
            PoissonClusterRates(), PRIORS,
            PriorProposal(), Resample(PriorProposal()),
            "prior", "resample",
        ),
        make_cfg(
            "Pois NMM σ=0.50 + NoUpdate", sname, sim,
            PoissonClusterRates(), PRIORS,
            NormalMomentMatch(σ_NMM), NoUpdate(),
            "nmm", "none";
            tune_param = σ_NMM,
        ),
        make_cfg(
            "Pois NMM σ=0.50 + Resample", sname, sim,
            PoissonClusterRates(), PRIORS,
            NormalMomentMatch(σ_NMM), Resample(NormalMomentMatch(σ_NMM)),
            "nmm", "resample";
            tune_param = σ_NMM, tune_param_r = σ_NMM,
        ),
        make_cfg(
            "Pois LNMM σ=0.50 + NoUpdate", sname, sim,
            PoissonClusterRates(), PRIORS,
            LogNormalMomentMatch(σ_LNMM), NoUpdate(),
            "lnmm", "none";
            tune_param = σ_LNMM,
        ),
        make_cfg(
            "Pois LNMM σ=0.50 + Resample", sname, sim,
            PoissonClusterRates(), PRIORS,
            LogNormalMomentMatch(σ_LNMM), Resample(LogNormalMomentMatch(σ_LNMM)),
            "lnmm", "resample";
            tune_param = σ_LNMM, tune_param_r = σ_LNMM,
        ),
    ])
end

println("\n$(repeat('=', 80))")
println("Part 1: Proposal sweep ($(length(sweep_configs)) configs × $(N_SAMPLES_SWEEP) iterations)")
println("$(repeat('=', 80))\n")

sweep_rows = pmap(run_and_save, sweep_configs)
sweep_df = DataFrame(collect(sweep_rows))

for (sname, _) in SCENARIOS
    sub = filter(r -> r.scenario == sname, sweep_df)
    print_summary_table(sub, "Poisson proposal sweep [$sname]", THIN)
end

CSV.write(joinpath(POISSON_DIR, "summary_sweep.csv"), sweep_df)
println("Sweep summary → $(joinpath(POISSON_DIR, "summary_sweep.csv"))")

# ============================================================================
# Part 2: Tuning-parameter sweep for NMM and LNMM
# ============================================================================

const σ_BIRTH_SWEEP = IS_TEST ? [0.1, 0.5, 1.0] : [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
const σ_RESAMPLE_SWEEP = IS_TEST ? [0.05, 0.25, 1.0] : [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

tune_configs = Any[]

for (sname, sim) in SCENARIOS
    for (pname, make_prop) in [
            ("nmm",  s -> NormalMomentMatch(s)),
            ("lnmm", s -> LogNormalMomentMatch(s)),
        ]

        for σ_b in σ_BIRTH_SWEEP
            σ_b_tag = @sprintf("%.2f", σ_b)
            label = "Pois $(uppercase(pname)) σ=$(σ_b_tag) + NoUpdate"
            push!(tune_configs, make_cfg(
                label, sname, sim,
                PoissonClusterRates(), PRIORS,
                make_prop(σ_b), NoUpdate(),
                pname, "none";
                tune_param = σ_b,
                tune_param_r = NaN,
                n_samples = N_SAMPLES_TUNE,
                burn = BURN_TUNE,
                filename_suffix = "_tune",
            ))
        end

        for σ_b in σ_BIRTH_SWEEP, σ_r in σ_RESAMPLE_SWEEP
            σ_b_tag = @sprintf("%.2f", σ_b)
            σ_r_tag = @sprintf("%.2f", σ_r)
            label = "Pois $(uppercase(pname)) sb=$(σ_b_tag) sr=$(σ_r_tag) + Resample"
            push!(tune_configs, make_cfg(
                label, sname, sim,
                PoissonClusterRates(), PRIORS,
                make_prop(σ_b), Resample(make_prop(σ_r)),
                pname, "resample";
                tune_param = σ_b,
                tune_param_r = σ_r,
                n_samples = N_SAMPLES_TUNE,
                burn = BURN_TUNE,
                filename_suffix = "_tune",
            ))
        end
    end
end

n_noupdate = 2 * length(SCENARIOS) * length(σ_BIRTH_SWEEP)
n_resample = 2 * length(SCENARIOS) * length(σ_BIRTH_SWEEP) * length(σ_RESAMPLE_SWEEP)
println("\n$(repeat('=', 80))")
println("Part 2: Tuning sweep ($(length(tune_configs)) configs × $(N_SAMPLES_TUNE) iterations)")
println("  NoUpdate: σ_birth ∈ $(σ_BIRTH_SWEEP)  ($n_noupdate configs)")
println("  Resample: σ_birth × σ_resample grid   ($n_resample configs)")
println("  proposals: NMM, LNMM  ×  scenario: overlapping")
println("$(repeat('=', 80))\n")

tune_rows = pmap(run_and_save, tune_configs)
tune_df = DataFrame(collect(tune_rows))

for (sname, _) in SCENARIOS
    sub = filter(r -> r.scenario == sname, tune_df)
    print_summary_table(sub, "Poisson tuning sweep [$sname]", THIN)
end

# Print best parameters per group (per scenario)
function best_params(df, scenario_val, birth_val, fdim_val)
    mask = (df.scenario .== scenario_val) .& (df.birth .== birth_val) .& (df.fdim .== fdim_val)
    sub  = df[mask, :]
    isempty(sub) && return nothing
    esjd_sub = [isnan(r.esjd_K) ? -Inf : r.esjd_K for r in eachrow(sub)]
    best = sub[argmax(esjd_sub), :]
    return (σ_b = best.tune_param, σ_r = best.tune_param_r)
end

println("\nBest parameters per group (by ESJD(K) in tuning sweep):")
for (sname, _) in SCENARIOS
    println("  [$sname]")
    for pname in ["nmm", "lnmm"]
        p_nu = best_params(tune_df, sname, pname, "none")
        if !isnothing(p_nu)
            @printf(
                "    %-6s + NoUpdate   σ_b = %.2f             ESJD(K) = %.6f\n",
                uppercase(pname), p_nu.σ_b,
                maximum(
                    [isnan(r.esjd_K) ? -Inf : r.esjd_K
                        for r in eachrow(tune_df[
                            (tune_df.scenario .== sname) .&
                            (tune_df.birth .== pname) .&
                            (tune_df.fdim .== "none"), :])]))
        end
        p_rs = best_params(tune_df, sname, pname, "resample")
        if !isnothing(p_rs)
            @printf(
                "    %-6s + Resample   σ_b = %.2f  σ_r = %.2f  ESJD(K) = %.6f\n",
                uppercase(pname), p_rs.σ_b, p_rs.σ_r,
                maximum([isnan(r.esjd_K) ? -Inf : r.esjd_K
                    for r in eachrow(tune_df[
                        (tune_df.scenario .== sname) .&
                        (tune_df.birth .== pname) .&
                        (tune_df.fdim .== "resample"), :])]))
        end
    end
end

CSV.write(joinpath(POISSON_DIR, "summary_tune.csv"), tune_df)
println("\nTuning summary → $(joinpath(POISSON_DIR, "summary_tune.csv"))")

# ============================================================================
# Part 3: Full-length chains for best-tuned RJMCMC variants
# ============================================================================

final_rjmcmc_cfgs = Any[]
for (sname, sim) in SCENARIOS
    for (pname, fdim_name) in [("nmm","none"), ("nmm","resample"),
                                ("lnmm","none"), ("lnmm","resample")]
        p = best_params(tune_df, sname, pname, fdim_name)
        isnothing(p) && continue

        make_prop = pname == "nmm" ? (s -> NormalMomentMatch(s)) :
                                      (s -> LogNormalMomentMatch(s))
        prop = make_prop(p.σ_b)
        σ_b_tag = @sprintf("%.2f", p.σ_b)

        if fdim_name == "none"
            fdim_obj = NoUpdate()
            label = "Pois $(uppercase(pname)) σ=$(σ_b_tag) + NoUpdate [best]"
            push!(final_rjmcmc_cfgs, make_cfg(
                label, sname, sim,
                PoissonClusterRates(), PRIORS,
                prop, fdim_obj,
                pname, fdim_name;
                tune_param = p.σ_b,
                tune_param_r = NaN,
                filename_suffix = "_final",
            ))
        else
            σ_r_tag = @sprintf("%.2f", p.σ_r)
            fdim_obj = Resample(make_prop(p.σ_r))
            label = "Pois $(uppercase(pname)) sb=$(σ_b_tag) sr=$(σ_r_tag) + Resample [best]"
            push!(final_rjmcmc_cfgs, make_cfg(
                label, sname, sim,
                PoissonClusterRates(), PRIORS,
                prop, fdim_obj,
                pname, fdim_name;
                tune_param = p.σ_b,
                tune_param_r = p.σ_r,
                filename_suffix = "_final",
            ))
        end
    end
end

n_final = length(final_rjmcmc_cfgs) + 3 * length(SCENARIOS)
println("\n$(repeat('=', 80))")
println("Part 3: Final comparison (~$n_final chains × $(N_SAMPLES_SWEEP) iterations)")
println("  Gibbs + Prior baselines (from Part 1) + $(length(final_rjmcmc_cfgs)) best-tuned RJMCMC chains")
println("$(repeat('=', 80))\n")

final_rjmcmc_rows = pmap(run_and_save, final_rjmcmc_cfgs)
final_rjmcmc_df = DataFrame(collect(final_rjmcmc_rows))

baselines_df = filter(:birth => b -> b in ("conjugate", "prior"), sweep_df)
final_df = vcat(baselines_df, final_rjmcmc_df)

for (sname, _) in SCENARIOS
    sub = filter(r -> r.scenario == sname, final_df)
    print_summary_table(sub, "Poisson final model comparison [$sname]", THIN)
end

CSV.write(joinpath(POISSON_DIR, "summary_final.csv"), final_df)
println("Final summary → $(joinpath(POISSON_DIR, "summary_final.csv"))")

# Also write per-scenario CSVs for the analysis script
for (sname, _) in SCENARIOS
    for (tag, df) in [("sweep", sweep_df), ("tune", tune_df), ("final", final_df)]
        sub = filter(r -> r.scenario == sname, df)
        scenario_dir = joinpath(POISSON_DIR, sname)
        mkpath(scenario_dir)
        CSV.write(joinpath(scenario_dir, "summary_$(tag)_$(sname).csv"), sub)
    end
end

println("\nAll chains complete. Run src/analyse_poisson.jl to generate figures and tables.")

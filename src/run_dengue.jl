# ============================================================================
# run_dengue.jl
#
# Real-data application: spatial clustering of 2024 dengue incidence across the
# municipalities of a Brazilian state, under a genuinely non-conjugate Negative
# Binomial likelihood. This is the analysis reported in the paper.
#
#   y_i | ρ_k, r_k ~ NegativeBinomial(mean = E_i ρ_k, size = r_k)
#     E_i = pop_i / 1e5      ⇒ ρ_k is the cluster incidence per 100k
#     r_k                     the cluster overdispersion (r → ∞ ⇒ Poisson)
#
# BOTH cluster parameters are non-conjugate, so no collapsed sampler exists.
#
# THE ddCRP PRIOR is the hop window of radius HOP_RADIUS on the queen contiguity
# graph: link weight is uniform over municipalities within HOP_RADIUS steps and
# suppressed (by e^-30) beyond, encoded as a 0 / LOG_FLOOR distance matrix under
# kernel rate 1 (see `hop_window_D` in dengue_common.jl). The radius is FIXED;
# how much the clustering depends on it, and on the choice of metric, is the
# subject of `src/dengue_sensitivity.jl`.
#
# LINKS ARE PROPOSED FROM THE PRIOR (`link_proposal = PriorLink()`), not
# uniformly: under the window only ~19 of the 399 municipalities carry any
# weight, so a uniform draw would waste ~95% of proposals, and a likelihood
# evaluation each, on links the prior all but forbids. With the prior proposal
# the proposal ratio cancels the prior ratio, so acceptance depends only on the
# likelihood and the parameter proposal.
#
# Run `src/prepare_dengue_data.jl` and `src/prepare_dengue_adjacency.R` first.
#
# Parts, all via `pmap` over worker processes:
#   0. --pilot  : short chains at several subsample sizes, to measure cost.
#   1. Proposal-family sweep — coordinate design over the birth family for each
#      parameter, crossed with fixed-dim ∈ {NoUpdate, Resample}. The winner is
#      the configuration with the largest ESJD(K).
#   2. σ tuning — a grid over the winner's tunable proposal scales, again by
#      ESJD(K). Skipped when nothing in the winner is tunable.
#   3. Final — one chain per seed in FINAL_SEEDS for the selected sampler (their
#      agreement replaces the Gibbs baseline this model does not have), and a
#      non-marginalised Poisson comparator that exists only to be scored against
#      the NB by WAIC / LOO.
#
# Usage:
#   julia --project=. src/run_dengue.jl --pilot
#   julia --project=. src/run_dengue.jl --test
#   julia --project=. src/run_dengue.jl
#   julia --project=. src/run_dengue.jl --slurm
#   --workers N sets the local worker count (default 8).
# ============================================================================

using DistanceDependentCRP, CSV, DataFrames, Statistics, StatsBase, Random, JLD2, Printf, Distributions
using Distributed

const USE_SLURM = "--slurm" in ARGS
const IS_TEST = "--test" in ARGS
const IS_PILOT = "--pilot" in ARGS

function arg_value(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) ? default : parse(Int, ARGS[i+1])
end
const UF = arg_value("--uf", 41)      # 41 = Paraná (see prepare_dengue_data.jl)
const YEAR = arg_value("--year", 2024)

if USE_SLURM
    using SlurmClusterManager
    # launch_timeout defaults to 60s, which is not enough for a wide allocation
    # loading packages off /rds; the workers then die with "launch_timeout
    # exceeded" before they can call home.
    addprocs(SlurmManager(launch_timeout = 300.0))
else
    addprocs(arg_value("--workers", 8))
end

@everywhere begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using DistanceDependentCRP, Statistics, StatsBase, Distributions, JLD2, Random
    include(joinpath(@__DIR__, "negbin_population_model.jl"))
    include(joinpath(@__DIR__, "dengue_common.jl"))
end

# ============================================================================
# Proposal settings
#
# Within-cluster MH scales are fixed from short pilot runs. The birth and
# resample σ below are only the Part 1 defaults; Part 2 selects the values used
# in the final chains.
# ============================================================================

@everywhere const PROP_SDS = (ρ = 0.15, r = 0.30)   # within-cluster MH, log scale
const SIGMA_BIRTH = 0.50                            # NMM / LNMM birth σ
const SIGMA_RESAMPLE = 0.10                         # NMM / LNMM resample σ

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

@everywhere make_birth(fρ, fr, σ) =
    MixedProposal((ρ = _birth_family(fρ, σ), r = _birth_family(fr, σ)))

@everywhere make_fdim(fρ, fr, σr; resample::Bool) =
    resample ? MixedFixedDim((ρ = Resample(_birth_family(fρ, σr)),
                              r = Resample(_birth_family(fr, σr)))) : NoUpdate()

# Whether a family's proposal has a σ to tune.
has_sigma(fam::Symbol) = fam in (:nmm, :lnmm)

# ============================================================================
# Chain thinning for storage
#
# Only the post-burn-in, thinned draws are written to disk: a full-length raw
# chain is 250k × n per parameter matrix, several GB at n = 399.
# ============================================================================

@everywhere thin_chain(s::NegBinPopulationRatesSamples, keep) =
    NegBinPopulationRatesSamples(s.c[keep, :], s.ρ[keep, :], s.r[keep, :],
                                 s.logpost[keep], s.α_ddcrp[keep], s.s_ddcrp[keep])
@everywhere thin_chain(s::PoissonPopulationRatesSamples, keep) =
    PoissonPopulationRatesSamples(s.c[keep, :], s.ρ[keep, :], s.logpost[keep],
                                  s.α_ddcrp[keep], s.s_ddcrp[keep])

# ============================================================================
# Unit of work
# ============================================================================

@everywhere function run_and_save(cfg)
    Random.seed!(cfg.seed)

    data = CountDataWithPopulation(cfg.y, cfg.E, cfg.D)

    opts = MCMCOptions(
        n_samples = cfg.n_samples,
        verbose = false,
        track_diagnostics = true,
        infer_params = Dict{Symbol,Bool}(:α_ddcrp => true, :s_ddcrp => false),
        prop_sds = Dict{Symbol,Float64}(:ρ => PROP_SDS.ρ, :r => PROP_SDS.r),
        # Links are proposed from the ddCRP prior, not uniformly. Under the hop
        # window only about 19 of the 399 municipalities carry any weight, so a
        # uniform draw would spend 95% of proposals on links the prior all but
        # forbids, each costing a likelihood evaluation before being rejected.
        link_proposal = PriorLink(),
    )

    # α_ddcrp ~ Gamma(1, rate 0.01), inferred; the kernel rate is fixed at 1
    # because the hop window encodes the prior entirely in D.
    ddcrp = DDCRPParams(1.0, cfg.kernel_rate, DistanceDependentCRP.exp_decay, 1.0, 0.01, 1.0, 0.01)

    if cfg.family === :negbin
        model  = NegBinPopulationRates()
        priors = cfg.priors
        birth  = make_birth(cfg.fam_ρ, cfg.fam_r, cfg.σ_birth)
        fdim   = make_fdim(cfg.fam_ρ, cfg.fam_r, cfg.σ_resample; resample = cfg.resample)
    else # :poisson — the likelihood comparator
        model  = PoissonPopulationRates()
        priors = cfg.priors_pois
        birth  = PriorProposal()
        fdim   = NoUpdate()
    end

    total_time = @elapsed begin
        samples, diag = mcmc(model, data, ddcrp, priors, birth;
                             fixed_dim_proposal = fdim, opts = opts)
    end

    keep = (cfg.burn + 1):cfg.thin:cfg.n_samples
    post = thin_chain(samples, keep)

    k_trace = calculate_n_clusters(post.c)
    acc     = acceptance_rates(diag)
    ess_K   = effective_sample_size(Float64.(k_trace))
    esjd_K  = mean(diff(Float64.(k_trace)).^2)

    # Per-parameter mixing: median ESS across observations.
    med_ess(M) = median([effective_sample_size(M[:, i]) for i in axes(M, 2)])
    ess_ρ = med_ess(post.ρ)
    ess_r = cfg.family === :negbin ? med_ess(post.r) : NaN

    # Saved already burnt-in and thinned, so the analyse script's burn/thin are
    # no-ops on the stored object.
    chain_dir = joinpath(cfg.outdir, "chains"); mkpath(chain_dir)
    jldsave(joinpath(chain_dir, cfg.chain_filename);
            samples = post, burn = 0, thin = 1,
            burn_applied = cfg.burn, thin_applied = cfg.thin, label = cfg.label)

    return (
        label = cfg.label,
        family = String(cfg.family),
        fam_ρ = String(cfg.fam_ρ),
        fam_r = String(cfg.fam_r),
        birth = cfg.family === :negbin ? "$(cfg.fam_ρ)/$(cfg.fam_r)" : String(cfg.family),
        fdim = cfg.resample ? "resample" : "none",
        varied = cfg.varied,
        seed = cfg.seed,
        hop_radius = cfg.hop_radius,
        α_post = mean(post.α_ddcrp),
        σ_birth = cfg.σ_birth,
        σ_resample = cfg.resample ? cfg.σ_resample : NaN,
        n_obs = length(cfg.y),
        n_iter = cfg.n_samples,
        burn = cfg.burn,
        thin = cfg.thin,
        n_post = length(k_trace),
        mean_K = mean(k_trace),
        mode_K = Int(mode(k_trace)),
        std_K = std(Float64.(k_trace)),
        min_K = minimum(k_trace),
        max_K = maximum(k_trace),
        acc_birth = acc.birth,
        acc_death = acc.death,
        acc_fixed = acc.fixed,
        acc_overall = acc.overall,
        ess_K = ess_K,
        ess_logpost = effective_sample_size(post.logpost),
        med_ess_ρ = ess_ρ,
        med_ess_r = ess_r,
        esjd_K = esjd_K,
        ess_per_sec = ess_K / total_time,
        total_time = total_time,
        mean_logpost = mean(post.logpost),
        chain_file = cfg.chain_filename,
    )
end

# ============================================================================
# Constants
# ============================================================================

const N_SAMPLES_SWEEP = IS_TEST ? 1_000 :  60_000
const BURN_SWEEP      = IS_TEST ?   200 :  10_000
const THIN_SWEEP      = 5
# Final chains: N_POST_FINAL draws AFTER burn-in, so the chain length is the
# sum of the two. 250k is sized from the prior link proposal: a 20k-iteration
# run of the selected sampler gave ESS(K) ≈ 200, about fifteen times the
# per-second yield of the uniform proposal, so 250k should give ESS(K) in the
# low thousands per chain. Thinning by 10 keeps 25k draws per chain, about
# 240 MB saved and ~2.9 GB raw per worker.
const N_POST_FINAL    = IS_TEST ? 1_200 : 250_000
const BURN_FINAL      = IS_TEST ?   300 :  50_000
const N_SAMPLES_FINAL = N_POST_FINAL + BURN_FINAL
const THIN_FINAL      = IS_TEST ?     5 :      10

# Independent chains of the selected sampler. There is no collapsed sampler to
# compare against, so between-chain agreement is the convergence check, and R̂
# needs more than two chains to be informative.
const FINAL_SEEDS = IS_TEST ? [42, 202] : [42, 202, 7, 13]

const OUTDIR = IS_TEST ?
    joinpath(@__DIR__, "..", "results", "real_data", "dengue_test") :
    joinpath(@__DIR__, "..", "results", "real_data", "dengue")
mkpath(OUTDIR)

# ============================================================================
# Load data and the contiguity graph
# ============================================================================

const DATA_FILE = joinpath(@__DIR__, "..", "data", "cleaned", "dengue_uf$(UF)_$(YEAR).csv")
isfile(DATA_FILE) || error("""
    Missing $DATA_FILE
    Run: julia --project=. src/prepare_dengue_data.jl --uf $UF --year $YEAR""")

const ADJ_FILE = joinpath(@__DIR__, "..", "data", "cleaned", "dengue_uf$(UF)_adjacency.csv")
isfile(ADJ_FILE) || error("""
    Missing $ADJ_FILE
    Run: Rscript src/prepare_dengue_adjacency.R --uf $UF""")

const DF = CSV.read(DATA_FILE, DataFrame)
const Y_ALL = Int.(DF.y)
const E_ALL = Float64.(DF.pop) ./ 1e5          # exposure: population per 100k
const N = length(Y_ALL)

const A_ADJ = load_adjacency(ADJ_FILE, Int.(DF.geocode))
const NBRS  = neighbour_lists(A_ADJ)
const H_HOP = hop_distances(NBRS)
any(isinf, H_HOP) && error("contiguity graph is disconnected; rebuild the edge list")

# The ddCRP prior: a uniform hop window of radius HOP_RADIUS, under kernel rate 1.
const HOP_RADIUS  = 2
const KERNEL_RATE = 1.0
const D_ALL = hop_window_D(H_HOP, HOP_RADIUS)

@printf("Loaded %s: %d municipalities, %d total cases, %.2f M people\n",
        basename(DATA_FILE), N, sum(Y_ALL), sum(DF.pop) / 1e6)
@printf("  incidence/100k: min %.1f  median %.1f  max %.1f\n",
        minimum(Y_ALL ./ E_ALL), median(Y_ALL ./ E_ALL), maximum(Y_ALL ./ E_ALL))
@printf("  contiguity graph: %d edges, mean degree %.2f, diameter %d hops\n",
        Int(sum(A_ADJ) / 2), sum(A_ADJ) / N, Int(maximum(H_HOP)))
@printf("  hop window radius %d: mean %.1f municipalities in each window\n",
        HOP_RADIUS, mean(count(<=(HOP_RADIUS), view(H_HOP, i, :)) - 1 for i in 1:N))

# ============================================================================
# Shared model settings
# ============================================================================

# ρ prior centred on the state-wide incidence, deliberately diffuse (shape 1.5
# ⇒ sd > mean) so it does not pin the cluster rates; r prior weakly informative
# over the overdispersion range typical of areal count data.
const ρ_BAR = sum(Y_ALL) / sum(E_ALL)
const ρ_A = 1.5
const PRIORS = NegBinPopulationRatesPriors(ρ_a = ρ_A, ρ_b = ρ_A / ρ_BAR, r_a = 2.0, r_b = 0.5)
const PRIORS_POIS = PoissonPopulationRatesPriors(ρ_A, ρ_A / ρ_BAR)

@printf("  ρ prior: Gamma(%.1f, rate %.2e) ⇒ mean %.0f per 100k\n", ρ_A, ρ_A / ρ_BAR, ρ_BAR)

# ============================================================================
# Config builder
# ============================================================================

fmtlabel(s) = replace(lowercase(s), r"[^a-z0-9]+" => "_")
fmtσ(x) = replace(@sprintf("%g", x), "." => "p")

function make_cfg(; family = :negbin, fam_ρ = :lnmm, fam_r = :lnmm, resample = false,
                    varied = "sweep", label = nothing,
                    y = Y_ALL, E = E_ALL, D = D_ALL,
                    n_samples = N_SAMPLES_SWEEP, burn = BURN_SWEEP, thin = THIN_SWEEP,
                    σ_birth = SIGMA_BIRTH, σ_resample = SIGMA_RESAMPLE,
                    seed = 42, suffix = "")
    lbl = something(label,
        family === :negbin ? "NB ρ:$(fam_ρ) r:$(fam_r) + $(resample ? "Resample" : "NoUpdate")" :
        "Poisson + NoUpdate")
    (
        label = lbl,
        family = family, fam_ρ = fam_ρ, fam_r = fam_r,
        resample = resample, varied = varied,
        σ_birth = σ_birth, σ_resample = σ_resample,
        priors = PRIORS, priors_pois = PRIORS_POIS,
        y = y, E = E, D = D,
        hop_radius = HOP_RADIUS, kernel_rate = KERNEL_RATE,
        n_samples = n_samples, burn = burn, thin = thin,
        outdir = OUTDIR,
        chain_filename = fmtlabel(lbl) * suffix * ".jld2",
        seed = seed,
    )
end

# ============================================================================
# Summary printer and selection rule
# ============================================================================

function print_summary_table(df, title)
    W = 150
    println(); println("=" ^ W)
    println(title * "  (post burn-in, thinned)"); println("=" ^ W)
    println(rpad("ρ / r", 16) * rpad("Fixed-dim", 11) * "  σ_b   σ_r " *
            "Mean K  Mode K   sd K  Acc.B%  Acc.D%  Acc.F%   ESS(K)   ESS_ρ   ESS_r  ESJD(K)  ESS/s  Time(s)")
    println("-" ^ W)
    for r in eachrow(df)
        @printf("%-16s%-11s%5.2f %5s %6.2f  %6d %6.2f  %6.2f  %6.2f  %6.2f  %7.0f %7.0f %7.0f  %7.4f %6.3f  %7.1f\n",
            r.birth, r.fdim, r.σ_birth, isnan(r.σ_resample) ? "--" : @sprintf("%.2f", r.σ_resample),
            r.mean_K, r.mode_K, r.std_K,
            r.acc_birth*100, r.acc_death*100, r.acc_fixed*100,
            r.ess_K, r.med_ess_ρ, r.med_ess_r, r.esjd_K, r.ess_per_sec, r.total_time)
    end
    println("=" ^ W)
end

# The selection rule used throughout: largest ESJD(K), ties broken by ESS(K).
# ESJD(K) rewards chains that move between partitions with different numbers of
# clusters, which is exactly what the trans-dimensional proposals are for.
best_row(df) = df[sortperm(collect(zip(df.esjd_K, df.ess_K)); rev = true)[1], :]

# ============================================================================
# Part 0 — pilot: how does cost scale with n, and is the sampler moving?
# ============================================================================

if IS_PILOT
    sizes = filter(m -> m <= N, [100, 200, 300, N])
    n_iter = 4_000

    println("\n$(repeat('=', 80))")
    println("PILOT — $(length(sizes)) subsample sizes × $n_iter iterations")
    println("$(repeat('=', 80))")

    # Spatially-representative subsamples: sort by longitude and take an even
    # stride, so a subsample still spans the state. The hop window is kept from
    # the full graph, so each subsample sees the same prior restricted to it.
    order = sortperm(DF.lon)
    pilot_cfgs = [
        begin
            idx = sort(order[round.(Int, range(1, N; length = m))])
            make_cfg(; label = "pilot n=$m", y = Y_ALL[idx], E = E_ALL[idx],
                       D = D_ALL[idx, idx], n_samples = n_iter, burn = n_iter ÷ 4,
                       suffix = "_pilot_n$m")
        end
        for m in sizes
    ]

    pdf = DataFrame(collect(pmap(run_and_save, pilot_cfgs)))
    pdf.sec_per_1k = pdf.total_time ./ (n_iter / 1000)

    println()
    @printf("%6s %12s %8s %8s %8s %8s %10s\n",
            "n", "sec/1k iter", "mean K", "acc.B%", "acc.D%", "acc.F%", "ESS(K)")
    println("-"^70)
    for r in eachrow(pdf)
        @printf("%6d %12.2f %8.2f %8.2f %8.2f %8.2f %10.0f\n",
                r.n_obs, r.sec_per_1k, r.mean_K,
                r.acc_birth*100, r.acc_death*100, r.acc_fixed*100, r.ess_K)
    end
    @printf("\nExtrapolated wall-clock for one %dk-iteration chain at n=%d: %.1f h\n",
            N_SAMPLES_FINAL ÷ 1000, N, pdf.sec_per_1k[end] * N_SAMPLES_FINAL / 1000 / 3600)

    CSV.write(joinpath(OUTDIR, "summary_pilot.csv"), pdf)
    println("\nPilot summary → $(joinpath(OUTDIR, "summary_pilot.csv"))")
    exit(0)
end

# ============================================================================
# Part 1 — proposal-family sweep (coordinate design)
#
# Baseline (ρ:lnmm, r:lnmm); swap one parameter's family at a time; plus an
# all-prior reference. Each crossed with {NoUpdate, Resample}.
# ============================================================================

const BIRTH_SPECS = Tuple{Symbol,Symbol,String}[
    (:lnmm,  :lnmm,   "baseline"),
    (:prior, :lnmm,   "ρ:prior"),
    (:nmm,   :lnmm,   "ρ:nmm"),
    (:ig,    :lnmm,   "ρ:ig"),
    (:lnmm,  :prior,  "r:prior"),
    (:lnmm,  :nmm,    "r:nmm"),
    (:lnmm,  :ig,     "r:ig"),
    (:prior, :prior,  "all-prior"),
]

sweep_cfgs = [make_cfg(; fam_ρ = fρ, fam_r = fr, resample = resample, varied = varied,
                         suffix = "_sweep")
              for (fρ, fr, varied) in BIRTH_SPECS for resample in (false, true)]

println("\n$(repeat('=', 80))")
println("PART 1 — proposal-family sweep: $(length(sweep_cfgs)) configs × $N_SAMPLES_SWEEP iterations")
println("  hop radius=$HOP_RADIUS  σ_birth=$SIGMA_BIRTH  σ_resample=$SIGMA_RESAMPLE  prop_sds=$PROP_SDS")
println("$(repeat('=', 80))")

sweep_df = DataFrame(collect(pmap(run_and_save, sweep_cfgs)))
CSV.write(joinpath(OUTDIR, "summary_sweep.csv"), sweep_df)
print_summary_table(sweep_df, "PART 1 — dengue proposal-family sweep")

best_sweep = best_row(sweep_df)
const BEST_FAM_ρ = Symbol(best_sweep.fam_ρ)
const BEST_FAM_r = Symbol(best_sweep.fam_r)
const BEST_RESAMPLE = best_sweep.fdim == "resample"
@printf("\nSelected by ESJD(K): %s  (ESJD %.4f, ESS(K) %.0f)\n",
        best_sweep.label, best_sweep.esjd_K, best_sweep.ess_K)

# ============================================================================
# Part 2 — σ tuning for the selected family / fixed-dim pair
# ============================================================================

const SIGMA_GRID    = IS_TEST ? [0.25, 0.5] : [0.1, 0.25, 0.5, 1.0, 2.0]
const RESAMPLE_GRID = IS_TEST ? [0.1]       : [0.05, 0.1, 0.25]

const TUNE_BIRTH    = has_sigma(BEST_FAM_ρ) || has_sigma(BEST_FAM_r)
const TUNE_RESAMPLE = BEST_RESAMPLE && TUNE_BIRTH   # Resample reuses the families

sb_grid = TUNE_BIRTH ? SIGMA_GRID : [SIGMA_BIRTH]
sr_grid = TUNE_RESAMPLE ? RESAMPLE_GRID : [SIGMA_RESAMPLE]

sel_σ_birth, sel_σ_resample = SIGMA_BIRTH, SIGMA_RESAMPLE
if TUNE_BIRTH || TUNE_RESAMPLE
    # Part 1 already ran the default (σ_birth, σ_resample) point; do not rerun it.
    tune_cfgs = [make_cfg(; fam_ρ = BEST_FAM_ρ, fam_r = BEST_FAM_r, resample = BEST_RESAMPLE,
                            varied = "tune", σ_birth = σb, σ_resample = σr,
                            suffix = "_tune_b$(fmtσ(σb))" * (BEST_RESAMPLE ? "_r$(fmtσ(σr))" : ""))
                 for σb in sb_grid for σr in sr_grid
                 if !(σb == SIGMA_BIRTH && σr == SIGMA_RESAMPLE)]

    println("\n$(repeat('=', 80))")
    println("PART 2 — σ tuning for $(best_sweep.label): $(length(tune_cfgs)) configs × $N_SAMPLES_SWEEP iterations")
    println("$(repeat('=', 80))")

    tune_df = DataFrame(collect(pmap(run_and_save, tune_cfgs)))
    # The Part 1 winner is the default-σ grid point, so it competes too.
    tune_all = vcat(DataFrame(best_sweep), tune_df; cols = :union)
    CSV.write(joinpath(OUTDIR, "summary_tune.csv"), tune_all)
    print_summary_table(sort(tune_all, :esjd_K, rev = true), "PART 2 — σ tuning (sorted by ESJD(K))")

    best_tune = best_row(tune_all)
    global sel_σ_birth = best_tune.σ_birth
    global sel_σ_resample = BEST_RESAMPLE ? best_tune.σ_resample : SIGMA_RESAMPLE
    @printf("\nSelected by ESJD(K): σ_birth=%.2f%s\n", sel_σ_birth,
            BEST_RESAMPLE ? @sprintf("  σ_resample=%.2f", sel_σ_resample) : "")
    if (TUNE_BIRTH && sel_σ_birth in extrema(sb_grid)) ||
       (TUNE_RESAMPLE && sel_σ_resample in extrema(sr_grid))
        @warn "Selected σ sits on the edge of the grid; widen the grid and re-run Part 2."
    end
else
    println("\nPART 2 skipped: $(best_sweep.label) has no tunable proposal scale.")
end

# ============================================================================
# Part 3 — final chains at full length
# ============================================================================

best_cfg(; seed, label = nothing) =
    make_cfg(; fam_ρ = BEST_FAM_ρ, fam_r = BEST_FAM_r, resample = BEST_RESAMPLE,
               varied = "final", label = label,
               σ_birth = sel_σ_birth, σ_resample = sel_σ_resample,
               n_samples = N_SAMPLES_FINAL, burn = BURN_FINAL, thin = THIN_FINAL,
               seed = seed, suffix = "_final_seed$(seed)")

const BEST_LABEL = best_cfg(; seed = first(FINAL_SEEDS)).label

# One chain per seed, labelled "(chain k)" after the first so the analyse script
# can tell them apart, plus the Poisson comparator.
final_cfgs = Any[
    best_cfg(; seed = s, label = k == 1 ? BEST_LABEL : BEST_LABEL * " (chain $k)")
    for (k, s) in enumerate(FINAL_SEEDS)
]
push!(final_cfgs,
      make_cfg(; family = :poisson, varied = "comparator",
                 n_samples = N_SAMPLES_FINAL, burn = BURN_FINAL, thin = THIN_FINAL,
                 suffix = "_final"))

println("\n$(repeat('=', 80))")
@printf("PART 3 — final chains: %d chains of the selected sampler + Poisson comparator\n",
        length(FINAL_SEEDS))
@printf("  %d iterations each (%d burn-in + %d kept), thinned by %d ⇒ %d draws per chain\n",
        N_SAMPLES_FINAL, BURN_FINAL, N_POST_FINAL, THIN_FINAL, N_POST_FINAL ÷ THIN_FINAL)
println("$(repeat('=', 80))")

final_df = DataFrame(collect(pmap(run_and_save, final_cfgs)))
CSV.write(joinpath(OUTDIR, "summary_final.csv"), final_df)
print_summary_table(final_df, "PART 3 — dengue final chains")

# Metadata the analyse script needs to rebuild the dataset and find the chains.
jldsave(joinpath(OUTDIR, "run_metadata.jld2");
        uf = UF, year = YEAR, y = Y_ALL, E = E_ALL, D = D_ALL,
        adjacency = A_ADJ, hop = H_HOP,
        hop_radius = HOP_RADIUS, kernel_rate = KERNEL_RATE, log_floor = LOG_FLOOR,
        lat = Float64.(DF.lat), lon = Float64.(DF.lon),
        geocode = Int.(DF.geocode), name = String.(DF.name),
        prior_rho = (a = ρ_A, b = ρ_A / ρ_BAR), prior_r = (a = 2.0, b = 0.5),
        prop_sds = PROP_SDS,
        best_fam_rho = String(BEST_FAM_ρ), best_fam_r = String(BEST_FAM_r),
        best_resample = BEST_RESAMPLE,
        sigma_birth = sel_σ_birth, sigma_resample = sel_σ_resample,
        n_samples = N_SAMPLES_FINAL, burn = BURN_FINAL, thin = THIN_FINAL,
        n_post = N_POST_FINAL, final_seeds = FINAL_SEEDS,
        best_label = BEST_LABEL,
        chain_files = [c.chain_filename for c in final_cfgs],
        labels = [c.label for c in final_cfgs],
        seeds = [c.seed for c in final_cfgs])

println("\nAll chains complete. Run src/analyse_dengue.jl for figures, tables and model comparison.")

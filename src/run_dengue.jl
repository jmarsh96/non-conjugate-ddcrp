# ============================================================================
# run_dengue.jl
#
# Real-data application: spatial clustering of 2024 dengue incidence across the
# municipalities of a Brazilian state, under a genuinely non-conjugate Negative
# Binomial likelihood.
#
#   y_i | ρ_k, r_k ~ NegativeBinomial(mean = E_i ρ_k, size = r_k)
#     E_i = pop_i / 1e5      ⇒ ρ_k is the cluster incidence per 100k
#     r_k                     the cluster overdispersion (r → ∞ ⇒ Poisson)
#
# BOTH cluster parameters are non-conjugate, so this is the count-data analogue
# of the SkewNormal simulation study, and the honest counterpart of the Poisson
# simulation study's marginalised Gibbs baseline. The ddCRP distance is the
# great-circle distance between municipality centroids — a genuinely 2-D spatial
# covariate, unlike the 1-D covariates used elsewhere in the paper.
#
# Run `src/prepare_dengue_data.jl` first to build the dataset.
#
# Parts, all via `pmap` over worker processes:
#   0. --pilot  : short chains at several subsample sizes, to measure how cost
#                 scales with n and choose the state for the full study.
#   1. Proposal-family sweep — coordinate design over the birth family for each
#      parameter, crossed with fixed-dim ∈ {NoUpdate, Resample}, plus the two
#      Poisson comparators.
#   2. Tuning sweep — shorter chains over σ_birth / σ_resample, best per group
#      chosen by ESJD(K).
#   3. Final comparison — full-length chains for the best-tuned variants.
#
# Usage:
#   julia --project=. src/run_dengue.jl --pilot
#   julia --project=. src/run_dengue.jl --test
#   julia --project=. src/run_dengue.jl
#   julia --project=. src/run_dengue.jl --slurm
# ============================================================================

using DistanceDependentCRP, CSV, DataFrames, Distances, Statistics, StatsBase, Random, JLD2, Printf, Distributions
using Distributed

const USE_SLURM = "--slurm" in ARGS
const IS_TEST = "--test" in ARGS
const IS_PILOT = "--pilot" in ARGS
const INFER_S = "--infer_s" in ARGS

function arg_value(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) ? default : parse(Int, ARGS[i+1])
end
const UF = arg_value("--uf", 41)      # 41 = Paraná (see prepare_dengue_data.jl)
const YEAR = arg_value("--year", 2024)

if USE_SLURM
    using SlurmClusterManager
    addprocs(SlurmManager())
else
    addprocs(8)
end

@everywhere begin
    using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
    using DistanceDependentCRP, Statistics, StatsBase, Distributions, JLD2, Random
    include(joinpath(@__DIR__, "negbin_population_model.jl"))
end

# ============================================================================
# Manually-tuned proposal parameters
#
# Starting values from the pilot; Part 2 refines them and the winners are
# recorded here. Re-run the tuning sweep if the state or priors change.
# ============================================================================

@everywhere const PROP_SDS = (ρ = 0.15, r = 0.30)   # within-cluster MH, log scale
const SIGMA_BIRTH = 0.50                            # NMM / LNMM birth σ
const SIGMA_RESAMPLE = 0.25                         # NMM / LNMM resample σ

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

# ============================================================================
# Chain thinning for storage
#
# Only the post-burn-in, thinned draws are written to disk. At full length the
# raw chain is 250k × n per parameter matrix — about 2.4 GB per Negative
# Binomial chain at n=399 — so saving every draw would make the analyse script,
# which loads several chains at once, run out of memory.
# ============================================================================

@everywhere thin_chain(s::NegBinPopulationRatesSamples, keep) =
    NegBinPopulationRatesSamples(s.c[keep, :], s.ρ[keep, :], s.r[keep, :],
                                 s.logpost[keep], s.α_ddcrp[keep], s.s_ddcrp[keep])
@everywhere thin_chain(s::PoissonPopulationRatesSamples, keep) =
    PoissonPopulationRatesSamples(s.c[keep, :], s.ρ[keep, :], s.logpost[keep],
                                  s.α_ddcrp[keep], s.s_ddcrp[keep])
@everywhere thin_chain(s::PoissonPopulationRatesMargSamples, keep) =
    PoissonPopulationRatesMargSamples(s.c[keep, :], s.logpost[keep],
                                      s.α_ddcrp[keep], s.s_ddcrp[keep])

# ============================================================================
# Unit of work
# ============================================================================

@everywhere function run_and_save(cfg)
    Random.seed!(cfg.seed)

    data = CountDataWithPopulation(cfg.y, cfg.E, cfg.D)
    is_poisson = cfg.family !== :negbin

    opts = MCMCOptions(
        n_samples = cfg.n_samples,
        verbose = false,
        track_diagnostics = true,
        infer_params = Dict{Symbol,Bool}(:α_ddcrp => true, :s_ddcrp => cfg.infer_s),
        prop_sds = Dict{Symbol,Float64}(
            :ρ => PROP_SDS.ρ, :r => PROP_SDS.r, :s_ddcrp => 0.3),
    )

    ddcrp = DDCRPParams(1.0, cfg.s, DistanceDependentCRP.exp_decay, 1.0, 0.01, 1.0, 0.01)

    if cfg.family === :negbin
        model  = NegBinPopulationRates()
        priors = cfg.priors
        birth  = make_birth(cfg.fam_ρ, cfg.fam_r, cfg.σ_birth)
        fdim   = make_fdim(cfg.fam_ρ, cfg.fam_r, cfg.σ_resample; resample = cfg.resample)
    elseif cfg.family === :poisson
        model  = PoissonPopulationRates()
        priors = cfg.priors_pois
        birth  = PriorProposal()
        fdim   = cfg.resample ? Resample(PriorProposal()) : NoUpdate()
    else # :poisson_marg — the conjugate Gibbs anchor
        model  = PoissonPopulationRatesMarg()
        priors = cfg.priors_pois_marg
        birth  = ConjugateProposal()
        fdim   = NoUpdate()
    end

    total_time = @elapsed begin
        samples, diag = mcmc(model, data, ddcrp, priors, birth;
                             fixed_dim_proposal = fdim, opts = opts)
    end

    post_c  = samples.c[(cfg.burn + 1):cfg.thin:end, :]
    post_lp = samples.logpost[(cfg.burn + 1):cfg.thin:end]

    k_trace = calculate_n_clusters(post_c)
    acc     = acceptance_rates(diag)
    ess_K   = effective_sample_size(Float64.(k_trace))
    esjd_K  = mean(diff(Float64.(k_trace)).^2)

    # Per-parameter mixing: median ESS across observations. The marginalised
    # model stores no cluster parameters, hence the NaNs.
    keep = (cfg.burn + 1):cfg.thin:cfg.n_samples
    med_ess(M) = median([effective_sample_size(M[keep, i]) for i in 1:size(M, 2)])
    ess_ρ = cfg.family === :poisson_marg ? NaN : med_ess(samples.ρ)
    ess_r = cfg.family === :negbin ? med_ess(samples.r) : NaN

    # Saved already burnt-in and thinned, so the analyse script's burn/thin are
    # no-ops on the stored object.
    chain_dir = joinpath(cfg.outdir, "chains"); mkpath(chain_dir)
    jldsave(joinpath(chain_dir, cfg.chain_filename);
            samples = thin_chain(samples, keep), burn = 0, thin = 1,
            burn_applied = cfg.burn, thin_applied = cfg.thin, label = cfg.label)

    return (
        label = cfg.label,
        family = String(cfg.family),
        fam_ρ = String(cfg.fam_ρ),
        fam_r = String(cfg.fam_r),
        birth = cfg.family === :negbin ? "$(cfg.fam_ρ)/$(cfg.fam_r)" : String(cfg.family),
        fdim = cfg.resample ? "resample" : "none",
        varied = cfg.varied,
        s_decay = cfg.s,
        infer_s = cfg.infer_s,
        s_post = mean(samples.s_ddcrp[(cfg.burn + 1):cfg.thin:end]),
        α_post = mean(samples.α_ddcrp[(cfg.burn + 1):cfg.thin:end]),
        σ_birth = cfg.σ_birth,
        σ_resample = cfg.resample ? cfg.σ_resample : NaN,
        n_obs = length(cfg.y),
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
        ess_logpost = effective_sample_size(post_lp),
        med_ess_ρ = ess_ρ,
        med_ess_r = ess_r,
        esjd_K = esjd_K,
        ess_per_sec = ess_K / total_time,
        total_time = total_time,
        mean_logpost = mean(post_lp),
    )
end

# ============================================================================
# Constants
# ============================================================================

const N_SAMPLES_SWEEP = IS_TEST ? 1_000 : 110_000
const BURN_SWEEP      = IS_TEST ?   200 :  10_000
const N_SAMPLES_TUNE  = IS_TEST ? 1_000 :  40_000
const BURN_TUNE       = IS_TEST ?   200 :  10_000
const N_SAMPLES_FINAL = IS_TEST ? 1_500 : 250_000
const BURN_FINAL      = IS_TEST ?   300 :  50_000
const THIN = 5

const OUTDIR = IS_TEST ?
    joinpath(@__DIR__, "..", "results", "real_data", "dengue_test") :
    joinpath(@__DIR__, "..", "results", "real_data", "dengue")
mkpath(OUTDIR)

# ============================================================================
# Load data
# ============================================================================

const DATA_FILE = joinpath(@__DIR__, "..", "data", "dengue_uf$(UF)_$(YEAR).csv")
isfile(DATA_FILE) || error("""
    Missing $DATA_FILE
    Run: julia --project=. src/prepare_dengue_data.jl --uf $UF --year $YEAR""")

const DF = CSV.read(DATA_FILE, DataFrame)
const Y_ALL = Int.(DF.y)
const E_ALL = Float64.(DF.pop) ./ 1e5          # exposure: population per 100k

# Great-circle distances (km) between municipality centroids, then rescaled by
# the median pairwise distance so the ddCRP decay scale s is O(1) — the same
# convention the simulation studies get for free from standardised covariates.
const COORDS = permutedims(hcat(Float64.(DF.lon), Float64.(DF.lat)))
const D_KM = pairwise(Haversine(6371.0), COORDS)
const D_SCALE = median(D_KM[i, j] for i in 1:size(D_KM, 1) for j in 1:size(D_KM, 2) if i != j)
const D_ALL = D_KM ./ D_SCALE

@printf("Loaded %s: %d municipalities, %d total cases, %.2f M people\n",
        basename(DATA_FILE), nrow(DF), sum(Y_ALL), sum(DF.pop) / 1e6)
@printf("  incidence/100k: min %.1f  median %.1f  max %.1f\n",
        minimum(Y_ALL ./ E_ALL), median(Y_ALL ./ E_ALL), maximum(Y_ALL ./ E_ALL))
@printf("  pairwise distance: median %.0f km, max %.0f km (D rescaled by the median)\n",
        D_SCALE, maximum(D_KM))

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
const PRIORS_POIS_MARG = PoissonPopulationRatesMargPriors(ρ_A, ρ_A / ρ_BAR)

# ddCRP decay scale. FIXED rather than inferred, deliberately: with s free the
# chain drives s → ~15, which flattens exp(-d/s) to ≈1 over the whole state and
# degenerates the ddCRP into a distance-independent CRP (α → 0.03, K → 27, death
# acceptance → 0.7%). s = 0.5 on distances normalised by the median pairwise
# distance is a decay length of about half that median — an epidemiologically
# sensible neighbourhood. Part 4 reports the sensitivity of the results to this
# choice, and runs one s-inferred chain so the pathology is documented, not hidden.
const S_DECAY = 0.5

@printf("  ρ prior: Gamma(%.1f, rate %.2e) ⇒ mean %.0f per 100k\n", ρ_A, ρ_A / ρ_BAR, ρ_BAR)
@printf("  ddCRP decay scale s = %.2f fixed (≈ %.0f km decay length)%s\n",
        S_DECAY, S_DECAY * D_SCALE, INFER_S ? "  [OVERRIDDEN: --infer_s given]" : "")

# ============================================================================
# Config builder
# ============================================================================

fmtlabel(s) = replace(lowercase(s), r"[^a-z0-9]+" => "_")

function make_cfg(; family = :negbin, fam_ρ = :lnmm, fam_r = :lnmm, resample = false,
                    varied = "baseline", label = nothing,
                    y = Y_ALL, E = E_ALL, D = D_ALL,
                    n_samples = N_SAMPLES_SWEEP, burn = BURN_SWEEP,
                    σ_birth = SIGMA_BIRTH, σ_resample = SIGMA_RESAMPLE,
                    s = S_DECAY, infer_s = INFER_S, seed = 42, suffix = "")
    lbl = something(label,
        family === :negbin ? "NB ρ:$(fam_ρ) r:$(fam_r) + $(resample ? "Resample" : "NoUpdate")" :
        family === :poisson ? "Poisson + $(resample ? "Resample" : "NoUpdate")" :
        "Poisson Gibbs (marginalised)")
    (
        label = lbl,
        family = family, fam_ρ = fam_ρ, fam_r = fam_r,
        resample = resample, varied = varied,
        σ_birth = σ_birth, σ_resample = σ_resample,
        priors = PRIORS, priors_pois = PRIORS_POIS, priors_pois_marg = PRIORS_POIS_MARG,
        y = y, E = E, D = D,
        s = s,
        n_samples = n_samples, burn = burn, thin = THIN,
        outdir = OUTDIR,
        chain_filename = fmtlabel(lbl) * suffix * ".jld2",
        seed = seed, infer_s = infer_s,
    )
end

# ============================================================================
# Summary printer
# ============================================================================

function print_summary_table(df, title)
    W = 140
    println(); println("=" ^ W)
    println(title * "  (post burn-in, thinned by $THIN)"); println("=" ^ W)
    println(rpad("ρ / r", 16) * rpad("Fixed-dim", 11) *
            "Mean K  Mode K   sd K  Acc.B%  Acc.D%  Acc.F%   ESS(K)   ESS_ρ   ESS_r  ESJD(K)  ESS/s  Time(s)")
    println("-" ^ W)
    for r in eachrow(df)
        @printf("%-16s%-11s%6.2f  %6d %6.2f  %6.2f  %6.2f  %6.2f  %7.0f %7.0f %7.0f  %7.4f %6.3f  %7.1f\n",
            r.birth, r.fdim, r.mean_K, r.mode_K, r.std_K,
            r.acc_birth*100, r.acc_death*100, r.acc_fixed*100,
            r.ess_K, r.med_ess_ρ, r.med_ess_r, r.esjd_K, r.ess_per_sec, r.total_time)
    end
    println("=" ^ W)
end

# ============================================================================
# Part 0 — pilot: how does cost scale with n, and is the sampler moving?
# ============================================================================

if IS_PILOT
    n_full = length(Y_ALL)
    sizes = filter(m -> m <= n_full, [100, 200, 300, n_full])
    n_iter = 4_000

    println("\n$(repeat('=', 80))")
    println("PILOT — $(length(sizes)) subsample sizes × $n_iter iterations")
    println("  Purpose: measure cost scaling in n and check the RJMCMC is moving,")
    println("  so the state for the full study can be chosen on measured cost.")
    println("$(repeat('=', 80))")

    # Spatially-representative subsamples: sort by longitude and take an even
    # stride, so a subsample still spans the state rather than one corner of it.
    order = sortperm(DF.lon)
    pilot_cfgs = [
        begin
            idx = sort(order[round.(Int, range(1, n_full; length = m))])
            make_cfg(; label = "pilot n=$m", y = Y_ALL[idx], E = E_ALL[idx],
                       D = D_ALL[idx, idx], n_samples = n_iter, burn = n_iter ÷ 4,
                       suffix = "_pilot_n$m")
        end
        for m in sizes
    ]

    pilot_rows = pmap(run_and_save, pilot_cfgs)
    pdf = DataFrame(collect(pilot_rows))
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

    # Empirical scaling exponent: sec/1k ∝ n^p. Used to extrapolate to states we
    # have not downloaded (São Paulo n=645, Minas Gerais n=853).
    if nrow(pdf) >= 2
        ln, lt = log.(pdf.n_obs), log.(pdf.sec_per_1k)
        p = (length(ln) * sum(ln .* lt) - sum(ln) * sum(lt)) /
            (length(ln) * sum(ln .^ 2) - sum(ln)^2)
        a = exp(mean(lt) - p * mean(ln))
        @printf("\nFitted cost ≈ %.3g · n^%.2f sec per 1k iterations\n", a, p)
        println("Extrapolated wall-clock for one 250k-iteration chain:")
        for m in [399, 497, 645, 853]
            @printf("  n=%3d (%s): %5.1f h\n", m,
                    m == 399 ? "Paraná" : m == 497 ? "Rio Grande do Sul" :
                    m == 645 ? "São Paulo" : "Minas Gerais",
                    a * m^p * 250 / 3600)
        end
        println("\n(8 workers run in parallel, so a sweep of C configs costs ≈ ceil(C/8) chains.)")
    end

    CSV.write(joinpath(OUTDIR, "summary_pilot.csv"), pdf)
    println("\nPilot summary → $(joinpath(OUTDIR, "summary_pilot.csv"))")
    exit(0)
end

# ============================================================================
# Part 1 — proposal-family sweep (coordinate design)
#
# Baseline (ρ:lnmm, r:lnmm); swap one parameter's family at a time; plus an
# all-prior reference. Each crossed with {NoUpdate, Resample}. The two Poisson
# models are carried alongside as comparators.
# ============================================================================

const BASE = (:lnmm, :lnmm)
const BIRTH_SPECS = Tuple{Symbol,Symbol,String}[
    (BASE...,         "baseline"),
    (:prior, :lnmm,   "ρ:prior"),
    (:nmm,   :lnmm,   "ρ:nmm"),
    (:ig,    :lnmm,   "ρ:ig"),
    (:lnmm,  :prior,  "r:prior"),
    (:lnmm,  :nmm,    "r:nmm"),
    (:lnmm,  :ig,     "r:ig"),
    (:prior, :prior,  "all-prior"),
]

sweep_cfgs = Any[]
for (fρ, fr, varied) in BIRTH_SPECS, resample in (false, true)
    push!(sweep_cfgs, make_cfg(; fam_ρ = fρ, fam_r = fr, resample = resample, varied = varied))
end
push!(sweep_cfgs, make_cfg(; family = :poisson, varied = "comparator"))
push!(sweep_cfgs, make_cfg(; family = :poisson_marg, varied = "comparator"))

println("\n$(repeat('=', 80))")
println("PART 1 — proposal-family sweep: $(length(sweep_cfgs)) configs × $N_SAMPLES_SWEEP iterations")
println("  σ_birth=$SIGMA_BIRTH  σ_resample=$SIGMA_RESAMPLE  prop_sds=$PROP_SDS")
println("$(repeat('=', 80))")

sweep_df = DataFrame(collect(pmap(run_and_save, sweep_cfgs)))
CSV.write(joinpath(OUTDIR, "summary_sweep.csv"), sweep_df)
print_summary_table(sweep_df, "PART 1 — dengue proposal-family sweep")

# ── Best family per parameter, by that parameter's median ESS ────────────────
println("\nBest birth family per parameter (by median parameter ESS, NoUpdate rows):")
nb_nu = filter(r -> r.family == "negbin" && r.fdim == "none", sweep_df)
for (pname, esscol, famcol) in [("ρ", :med_ess_ρ, :fam_ρ), ("r", :med_ess_r, :fam_r)]
    cand = filter(r -> r.varied == "baseline" || startswith(r.varied, "$pname:"), nb_nu)
    isempty(cand) && continue
    best = cand[argmax([getproperty(r, esscol) for r in eachrow(cand)]), :]
    @printf("  %s: best = %-6s (median ESS %.0f, mean K %.2f, acc.birth %.2f%%)\n",
            pname, getproperty(best, famcol), getproperty(best, esscol),
            best.mean_K, best.acc_birth * 100)
end

# ============================================================================
# Part 2 — tuning sweep: choose σ_birth (and σ_resample) by ESJD(K)
# ============================================================================

const SIGMA_GRID   = IS_TEST ? [0.5] : [0.25, 0.5, 1.0, 2.0]
const RESAMPLE_GRID = IS_TEST ? [0.25] : [0.1, 0.25, 0.5]

tune_cfgs = Any[]
for σb in SIGMA_GRID
    push!(tune_cfgs, make_cfg(; fam_ρ = BASE[1], fam_r = BASE[2], resample = false,
                                varied = "tune", σ_birth = σb,
                                n_samples = N_SAMPLES_TUNE, burn = BURN_TUNE,
                                suffix = "_tune_b$(σb)"))
    for σr in RESAMPLE_GRID
        push!(tune_cfgs, make_cfg(; fam_ρ = BASE[1], fam_r = BASE[2], resample = true,
                                    varied = "tune", σ_birth = σb, σ_resample = σr,
                                    n_samples = N_SAMPLES_TUNE, burn = BURN_TUNE,
                                    suffix = "_tune_b$(σb)_r$(σr)"))
    end
end

println("\n$(repeat('=', 80))")
println("PART 2 — tuning sweep: $(length(tune_cfgs)) configs × $N_SAMPLES_TUNE iterations")
println("$(repeat('=', 80))")

tune_df = DataFrame(collect(pmap(run_and_save, tune_cfgs)))
CSV.write(joinpath(OUTDIR, "summary_tune.csv"), tune_df)

println()
@printf("%10s %12s %10s %10s %10s %10s\n",
        "σ_birth", "σ_resample", "ESJD(K)", "ESS(K)", "acc.B%", "acc.F%")
println("-"^70)
for r in eachrow(sort(tune_df, :esjd_K, rev = true))
    @printf("%10.2f %12s %10.4f %10.0f %10.2f %10.2f\n",
            r.σ_birth, isnan(r.σ_resample) ? "—" : @sprintf("%.2f", r.σ_resample),
            r.esjd_K, r.ess_K, r.acc_birth*100, r.acc_fixed*100)
end

# Best σ per fixed-dim group, by ESJD(K).
best_nu = let sub = filter(r -> r.fdim == "none", tune_df)
    isempty(sub) ? (σ_birth = SIGMA_BIRTH,) : sub[argmax(sub.esjd_K), :]
end
best_rs = let sub = filter(r -> r.fdim == "resample", tune_df)
    isempty(sub) ? (σ_birth = SIGMA_BIRTH, σ_resample = SIGMA_RESAMPLE) : sub[argmax(sub.esjd_K), :]
end
@printf("\nSelected by ESJD(K):  NoUpdate σ_birth=%.2f   Resample σ_birth=%.2f σ_resample=%.2f\n",
        best_nu.σ_birth, best_rs.σ_birth, best_rs.σ_resample)
println("(If either sits on a grid boundary, widen the grid and re-run Part 2.)")

# ============================================================================
# Part 3 — final comparison at full length
# ============================================================================

final_cfgs = Any[
    make_cfg(; fam_ρ = BASE[1], fam_r = BASE[2], resample = false, varied = "final",
               σ_birth = best_nu.σ_birth,
               n_samples = N_SAMPLES_FINAL, burn = BURN_FINAL, suffix = "_final"),
    make_cfg(; fam_ρ = BASE[1], fam_r = BASE[2], resample = true, varied = "final",
               σ_birth = best_rs.σ_birth, σ_resample = best_rs.σ_resample,
               n_samples = N_SAMPLES_FINAL, burn = BURN_FINAL, suffix = "_final"),
    # Independent-proposal reference: agreement with the moment-matched runs is
    # the correctness check, there being no ground truth on real data.
    make_cfg(; fam_ρ = :prior, fam_r = :prior, resample = false, varied = "final",
               n_samples = N_SAMPLES_FINAL, burn = BURN_FINAL, suffix = "_final"),
    make_cfg(; family = :poisson, varied = "final",
               n_samples = N_SAMPLES_FINAL, burn = BURN_FINAL, suffix = "_final"),
    make_cfg(; family = :poisson_marg, varied = "final",
               n_samples = N_SAMPLES_FINAL, burn = BURN_FINAL, suffix = "_final"),
]
# A second seed for the best variant, to separate Monte Carlo error from
# genuine disagreement between proposal families.
push!(final_cfgs, make_cfg(; fam_ρ = BASE[1], fam_r = BASE[2], resample = true,
                             varied = "final-seed2", label = "NB ρ:lnmm r:lnmm + Resample (seed 2)",
                             σ_birth = best_rs.σ_birth, σ_resample = best_rs.σ_resample,
                             n_samples = N_SAMPLES_FINAL, burn = BURN_FINAL,
                             seed = 202, suffix = "_final"))

println("\n$(repeat('=', 80))")
println("PART 3 — final comparison: $(length(final_cfgs)) configs × $N_SAMPLES_FINAL iterations")
println("$(repeat('=', 80))")

final_df = DataFrame(collect(pmap(run_and_save, final_cfgs)))
CSV.write(joinpath(OUTDIR, "summary_final.csv"), final_df)
print_summary_table(final_df, "PART 3 — dengue final comparison")

# ============================================================================
# Part 4 — sensitivity to the ddCRP decay scale s
#
# s is fixed rather than inferred (see S_DECAY), so the results must be shown to
# be robust to that choice. One chain per s on the grid, plus one chain with s
# inferred so the degeneracy is on the record rather than quietly avoided.
# ============================================================================

const S_GRID = IS_TEST ? [0.5, 2.0] : [0.25, 0.5, 1.0, 2.0, 4.0]

sens_cfgs = Any[
    make_cfg(; fam_ρ = BASE[1], fam_r = BASE[2], resample = true, varied = "sens",
               label = "NB s=$(s) (fixed)", s = s, infer_s = false,
               σ_birth = best_rs.σ_birth, σ_resample = best_rs.σ_resample,
               n_samples = N_SAMPLES_TUNE, burn = BURN_TUNE, suffix = "_sens_s$(s)")
    for s in S_GRID
]
push!(sens_cfgs, make_cfg(; fam_ρ = BASE[1], fam_r = BASE[2], resample = true,
                            varied = "sens", label = "NB s inferred",
                            s = S_DECAY, infer_s = true,
                            σ_birth = best_rs.σ_birth, σ_resample = best_rs.σ_resample,
                            n_samples = N_SAMPLES_TUNE, burn = BURN_TUNE,
                            suffix = "_sens_infer_s"))

println("\n$(repeat('=', 80))")
println("PART 4 — sensitivity to the decay scale s: $(length(sens_cfgs)) configs × $N_SAMPLES_TUNE iterations")
println("$(repeat('=', 80))")

sens_df = DataFrame(collect(pmap(run_and_save, sens_cfgs)))
CSV.write(joinpath(OUTDIR, "summary_sensitivity.csv"), sens_df)

println()
@printf("%10s %9s %8s %8s %9s %9s %9s %9s\n",
        "s (fixed)", "infer_s", "mean K", "mode K", "s post", "α post", "acc.D%", "ESS(K)")
println("-"^80)
for r in eachrow(sens_df)
    @printf("%10s %9s %8.2f %8d %9.3f %9.3f %9.2f %9.0f\n",
            r.infer_s ? "—" : @sprintf("%.2f", r.s_decay), r.infer_s,
            r.mean_K, r.mode_K, r.s_post, r.α_post, r.acc_death*100, r.ess_K)
end
println("\nK should move only modestly across the fixed-s grid. If the inferred-s row")
println("shows s ≫ 1 with α → 0, the ddCRP has collapsed to a distance-independent CRP.")

# Metadata the analyse script needs to rebuild the dataset and name the chains.
jldsave(joinpath(OUTDIR, "run_metadata.jld2");
        uf = UF, year = YEAR, y = Y_ALL, E = E_ALL, D = D_ALL, D_km = D_KM,
        lat = Float64.(DF.lat), lon = Float64.(DF.lon),
        geocode = Int.(DF.geocode), name = String.(DF.name),
        burn = BURN_FINAL, thin = THIN, s_decay = S_DECAY, d_scale_km = D_SCALE,
        chain_files = [c.chain_filename for c in final_cfgs],
        labels = [c.label for c in final_cfgs])

println("\nAll chains complete. Run src/analyse_dengue.jl for figures, tables and model comparison.")

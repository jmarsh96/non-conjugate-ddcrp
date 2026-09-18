# ============================================================================
# run_aqs.jl
#
# Real-data application: spatial clustering of US PM2.5 monitoring sites under
# the 3-parameter, fully non-conjugate SkewNormal likelihood.
#
#   y_i | ξ_k, ω_k, α_k ~ SkewNormal(ξ_k, ω_k, α_k)     for site i in cluster k
#     y_i = the site's 2023 98th-percentile daily PM2.5 (the NAAQS design-value
#           metric), which is genuinely right-skewed across sites
#
# The distance driving the ddCRP is the great-circle distance between monitors.
# There is NO Gibbs baseline — the model is non-conjugate in all three
# parameters — so correctness is judged by mutual agreement across proposal
# families, as in the SkewNormal simulation study and the Old Faithful analysis.
# The model itself is reused verbatim from `src/skewnormal_model.jl`.
#
# Run `src/prepare_aqs_data.jl` first to build the dataset.
#
# Usage:
#   julia --project=. src/run_aqs.jl --pilot
#   julia --project=. src/run_aqs.jl --test
#   julia --project=. src/run_aqs.jl --region east
#   julia --project=. src/run_aqs.jl --slurm
# ============================================================================

using DistanceDependentCRP, CSV, DataFrames, Distances, Statistics, StatsBase, Random, JLD2, Printf, Distributions
using Distributed

const USE_SLURM = "--slurm" in ARGS
const IS_TEST = "--test" in ARGS
const IS_PILOT = "--pilot" in ARGS
const INFER_S = "--infer_s" in ARGS

function arg_str(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) ? default : ARGS[i+1]
end
const REGION = arg_str("--region", "conus")
const YEAR = parse(Int, arg_str("--year", "2023"))

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

# ============================================================================
# Manually-tuned proposal parameters
#
# Starting values carried over from the SkewNormal simulation study, rescaled
# for this response (98th-percentile PM2.5, roughly 10–60 µg/m³ across sites,
# an order of magnitude wider than the simulated data). Part 2 refines them.
# ============================================================================

@everywhere const PROP_SDS = (ξ = 1.5, ω = 0.15, α = 1.2)   # within-cluster MH
const SIGMA_BIRTH = 2.0
const SIGMA_RESAMPLE = 0.5

# ============================================================================
# Per-parameter proposal-family builders (identical design to run_skewnormal.jl)
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

# Only post-burn-in, thinned draws are written to disk. At full length the raw
# chain is 250k × n per parameter matrix — about 5.6 GB per chain at n=697 —
# so saving every draw would exhaust memory in the analyse script, which loads
# several chains at once.
@everywhere thin_chain(s::SkewNormalClustersSamples, keep) =
    SkewNormalClustersSamples(s.c[keep, :], s.ξ[keep, :], s.ω[keep, :], s.α[keep, :],
                              s.logpost[keep], s.α_ddcrp[keep], s.s_ddcrp[keep])

# ============================================================================
# Unit of work
# ============================================================================

@everywhere function run_and_save(cfg)
    Random.seed!(cfg.seed)

    data = ContinuousData(cfg.y, cfg.D)
    ddcrp = DDCRPParams(1.0, cfg.s, DistanceDependentCRP.exp_decay, 1.0, 0.01, 1.0, 0.01)

    opts = MCMCOptions(
        n_samples = cfg.n_samples,
        verbose = false,
        track_diagnostics = true,
        infer_params = Dict{Symbol,Bool}(
            :α_ddcrp => true, :s_ddcrp => cfg.infer_s,
            # The symmetric comparator additionally pins α ≈ 0 through a
            # near-degenerate α prior (see PRIORS_SYM), so that births as well as
            # within-cluster updates leave it there.
            :ξ => true, :ω => true, :α => cfg.infer_skew),
        prop_sds = Dict{Symbol,Float64}(
            :ξ => PROP_SDS.ξ, :ω => PROP_SDS.ω, :α => PROP_SDS.α, :s_ddcrp => 0.3),
    )

    birth = make_birth(cfg.fam_ξ, cfg.fam_ω, cfg.fam_α, cfg.σ_birth)
    fdim  = make_fdim(cfg.fam_ξ, cfg.fam_ω, cfg.fam_α, cfg.σ_resample; resample = cfg.resample)

    total_time = @elapsed begin
        samples, diag = mcmc(SkewNormalClusters(), data, ddcrp, cfg.priors, birth;
                             fixed_dim_proposal = fdim, opts = opts)
    end

    post_c  = samples.c[(cfg.burn + 1):cfg.thin:end, :]
    post_lp = samples.logpost[(cfg.burn + 1):cfg.thin:end]

    k_trace = calculate_n_clusters(post_c)
    acc     = acceptance_rates(diag)
    ess_K   = effective_sample_size(Float64.(k_trace))

    keep = (cfg.burn + 1):cfg.thin:cfg.n_samples
    med_ess(M) = median([effective_sample_size(M[keep, i]) for i in 1:size(M, 2)])

    chain_dir = joinpath(cfg.outdir, "chains"); mkpath(chain_dir)
    jldsave(joinpath(chain_dir, cfg.chain_filename);
            samples = thin_chain(samples, keep), burn = 0, thin = 1,
            burn_applied = cfg.burn, thin_applied = cfg.thin, label = cfg.label)

    return (
        label = cfg.label,
        fam_ξ = String(cfg.fam_ξ), fam_ω = String(cfg.fam_ω), fam_α = String(cfg.fam_α),
        birth = "$(cfg.fam_ξ)/$(cfg.fam_ω)/$(cfg.fam_α)",
        fdim = cfg.resample ? "resample" : "none",
        varied = cfg.varied,
        infer_skew = cfg.infer_skew,
        s_decay = cfg.s, infer_s = cfg.infer_s,
        s_post = mean(samples.s_ddcrp[keep]),
        α_post_ddcrp = mean(samples.α_ddcrp[keep]),
        σ_birth = cfg.σ_birth,
        σ_resample = cfg.resample ? cfg.σ_resample : NaN,
        n_obs = length(cfg.y),
        n_post = length(k_trace),
        mean_K = mean(k_trace), mode_K = Int(mode(k_trace)),
        std_K = std(Float64.(k_trace)),
        min_K = minimum(k_trace), max_K = maximum(k_trace),
        acc_birth = acc.birth, acc_death = acc.death,
        acc_fixed = acc.fixed, acc_overall = acc.overall,
        ess_K = ess_K,
        ess_logpost = effective_sample_size(post_lp),
        med_ess_ξ = med_ess(samples.ξ),
        med_ess_ω = med_ess(samples.ω),
        med_ess_α = med_ess(samples.α),
        esjd_K = mean(diff(Float64.(k_trace)).^2),
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
    joinpath(@__DIR__, "..", "results", "real_data", "aqs_test") :
    joinpath(@__DIR__, "..", "results", "real_data", "aqs")
mkpath(OUTDIR)

# ============================================================================
# Load data
# ============================================================================

const DATA_FILE = joinpath(@__DIR__, "..", "data", "cleaned", "aqs_pm25_$(YEAR)_$(REGION).csv")
isfile(DATA_FILE) || error("""
    Missing $DATA_FILE
    Run: julia --project=. src/prepare_aqs_data.jl --year $YEAR --region $REGION""")

const DF = CSV.read(DATA_FILE, DataFrame)

# The 98th percentile of daily PM2.5 is the response: it is the NAAQS design
# metric and, unlike the mean of log daily values, it is right-skewed across
# sites with |γ₁| inside the SkewNormal's attainable range. See the skewness
# report printed by prepare_aqs_data.jl.
const Y_ALL = Float64.(DF.y_p98)

const COORDS = permutedims(hcat(Float64.(DF.lon), Float64.(DF.lat)))
const D_KM = pairwise(Haversine(6371.0), COORDS)
const D_SCALE = median(D_KM[i, j] for i in 1:size(D_KM, 1) for j in 1:size(D_KM, 2) if i != j)
const D_ALL = D_KM ./ D_SCALE

@printf("Loaded %s: %d monitoring sites in %d states\n",
        basename(DATA_FILE), nrow(DF), length(unique(DF.state)))
@printf("  y (98th pct daily PM2.5): min %.1f  median %.1f  max %.1f µg/m³\n",
        minimum(Y_ALL), median(Y_ALL), maximum(Y_ALL))
let m = mean(Y_ALL), s = std(Y_ALL)
    @printf("  skewness across sites: %.3f  (SkewNormal attains |γ₁| < 0.995)\n",
            mean(((x - m) / s)^3 for x in Y_ALL))
end
@printf("  pairwise distance: median %.0f km, max %.0f km (D rescaled by the median)\n",
        D_SCALE, maximum(D_KM))

# ============================================================================
# Shared model settings
# ============================================================================

# Scaled to the observed spread of the response but deliberately uninformative,
# matching the convention used in run_skewnormal.jl: wide location, heavy-tailed
# scale (InverseGamma shape 2 ⇒ infinite variance), wide skew.
const PRIORS = SkewNormalClustersPriors(
    ξ_m = median(Y_ALL), ξ_s = 10 * std(Y_ALL),
    ω_a = 2.0, ω_b = std(Y_ALL),
    α_m = 0.0, α_s = 10.0,
)

# Symmetric (Normal) comparator: α is pinned at ≈0 by a near-degenerate prior,
# so births draw α ≈ 0 and the within-cluster α update is switched off. Scoring
# this against the full SkewNormal fit tests whether the skew parameter — the
# thing that makes the model non-conjugate — is earning its place.
const PRIORS_SYM = SkewNormalClustersPriors(
    ξ_m = median(Y_ALL), ξ_s = 10 * std(Y_ALL),
    ω_a = 2.0, ω_b = std(Y_ALL),
    α_m = 0.0, α_s = 1e-4,
)

# Fixed for the same reason as in run_dengue.jl: an inferred decay scale
# collapses the ddCRP toward a distance-independent CRP. Part 4 reports the
# sensitivity to this choice.
const S_DECAY = 0.5

# s is a RATE: the kernel is exp(-d*s), so the decay length is D_SCALE / s.
@printf("  ddCRP decay scale s = %.2f fixed, a rate (≈ %.0f km decay length)\n",
        S_DECAY, D_SCALE / S_DECAY)

# ============================================================================
# Config builder
# ============================================================================

fmtlabel(s) = replace(lowercase(s), r"[^a-z0-9]+" => "_")

function make_cfg(; fam_ξ = :nmm, fam_ω = :lnmm, fam_α = :nmm, resample = false,
                    varied = "baseline", label = nothing, infer_skew = true,
                    y = Y_ALL, D = D_ALL,
                    n_samples = N_SAMPLES_SWEEP, burn = BURN_SWEEP,
                    σ_birth = SIGMA_BIRTH, σ_resample = SIGMA_RESAMPLE,
                    s = S_DECAY, infer_s = INFER_S, seed = 42, suffix = "")
    lbl = something(label,
        "Skew ξ:$(fam_ξ) ω:$(fam_ω) α:$(fam_α) + $(resample ? "Resample" : "NoUpdate")")
    # The symmetric comparator must draw α from the degenerate prior, not from a
    # moment-matched family, or births would reintroduce skewness.
    infer_skew || (fam_α = :prior)
    (
        label = lbl,
        fam_ξ = fam_ξ, fam_ω = fam_ω, fam_α = fam_α,
        resample = resample, varied = varied, infer_skew = infer_skew,
        σ_birth = σ_birth, σ_resample = σ_resample,
        priors = infer_skew ? PRIORS : PRIORS_SYM, y = y, D = D, s = s,
        n_samples = n_samples, burn = burn, thin = THIN,
        outdir = OUTDIR,
        chain_filename = fmtlabel(lbl) * suffix * ".jld2",
        seed = seed, infer_s = infer_s,
    )
end

function print_summary_table(df, title)
    W = 142
    println(); println("=" ^ W)
    println(title * "  (post burn-in, thinned by $THIN)"); println("=" ^ W)
    println(rpad("ξ / ω / α", 22) * rpad("Fixed-dim", 11) *
            "Mean K  Mode K  Acc.B%  Acc.D%  Acc.F%   ESS(K)   ESS_ξ   ESS_ω   ESS_α  ESJD(K)  Time(s)")
    println("-" ^ W)
    for r in eachrow(df)
        @printf("%-22s%-11s%6.2f  %6d  %6.2f  %6.2f  %6.2f  %7.0f %7.0f %7.0f %7.0f  %7.4f  %7.1f\n",
            r.birth, r.fdim, r.mean_K, r.mode_K,
            r.acc_birth*100, r.acc_death*100, r.acc_fixed*100, r.ess_K,
            r.med_ess_ξ, r.med_ess_ω, r.med_ess_α, r.esjd_K, r.total_time)
    end
    println("=" ^ W)
end

# ============================================================================
# Part 0 — pilot: cost scaling in n, and is the sampler moving?
# ============================================================================

if IS_PILOT
    n_full = length(Y_ALL)
    sizes = unique(filter(m -> m <= n_full, [100, 200, 350, n_full]))
    n_iter = 8_000     # long enough to clear burn-in; 4k was not, for dengue

    println("\n$(repeat('=', 80))")
    println("PILOT — $(length(sizes)) subsample sizes × $n_iter iterations")
    println("  Chooses the AQS region for the full study on measured cost.")
    println("$(repeat('=', 80))")

    # Longitude-ordered stride, so a subsample still spans the region.
    order = sortperm(DF.lon)
    pilot_cfgs = [
        begin
            idx = sort(order[round.(Int, range(1, n_full; length = m))])
            make_cfg(; label = "pilot n=$m", y = Y_ALL[idx], D = D_ALL[idx, idx],
                       resample = true, n_samples = n_iter, burn = n_iter ÷ 2,
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

    if nrow(pdf) >= 2
        ln, lt = log.(pdf.n_obs), log.(pdf.sec_per_1k)
        p = (length(ln) * sum(ln .* lt) - sum(ln) * sum(lt)) /
            (length(ln) * sum(ln .^ 2) - sum(ln)^2)
        a = exp(mean(lt) - p * mean(ln))
        @printf("\nFitted cost ≈ %.3g · n^%.2f sec per 1k iterations\n", a, p)
        println("Extrapolated wall-clock for one 250k-iteration chain:")
        for (m, nm) in [(90, "ca"), (227, "west"), (470, "east"), (697, "conus")]
            @printf("  n=%3d (%-5s): %5.1f h\n", m, nm, a * m^p * 250 / 3600)
        end
        println("\n(8 workers run in parallel, so a sweep of C configs costs ≈ ceil(C/8) chains.)")
    end

    CSV.write(joinpath(OUTDIR, "summary_pilot_$(REGION).csv"), pdf)
    println("\nPilot summary → $(joinpath(OUTDIR, "summary_pilot_$(REGION).csv"))")
    exit(0)
end

# ============================================================================
# Part 1 — proposal-family sweep (same coordinate design as run_skewnormal.jl)
# ============================================================================

const BASE = (:nmm, :lnmm, :nmm)
const BIRTH_SPECS = Tuple{Symbol,Symbol,Symbol,String}[
    (BASE...,               "baseline"),
    (:prior, :lnmm,  :nmm,  "ξ:prior"),
    (:nmm,   :prior, :nmm,  "ω:prior"),
    (:nmm,   :nmm,   :nmm,  "ω:nmm"),
    (:nmm,   :ig,    :nmm,  "ω:ig"),
    (:nmm,   :lnmm,  :prior, "α:prior"),
    (:prior, :prior, :prior, "all-prior"),
]

sweep_cfgs = Any[]
for (fξ, fω, fα, varied) in BIRTH_SPECS, resample in (false, true)
    push!(sweep_cfgs, make_cfg(; fam_ξ = fξ, fam_ω = fω, fam_α = fα,
                                 resample = resample, varied = varied))
end

println("\n$(repeat('=', 80))")
println("PART 1 — proposal-family sweep: $(length(sweep_cfgs)) configs × $N_SAMPLES_SWEEP iterations")
println("$(repeat('=', 80))")

sweep_df = DataFrame(collect(pmap(run_and_save, sweep_cfgs)))
CSV.write(joinpath(OUTDIR, "summary_sweep.csv"), sweep_df)
print_summary_table(sweep_df, "PART 1 — AQS proposal-family sweep")

println("\nBest birth family per parameter (by median parameter ESS, NoUpdate rows):")
nu = filter(r -> r.fdim == "none", sweep_df)
for (pname, esscol, famcol) in [("ξ", :med_ess_ξ, :fam_ξ), ("ω", :med_ess_ω, :fam_ω),
                                ("α", :med_ess_α, :fam_α)]
    cand = filter(r -> r.varied == "baseline" || startswith(r.varied, "$pname:"), nu)
    isempty(cand) && continue
    best = cand[argmax([getproperty(r, esscol) for r in eachrow(cand)]), :]
    @printf("  %s: best = %-6s (median ESS %.0f, mean K %.2f)\n",
            pname, getproperty(best, famcol), getproperty(best, esscol), best.mean_K)
end

# ============================================================================
# Part 2 — tuning sweep by ESJD(K)
# ============================================================================

# Grids centred on what the pilot measured at σ_birth=2.0, σ_resample=0.5:
# birth acceptance ≈ 1% (in line with the SkewNormal simulation study, so the
# grid extends downward rather than assuming a bug) and fixed-dim acceptance
# ≈ 79%, which is far too high — the resample step is barely moving, so that
# grid extends upward.
const SIGMA_GRID    = IS_TEST ? [2.0] : [0.5, 1.0, 2.0, 4.0]
const RESAMPLE_GRID = IS_TEST ? [0.5] : [0.25, 0.5, 1.0, 2.0]

tune_cfgs = Any[]
for σb in SIGMA_GRID
    push!(tune_cfgs, make_cfg(; fam_ξ = BASE[1], fam_ω = BASE[2], fam_α = BASE[3],
                                resample = false, varied = "tune", σ_birth = σb,
                                n_samples = N_SAMPLES_TUNE, burn = BURN_TUNE,
                                suffix = "_tune_b$(σb)"))
    for σr in RESAMPLE_GRID
        push!(tune_cfgs, make_cfg(; fam_ξ = BASE[1], fam_ω = BASE[2], fam_α = BASE[3],
                                    resample = true, varied = "tune",
                                    σ_birth = σb, σ_resample = σr,
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
# Part 3 — final comparison
# ============================================================================

final_cfgs = Any[
    make_cfg(; fam_ξ = BASE[1], fam_ω = BASE[2], fam_α = BASE[3], resample = false,
               varied = "final", σ_birth = best_nu.σ_birth,
               n_samples = N_SAMPLES_FINAL, burn = BURN_FINAL, suffix = "_final"),
    make_cfg(; fam_ξ = BASE[1], fam_ω = BASE[2], fam_α = BASE[3], resample = true,
               varied = "final", σ_birth = best_rs.σ_birth, σ_resample = best_rs.σ_resample,
               n_samples = N_SAMPLES_FINAL, burn = BURN_FINAL, suffix = "_final"),
    make_cfg(; fam_ξ = :prior, fam_ω = :prior, fam_α = :prior, resample = false,
               varied = "final", n_samples = N_SAMPLES_FINAL, burn = BURN_FINAL,
               suffix = "_final"),
    # Symmetric comparator: α held at its initial value throughout. Scoring this
    # against the full fit is the evidence that the skew parameter — and hence
    # the non-conjugacy — is doing real work.
    make_cfg(; fam_ξ = BASE[1], fam_ω = BASE[2], fam_α = BASE[3], resample = true,
               varied = "final-symmetric", label = "Skew symmetric (α fixed)",
               infer_skew = false, σ_birth = best_rs.σ_birth, σ_resample = best_rs.σ_resample,
               n_samples = N_SAMPLES_FINAL, burn = BURN_FINAL, suffix = "_final"),
    # Second seed for the best variant, to separate Monte Carlo error from
    # genuine disagreement between proposal families.
    make_cfg(; fam_ξ = BASE[1], fam_ω = BASE[2], fam_α = BASE[3], resample = true,
               varied = "final-seed2", label = "Skew ξ:nmm ω:lnmm α:nmm + Resample (seed 2)",
               σ_birth = best_rs.σ_birth, σ_resample = best_rs.σ_resample,
               n_samples = N_SAMPLES_FINAL, burn = BURN_FINAL, seed = 202, suffix = "_final"),
]

println("\n$(repeat('=', 80))")
println("PART 3 — final comparison: $(length(final_cfgs)) configs × $N_SAMPLES_FINAL iterations")
println("$(repeat('=', 80))")

final_df = DataFrame(collect(pmap(run_and_save, final_cfgs)))
CSV.write(joinpath(OUTDIR, "summary_final.csv"), final_df)
print_summary_table(final_df, "PART 3 — AQS final comparison")

# ============================================================================
# Part 4 — sensitivity to the decay scale s
# ============================================================================

const S_GRID = IS_TEST ? [0.5, 2.0] : [0.25, 0.5, 1.0, 2.0, 4.0]

sens_cfgs = Any[
    make_cfg(; fam_ξ = BASE[1], fam_ω = BASE[2], fam_α = BASE[3], resample = true,
               varied = "sens", label = "Skew s=$(s) (fixed)", s = s, infer_s = false,
               σ_birth = best_rs.σ_birth, σ_resample = best_rs.σ_resample,
               n_samples = N_SAMPLES_TUNE, burn = BURN_TUNE, suffix = "_sens_s$(s)")
    for s in S_GRID
]
push!(sens_cfgs, make_cfg(; fam_ξ = BASE[1], fam_ω = BASE[2], fam_α = BASE[3],
                            resample = true, varied = "sens", label = "Skew s inferred",
                            s = S_DECAY, infer_s = true,
                            σ_birth = best_rs.σ_birth, σ_resample = best_rs.σ_resample,
                            n_samples = N_SAMPLES_TUNE, burn = BURN_TUNE,
                            suffix = "_sens_infer_s"))

println("\n$(repeat('=', 80))")
println("PART 4 — sensitivity to the decay scale s: $(length(sens_cfgs)) configs")
println("$(repeat('=', 80))")

sens_df = DataFrame(collect(pmap(run_and_save, sens_cfgs)))
CSV.write(joinpath(OUTDIR, "summary_sensitivity.csv"), sens_df)

println()
@printf("%10s %9s %8s %8s %9s %9s %9s\n",
        "s (fixed)", "infer_s", "mean K", "mode K", "s post", "α post", "ESS(K)")
println("-"^72)
for r in eachrow(sens_df)
    @printf("%10s %9s %8.2f %8d %9.3f %9.3f %9.0f\n",
            r.infer_s ? "—" : @sprintf("%.2f", r.s_decay), r.infer_s,
            r.mean_K, r.mode_K, r.s_post, r.α_post_ddcrp, r.ess_K)
end

# ============================================================================

jldsave(joinpath(OUTDIR, "run_metadata.jld2");
        region = REGION, year = YEAR, y = Y_ALL, D = D_ALL, D_km = D_KM,
        lat = Float64.(DF.lat), lon = Float64.(DF.lon),
        site_id = String.(DF.site_id), state = String.(DF.state_name),
        burn = BURN_FINAL, thin = THIN, s_decay = S_DECAY, d_scale_km = D_SCALE,
        chain_files = [c.chain_filename for c in final_cfgs],
        labels = [c.label for c in final_cfgs])

println("\nAll chains complete. Run src/analyse_aqs.jl for figures, tables and model comparison.")

# ============================================================================
# analyse_dengue.jl
#
# Figures, tables and summary numbers for the Brazil dengue application, as
# reported in the paper. Run AFTER `src/run_dengue.jl`; it reads the final
# chains that script wrote.
#
# Writes to results/real_data/dengue/ (dengue_test/ under --test):
#   figures/dengue_k_traces.png   P(K) per chain, traces of K and α_ddcrp
#   figures/dengue_ppc.png        posterior predictive checks, NB and Poisson
#   table_dengue.tex              mixing of every chain of the selected sampler
#   dengue_numbers.tex            \newcommand macros for every number the
#                                 paper quotes, so the text updates with a rerun
#   model_comparison.csv          WAIC / LPML / PSIS-LOO, NB vs Poisson
#   ppc_stats.csv, cluster_assignments.csv (input to src/map_dengue.R)
#
# and, unless --test, copies the paper assets into docs/paper/{figures,tables}/.
#
# Usage:
#   julia --project=. src/analyse_dengue.jl
#   julia --project=. src/analyse_dengue.jl --test
# ============================================================================

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using DistanceDependentCRP, StatsPlots, DataFrames, CSV, Statistics, StatsBase,
      JLD2, Printf, Distributions, Random

# Brings the NegBinPopulationRates* types into scope so JLD2 deserialises the
# saved chains as their real types.
include(joinpath(@__DIR__, "negbin_population_model.jl"))
include(joinpath(@__DIR__, "model_comparison.jl"))
include(joinpath(@__DIR__, "dengue_common.jl"))

ENV["GKSwstype"] = "nul"   # headless GR

const IS_TEST = "--test" in ARGS
const OUTDIR = IS_TEST ?
    joinpath(@__DIR__, "..", "results", "real_data", "dengue_test") :
    joinpath(@__DIR__, "..", "results", "real_data", "dengue")
const FIGDIR = joinpath(OUTDIR, "figures"); mkpath(FIGDIR)
const PAPER = joinpath(@__DIR__, "..", "docs", "paper")

fmt(x::Real, d::Int) = Printf.format(Printf.Format("%.$(d)f"), Float64(x))
# Thousands separator in LaTeX's {,} form, for counts quoted in the text.
function fmtint(x::Real)
    x < 0 && return "-" * fmtint(-x)
    s = string(round(Int, x))
    parts = String[]
    while length(s) > 3
        pushfirst!(parts, s[end-2:end]); s = s[1:end-3]
    end
    pushfirst!(parts, s)
    return join(parts, "{,}")
end

# Prose names, as they read in "uses the ... proposal for ρ_k".
const FAMILY_NAME = Dict("prior" => "prior", "nmm" => "truncated-normal moment-matched",
                         "lnmm" => "log-normal moment-matched",
                         "ig" => "inverse-gamma moment-matched")

# ============================================================================
# Load run outputs
# ============================================================================

const META_FILE = joinpath(OUTDIR, "run_metadata.jld2")
isfile(META_FILE) || error("""
    Missing $META_FILE
    Run: julia --project=. src/run_dengue.jl$(IS_TEST ? " --test" : "")""")

meta = load(META_FILE)
const Y = meta["y"]; const E = meta["E"]; const D = meta["D"]
const LAT = meta["lat"]; const LON = meta["lon"]
const GEOCODE = meta["geocode"]; const NAME = meta["name"]
const DATA = CountDataWithPopulation(Y, E, D)
const N = length(Y)
const NBRS = neighbour_lists(meta["adjacency"])

final_df = DataFrame(CSV.File(joinpath(OUTDIR, "summary_final.csv")))
sweep_df = DataFrame(CSV.File(joinpath(OUTDIR, "summary_sweep.csv")))
tune_file = joinpath(OUTDIR, "summary_tune.csv")
tune_df = isfile(tune_file) ? DataFrame(CSV.File(tune_file)) : nothing

chains = Dict{String,Any}()
for (label, fname) in zip(meta["labels"], meta["chain_files"])
    path = joinpath(OUTDIR, "chains", fname)
    if isfile(path)
        chains[label] = load(path)["samples"]
    else
        @warn "Chain file not found; skipping" label path
    end
end

const BEST = meta["best_label"]
const POIS = only(l for l in meta["labels"] if startswith(l, "Poisson"))
haskey(chains, BEST) || error("Selected chain $BEST not found")
# Every chain of the selected sampler, in the order the run script produced
# them: the first is the one summarised in the maps and the model comparison,
# the rest are replicate chains from other seeds.
nb_labels = [l for l in meta["labels"] if l != POIS && haskey(chains, l)]
const NCHAIN = length(nb_labels)

println("Loaded $(length(chains)) chains over $N municipalities; selected sampler: $BEST\n")

# ============================================================================
# Figure: posterior of K and traces
# ============================================================================

println("Generating P(K) and traces...")
const CHAIN_COLOURS = [:steelblue, :darkorange, :seagreen, :purple, :firebrick]
chain_colour(i) = CHAIN_COLOURS[mod1(i, length(CHAIN_COLOURS))]
k_traces = Dict(l => calculate_n_clusters(chains[l].c) for l in nb_labels)
let
    # P(K) overlaid rather than dodged: with several chains, side-by-side bars
    # offset every pair and make Monte Carlo noise read as a systematic shift.
    allK = sort(unique(vcat(values(k_traces)...)))
    p1 = plot(; xlabel = "number of clusters K", ylabel = "posterior probability",
                legend = :topright)
    for (i, l) in enumerate(nb_labels)
        kt = k_traces[l]
        bar!(p1, allK, [count(==(k), kt) / length(kt) for k in allK];
             bar_width = 0.85, lw = 0, fillalpha = 0.45, color = chain_colour(i),
             label = "chain $i")
    end
    # One panel per chain, each with its running mean, so slow excursions are
    # visible instead of being hidden under whichever chain is drawn last.
    ps = Any[]
    for (i, l) in enumerate(nb_labels)
        kt = Float64.(k_traces[l])
        run_mean = cumsum(kt) ./ (1:length(kt))
        p = plot(kt; lw = 0.4, color = chain_colour(i), alpha = 0.75, label = "",
                 ylabel = "K", title = "chain $i", titlefontsize = 9,
                 xformatter = x -> string(round(Int, x)),
                 xlabel = i == NCHAIN ? "draw (post burn-in, thinned)" : "")
        plot!(p, run_mean; lw = 2, color = :black, label = "")
        push!(ps, p)
    end
    pα = plot(; xlabel = "draw (post burn-in, thinned)", ylabel = "α (ddCRP)",
                legend = false, xformatter = x -> string(round(Int, x)))
    for (i, l) in enumerate(nb_labels)
        plot!(pα, chains[l].α_ddcrp; lw = 0.4, color = chain_colour(i), alpha = 0.6)
    end
    push!(ps, pα)
    plot(p1, plot(ps...; layout = (length(ps), 1)); layout = (1, 2),
         size = (1500, 220 * length(ps)), left_margin = 6Plots.mm,
         bottom_margin = 6Plots.mm, guidefontsize = 11, tickfontsize = 9,
         legendfontsize = 10)
    savefig(joinpath(FIGDIR, "dengue_k_traces.png"))
end

# ============================================================================
# Point estimates and agreement between seeds
# ============================================================================

println("Computing point estimates...")
sims = Dict(l => compute_similarity_matrix(chains[l].c) for l in nb_labels)
modes = Dict(l => mode_point_estimate(chains[l].c, chains[l].logpost) for l in nb_labels)
pairs = [(a, b) for a in 1:NCHAIN for b in (a+1):NCHAIN]
# Agreement between chains, over every pair: the point estimates (ARI) and the
# full posteriors (co-clustering probabilities). The latter is the one to trust,
# since a single partition is a noisy summary of a diffuse posterior.
seed_ari = isempty(pairs) ? NaN :
    minimum(ari(modes[nb_labels[a]], modes[nb_labels[b]]) for (a, b) in pairs)
seed_sim_mad = isempty(pairs) ? NaN :
    maximum(mean(abs.(sims[nb_labels[a]] .- sims[nb_labels[b]])) for (a, b) in pairs)

# Rank-normalised split-R̂ across chains: the convergence check that replaces
# the comparison with a collapsed sampler.
rhat_K = split_rhat([Float64.(k_traces[l]) for l in nb_labels])
rhat_α = split_rhat([chains[l].α_ddcrp for l in nb_labels])
rhat_lp = split_rhat([chains[l].logpost for l in nb_labels])
@printf("  %d chains: worst pairwise ARI of mode partitions %.3f, worst mean |Δ co-clustering| %.4f\n",
        NCHAIN, seed_ari, seed_sim_mad)
@printf("  split-R̂: K %.4f, α %.4f, log posterior %.4f\n", rhat_K, rhat_α, rhat_lp)

# ============================================================================
# Mixing table
# ============================================================================

println("\nMixing and acceptance summary:")
nb_rows = filter(r -> r.family == "negbin", final_df)
for r in eachrow(final_df)
    @printf("  %-40s mean K %6.2f  mode K %3d  ESS(K) %6.0f  acc B/D/F %.2f/%.2f/%.2f%%\n",
            first(r.label, 40), r.mean_K, r.mode_K, r.ess_K,
            r.acc_birth*100, r.acc_death*100, r.acc_fixed*100)
end

let
    lines = String[
        "% Required packages: booktabs",
        "% Generated by src/analyse_dengue.jl; re-run it to update.",
        "\\begin{tabular}{l r r r r r r r r r}",
        "  \\toprule",
        "  Chain & Mean \$K\$ & Mode \$K\$ & ESS\$(K)\$ & ESS\$_\\rho\$ & ESS\$_r\$ & " *
            "Birth (\\%) & Death (\\%) & Fixed (\\%) & Time (h) \\\\",
        "  \\midrule",
    ]
    for (i, r) in enumerate(eachrow(nb_rows))
        push!(lines, "  Chain $i & " * fmt(r.mean_K, 1) * " & " * string(r.mode_K) * " & " *
              string(round(Int, r.ess_K)) * " & " * string(round(Int, r.med_ess_ρ)) * " & " *
              string(round(Int, r.med_ess_r)) * " & " * fmt(r.acc_birth * 100, 2) * " & " *
              fmt(r.acc_death * 100, 2) * " & " * fmt(r.acc_fixed * 100, 2) * " & " *
              fmt(r.total_time / 3600, 1) * " \\\\")
    end
    push!(lines, "  \\bottomrule", "\\end{tabular}")
    open(io -> println(io, join(lines, "\n")), joinpath(OUTDIR, "table_dengue.tex"), "w")
end

# ============================================================================
# Model comparison: is the non-conjugate NB needed?
# ============================================================================

println("\nModel comparison (WAIC / LPML / PSIS-LOO):")
scores = Any[]; pointwise = Dict{String,Any}()
for l in vcat(nb_labels[1:1], haskey(chains, POIS) ? [POIS] : String[])
    s = chains[l]
    model = s isa NegBinPopulationRatesSamples ? NegBinPopulationRates() : PoissonPopulationRates()
    sc, pw = score_fit_pointwise(l, loglik_matrix(model, s, DATA))
    push!(scores, sc); pointwise[l] = pw
end
print_comparison(scores)
CSV.write(joinpath(OUTDIR, "model_comparison.csv"), DataFrame(scores))
loo_diff = haskey(pointwise, POIS) ?
    elpd_difference(pointwise[BEST].elpd_loo_i, pointwise[POIS].elpd_loo_i) :
    (diff = NaN, se = NaN)
@printf("  elpd_loo(NB) - elpd_loo(Poisson) = %.1f (SE %.1f)\n", loo_diff.diff, loo_diff.se)

# ============================================================================
# Posterior predictive checks
# ============================================================================

println("\nPosterior predictive checks...")
ppc_rows = DataFrame(label = String[], statistic = String[], T_obs = Float64[], ppp = Float64[],
                     q025 = Float64[], q50 = Float64[], q975 = Float64[])
let
    stats = ppc_statistics(E, NBRS)
    short = Dict("dispersion index Var/Mean" => "dispersion index",
                 "maximum count" => "maximum count",
                 "n municipalities < 100/100k" => "# below 100 per 100k",
                 "90th pct of incidence" => "90th pct incidence",
                 "Moran's I of log incidence" => "Moran's I")
    ps = []
    ppc_labels = [l for l in (BEST, POIS) if haskey(chains, l)]
    for l in ppc_labels
        s = chains[l]
        model = s isa NegBinPopulationRatesSamples ? NegBinPopulationRates() : PoissonPopulationRates()
        Random.seed!(11)
        y_rep = posterior_predictive(model, s, DATA; n_rep = IS_TEST ? 200 : 1000)
        rowname = s isa NegBinPopulationRatesSamples ? "Negative binomial" : "Poisson"
        for (j, (sname, f)) in enumerate(stats)
            t_obs = f(Y)
            t_rep = [f(view(y_rep, r, :)) for r in axes(y_rep, 1)]
            p = ppc_pvalue(t_obs, t_rep)
            push!(ppc_rows, (l, sname, t_obs, p, quantile(t_rep, 0.025),
                             quantile(t_rep, 0.5), quantile(t_rep, 0.975)))
            @printf("  %-22s %-30s observed %12.3f  ppp = %.3f\n", first(rowname, 22), sname, t_obs, p)
            # A Poisson replicate can sit orders of magnitude from the data, so
            # the axis always spans both the replicates and the observed value.
            lo, hi = extrema(vcat(t_rep, t_obs)); pad = 0.05 * (hi - lo + eps())
            h = histogram(t_rep; bins = 30, color = :grey70, lw = 0, legend = false,
                          xlims = (lo - pad, hi + pad), xticks = 3,
                          xformatter = x -> @sprintf("%.3g", x),
                          title = j == 1 ? "$rowname\n$(short[sname])" : "\n$(short[sname])",
                          titlefontsize = 11, yticks = false)
            vline!(h, [t_obs]; lw = 2.5, color = :firebrick)
            push!(ps, h)
        end
    end
    plot(ps...; layout = (length(ppc_labels), length(stats)),
         size = (1600, 300 * length(ppc_labels)), left_margin = 6Plots.mm,
         bottom_margin = 4Plots.mm, tickfontsize = 8, guidefontsize = 11)
    savefig(joinpath(FIGDIR, "dengue_ppc.png"))
end
CSV.write(joinpath(OUTDIR, "ppc_stats.csv"), ppc_rows)

# ============================================================================
# Outputs for the R/sf maps
# ============================================================================

println("\nWriting map inputs...")
cluster_df = let
    s = chains[BEST]; z = modes[BEST]; sim = sims[BEST]
    conf = [mean(sim[i, findall(==(z[i]), z)]) for i in 1:N]
    DataFrame(geocode = GEOCODE, name = NAME, lat = LAT, lon = LON,
              y = Y, pop = E .* 1e5, incidence = Y ./ E,
              cluster = z, cocluster_confidence = conf,
              rho_mean = vec(mean(s.ρ, dims = 1)), r_mean = vec(mean(s.r, dims = 1)))
end
CSV.write(joinpath(OUTDIR, "cluster_assignments.csv"), cluster_df)
jldsave(joinpath(OUTDIR, "similarity_matrix.jld2"); sim = sims[BEST], label = BEST, geocode = GEOCODE)

# Cluster-level summary of the mode partition: size, and the posterior mean
# incidence and dispersion of its members.
clusters = combine(groupby(cluster_df, :cluster), nrow => :size,
                   :rho_mean => median => :rho, :r_mean => median => :r,
                   [:y, :pop] => ((y, p) -> sum(y) / sum(p) * 1e5) => :pooled_incidence)
sort!(clusters, :rho)
CSV.write(joinpath(OUTDIR, "cluster_summary.csv"), clusters)
@printf("  mode partition: %d clusters; %d singletons; fitted incidence %.0f to %.0f per 100k\n",
        nrow(clusters), count(==(1), clusters.size), minimum(clusters.rho), maximum(clusters.rho))

# ============================================================================
# Numbers quoted in the paper
# ============================================================================

println("\nWriting dengue_numbers.tex...")
let
    b = only(eachrow(filter(r -> r.label == BEST, final_df)))
    nb_final = filter(r -> r.family == "negbin", final_df)
    kt = calculate_n_clusters(chains[BEST].c)
    ci = quantile(Float64.(kt), [0.025, 0.975])
    α = chains[BEST].α_ddcrp
    ppc_nb = filter(r -> r.label == BEST, ppc_rows)
    ppc_po = filter(r -> r.label == POIS, ppc_rows)
    ppp(df, stat) = (r = filter(x -> x.statistic == stat, df); nrow(r) == 1 ? r.ppp[1] : NaN)
    sc = Dict(x.label => x for x in scores)
    pois_final = filter(r -> r.label == POIS, final_df)
    ppp_row(df, stat, col) = (r = filter(x -> x.statistic == stat, df); nrow(r) == 1 ? r[1, col] : NaN)
    # Share of municipalities in singleton clusters under the Poisson fit, over
    # an evenly spaced subsample of draws: a direct measure of how close that fit
    # comes to one rate per municipality.
    function pois_singleton_share()
        c = chains[POIS].c
        draws = round.(Int, range(1, size(c, 1); length = min(500, size(c, 1))))
        mean(begin
                 z = compute_table_assignments(c[s, :]); cm = countmap(z)
                 count(i -> cm[z[i]] == 1, eachindex(z)) / length(z)
             end for s in draws)
    end
    win = mean(count(<=(meta["hop_radius"]), view(meta["hop"], i, :)) - 1 for i in 1:N)

    macros = [
        # data
        "dengueN" => string(N),
        "dengueCases" => fmtint(sum(Y)),
        "denguePopM" => fmt(sum(E) / 10, 1),
        "dengueIncMedian" => fmtint(median(Y ./ E)),
        "dengueIncMin" => fmtint(minimum(Y ./ E)),
        "dengueIncMax" => fmtint(maximum(Y ./ E)),
        "dengueRhoBar" => fmtint(sum(Y) / sum(E)),
        "dengueEdges" => fmtint(sum(meta["adjacency"]) / 2),
        "dengueDegree" => fmt(sum(meta["adjacency"]) / N, 2),
        "dengueDiameter" => string(round(Int, maximum(meta["hop"]))),
        "dengueHopRadius" => string(meta["hop_radius"]),
        "dengueWindowSize" => fmt(win, 1),
        # sampler selection and run length
        "dengueSweepIter" => fmtint(sweep_df.n_iter[1]),
        "dengueSweepBurn" => fmtint(sweep_df.burn[1]),
        "dengueNSweep" => string(nrow(sweep_df)),
        "dengueNTune" => string(isnothing(tune_df) ? 0 : nrow(tune_df)),
        "dengueFamRho" => FAMILY_NAME[meta["best_fam_rho"]],
        "dengueFamR" => FAMILY_NAME[meta["best_fam_r"]],
        "dengueFixedDim" => meta["best_resample"] ? "Resample" : "No Update",
        "dengueSigmaBirth" => fmt(meta["sigma_birth"], 2),
        "dengueSigmaResample" => fmt(meta["sigma_resample"], 2),
        # The σ clause of the sentence, since σ_r only exists under Resample.
        "dengueSigmaText" => "\$\\sigma_b = $(fmt(meta["sigma_birth"], 2))\$" *
            (meta["best_resample"] ? " and \$\\sigma_r = $(fmt(meta["sigma_resample"], 2))\$" : ""),
        "dengueFinalIter" => fmtint(meta["n_samples"]),
        "dengueFinalBurn" => fmtint(meta["burn"]),
        "dengueFinalThin" => string(meta["thin"]),
        "dengueNPost" => fmtint(b.n_post),
        "dengueHours" => fmt(b.total_time / 3600, 1),
        # posterior of K and α
        "dengueMeanK" => fmt(b.mean_K, 1),
        "dengueModeK" => string(b.mode_K),
        "dengueKlo" => string(round(Int, ci[1])),
        "dengueKhi" => string(round(Int, ci[2])),
        "dengueNChains" => string(NCHAIN),
        # Range of the per-chain posterior means and modes of K, so the text can
        # quote the spread across chains rather than one replicate.
        "dengueMeanKLo" => fmt(minimum(nb_final.mean_K), 1),
        "dengueMeanKHi" => fmt(maximum(nb_final.mean_K), 1),
        "dengueModeKLo" => string(minimum(nb_final.mode_K)),
        "dengueModeKHi" => string(maximum(nb_final.mode_K)),
        "dengueRhatK" => fmt(rhat_K, 3),
        "dengueRhatAlpha" => fmt(rhat_α, 3),
        "dengueRhatLogpost" => fmt(rhat_lp, 3),
        "dengueEssKLo" => string(round(Int, minimum(nb_final.ess_K))),
        "dengueEssKHi" => string(round(Int, maximum(nb_final.ess_K))),
        "dengueAlphaMean" => fmt(mean(α), 2),
        "dengueAlphaLo" => fmt(quantile(α, 0.025), 2),
        "dengueAlphaHi" => fmt(quantile(α, 0.975), 2),
        "dengueSeedARI" => fmt(seed_ari, 2),
        "dengueSeedSimMAD" => fmt(seed_sim_mad, 3),
        # mixing
        "dengueAccBirth" => fmt(b.acc_birth * 100, 2),
        "dengueAccDeath" => fmt(b.acc_death * 100, 2),
        "dengueAccFixed" => fmt(b.acc_fixed * 100, 1),
        "dengueEssK" => string(round(Int, b.ess_K)),
        "dengueEssRho" => string(round(Int, b.med_ess_ρ)),
        "dengueEssR" => string(round(Int, b.med_ess_r)),
        # mode partition
        "dengueNClusters" => string(nrow(clusters)),
        "dengueNSingletons" => string(count(==(1), clusters.size)),
        "dengueLargestCluster" => string(maximum(clusters.size)),
        "dengueRhoLow" => fmtint(minimum(clusters.rho)),
        "dengueRhoHigh" => fmtint(maximum(clusters.rho)),
        "dengueConfMedian" => fmt(median(cluster_df.cocluster_confidence), 2),
        # Poisson comparator and model comparison
        "denguePoisMeanK" => nrow(pois_final) == 1 ? fmt(pois_final.mean_K[1], 0) : "--",
        "dengueWaicNB" => haskey(sc, BEST) ? fmtint(sc[BEST].waic) : "--",
        "dengueWaicPois" => haskey(sc, POIS) ? fmtint(sc[POIS].waic) : "--",
        "dengueLooDiff" => fmtint(loo_diff.diff),
        "dengueLooDiffSE" => fmtint(loo_diff.se),
        "dengueBadK" => haskey(sc, BEST) ? string(sc[BEST].n_bad_k) : "--",
        "dengueBadKPois" => haskey(sc, POIS) ? string(sc[POIS].n_bad_k) : "--",
        # posterior predictive p-values
        "denguePppDispNB" => fmt(ppp(ppc_nb, "dispersion index Var/Mean"), 2),
        "denguePppMaxNB" => fmt(ppp(ppc_nb, "maximum count"), 2),
        "denguePppLowNB" => fmt(ppp(ppc_nb, "n municipalities < 100/100k"), 2),
        "denguePppQNB" => fmt(ppp(ppc_nb, "90th pct of incidence"), 2),
        "denguePppMoranNB" => fmt(ppp(ppc_nb, "Moran's I of log incidence"), 2),
        "denguePppDispPois" => fmt(ppp(ppc_po, "dispersion index Var/Mean"), 2),
        "denguePppLowPois" => fmt(ppp(ppc_po, "n municipalities < 100/100k"), 2),
        "dengueMoranObs" => fmt(ppp_row(ppc_nb, "Moran's I of log incidence", :T_obs), 2),
        "dengueMoranRepNB" => fmt(ppp_row(ppc_nb, "Moran's I of log incidence", :q50), 2),
        "denguePoisMaxLo" => fmtint(ppp_row(ppc_po, "maximum count", :q025)),
        "denguePoisMaxHi" => fmtint(ppp_row(ppc_po, "maximum count", :q975)),
        "dengueMaxObs" => fmtint(maximum(Y)),
        "denguePwaicNB" => haskey(sc, BEST) ? fmt(sc[BEST].p_waic, 0) : "--",
        "denguePwaicPois" => haskey(sc, POIS) ? fmt(sc[POIS].p_waic, 0) : "--",
        "denguePoisAlpha" => haskey(chains, POIS) ? fmt(mean(chains[POIS].α_ddcrp), 0) : "--",
        "denguePoisSingletons" => haskey(chains, POIS) ? string(round(Int, 100 * pois_singleton_share())) : "--",
    ]
    lines = ["% Generated by src/analyse_dengue.jl; re-run it to update.",
             "% Numbers quoted in the dengue section of the paper."]
    for (k, v) in macros
        push!(lines, "\\newcommand{\\$k}{$v}")
    end
    open(io -> println(io, join(lines, "\n")), joinpath(OUTDIR, "dengue_numbers.tex"), "w")
end

# ============================================================================
# Copy the paper assets
# ============================================================================

if IS_TEST
    println("\n--test: paper assets left in $OUTDIR (docs/paper/ untouched).")
else
    for f in ("dengue_k_traces.png", "dengue_ppc.png")
        cp(joinpath(FIGDIR, f), joinpath(PAPER, "figures", f); force = true)
    end
    for f in ("table_dengue.tex", "dengue_numbers.tex")
        cp(joinpath(OUTDIR, f), joinpath(PAPER, "tables", f); force = true)
    end
    println("\nPaper assets copied to $PAPER/{figures,tables}/")
end

println("\nDone. Run `Rscript src/map_dengue.R$(IS_TEST ? " --test" : "")` for the maps.")

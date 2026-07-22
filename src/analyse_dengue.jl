# ============================================================================
# analyse_dengue.jl
#
# Figures, tables and model comparison for the Brazil dengue application.
# Run AFTER `src/run_dengue.jl`; it reads the chains that script wrote and
# warns-and-skips anything missing.
#
# Answers the three questions the application has to answer:
#   1. Does the ddCRP find MULTIPLE clusters?      → posterior of K, maps
#   2. Do they MIX?                                → ESS, traces, ESJD
#   3. Is the RJMCMC DOING ITS JOB?                → birth/death acceptance,
#                                                    agreement across proposal
#                                                    families and with the
#                                                    conjugate Gibbs anchor
# plus: is the non-conjugate NB actually NEEDED?   → WAIC / LPML / PSIS-LOO
#       are the clusters real structure?           → posterior predictive checks
#
# Also writes `cluster_assignments.csv` and `similarity_matrix.jld2`, which
# `src/map_dengue.R` turns into maps.
#
# Usage:
#   julia --project=. src/analyse_dengue.jl
#   julia --project=. src/analyse_dengue.jl --test
# ============================================================================

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using DistanceDependentCRP, StatsPlots, DataFrames, CSV, Statistics, StatsBase,
      JLD2, Printf, Distributions, Random

# Brings the NegBinPopulationRates* types into scope so JLD2 deserialises the
# saved chains as their real types (needed for samples.ρ / samples.r access).
include(joinpath(@__DIR__, "negbin_population_model.jl"))
include(joinpath(@__DIR__, "model_comparison.jl"))

ENV["GKSwstype"] = "nul"   # headless GR

const IS_TEST = "--test" in ARGS
const OUTDIR = IS_TEST ?
    joinpath(@__DIR__, "..", "results", "real_data", "dengue_test") :
    joinpath(@__DIR__, "..", "results", "real_data", "dengue")
const FIGDIR = joinpath(OUTDIR, "figures"); mkpath(FIGDIR)

latex_fmt(x::Real, d::Int) = Printf.format(Printf.Format("%.$(d)f"), Float64(x))

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
const BURN = meta["burn"]; const THIN = meta["thin"]
const DATA = CountDataWithPopulation(Y, E, D)
const N = length(Y)

const FINAL_CSV = joinpath(OUTDIR, "summary_final.csv")
isfile(FINAL_CSV) || error("Missing $FINAL_CSV")
final_df = DataFrame(CSV.File(FINAL_CSV))

chain_dir = joinpath(OUTDIR, "chains")
chains = Dict{String,Any}()
for (label, fname) in zip(meta["labels"], meta["chain_files"])
    path = joinpath(chain_dir, fname)
    if isfile(path)
        d = load(path)
        chains[label] = (samples = d["samples"], burn = d["burn"], thin = d["thin"])
    else
        @warn "Chain file not found — skipping" label path
    end
end
isempty(chains) && error("No chain files found in $chain_dir")

labels = [l for l in meta["labels"] if haskey(chains, l)]
keep_idx(ch) = (ch.burn + 1):ch.thin:size(ch.samples.c, 1)
post_c(ch) = ch.samples.c[keep_idx(ch), :]

println("Loaded $(length(labels)) chains over $N municipalities.\n")

# ============================================================================
# Figure 1 — the data itself
# ============================================================================

println("Generating data overview...")
let
    inc = Y ./ E
    p1 = scatter(LON, LAT; marker_z = log10.(inc), c = :viridis, ms = 4,
                 msw = 0, colorbar_title = "log₁₀ incidence /100k",
                 xlabel = "longitude", ylabel = "latitude",
                 title = "2024 dengue incidence by municipality", legend = false)
    p2 = histogram(log10.(inc); bins = 40, xlabel = "log₁₀ incidence per 100k",
                   ylabel = "municipalities", legend = false,
                   title = "Incidence is multi-modal")
    p3 = scatter(E .* 1e5, inc; xscale = :log10, yscale = :log10, ms = 3, msw = 0,
                 xlabel = "population", ylabel = "incidence per 100k", legend = false,
                 title = "Incidence vs population")
    plot(p1, p2, p3; layout = (1, 3), size = (1500, 430), left_margin = 6Plots.mm,
         bottom_margin = 6Plots.mm)
    savefig(joinpath(FIGDIR, "data_overview.png"))
end

# ============================================================================
# Figure 2 — posterior of K: are there multiple clusters?
# ============================================================================

println("Generating P(K) distribution...")
let
    plt = plot(; xlabel = "number of clusters K", ylabel = "posterior probability",
                 title = "Posterior of K", size = (900, 500), legend = :topright)
    for l in labels
        kt = calculate_n_clusters(post_c(chains[l]))
        cm = countmap(kt)
        ks = sort(collect(keys(cm)))
        plot!(plt, ks, [cm[k] / length(kt) for k in ks]; lw = 2, marker = :circle,
              ms = 3, label = l)
    end
    savefig(plt, joinpath(FIGDIR, "pk_distribution.png"))
end

# ============================================================================
# Figure 3 — traces: does it mix?
# ============================================================================

println("Generating trace plots...")
let
    ps = []
    for l in labels
        ch = chains[l]
        kt = calculate_n_clusters(post_c(ch))
        push!(ps, plot(kt; lw = 0.6, legend = false, ylabel = "K", title = l,
                       titlefontsize = 8))
    end
    ch1 = chains[labels[1]]
    idx = keep_idx(ch1)
    push!(ps, plot(ch1.samples.α_ddcrp[idx]; lw = 0.6, legend = false,
                   ylabel = "α_ddcrp", title = "α_ddcrp ($(labels[1]))", titlefontsize = 8))
    push!(ps, plot(ch1.samples.logpost[idx]; lw = 0.6, legend = false,
                   ylabel = "log posterior", title = "log posterior", titlefontsize = 8))
    plot(ps...; layout = (length(ps), 1), size = (1000, 220 * length(ps)),
         left_margin = 8Plots.mm)
    savefig(joinpath(FIGDIR, "k_alpha_traces.png"))
end

# ============================================================================
# Cross-variant agreement — the correctness check, there being no ground truth
# ============================================================================

"""
    dahl_point_estimate(c_samples, sim) -> Vector{Int}

Least-squares (Dahl 2006) point estimate: the sampled partition whose own
co-clustering matrix is closest to the posterior similarity matrix. Preferred
over `:MAP` here because with n in the hundreds every draw is a distinct
configuration, so the modal configuration is not meaningful.
"""
function dahl_point_estimate(c_samples::Matrix{Int}, sim::Matrix{Float64};
                             max_draws::Int = 2000)
    S = size(c_samples, 1)
    # The search is O(draws x n^2); capping the candidate set keeps it seconds
    # rather than minutes on a full-length chain, with no practical loss since
    # any of thousands of draws is an adequate candidate partition.
    cand = S <= max_draws ? (1:S) : round.(Int, range(1, S; length = max_draws))
    best, best_loss = first(cand), Inf
    for s in cand
        z = compute_table_assignments(c_samples[s, :])
        loss = 0.0
        @inbounds for i in 1:length(z), j in (i+1):length(z)
            loss += ((z[i] == z[j] ? 1.0 : 0.0) - sim[i, j])^2
        end
        loss < best_loss && (best_loss = loss; best = s)
    end
    return compute_table_assignments(c_samples[best, :])
end

println("Computing point estimates and cross-variant agreement...")
sims = Dict(l => compute_similarity_matrix(post_c(chains[l])) for l in labels)
points = Dict(l => dahl_point_estimate(post_c(chains[l]), sims[l]) for l in labels)

# ARI between every pair of point estimates. High agreement between independent
# proposal families (and with the conjugate Gibbs anchor) is what stands in for
# a ground-truth check on real data.
function ari(a::Vector{Int}, b::Vector{Int})
    n = length(a)
    tab = Dict{Tuple{Int,Int},Int}()
    for i in 1:n
        tab[(a[i], b[i])] = get(tab, (a[i], b[i]), 0) + 1
    end
    ai = countmap(a); bj = countmap(b)
    c2(x) = x * (x - 1) / 2
    sij = sum(c2(v) for v in values(tab))
    sa = sum(c2(v) for v in values(ai))
    sb = sum(c2(v) for v in values(bj))
    exp_idx = sa * sb / c2(n)
    max_idx = (sa + sb) / 2
    return (sij - exp_idx) / (max_idx - exp_idx)
end

ari_mat = [ari(points[a], points[b]) for a in labels, b in labels]
println("\nPairwise ARI between point-estimate clusterings:")
println(rpad("", 42) * join([rpad(string(i), 8) for i in 1:length(labels)]))
for (i, l) in enumerate(labels)
    @printf("%2d %-39s%s\n", i, first(l, 38),
            join([rpad(latex_fmt(ari_mat[i, j], 3), 8) for j in 1:length(labels)]))
end
CSV.write(joinpath(OUTDIR, "cross_variant_ari.csv"),
          DataFrame([Symbol("v$j") => ari_mat[:, j] for j in 1:length(labels)]...;
                    makeunique = true))

let
    heatmap(1:length(labels), 1:length(labels), ari_mat; c = :viridis, clims = (0, 1),
            xlabel = "variant", ylabel = "variant", title = "Agreement (ARI) between variants",
            size = (700, 600))
    savefig(joinpath(FIGDIR, "cross_variant_ari.png"))
end

# ============================================================================
# Mixing / acceptance table
# ============================================================================

println("\nMixing and acceptance summary:")
@printf("%-42s %7s %7s %8s %8s %8s %9s %9s\n",
        "variant", "mean K", "ESS(K)", "acc.B%", "acc.D%", "acc.F%", "ESS_ρ", "ESS_r")
println("-"^105)
for r in eachrow(final_df)
    @printf("%-42s %7.2f %7.0f %8.2f %8.2f %8.2f %9.0f %9.0f\n",
            first(r.label, 41), r.mean_K, r.ess_K,
            r.acc_birth*100, r.acc_death*100, r.acc_fixed*100,
            r.med_ess_ρ, r.med_ess_r)
end

let
    lines = String[]
    push!(lines, "% Required packages: booktabs")
    push!(lines, "% Generated automatically — re-run analyse_dengue.jl to update")
    push!(lines, "\\begin{tabular}{l r r r r r r}")
    push!(lines, "  \\toprule")
    push!(lines, "  Sampler & Mean \$K\$ & ESS\$(K)\$ & Birth (\\%) & Death (\\%) & " *
                 "Fixed (\\%) & ESS/s \\\\")
    push!(lines, "  \\midrule")
    for r in eachrow(final_df)
        push!(lines, "  " * replace(r.label, "_" => "\\_") * " & " *
              latex_fmt(r.mean_K, 2) * " & " * string(round(Int, r.ess_K)) * " & " *
              latex_fmt(r.acc_birth * 100, 2) * " & " * latex_fmt(r.acc_death * 100, 2) * " & " *
              latex_fmt(r.acc_fixed * 100, 2) * " & " * latex_fmt(r.ess_per_sec, 3) * " \\\\")
    end
    push!(lines, "  \\bottomrule"); push!(lines, "\\end{tabular}")
    open(io -> println(io, join(lines, "\n")), joinpath(OUTDIR, "table_dengue_mixing.tex"), "w")
    println("\n  LaTeX table → $(joinpath(OUTDIR, "table_dengue_mixing.tex"))")
end

# ============================================================================
# Model comparison — is the non-conjugate NB actually needed?
# ============================================================================

println("\nModel comparison (WAIC / LPML / PSIS-LOO):")
scores = []
for l in labels
    ch = chains[l]
    s = ch.samples
    idx = keep_idx(ch)
    if s isa NegBinPopulationRatesSamples
        thin = NegBinPopulationRatesSamples(s.c[idx, :], s.ρ[idx, :], s.r[idx, :],
                                            s.logpost[idx], s.α_ddcrp[idx], s.s_ddcrp[idx])
        push!(scores, score_fit(l, loglik_matrix(NegBinPopulationRates(), thin, DATA)))
    elseif s isa PoissonPopulationRatesSamples
        thin = PoissonPopulationRatesSamples(s.c[idx, :], s.ρ[idx, :], s.logpost[idx],
                                             s.α_ddcrp[idx], s.s_ddcrp[idx])
        push!(scores, score_fit(l, loglik_matrix(PoissonPopulationRates(), thin, DATA)))
    else
        # PoissonPopulationRatesMarg stores no per-observation rate (λ is
        # integrated out), so it cannot be scored here. It serves as the
        # clustering benchmark in the ARI table above instead.
        println("  (skipping $l — marginalised model, no pointwise likelihood)")
    end
end
if !isempty(scores)
    print_comparison(scores)
    CSV.write(joinpath(OUTDIR, "model_comparison.csv"), DataFrame(scores))
end

# ============================================================================
# Posterior predictive checks — are the clusters reproducing the data?
# ============================================================================

println("\nPosterior predictive checks...")
let
    nb_labels = [l for l in labels if chains[l].samples isa NegBinPopulationRatesSamples]
    pois_labels = [l for l in labels if chains[l].samples isa PoissonPopulationRatesSamples]
    isempty(nb_labels) && @warn "no Negative Binomial chain — skipping PPC"

    # Discrepancy statistics chosen to probe what a Poisson model gets wrong:
    # the spread of the counts, the extreme upper tail, and how many
    # municipalities sit at very low incidence.
    stats = Dict(
        "dispersion index Var/Mean" => v -> var(v) / mean(v),
        "maximum count"             => v -> float(maximum(v)),
        "n municipalities < 100/100k" => v -> float(count(i -> v[i] / E[i] < 100, eachindex(v))),
        "90th pct of incidence"     => v -> quantile(v ./ E, 0.9),
    )
    stat_names = collect(keys(stats))

    ps = []
    for l in vcat(isempty(nb_labels) ? String[] : [nb_labels[1]],
                  isempty(pois_labels) ? String[] : [pois_labels[1]])
        ch = chains[l]; s = ch.samples; idx = keep_idx(ch)
        if s isa NegBinPopulationRatesSamples
            thin = NegBinPopulationRatesSamples(s.c[idx, :], s.ρ[idx, :], s.r[idx, :],
                                                s.logpost[idx], s.α_ddcrp[idx], s.s_ddcrp[idx])
            model = NegBinPopulationRates()
        else
            thin = PoissonPopulationRatesSamples(s.c[idx, :], s.ρ[idx, :], s.logpost[idx],
                                                 s.α_ddcrp[idx], s.s_ddcrp[idx])
            model = PoissonPopulationRates()
        end
        Random.seed!(11)
        y_rep = posterior_predictive(model, thin, DATA; n_rep = 1000)

        println("  $l")
        for sname in stat_names
            f = stats[sname]
            T_obs = f(Y)
            T_rep = [f(view(y_rep, s, :)) for s in axes(y_rep, 1)]
            p = ppc_pvalue(T_obs, T_rep)
            @printf("    %-30s observed %12.2f   ppp = %.3f%s\n",
                    sname, T_obs, p, (p < 0.05 || p > 0.95) ? "  ← model misfit" : "")
            push!(ps, histogram(T_rep; bins = 40, legend = false, title = "$sname\n$l",
                                titlefontsize = 7, xlabel = "replicate"))
            vline!(ps[end], [T_obs]; lw = 2, c = :red)
        end
    end
    if !isempty(ps)
        plot(ps...; layout = (length(ps) ÷ length(stat_names), length(stat_names)),
             size = (400 * length(stat_names), 320 * (length(ps) ÷ length(stat_names))))
        savefig(joinpath(FIGDIR, "ppc.png"))
    end
end

# ============================================================================
# Outputs for the R/sf maps
# ============================================================================

println("\nWriting map inputs...")
let
    # The best-mixing Negative Binomial variant drives the maps.
    nb_rows = filter(r -> startswith(r.family, "negbin"), final_df)
    best_label = isempty(nb_rows) ? labels[1] : nb_rows[argmax(nb_rows.ess_K), :].label
    haskey(chains, best_label) || (best_label = labels[1])
    println("  mapping variant: $best_label")

    ch = chains[best_label]; s = ch.samples; idx = keep_idx(ch)
    z = points[best_label]
    sim = sims[best_label]

    # Per-municipality posterior means, plus how confidently each municipality
    # is placed: the mean co-clustering probability with its own cluster.
    cocluster_conf = [mean(sim[i, findall(==(z[i]), z)]) for i in 1:N]

    df = DataFrame(
        geocode = GEOCODE, name = NAME, lat = LAT, lon = LON,
        y = Y, pop = E .* 1e5, incidence = Y ./ E,
        cluster = z,
        cocluster_confidence = cocluster_conf,
        rho_mean = s isa NegBinPopulationRatesSamples ? vec(mean(s.ρ[idx, :], dims = 1)) : fill(NaN, N),
        r_mean = s isa NegBinPopulationRatesSamples ? vec(mean(s.r[idx, :], dims = 1)) : fill(NaN, N),
    )
    CSV.write(joinpath(OUTDIR, "cluster_assignments.csv"), df)
    jldsave(joinpath(OUTDIR, "similarity_matrix.jld2");
            sim = sim, label = best_label, geocode = GEOCODE)

    @printf("  point estimate has %d clusters; sizes: %s\n",
            length(unique(z)), string(sort(collect(values(countmap(z))), rev = true)))
    println("  → $(joinpath(OUTDIR, "cluster_assignments.csv"))")
    println("  → $(joinpath(OUTDIR, "similarity_matrix.jld2"))")
end

println("\nDone. Figures in $FIGDIR")
println("Run `Rscript src/map_dengue.R` for the spatial maps.")

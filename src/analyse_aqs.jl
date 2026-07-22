# ============================================================================
# analyse_aqs.jl
#
# Figures, tables and model comparison for the EPA AQS PM2.5 application.
# Run AFTER `src/run_aqs.jl`.
#
# There is no Gibbs baseline and no ground truth here — the SkewNormal model is
# non-conjugate in all three parameters — so correctness rests on mutual
# agreement between independent proposal families, exactly as in the SkewNormal
# simulation study.
#
# Also writes `cluster_assignments.csv` and `similarity_matrix.jld2` for
# `src/map_aqs.R`.
#
# Usage:
#   julia --project=. src/analyse_aqs.jl
#   julia --project=. src/analyse_aqs.jl --test
# ============================================================================

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using DistanceDependentCRP, StatsPlots, DataFrames, CSV, Statistics, StatsBase,
      JLD2, Printf, Distributions, Random

# Brings SkewNormalClusters* into scope so JLD2 deserialises the chains as their
# real types (needed for samples.ξ/.ω/.α access).
include(joinpath(@__DIR__, "skewnormal_model.jl"))
include(joinpath(@__DIR__, "model_comparison.jl"))

ENV["GKSwstype"] = "nul"   # headless GR

const IS_TEST = "--test" in ARGS
const OUTDIR = IS_TEST ?
    joinpath(@__DIR__, "..", "results", "real_data", "aqs_test") :
    joinpath(@__DIR__, "..", "results", "real_data", "aqs")
const FIGDIR = joinpath(OUTDIR, "figures"); mkpath(FIGDIR)

latex_fmt(x::Real, d::Int) = Printf.format(Printf.Format("%.$(d)f"), Float64(x))

# ============================================================================
# Load run outputs
# ============================================================================

const META_FILE = joinpath(OUTDIR, "run_metadata.jld2")
isfile(META_FILE) || error("""
    Missing $META_FILE
    Run: julia --project=. src/run_aqs.jl$(IS_TEST ? " --test" : "")""")

meta = load(META_FILE)
const Y = meta["y"]; const D = meta["D"]
const LAT = meta["lat"]; const LON = meta["lon"]
const SITE = meta["site_id"]; const STATE = meta["state"]
const DATA = ContinuousData(Y, D)
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
thin_samples(ch) = (s = ch.samples; i = keep_idx(ch);
    SkewNormalClustersSamples(s.c[i, :], s.ξ[i, :], s.ω[i, :], s.α[i, :],
                              s.logpost[i], s.α_ddcrp[i], s.s_ddcrp[i]))

println("Loaded $(length(labels)) chains over $N monitoring sites.\n")

# ============================================================================
# Figure 1 — the data
# ============================================================================

println("Generating data overview...")
let
    p1 = scatter(LON, LAT; marker_z = Y, c = :inferno, ms = 4, msw = 0,
                 colorbar_title = "98th pct PM2.5 (µg/m³)",
                 xlabel = "longitude", ylabel = "latitude", legend = false,
                 title = "2023 PM2.5 design-value metric by monitor")
    m, s = mean(Y), std(Y)
    g = mean(((x - m) / s)^3 for x in Y)
    p2 = histogram(Y; bins = 40, xlabel = "98th pct daily PM2.5 (µg/m³)",
                   ylabel = "monitors", legend = false,
                   title = @sprintf("Right-skewed across sites (γ₁ = %.2f)", g))
    plot(p1, p2; layout = (1, 2), size = (1300, 470), left_margin = 6Plots.mm,
         bottom_margin = 6Plots.mm)
    savefig(joinpath(FIGDIR, "data_overview.png"))
end

# ============================================================================
# Figure 2 — posterior of K
# ============================================================================

println("Generating P(K) distribution...")
let
    plt = plot(; xlabel = "number of clusters K", ylabel = "posterior probability",
                 title = "Posterior of K", size = (900, 500), legend = :topright)
    for l in labels
        kt = calculate_n_clusters(post_c(chains[l]))
        cm = countmap(kt); ks = sort(collect(keys(cm)))
        plot!(plt, ks, [cm[k] / length(kt) for k in ks]; lw = 2, marker = :circle,
              ms = 3, label = l)
    end
    savefig(plt, joinpath(FIGDIR, "pk_distribution.png"))
end

# ============================================================================
# Figure 3 — traces
# ============================================================================

println("Generating trace plots...")
let
    ps = []
    for l in labels
        kt = calculate_n_clusters(post_c(chains[l]))
        push!(ps, plot(kt; lw = 0.6, legend = false, ylabel = "K", title = l,
                       titlefontsize = 8))
    end
    ch1 = chains[labels[1]]; idx = keep_idx(ch1)
    push!(ps, plot(ch1.samples.α_ddcrp[idx]; lw = 0.6, legend = false,
                   ylabel = "α_ddcrp", title = "α_ddcrp ($(labels[1]))", titlefontsize = 8))
    push!(ps, plot(ch1.samples.logpost[idx]; lw = 0.6, legend = false,
                   ylabel = "log posterior", title = "log posterior", titlefontsize = 8))
    plot(ps...; layout = (length(ps), 1), size = (1000, 220 * length(ps)),
         left_margin = 8Plots.mm)
    savefig(joinpath(FIGDIR, "k_alpha_traces.png"))
end

# ============================================================================
# Cross-variant agreement
# ============================================================================

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

function ari(a::Vector{Int}, b::Vector{Int})
    n = length(a)
    tab = Dict{Tuple{Int,Int},Int}()
    for i in 1:n
        tab[(a[i], b[i])] = get(tab, (a[i], b[i]), 0) + 1
    end
    c2(x) = x * (x - 1) / 2
    sij = sum(c2(v) for v in values(tab))
    sa = sum(c2(v) for v in values(countmap(a)))
    sb = sum(c2(v) for v in values(countmap(b)))
    exp_idx = sa * sb / c2(n); max_idx = (sa + sb) / 2
    return (sij - exp_idx) / (max_idx - exp_idx)
end

println("Computing point estimates and cross-variant agreement...")
sims = Dict(l => compute_similarity_matrix(post_c(chains[l])) for l in labels)
points = Dict(l => dahl_point_estimate(post_c(chains[l]), sims[l]) for l in labels)

ari_mat = [ari(points[a], points[b]) for a in labels, b in labels]
println("\nPairwise ARI between point-estimate clusterings:")
println(rpad("", 46) * join([rpad(string(i), 8) for i in 1:length(labels)]))
for (i, l) in enumerate(labels)
    @printf("%2d %-43s%s\n", i, first(l, 42),
            join([rpad(latex_fmt(ari_mat[i, j], 3), 8) for j in 1:length(labels)]))
end
println("\nWith no ground truth, high off-diagonal ARI between independently")
println("proposed samplers is the correctness evidence. Low values mean at least")
println("one chain has not converged.")

let
    heatmap(1:length(labels), 1:length(labels), ari_mat; c = :viridis, clims = (0, 1),
            xlabel = "variant", ylabel = "variant",
            title = "Agreement (ARI) between variants", size = (700, 600))
    savefig(joinpath(FIGDIR, "cross_variant_ari.png"))
end

# ============================================================================
# Mixing table
# ============================================================================

println("\nMixing and acceptance summary:")
@printf("%-44s %7s %7s %8s %8s %8s %8s\n",
        "variant", "mean K", "ESS(K)", "acc.B%", "acc.D%", "ESS_ω", "ESS_α")
println("-"^100)
for r in eachrow(final_df)
    @printf("%-44s %7.2f %7.0f %8.2f %8.2f %8.0f %8.0f\n",
            first(r.label, 43), r.mean_K, r.ess_K,
            r.acc_birth*100, r.acc_death*100, r.med_ess_ω, r.med_ess_α)
end

let
    lines = String[]
    push!(lines, "% Required packages: booktabs")
    push!(lines, "% Generated automatically — re-run analyse_aqs.jl to update")
    push!(lines, "\\begin{tabular}{l r r r r r r}")
    push!(lines, "  \\toprule")
    push!(lines, "  Sampler & Mean \$K\$ & ESS\$(K)\$ & Birth (\\%) & Death (\\%) & " *
                 "ESS\$_\\omega\$ & ESS\$_\\alpha\$ \\\\")
    push!(lines, "  \\midrule")
    for r in eachrow(final_df)
        push!(lines, "  " * replace(r.label, "_" => "\\_") * " & " *
              latex_fmt(r.mean_K, 2) * " & " * string(round(Int, r.ess_K)) * " & " *
              latex_fmt(r.acc_birth * 100, 2) * " & " * latex_fmt(r.acc_death * 100, 2) * " & " *
              string(round(Int, r.med_ess_ω)) * " & " * string(round(Int, r.med_ess_α)) * " \\\\")
    end
    push!(lines, "  \\bottomrule"); push!(lines, "\\end{tabular}")
    open(io -> println(io, join(lines, "\n")), joinpath(OUTDIR, "table_aqs_mixing.tex"), "w")
    println("\n  LaTeX table → $(joinpath(OUTDIR, "table_aqs_mixing.tex"))")
end

# ============================================================================
# Model comparison — is the skew parameter earning its place?
# ============================================================================

println("\nModel comparison (WAIC / LPML / PSIS-LOO):")
scores = [score_fit(l, loglik_matrix(SkewNormalClusters(), thin_samples(chains[l]), DATA))
          for l in labels]
print_comparison(scores)
CSV.write(joinpath(OUTDIR, "model_comparison.csv"), DataFrame(scores))
println("\nA symmetric fit (α pinned at 0) scoring clearly worse is the evidence")
println("that the skewness — and hence the non-conjugacy — is real.")

# ============================================================================
# Posterior predictive checks
# ============================================================================

println("\nPosterior predictive checks...")
let
    stats = Dict(
        "skewness"      => v -> (m = mean(v); s = std(v); mean(((x - m) / s)^3 for x in v)),
        "95th pct"      => v -> quantile(v, 0.95),
        "5th pct"       => v -> quantile(v, 0.05),
        "maximum"       => v -> maximum(v),
    )
    stat_names = collect(keys(stats))
    show_labels = labels[1:min(2, length(labels))]

    ps = []
    for l in show_labels
        Random.seed!(11)
        y_rep = posterior_predictive(SkewNormalClusters(), thin_samples(chains[l]), DATA;
                                     n_rep = 1000)
        println("  $l")
        for sname in stat_names
            f = stats[sname]
            T_obs = f(Y)
            T_rep = [f(vec(view(y_rep, s, :))) for s in axes(y_rep, 1)]
            p = ppc_pvalue(T_obs, T_rep)
            @printf("    %-12s observed %10.3f   ppp = %.3f%s\n",
                    sname, T_obs, p, (p < 0.05 || p > 0.95) ? "  ← model misfit" : "")
            push!(ps, histogram(T_rep; bins = 40, legend = false,
                                title = "$sname\n$(first(l, 30))", titlefontsize = 7,
                                xlabel = "replicate"))
            vline!(ps[end], [T_obs]; lw = 2, c = :red)
        end
    end
    plot(ps...; layout = (length(show_labels), length(stat_names)),
         size = (380 * length(stat_names), 320 * length(show_labels)))
    savefig(joinpath(FIGDIR, "ppc.png"))
end

# ============================================================================
# Outputs for the R/sf maps
# ============================================================================

println("\nWriting map inputs...")
let
    full = filter(r -> r.infer_skew === true, final_df)
    best_label = isempty(full) ? labels[1] : full[argmax(full.ess_K), :].label
    haskey(chains, best_label) || (best_label = labels[1])
    println("  mapping variant: $best_label")

    ch = chains[best_label]; ts = thin_samples(ch)
    z = points[best_label]; sim = sims[best_label]
    cocluster_conf = [mean(sim[i, findall(==(z[i]), z)]) for i in 1:N]

    df = DataFrame(
        site_id = SITE, state = STATE, lat = LAT, lon = LON, y = Y,
        cluster = z, cocluster_confidence = cocluster_conf,
        xi_mean = vec(mean(ts.ξ, dims = 1)),
        omega_mean = vec(mean(ts.ω, dims = 1)),
        alpha_mean = vec(mean(ts.α, dims = 1)),
    )
    CSV.write(joinpath(OUTDIR, "cluster_assignments.csv"), df)
    jldsave(joinpath(OUTDIR, "similarity_matrix.jld2");
            sim = sim, label = best_label, site_id = SITE)

    @printf("  point estimate has %d clusters; sizes: %s\n",
            length(unique(z)), string(sort(collect(values(countmap(z))), rev = true)))
    println("  → $(joinpath(OUTDIR, "cluster_assignments.csv"))")
end

println("\nDone. Figures in $FIGDIR")
println("Run `Rscript src/map_aqs.R` for the spatial maps.")

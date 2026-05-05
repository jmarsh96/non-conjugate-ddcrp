using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
ENV["GKSwstype"] = "nul"   # use GR's null backend; avoids Qt QPainterPath warnings
using DistanceDependentCRP, StatsPlots, DataFrames, CSV, Statistics, StatsBase, JLD2, Printf, Distributions

const OUTDIR = joinpath(@__DIR__, "..", "results", "old_faithful")
const FIGDIR = joinpath(OUTDIR, "figures")
mkpath(OUTDIR)
mkpath(FIGDIR)

# ── Load data ─────────────────────────────────────────────────────────────────

println("Loading Old Faithful data...")
faithful = DataFrame(CSV.File(joinpath(@__DIR__, "..", "data", "faithful.csv")))
y = Float64.(faithful.eruptions)
w = Float64.(faithful.waiting)
n = length(y)
println("  n = $n observations")

# ── Load MCMC samples ─────────────────────────────────────────────────────────

println("Loading MCMC samples...")
chain_data = load(joinpath(OUTDIR, "mcmc_samples.jld2"))
samples = chain_data["samples"]
n_samp = size(samples.c, 1)
println("  $n_samp samples")

# ── MCMC diagnostics ─────────────────────────────────────────────────────────

if haskey(chain_data, "diagnostics")
    diag = chain_data["diagnostics"]
    println("\nMCMC Diagnostics")
    println("=" ^ 40)
    rates = acceptance_rates(diag)
    @printf("Acceptance rates:\n")
    @printf("  Birth:   %5.1f%%  (%d / %d)\n", rates.birth * 100, diag.birth_accepts, diag.birth_proposes)
    @printf("  Death:   %5.1f%%  (%d / %d)\n", rates.death * 100, diag.death_accepts, diag.death_proposes)
    @printf("  Fixed:   %5.1f%%  (%d / %d)\n", rates.fixed * 100, diag.fixed_accepts, diag.fixed_proposes)
    @printf("  Overall: %5.1f%%\n", rates.overall * 100)
else
    println("\n  (No diagnostics found — re-run run_old_faithful.jl to save them)")
end

k_trace_diag = calculate_n_clusters(samples.c)
ess_lp = effective_sample_size(samples.logpost)
ess_k = effective_sample_size(Float64.(k_trace_diag))
println("\nEffective Sample Size (post-thinning):")
@printf("  Log-posterior: %.1f\n", ess_lp)
@printf("  K:             %.1f\n", ess_k)
println()

# ── ESS for shape parameters ──────────────────────────────────────────────────

println("Computing per-observation α ESS...")
ess_alpha_vec = [effective_sample_size(samples.α[:, i]) for i in 1:n]
ess_alpha_min = minimum(ess_alpha_vec)
ess_alpha_max = maximum(ess_alpha_vec)
ess_alpha_median = median(ess_alpha_vec)
@printf("  α ESS: min=%.1f  median=%.1f  max=%.1f\n",
        ess_alpha_min, ess_alpha_median, ess_alpha_max)

ess_df = DataFrame(
    metric = ["ess_logpost", "ess_K", "ess_alpha_min", "ess_alpha_max", "ess_alpha_median"],
    value = [ess_lp, ess_k, ess_alpha_min, ess_alpha_max, ess_alpha_median],
)
CSV.write(joinpath(OUTDIR, "ess_summary.csv"), ess_df)
println("  Saved ess_summary.csv")

# ── Figure 1: K posterior bar chart ──────────────────────────────────────────

println("\nGenerating K posterior figure...")
k_trace = calculate_n_clusters(samples.c)
k_vals = sort(unique(k_trace))
k_freq = [count(==(k), k_trace) / n_samp for k in k_vals]

fig1 = bar(k_vals, k_freq;
    xlabel = "K",
    ylabel = "Posterior probability",
    legend = false,
    color = :steelblue,
    bar_width = 0.6,
    tickfontsize = 10,
    guidefontsize = 12,
    left_margin = 8Plots.mm,
    right_margin = 5Plots.mm,
    top_margin = 5Plots.mm,
    bottom_margin = 8Plots.mm,
    size = (508, 320),
)
savefig(fig1, joinpath(FIGDIR, "k_posterior.png"))
println("  Saved k_posterior.png")

# ── Figure 2: Trace plots (K + log-posterior, 1×2) ────────────────────────────

println("\nGenerating trace plots figure...")
common_trace_kw = (
    legend = false,
    color = :steelblue,
    linewidth = 0.8,
    tickfontsize = 10,
    guidefontsize = 12,
    left_margin = 8Plots.mm,
    right_margin = 5Plots.mm,
    top_margin = 5Plots.mm,
    bottom_margin = 8Plots.mm,
)

p_k_trace = plot(k_trace; xlabel="Iteration", ylabel="Number of clusters",
                 common_trace_kw...)
p_lp_trace = plot(samples.logpost; xlabel="Iteration", ylabel="Log-posterior",
                  common_trace_kw...)

fig2 = plot(p_k_trace, p_lp_trace; layout=(1, 2), size=(900, 320))  # width ∝ 0.62
savefig(fig2, joinpath(FIGDIR, "traces.png"))
println("  Saved traces.png")

# ── Figure 3: MAP Clustering + Posterior Probability Arrows ──────────────────

println("\nGenerating MAP clustering + arrows figure...")

# Restrict to posterior mode K
k_mode = mode(k_trace)
k3_idx = findall(k_trace .== k_mode)
map_idx = k3_idx[argmax(samples.logpost[k3_idx])]
c_map = samples.c[map_idx, :]
z_map = compute_table_assignments(c_map)
clusters = sort(unique(z_map))

p_map = plot(
    xlabel = "Waiting time (min)",
    ylabel = "Eruption duration (min)",
    title = "MAP clustering with posterior link probabilities",
    legend = :topleft,
    tickfontsize = 10,
    guidefontsize = 12,
    titlefontsize = 12,
    left_margin = 8Plots.mm,
    right_margin = 20Plots.mm,
    bottom_margin = 5Plots.mm,
)

# Compute posterior link probabilities (conditioning on posterior mode K)
println("  Computing link probabilities (K=$k_mode samples)...")
link_prob = zeros(n, n)
for iter in k3_idx
    for i in 1:n
        j = samples.c[iter, i]
        link_prob[i, j] += 1
    end
end
link_prob ./= length(k3_idx)

# Draw arrows first (underneath scatter), opacity driven by posterior probability
threshold = 0.02
p_max = maximum(link_prob[i, j] for i in 1:n, j in 1:n if i != j)
arrow_cmap = cgrad(:YlOrRd, [0.0, 1.0])

for i in 1:n
    # Always show the strongest outgoing link; additionally show any above threshold
    row = [j == i ? -Inf : link_prob[i, j] for j in 1:n]
    top_js = Set(partialsortperm(row, 1:3, rev=true))
    for j in 1:n
        i == j && continue
        p = link_prob[i, j]
        p < threshold && j ∉ top_js && continue
        quiver!(p_map, [w[i]], [y[i]];
                quiver = ([w[j] - w[i]], [y[j] - y[i]]),
                color = get(arrow_cmap, 0.3 + 0.7 * p / p_max),
                linewidth = clamp(p / p_max * 2.5, 0.6, 2.5),
                label = false)
    end
end

# Empty scatter to register the colorbar (no points drawn, no arcs)
scatter!(p_map, Float64[], Float64[];
         zcolor=Float64[], color=arrow_cmap, colorbar=true,
         colorbar_title="\nPosterior link prob.", clims=(0.0, p_max),
         label=false)

# Scatter colored by MAP cluster
for (ci, k) in enumerate(clusters)
    mask = z_map .== k
    scatter!(p_map, w[mask], y[mask];
             label="Cluster $ci", color=palette(:tab10)[ci],
             markersize=4, alpha=0.85, markerstrokewidth=0.3)
end

fig3 = plot(p_map; size=(800, 560))
savefig(fig3, joinpath(FIGDIR, "map_clustering_arrows.png"))
println("  Saved map_clustering_arrows.png")

# ── Figure 3b: Unconditioned MAP clustering + link probabilities ──────────────

println("\nGenerating unconditioned MAP clustering figure...")

# Overall MAP sample (no K conditioning)
map_idx_all = argmax(samples.logpost)
c_map_all = samples.c[map_idx_all, :]
z_map_all = compute_table_assignments(c_map_all)
clusters_all = sort(unique(z_map_all))

# Link probabilities over all iterations
println("  Computing link probabilities (all samples)...")
link_prob_all = zeros(n, n)
for iter in 1:n_samp
    for i in 1:n
        j = samples.c[iter, i]
        link_prob_all[i, j] += 1
    end
end
link_prob_all ./= n_samp

p_map_all = plot(
    xlabel = "Waiting time (min)",
    ylabel = "Eruption duration (min)",
    title = "MAP clustering with posterior link probabilities (unconditioned)",
    legend = :topleft,
    tickfontsize = 10,
    guidefontsize = 12,
    titlefontsize = 12,
    left_margin = 8Plots.mm,
    right_margin = 20Plots.mm,
    bottom_margin = 5Plots.mm,
)

threshold_all = 0.02
p_max_all = maximum(link_prob_all[i, j] for i in 1:n, j in 1:n if i != j)

for i in 1:n
    top_j = argmax([j == i ? -Inf : link_prob_all[i, j] for j in 1:n])
    for j in 1:n
        i == j && continue
        p = link_prob_all[i, j]
        p < threshold_all && j != top_j && continue
        quiver!(p_map_all, [w[i]], [y[i]];
                quiver = ([w[j] - w[i]], [y[j] - y[i]]),
                color = get(arrow_cmap, 0.3 + 0.7 * p / p_max_all),
                linewidth = clamp(p / p_max_all * 2.5, 0.6, 2.5),
                label = false)
    end
end

scatter!(p_map_all, Float64[], Float64[];
         zcolor=Float64[], color=arrow_cmap, colorbar=true,
         colorbar_title="\nPosterior link prob.", clims=(0.0, p_max_all),
         label=false)

for (ci, k) in enumerate(clusters_all)
    mask = z_map_all .== k
    scatter!(p_map_all, w[mask], y[mask];
             label="Cluster $ci", color=palette(:tab10)[ci],
             markersize=4, alpha=0.85, markerstrokewidth=0.3)
end

fig3b = plot(p_map_all; size=(800, 560))
savefig(fig3b, joinpath(FIGDIR, "map_clustering_arrows_unconditioned.png"))
println("  Saved map_clustering_arrows_unconditioned.png")

# ── Figure 4: Posterior Predictive Check ─────────────────────────────────────

println("\nComputing posterior predictive check...")

priors = (
    α_a = 2.0,
    α_b = 0.5,
    β_a = 2.0,
    β_b = 0.5
)

ppd = Matrix{Float64}(undef, n, n_samp)
for iter in 1:n_samp
    c_cur = samples.c[iter, :]
    tables = table_vector(c_cur)
    for table in tables
        S_k = sum(y[table])
        n_k = length(table)
        for i in table
            α_i = samples.α[iter, i]
            β_i = rand(Gamma(priors.β_a + n_k * α_i, 1 / (priors.β_b + S_k)))
            ppd[i, iter] = rand(Gamma(α_i, 1 / β_i))
        end
    end
end

ppd_mean = vec(mean(ppd, dims=2))
ppd_lower = vec(mapslices(col -> quantile(col, 0.025), ppd, dims=2))
ppd_upper = vec(mapslices(col -> quantile(col, 0.975), ppd, dims=2))

p_ppc = plot(
    xlabel = "Observation index",
    ylabel = "Eruption duration (min)",
    title = "Posterior predictive check",
    legend = :topright,
    tickfontsize = 10,
    guidefontsize = 12,
    titlefontsize = 12,
    left_margin = 8Plots.mm,
    bottom_margin = 5Plots.mm,
)

# 95% credible interval lines
for i in 1:n
    plot!(p_ppc, [i, i], [ppd_lower[i], ppd_upper[i]];
          color=:lightblue, alpha=0.6, linewidth=1.2, label=false)
end

scatter!(p_ppc, 1:n, ppd_mean;
         color=:steelblue, markersize=3, label="PPD mean", alpha=0.9,
         markerstrokewidth=0.3)
scatter!(p_ppc, 1:n, y;
         color=:red, markersize=3, label="Observed", alpha=0.6,
         markerstrokewidth=0.3)

fig4 = plot(p_ppc; size=(900, 480))
savefig(fig4, joinpath(FIGDIR, "ppc.png"))
println("  Saved ppc.png")

println("\nAnalysis complete. Figures in: $FIGDIR")

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using DistanceDependentCRP, StatsPlots, DataFrames, CSV, Statistics, StatsBase, JLD2, Printf, KernelDensity

const OUTDIR = joinpath(@__DIR__, "..", "results", "simulation_study", "poisson")
const K_TRUE = 3
const DPI = 200

fmtlabel(s) = replace(lowercase(s), r"[^a-z0-9]+" => "_")

const CLUSTER_COLS = palette(:tab10)[1:K_TRUE]

const BIRTH_COLOURS = Dict(
    "conjugate" => :black,
    "prior" => :gray,
    "nmm" => palette(:blues, 5)[3:5],
    "lnmm" => palette(:reds, 5)[3:5],
)

method_colour(birth, fdim) =
    birth in ("conjugate", "prior") ? BIRTH_COLOURS[birth] :
    fdim == "none" ? BIRTH_COLOURS[birth][1] : BIRTH_COLOURS[birth][2]

method_style(fdim) = fdim == "resample" ? :dash : :solid

# ── Utility: LaTeX formatting ──────────────────────────────────────────────────

function latex_fmt(x::Float64, decimals::Int)
    Printf.format(Printf.Format("%.$(decimals)f"), x)
end

function generate_latex_table(scenario_name::String, df::DataFrame, outdir::String)
    tex_path = joinpath(outdir, "table_$(scenario_name).tex")

    birth_groups = ["conjugate", "prior", "nmm", "lnmm"]
    present = [b for b in birth_groups if any(==(b), df.birth)]
    shorten(s) = Dict("conjugate"=>"Gibbs", "prior"=>"Prior",
                      "nmm"=>"NMM", "lnmm"=>"LNMM")[s]
    bf(s, bold) = bold ? "\\textbf{$s}" : s

    lines = String[]
    push!(lines, "% Required packages: booktabs, multirow")
    push!(lines, "% Generated automatically — re-run script to update")
    push!(lines, "")
    push!(lines, "\\begin{tabular}{l c c r r r r}")
    push!(lines, "  \\toprule")
    push!(lines, "  Birth & Fixed-dim & \$\\bar{K}\$ & Mode \$K\$ " *
                 "& \$P(K{=}K^*)\$ & ESS\$(K)\$ & Birth acc (\\%) \\\\")
    push!(lines, "  \\midrule")

    for blabel in present
        sub = filter(r -> r.birth == blabel, df)
        for (i, row) in enumerate(eachrow(sub))
            bold = false
            bacc_s = isnan(row.acc_birth) ? "—" :
                     bf(latex_fmt(row.acc_birth * 100, 1), bold)
            push!(lines, "  " *
                bf(shorten(blabel), bold) * " & " *
                bf(row.fdim,        bold) * " & " *
                bf(latex_fmt(row.mean_K, 2),        bold) * " & " *
                bf(string(row.mode_K),              bold) * " & " *
                bf(latex_fmt(row.prob_K_true, 3),   bold) * " & " *
                bf(string(round(Int, row.ess_K)),   bold) * " & " *
                bacc_s * " \\\\"
            )
        end
        blabel != last(present) && push!(lines, "  \\midrule")
    end

    push!(lines, "  \\bottomrule")
    push!(lines, "\\end{tabular}")

    open(tex_path, "w") do io
        println(io, join(lines, "\n"))
    end
    println("  LaTeX table → $tex_path")
end

# ── Load sim data ──────────────────────────────────────────────────────────────

sim_data = load(joinpath(OUTDIR, "sim_data.jld2"))
SIM_OVERLAPPING = sim_data["sim_overlapping"]

# ── Per-scenario analysis ──────────────────────────────────────────────────────

for (sname, sim) in [("overlapping", SIM_OVERLAPPING)]
    println("=" ^ 65)
    println("Scenario: $sname")
    println("=" ^ 65)

    scenario_dir = joinpath(OUTDIR, sname)
    figdir = joinpath(scenario_dir, "figures")
    mkpath(figdir)

    final_csv = joinpath(scenario_dir, "summary_final_$(sname).csv")
    if !isfile(final_csv)
        @warn "Missing $final_csv — skipping $sname"
        continue
    end
    final_df = DataFrame(CSV.File(final_csv))

    # ── LaTeX table ─────────────────────────────────────────────────────────

    generate_latex_table(sname, final_df, scenario_dir)
    println()

    # Load chains for the final models
    chain_dir = joinpath(scenario_dir, "chains")
    chains = Dict{String, Any}()
    for row in eachrow(final_df)
        fname = joinpath(chain_dir, fmtlabel(row.label) * "_" * sname * "_final.jld2")
        if !isfile(fname)
            fname = joinpath(chain_dir, fmtlabel(row.label) * "_" * sname * ".jld2")
        end
        if isfile(fname)
            d = load(fname)
            chains[row.label] = (
                samples = d["samples"],
                burn = d["burn"],
                thin = d["thin"],
            )
        else
            @warn "Chain file not found: $fname"
        end
    end

    function post_slice(label)
        ch = chains[label]
        b, th = ch.burn, ch.thin
        return ch.samples, (b + 1):th:size(ch.samples.c, 1)
    end

    # ── Figure 1: Simulated data overview ───────────────────────────────────

    println("  Generating simulated data plot...")
    let
        labels = compute_table_assignments(sim.c)
        ys = minimum(sim.y):maximum(sim.y)
        freq_matrix = [count(sim.y[labels .== k] .== y) for k in 1:K_TRUE, y in ys]
        p = groupedbar(
            freq_matrix';
            bar_position = :dodge,
            xticks = (1:length(ys), string.(ys)),
            xlabel = "Count y",
            ylabel = "Frequency",
            title = "$sname (λ=$(join(sim.λs, ", ")))",
            label = reshape(["λ=$(sim.λs[k])" for k in 1:K_TRUE], 1, :),
            color = reshape([palette(:tab10)[k] for k in 1:K_TRUE], 1, :),
            legend = :topright,
            tickfontsize = 10,
            guidefontsize = 12,
            titlefontsize = 13,
        )
        savefig(p, joinpath(figdir, "simulated_data.png"))
        println("    Saved simulated_data.png")
    end

    # ── Figure 2: K trace ───────────────────────────────────────────────────

    println("  Generating K trace figure...")
    let
        p = plot(xlabel="Iteration (thinned)", ylabel="K",
                 title="K trace — $sname", legend=:topright)
        hline!(p, [K_TRUE]; color=:black, linestyle=:dash, linewidth=1.5,
               label="K_true=$K_TRUE")
        for row in eachrow(final_df)
            haskey(chains, row.label) || continue
            ch = chains[row.label]
            post_c = ch.samples.c[(ch.burn + 1):ch.thin:end, :]
            k_full = calculate_n_clusters(post_c)
            plot!(p, k_full;
                  label = "$(row.birth)/$(row.fdim)",
                  color = method_colour(row.birth, row.fdim),
                  linestyle = method_style(row.fdim),
                  alpha = 0.7, linewidth = 1.0)
        end
        savefig(p, joinpath(figdir, "k_trace.png"))
        println("    Saved k_trace.png")
    end

    # ── Figure 3: P(K) distribution ─────────────────────────────────────────

    println("  Generating P(K) distribution figure...")
    let
        k_max = K_TRUE + 4
        ks = 1:k_max
        method_labels = String[]
        pk_matrix = Matrix{Float64}(undef, 0, length(ks))

        for row in eachrow(final_df)
            haskey(chains, row.label) || continue
            ch = chains[row.label]
            post_c = ch.samples.c[(ch.burn + 1):ch.thin:end, :]
            k_tr = Float64.(calculate_n_clusters(post_c))
            pk = [mean(k_tr .== k) for k in ks]
            push!(method_labels, "$(row.birth)/$(row.fdim)")
            pk_matrix = vcat(pk_matrix, pk')
        end

        fig = groupedbar(
            pk_matrix';
            bar_position = :dodge,
            xticks = (1:length(ks), string.(ks)),
            xlabel = "K",
            ylabel = "P(K = k)",
            label = reshape(method_labels, 1, :),
            legend = :topleft,
            legendfontsize = 7,
            size = (900, 420),
            left_margin = 10Plots.mm,
            bottom_margin = 8Plots.mm,
        )
        vline!(fig, [K_TRUE]; color=:black, linestyle=:dash,
               linewidth=1.5, label="K*=$K_TRUE")
        savefig(fig, joinpath(figdir, "pk_distribution.png"))
        println("    Saved pk_distribution.png")
    end

    # ── Figure 4: ARI trace ──────────────────────────────────────────────────

    println("  Generating ARI trace figure...")
    let
        p = plot(xlabel="Post-burnin iteration (thinned)", ylabel="ARI",
                 title="ARI trace — $sname", legend=:bottomright,
                 ylims=(-0.1, 1.05))
        hline!(p, [1.0]; color=:black, linestyle=:dash, linewidth=1.5,
               label="Perfect (1.0)")
        for row in eachrow(final_df)
            haskey(chains, row.label) || continue
            ch = chains[row.label]
            post_c = ch.samples.c[(ch.burn + 1):ch.thin:end, :]
            ari_tr = compute_ari_trace(post_c, sim.c)
            plot!(p, ari_tr;
                  label = "$(row.birth)/$(row.fdim)",
                  color = method_colour(row.birth, row.fdim),
                  linestyle = method_style(row.fdim),
                  alpha = 0.7, linewidth = 1.0)
        end
        savefig(p, joinpath(figdir, "ari_trace.png"))
        println("    Saved ari_trace.png")
    end

    # ── Figure 5: Acceptance rates ───────────────────────────────────────────

    println("  Generating acceptance rates figure...")
    let
        rjmcmc_rows = filter(r -> r.birth != "conjugate", final_df)
        if !isempty(rjmcmc_rows)
            xlabels = string.(rjmcmc_rows.birth, "/", rjmcmc_rows.fdim)
            births = [isnan(r.acc_birth) ? 0.0 : r.acc_birth * 100 for r in eachrow(rjmcmc_rows)]
            deaths = [isnan(r.acc_death) ? 0.0 : r.acc_death * 100 for r in eachrow(rjmcmc_rows)]
            overall = [isnan(r.acc_overall) ? 0.0 : r.acc_overall * 100 for r in eachrow(rjmcmc_rows)]

            p = groupedbar(
                hcat(births, deaths, overall);
                bar_position = :dodge,
                xticks = (1:nrow(rjmcmc_rows), xlabels),
                xrotation = 30,
                label = ["Birth" "Death" "Overall"],
                ylabel = "Acceptance rate (%)",
                title = "Acceptance rates — $sname",
                legend = :topright,
                ylims = (0, 100),
            )
            savefig(p, joinpath(figdir, "acceptance_rates.png"))
            println("    Saved acceptance_rates.png")
        end
    end

    # ── Figure 6: ESS comparison ─────────────────────────────────────────────

    println("  Generating ESS comparison figure...")
    let
        xlabels = string.(final_df.birth, "/", final_df.fdim)
        ess_K_v = final_df.ess_K
        ess_ps_v = final_df.ess_per_sec

        p = groupedbar(
            hcat(ess_K_v, ess_ps_v);
            bar_position = :dodge,
            xticks = (1:nrow(final_df), xlabels),
            xrotation = 30,
            label = ["ESS(K)" "ESS/sec"],
            ylabel = "ESS",
            title = "ESS comparison — $sname",
            legend = :topright,
        )
        savefig(p, joinpath(figdir, "ess_comparison.png"))
        println("    Saved ess_comparison.png")
    end

    # ── Figure 7: λ recovery ────────────────────────────────────────────────

    println("  Generating lambda recovery figure...")
    let
        true_labels = compute_table_assignments(sim.c)
        true_λ_obs = sim.λs[true_labels]

        panels = []
        for row in eachrow(final_df)
            haskey(chains, row.label) || continue
            ch = chains[row.label]
            samp = ch.samples
            hasproperty(samp, :λ) || continue

            post_λ = samp.λ[(ch.burn + 1):ch.thin:end, :]
            post_λ_mean = vec(mean(post_λ, dims=1))

            λ_lim = max(maximum(true_λ_obs), maximum(post_λ_mean)) * 1.05
            p = scatter(true_λ_obs, post_λ_mean;
                        xlabel = "True λ",
                        ylabel = "Posterior mean λ",
                        title = "$(row.birth)/$(row.fdim)",
                        legend = false,
                        markersize = 3, alpha = 0.5,
                        color = method_colour(row.birth, row.fdim))
            plot!(p, [0, λ_lim], [0, λ_lim];
                  color=:black, linestyle=:dash, linewidth=1.0)
            push!(panels, p)
        end

        if !isempty(panels)
            ncols = min(length(panels), 3)
            nrows = ceil(Int, length(panels) / ncols)
            fig = plot(panels...;
                       layout = (nrows, ncols),
                       size = (400 * ncols, 380 * nrows),
                       plot_title = "λ recovery — $sname")
            savefig(fig, joinpath(figdir, "lambda_recovery.png"))
            println("    Saved lambda_recovery.png")
        end
    end

    # ── Figure 8: Data overview (histogram + scatter) — paper figure ─────────────

    println("  Generating data overview figure...")
    let
        true_labels = compute_table_assignments(sim.c)
        λ_int = round.(Int, sim.λs)

        ys = minimum(sim.y):maximum(sim.y)
        freq_matrix = [count(sim.y[true_labels .== k] .== y) for k in 1:K_TRUE, y in ys]
        p_hist = groupedbar(
            freq_matrix';
            bar_position = :dodge,
            xticks = (1:length(ys), string.(ys)),
            xlabel = "Count y",
            ylabel = "Frequency",
            label = reshape(["Cluster $k  (λ=$(λ_int[k]))" for k in 1:K_TRUE], 1, :),
            color = reshape([CLUSTER_COLS[k] for k in 1:K_TRUE], 1, :),
            legend = :topright,
            tickfontsize = 10,
            guidefontsize = 12,
            left_margin = 10Plots.mm,
            bottom_margin = 8Plots.mm,
        )

        p_scatter = plot(xlabel="Covariate x", ylabel="Count y",
                         legend=:topleft, tickfontsize=10, guidefontsize=12,
                         left_margin=10Plots.mm, bottom_margin=8Plots.mm)
        for k in 1:K_TRUE
            mask = true_labels .== k
            scatter!(p_scatter, sim.x[mask], sim.y[mask];
                     label="Cluster $k  (λ=$(λ_int[k]))",
                     color=CLUSTER_COLS[k], markersize=4, alpha=0.7,
                     markerstrokewidth=0)
        end

        fig = plot(p_hist, p_scatter; layout=(1, 2), size=(960, 400))
        savefig(fig, joinpath(figdir, "data_overview.png"))
        println("    Saved data_overview.png")
    end

    # ── Figure 9: K + α_ddcrp traces — paper figure ──────────────────────────────

    println("  Generating K and alpha trace figure...")
    let
        p_k = plot(xlabel="", ylabel="K",
                   legend=:topright, tickfontsize=10, guidefontsize=12,
                   left_margin=10Plots.mm, bottom_margin=2Plots.mm)
        hline!(p_k, [K_TRUE]; color=:black, linestyle=:dash,
               linewidth=1.5, label="K* = $K_TRUE")

        p_alpha = plot(xlabel="Post-burnin iteration (thinned)", ylabel="α (ddCRP)",
                       legend=:topright, tickfontsize=10, guidefontsize=12,
                       left_margin=10Plots.mm, bottom_margin=8Plots.mm)

        for row in eachrow(final_df)
            haskey(chains, row.label) || continue
            ch = chains[row.label]
            post_c = ch.samples.c[(ch.burn + 1):ch.thin:end, :]
            k_tr = calculate_n_clusters(post_c)
            alpha_tr = ch.samples.α_ddcrp[(ch.burn + 1):ch.thin:end]
            col = method_colour(row.birth, row.fdim)
            ls = method_style(row.fdim)
            lbl = "$(row.birth)/$(row.fdim)"
            plot!(p_k, k_tr; label=lbl, color=col, linestyle=ls, alpha=0.7, linewidth=0.9)
            plot!(p_alpha, alpha_tr; label=lbl, color=col, linestyle=ls, alpha=0.7, linewidth=0.9)
        end

        fig = plot(p_k, p_alpha; layout=(2, 1), size=(1000, 600))
        savefig(fig, joinpath(figdir, "k_alpha_traces.png"))
        println("    Saved k_alpha_traces.png")
    end

    # ── Summary panel (3×3) ──────────────────────────────────────────────────────

    println("  Generating summary panel figure...")

    gibbs_row = findfirst(r -> r.birth == "conjugate", eachrow(final_df))
    gibbs_label = isnothing(gibbs_row) ? nothing : final_df[gibbs_row, :label]
    rjmcmc_rows = filter(r -> r.birth != "conjugate" && haskey(chains, r.label), final_df)
    best_rjmcmc_label = isempty(rjmcmc_rows) ? nothing :
        rjmcmc_rows[argmax([isnan(r.ess_per_sec) ? -Inf : r.ess_per_sec
            for r in eachrow(rjmcmc_rows)]), :label]
    map_label = !isnothing(gibbs_label) && haskey(chains, gibbs_label) ?
                gibbs_label : best_rjmcmc_label

    true_labels = compute_table_assignments(sim.c)

    # Panel 1: Histogram by true cluster
    p_hist = plot(xlabel="Count (y)", ylabel="Frequency",
                  title="Data: counts by true cluster",
                  legend=:topright, tickfontsize=10, guidefontsize=11)
    for k in 1:K_TRUE
        mask = true_labels .== k
        histogram!(p_hist, sim.y[mask];
                   label="Cluster $k  (λ=$(sim.λs[k]))",
                   color=CLUSTER_COLS[k], alpha=0.7, linewidth=0, normalize=false)
    end

    # Panel 2: Scatter by true cluster
    p_scatter_true = plot(xlabel="Covariate x", ylabel="Count y",
                          title="Simulated data (true clusters)",
                          legend=:topleft, tickfontsize=10, guidefontsize=11)
    for k in 1:K_TRUE
        mask = true_labels .== k
        scatter!(p_scatter_true, sim.x[mask], sim.y[mask];
                 label="Cluster $k", color=CLUSTER_COLS[k],
                 markersize=4, alpha=0.7, markerstrokewidth=0)
    end

    # Panel 3: Scatter by MAP cluster
    p_scatter_map = plot(xlabel="Covariate x", ylabel="Count y",
                         title="MAP clustering ($(isnothing(map_label) ? "N/A" : "Gibbs"))",
                         legend=:topleft, tickfontsize=10, guidefontsize=11)
    if !isnothing(map_label)
        samp, idx = post_slice(map_label)
        idx_arr = collect(idx)
        post_c_arr = samp.c[idx_arr, :]
        post_lp = samp.logpost[idx_arr]
        k_tr = calculate_n_clusters(post_c_arr)
        mode_K = Int(mode(k_tr))
        mode_pos = findall(k_tr .== mode_K)
        best_pos = mode_pos[argmax(post_lp[mode_pos])]
        c_map = post_c_arr[best_pos, :]
        map_labels = compute_table_assignments(c_map)
        K_map = maximum(map_labels)
        col20 = palette(:tab20)
        for k in 1:K_map
            mask = map_labels .== k
            scatter!(p_scatter_map, sim.x[mask], sim.y[mask];
                     label="Cluster $k", color=col20[mod1(k, length(col20))],
                     markersize=4, alpha=0.7, markerstrokewidth=0)
        end
    end

    # Panel 4: K trace
    p_ktrace = plot(xlabel="Post-burnin iteration (thinned)", ylabel="K",
                    title="K trace", legend=:topright,
                    tickfontsize=10, guidefontsize=11)
    hline!(p_ktrace, [K_TRUE]; color=:black, linestyle=:dash,
           linewidth=1.5, label="K* = $K_TRUE")
    for row in eachrow(final_df)
        haskey(chains, row.label) || continue
        samp, idx = post_slice(row.label)
        k_tr = calculate_n_clusters(samp.c[idx, :])
        plot!(p_ktrace, k_tr;
              label = "$(row.birth)/$(row.fdim)",
              color = method_colour(row.birth, row.fdim),
              linestyle = method_style(row.fdim),
              alpha = 0.7, linewidth = 0.9)
    end

    # Panel 5: Log-posterior trace
    p_lp_trace = plot(xlabel="Post-burnin iteration (thinned)", ylabel="Log-posterior",
                      title="Log-posterior trace", legend=:bottomright,
                      tickfontsize=10, guidefontsize=11)
    for row in eachrow(final_df)
        haskey(chains, row.label) || continue
        samp, idx = post_slice(row.label)
        plot!(p_lp_trace, samp.logpost[idx];
              label = "$(row.birth)/$(row.fdim)",
              color = method_colour(row.birth, row.fdim),
              linestyle = method_style(row.fdim),
              alpha = 0.7, linewidth = 0.9)
    end

    # Panel 6: Log-posterior density
    p_lp_dens = plot(xlabel="Log-posterior", ylabel="Density",
                     title="Log-posterior distribution", legend=:topleft,
                     tickfontsize=10, guidefontsize=11)
    for row in eachrow(final_df)
        haskey(chains, row.label) || continue
        samp, idx = post_slice(row.label)
        density!(p_lp_dens, samp.logpost[idx];
                 label = "$(row.birth)/$(row.fdim)",
                 color = method_colour(row.birth, row.fdim),
                 linestyle = method_style(row.fdim),
                 linewidth = 1.5)
    end

    # Panel 7: α_ddcrp trace
    p_alpha_trace = plot(xlabel="Post-burnin iteration (thinned)", ylabel="α (DDCRP)",
                         title="α_DDCRP trace", legend=:topright,
                         tickfontsize=10, guidefontsize=11)
    for row in eachrow(final_df)
        haskey(chains, row.label) || continue
        samp, idx = post_slice(row.label)
        plot!(p_alpha_trace, samp.α_ddcrp[idx];
              label = "$(row.birth)/$(row.fdim)",
              color = method_colour(row.birth, row.fdim),
              linestyle = method_style(row.fdim),
              alpha = 0.7, linewidth = 0.9)
    end

    # Panel 8: α_ddcrp density
    p_alpha_dens = plot(xlabel="α (DDCRP)", ylabel="Density",
                        title="α_DDCRP distribution", legend=:topright,
                        tickfontsize=10, guidefontsize=11)
    for row in eachrow(final_df)
        haskey(chains, row.label) || continue
        samp, idx = post_slice(row.label)
        density!(p_alpha_dens, samp.α_ddcrp[idx];
                 label = "$(row.birth)/$(row.fdim)",
                 color = method_colour(row.birth, row.fdim),
                 linestyle = method_style(row.fdim),
                 linewidth = 1.5)
    end

    # Panel 9: P(K)
    let
        k_max = K_TRUE + 4
        ks = 1:k_max
        method_labels = String[]
        pk_matrix = Matrix{Float64}(undef, 0, length(ks))
        for row in eachrow(final_df)
            haskey(chains, row.label) || continue
            samp, idx = post_slice(row.label)
            k_tr = Float64.(calculate_n_clusters(samp.c[idx, :]))
            pk = [mean(k_tr .== k) for k in ks]
            push!(method_labels, "$(row.birth)/$(row.fdim)")
            pk_matrix = vcat(pk_matrix, pk')
        end

        global p_pk = groupedbar(
            pk_matrix';
            bar_position = :dodge,
            xticks = (1:length(ks), string.(ks)),
            xlabel = "K",
            ylabel = "P(K = k)",
            title = "P(K) distribution",
            label = reshape(method_labels, 1, :),
            legend = :topleft,
            legendfontsize = 6,
        )
        vline!(p_pk, [K_TRUE]; color=:black, linestyle=:dash,
               linewidth=1.5, label="K* = $K_TRUE")
    end

    fig_summary = plot(
        p_hist, p_scatter_true, p_scatter_map,
        p_ktrace, p_lp_trace, p_lp_dens,
        p_alpha_trace, p_alpha_dens, p_pk;
        layout = (3, 3),
        size = (1800, 1500),
        dpi = DPI,
        plot_titlefontsize = 13,
        left_margin = 8Plots.mm,
        bottom_margin = 8Plots.mm,
        right_margin = 4Plots.mm,
        top_margin = 4Plots.mm,
    )
    savefig(fig_summary, joinpath(scenario_dir, "summary_$(sname).png"))
    println("    Saved summary_$(sname).png")

    println()
end

println("Analysis complete. Figures and tables in: $OUTDIR")

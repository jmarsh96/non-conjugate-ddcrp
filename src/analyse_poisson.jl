using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using DistanceDependentCRP, StatsPlots, DataFrames, CSV, Statistics, StatsBase, JLD2, Printf

const OUTDIR = joinpath(@__DIR__, "..", "results", "simulation_study", "poisson")
const K_TRUE = 3

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

    # ── Figure 1: P(K) distribution ─────────────────────────────────────────

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

    # ── Figure 2: Data overview (histogram + scatter) — paper figure ─────────────

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

    # ── Figure 3: K + α_ddcrp traces — paper figure ──────────────────────────────

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

    println()
end

println("Analysis complete. Figures and tables in: $OUTDIR")

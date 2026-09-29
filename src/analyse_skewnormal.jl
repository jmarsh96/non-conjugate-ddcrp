using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using DistanceDependentCRP, StatsPlots, DataFrames, CSV, Statistics, StatsBase, JLD2, Printf, Distributions

# Bring the SkewNormalClusters* types into scope so JLD2 deserialises the saved
# chains as their real types (needed for field access, e.g. samples.ξ/.ω/.α).
include(joinpath(@__DIR__, "skewnormal_model.jl"))

ENV["GKSwstype"] = "nul"   # headless GR

# Reads the pooled summaries written by run_skewnormal.jl (summary_final_*.csv,
# one row per config over all seeds) for the tables and the P(K) figure, and the
# first seed's saved chain for the trace and parameter-recovery figures. Unless
# --test, copies the paper's skew-normal figures and tables into
# docs/ddcrp-paper/{figures,tables}/, including the prior-recovery table from
# geweke_skewnormal.jl when it exists.
const IS_TEST = "--test" in ARGS
const OUTDIR = joinpath(@__DIR__, "..", "results", "simulation_study",
                        IS_TEST ? "skewnormal_test" : "skewnormal")
const PAPER = joinpath(@__DIR__, "..", "docs", "ddcrp-paper")
const K_TRUE = 3

fmtlabel(s) = replace(lowercase(s), r"[^a-z0-9]+" => "_")

const CLUSTER_COLS = palette(:tab10)[1:K_TRUE]

# ── LaTeX comparison table (per-parameter family columns) ────────────────────

latex_fmt(x::Real, d::Int) = Printf.format(Printf.Format("%.$(d)f"), Float64(x))

const FAM_TEX = Dict("prior"=>"Prior", "nmm"=>"Normal", "lnmm"=>"LogNormal", "ig"=>"InvGamma")

function generate_latex_table(scenario_name::String, df::DataFrame, outdir::String)
    tex_path = joinpath(outdir, "table_$(scenario_name).tex")
    d = sort(df, [:fdim, :fam_ξ, :fam_ω, :fam_α])

    lines = String[]
    push!(lines, "% Required packages: booktabs")
    push!(lines, "% Generated automatically; re-run analyse_skewnormal.jl to update")
    push!(lines, "% Pooled over seeds: P(K=K*) with its Monte Carlo SE (spread of the per-seed")
    push!(lines, "% estimates), split-R-hat of K across seeds, and ESS columns as means per chain")
    push!(lines, "% (parameter ESS: median over observations).")
    push!(lines, "")
    # ESJD(K) is the selection criterion and ESS(K) is reported beside it; the
    # mean ARI is quoted in the text instead.
    push!(lines, "\\begin{tabular}{l l l l r r r r r r r r r}")
    push!(lines, "  \\toprule")
    push!(lines, "  \$\\xi\$ & \$\\omega\$ & \$\\kappa\$ & Fixed-dim & \$P(K{=}K^*)\$ & \$\\hat{R}(K)\$ & ESJD\$(K)\$ & ESS\$(K)\$ " *
                 "& ESS\$_\\xi\$ & ESS\$_\\omega\$ & ESS\$_\\kappa\$ & Birth (\\%) & Fixed (\\%) \\\\")
    push!(lines, "  \\midrule")
    for (i, row) in enumerate(eachrow(d))
        # A rule between the No Update and Resample blocks.
        i > 1 && row.fdim != d.fdim[i-1] && push!(lines, "  \\midrule")
        push!(lines, "  " *
            FAM_TEX[row.fam_ξ] * " & " * FAM_TEX[row.fam_ω] * " & " * FAM_TEX[row.fam_α] * " & " *
            (row.fdim == "resample" ? "Resample" : "No Update") * " & " *
            latex_fmt(row.prob_K_true, 2) * " (" * latex_fmt(row.prob_K_true_se, 2) * ") & " *
            latex_fmt(row.rhat_K, 2) * " & " * latex_fmt(row.esjd_K, 3) * " & " *
            string(round(Int, row.ess_K)) * " & " *
            string(round(Int, row.med_ess_ξ)) * " & " * string(round(Int, row.med_ess_ω)) * " & " *
            string(round(Int, row.med_ess_α)) * " & " *
            latex_fmt(row.acc_birth * 100, 2) * " & " * latex_fmt(row.acc_fixed * 100, 1) * " \\\\")
    end
    push!(lines, "  \\bottomrule")
    push!(lines, "\\end{tabular}")
    open(io -> println(io, join(lines, "\n")), tex_path, "w")
    println("  LaTeX table → $tex_path")
end

# ── Load sim data ─────────────────────────────────────────────────────────────

sim_data = load(joinpath(OUTDIR, "sim_data.jld2"))
SIMS = Dict("overlapping" => sim_data["sim_overlapping"], "separated" => sim_data["sim_separated"])

# ── Per-scenario analysis ─────────────────────────────────────────────────────

for sname in ["overlapping", "separated"]
    sim = SIMS[sname]
    println("=" ^ 65); println("Scenario: $sname"); println("=" ^ 65)

    scenario_dir = joinpath(OUTDIR, sname)
    figdir = joinpath(scenario_dir, "figures"); mkpath(figdir)

    final_csv = joinpath(scenario_dir, "summary_final_$(sname).csv")
    if !isfile(final_csv)
        @warn "Missing $final_csv — skipping $sname"; continue
    end
    final_df = DataFrame(CSV.File(final_csv))

    generate_latex_table(sname, final_df, scenario_dir); println()

    # Load chains
    chain_dir = joinpath(scenario_dir, "chains")
    chains = Dict{String, Any}()
    for row in eachrow(final_df)
        fname = joinpath(chain_dir, fmtlabel(row.label) * "_" * sname * ".jld2")
        if isfile(fname)
            d = load(fname)
            chains[row.label] = (samples=d["samples"], burn=d["burn"], thin=d["thin"])
        else
            @warn "Chain file not found: $fname"
        end
    end

    # ── Figure 1: P(K) distribution, split by fixed-dim strategy ─────────────
    println("  Generating P(K) distribution figure...")
    let
        ks = 1:(K_TRUE + 4)
        panels = Any[]
        for fd in ["none", "resample"]
            sub = filter(r -> r.fdim == fd, final_df)
            labs = String[]; pk_mat = Matrix{Float64}(undef, 0, length(ks))
            # Pooled over every seed, from the pk_k columns of the summary.
            for row in eachrow(sub)
                pk_mat = vcat(pk_mat, [row[Symbol("pk_$k")] for k in ks]')
                push!(labs, row.birth)
            end
            p = groupedbar(pk_mat'; bar_position=:dodge,
                xticks=(1:length(ks), string.(ks)), xlabel="K", ylabel="P(K=k)",
                label=reshape(labs, 1, :), legend=:topright, legendfontsize=6,
                title="fixed-dim: $fd", titlefontsize=10)
            vline!(p, [K_TRUE]; color=:black, linestyle=:dash, linewidth=1.5, label="K*=$K_TRUE")
            push!(panels, p)
        end
        fig = plot(panels...; layout=(2, 1), size=(950, 700),
                   left_margin=10Plots.mm, bottom_margin=8Plots.mm)
        savefig(fig, joinpath(figdir, "pk_distribution.png"))
        println("    Saved pk_distribution.png")
    end

    # ── Figure 2: Data overview (density by cluster + scatter) ───────────────
    println("  Generating data overview figure...")
    let
        true_labels = compute_table_assignments(sim.c)
        p_dens = plot(xlabel="Response y", ylabel="Density", legend=:topright,
                      tickfontsize=10, guidefontsize=12, left_margin=10Plots.mm, bottom_margin=8Plots.mm)
        xgrid = range(minimum(sim.y) - 1, maximum(sim.y) + 1; length=400)
        for k in 1:K_TRUE
            mask = true_labels .== k
            ξk, ωk, αk = sim.pars[k]
            histogram!(p_dens, sim.y[mask]; bins=20, normalize=:pdf, color=CLUSTER_COLS[k],
                       alpha=0.35, label="Cluster $k data")
            plot!(p_dens, xgrid, pdf.(SkewNormal(ξk, ωk, αk), xgrid); color=CLUSTER_COLS[k],
                  linewidth=2, label="SkewNormal(ξ=$(ξk), ω=$(ωk), α=$(αk))")
        end
        p_scatter = plot(xlabel="Covariate x", ylabel="Response y", legend=:topleft,
                         tickfontsize=10, guidefontsize=12, left_margin=10Plots.mm, bottom_margin=8Plots.mm)
        for k in 1:K_TRUE
            mask = true_labels .== k
            scatter!(p_scatter, sim.x[mask], sim.y[mask]; label="Cluster $k", color=CLUSTER_COLS[k],
                     markersize=4, alpha=0.7, markerstrokewidth=0)
        end
        fig = plot(p_dens, p_scatter; layout=(1, 2), size=(980, 400))
        savefig(fig, joinpath(figdir, "data_overview.png"))
        println("    Saved data_overview.png")
    end

    # ── Figure 3: K + α_ddcrp traces (baseline + all-prior, both fixed-dim) ──
    println("  Generating K and alpha trace figure...")
    let
        show_rows = filter(r -> r.varied in ["baseline", "all-prior"], final_df)
        p_k = plot(xlabel="", ylabel="K", legend=:topright, tickfontsize=10, guidefontsize=12,
                   left_margin=10Plots.mm, bottom_margin=2Plots.mm)
        hline!(p_k, [K_TRUE]; color=:black, linestyle=:dash, linewidth=1.5, label="K* = $K_TRUE")
        p_alpha = plot(xlabel="Post-burnin iteration (thinned)", ylabel="α (ddCRP)", legend=:topright,
                       tickfontsize=10, guidefontsize=12, left_margin=10Plots.mm, bottom_margin=8Plots.mm)
        cols = palette(:tab10)
        for (i, row) in enumerate(eachrow(show_rows))
            haskey(chains, row.label) || continue
            ch = chains[row.label]
            pc = ch.samples.c[(ch.burn + 1):ch.thin:end, :]
            ktr = calculate_n_clusters(pc)
            atr = ch.samples.α_ddcrp[(ch.burn + 1):ch.thin:end]
            ls = row.fdim == "resample" ? :dash : :solid
            lbl = "$(row.birth)/$(row.fdim)"
            plot!(p_k, ktr; label=lbl, color=cols[i], linestyle=ls, alpha=0.7, linewidth=0.9)
            plot!(p_alpha, atr; label=lbl, color=cols[i], linestyle=ls, alpha=0.7, linewidth=0.9)
        end
        fig = plot(p_k, p_alpha; layout=(2, 1), size=(1000, 600))
        savefig(fig, joinpath(figdir, "k_alpha_traces.png"))
        println("    Saved k_alpha_traces.png")
    end

    # ── Figure 4: parameter recovery (best ESS(K) variant, K=K* draws) ───────
    println("  Generating parameter recovery figure...")
    let
        # No Update only: the Resample chains fail to mix across partitions, so
        # their ESS(K) overstates how well they explore the posterior.
        cand = filter(r -> haskey(chains, r.label) && r.fdim == "none", collect(eachrow(final_df)))
        if isempty(cand)
            @warn "No chains available for parameter recovery"
        else
            best_row = cand[argmax([r.ess_K for r in cand])]
            ch = chains[best_row.label]
            keep = (ch.burn + 1):ch.thin:size(ch.samples.c, 1)
            k_tr = calculate_n_clusters(ch.samples.c[keep, :])
            true_labels = compute_table_assignments(sim.c)
            sel = findall(==(K_TRUE), k_tr)
            panels = Any[]
            for (pi, psym) in enumerate((:ξ, :ω, :α))
                pmat = getfield(ch.samples, psym)[keep, :]
                p = plot(xlabel="True cluster", ylabel=(psym === :α ? "κ (shape)" : string(psym)), legend=false,
                         tickfontsize=9, guidefontsize=11, xticks=(1:K_TRUE, string.(1:K_TRUE)))
                for k in 1:K_TRUE
                    idx = findfirst(==(k), true_labels)
                    vals = isempty(sel) ? pmat[:, idx] : pmat[sel, idx]
                    violin!(p, fill(k, length(vals)), vals; color=CLUSTER_COLS[k], alpha=0.6)
                    scatter!(p, [k], [sim.pars[k][pi]]; color=:black, markershape=:diamond, markersize=6)
                end
                push!(panels, p)
            end
            fig = plot(panels...; layout=(1, 3), size=(1000, 340),
                       plot_title="Parameter recovery — $(best_row.birth)/$(best_row.fdim) (black ◆ = truth)",
                       plot_titlefontsize=11, bottom_margin=8Plots.mm, left_margin=6Plots.mm)
            savefig(fig, joinpath(figdir, "param_recovery.png"))
            println("    Saved param_recovery.png")
        end
    end
    println()
end

println("Analysis complete. Figures and tables in: $OUTDIR")

# ── Copy the paper's skew-normal assets ──────────────────────────────────────
# docs/ is git-ignored, so a cluster checkout has no paper directory: the
# assets then stay in results/ and are copied when the script is rerun locally.
if IS_TEST
    println("--test: paper assets left in $OUTDIR (docs/ddcrp-paper/ untouched).")
elseif !isdir(PAPER)
    println("No paper directory at $PAPER (docs/ is git-ignored): assets left in $OUTDIR.")
else
    copies = Pair{String,String}[]
    for sname in ["overlapping", "separated"]
        sdir = joinpath(OUTDIR, sname)
        push!(copies,
            joinpath(sdir, "figures", "pk_distribution.png") => joinpath("figures", "skew_pk_$sname.png"),
            joinpath(sdir, "figures", "data_overview.png") => joinpath("figures", "skew_data_overview_$sname.png"),
            joinpath(sdir, "table_$sname.tex") => joinpath("tables", "table_skew_$sname.tex"))
    end
    push!(copies,
        joinpath(OUTDIR, "separated", "figures", "param_recovery.png") => joinpath("figures", "skew_param_recovery.png"),
        joinpath(OUTDIR, "geweke", "table_geweke.tex") => joinpath("tables", "table_skew_geweke.tex"))
    for (src, dst) in copies
        if isfile(src)
            cp(src, joinpath(PAPER, dst); force = true)
        else
            @warn "Not found, not copied: $src"
        end
    end
    println("Paper assets copied to $PAPER/{figures,tables}/")
end

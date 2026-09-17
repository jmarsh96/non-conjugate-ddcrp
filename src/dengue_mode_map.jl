# ============================================================================
# dengue_mode_map.jl
#
# Produce a POSTERIOR-MODE cluster partition for the Brazil dengue NB log-normal
# chain, as an alternative to the Dahl least-squares point estimate written by
# analyse_dengue.jl (which over-fragments to ~80 clusters because the posterior
# co-clustering signal is weak).
#
# Representative partition = the highest log-posterior draw among those whose
# number of clusters equals the posterior mode of K — the "MAP-at-mode"
# partition, the same summary used in the Old Faithful analysis. Its K matches
# the reported posterior mode, so the map and the number quoted on the slide
# agree.
#
# Writes results/real_data/dengue/cluster_assignments_mode.csv (same columns as
# cluster_assignments.csv). Render the map with src/map_dengue_mode.R. Neither
# this script nor the R one touches the existing map_clusters.png / .csv.
#
#   julia --project=. src/dengue_mode_map.jl
# ============================================================================

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using DistanceDependentCRP, DataFrames, CSV, Statistics, StatsBase, JLD2, Printf

# Brings NegBinPopulationRatesSamples into scope so JLD2 deserialises the chain
# as its real type (needed for samples.ρ / samples.r).
include(joinpath(@__DIR__, "negbin_population_model.jl"))

const OUTDIR = joinpath(@__DIR__, "..", "results", "real_data", "dengue")
const CHAIN  = joinpath(OUTDIR, "chains", "nb_lnmm_r_lnmm_noupdate_final.jld2")
isfile(CHAIN) || error("Missing chain: $CHAIN")

meta = load(joinpath(OUTDIR, "run_metadata.jld2"))
GEOCODE = meta["geocode"]; NAME = meta["name"]
LAT = meta["lat"]; LON = meta["lon"]
Y = meta["y"]; E = meta["E"]
N = length(Y)

println("Loading chain (~384 MB, may take a moment)...")
d = load(CHAIN)
s = d["samples"]
burn = get(d, "burn", meta["burn"])
thin = get(d, "thin", meta["thin"])
keep = (burn + 1):thin:size(s.c, 1)

C   = s.c[keep, :]
LP  = s.logpost[keep]
RHO = s.ρ[keep, :]
Rr  = s.r[keep, :]
@printf("  %d post-burn-in draws over %d municipalities\n", size(C, 1), N)

# 1..K table labels for every draw, and the posterior of K.
z_all = [compute_table_assignments(C[i, :]) for i in 1:size(C, 1)]
Kt = [maximum(z) for z in z_all]
Kmode = mode(Kt)
@printf("  posterior of K: mode %d, mean %.1f, range %d-%d\n",
        Kmode, mean(Kt), minimum(Kt), maximum(Kt))

# Representative partition: highest log-posterior draw with K == mode.
cand = findall(==(Kmode), Kt)
best = cand[argmax(LP[cand])]
z = z_all[best]
@printf("  MAP-at-mode partition: draw %d of the K=%d draws, logpost %.1f\n",
        best, Kmode, LP[best])

# Posterior co-clustering matrix -> per-municipality placement confidence
# (identical definition to analyse_dengue.jl, for a comparable uncertainty map).
println("Computing posterior co-clustering matrix (this is the slow step)...")
sim = compute_similarity_matrix(C)
cocluster_conf = [mean(sim[i, findall(==(z[i]), z)]) for i in 1:N]

df = DataFrame(
    geocode = GEOCODE, name = NAME, lat = LAT, lon = LON,
    y = Y, pop = E .* 1e5, incidence = Y ./ E,
    cluster = z,
    cocluster_confidence = cocluster_conf,
    rho_mean = vec(mean(RHO, dims = 1)),
    r_mean   = vec(mean(Rr,  dims = 1)),
)
out = joinpath(OUTDIR, "cluster_assignments_mode.csv")
CSV.write(out, df)
@printf("  wrote %s\n", out)
@printf("  %d clusters; sizes %s\n", length(unique(z)),
        string(sort(collect(values(countmap(z))), rev = true)))
println("\nNow render the map with:  Rscript src/map_dengue_mode.R")

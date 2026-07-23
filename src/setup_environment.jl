# ============================================================================
# setup_environment.jl
#
# Installs everything this project needs, from within Julia, and verifies the
# result. Intended to be the first thing run on a new machine — in particular
# on a cluster login node, since compute nodes generally have no internet
# access and `Pkg` cannot download anything from them.
#
#   julia --project=. src/setup_environment.jl              # reproduce Manifest.toml
#   julia --project=. src/setup_environment.jl --fresh      # re-resolve from scratch
#   julia --project=. src/setup_environment.jl --update     # upgrade to newest compatible
#
# MODES
#
#   (default)  Pkg.instantiate() against the committed Manifest.toml, which
#              reproduces the exact package versions this project was developed
#              and tested against. Use this whenever it works.
#
#   --fresh    Moves Manifest.toml aside and re-resolves from Project.toml,
#              generating a new Manifest. Needed when the committed Manifest
#              cannot be satisfied — most often a different Julia version on the
#              cluster than the 1.12 used locally, or a platform with different
#              binary artifacts. The old Manifest is kept as Manifest.toml.bak
#              and restored automatically if the fresh resolve fails.
#
#   --update   Pkg.update(): newest versions compatible with Project.toml, and
#              rewrites the Manifest. Use deliberately, not as a fix-it — it can
#              change results by changing package versions.
#
# All three end by precompiling and then actually loading every dependency plus
# the two local model prototypes, so a green run means the analysis will start,
# not merely that the packages downloaded.
# ============================================================================

using Pkg

const REPO = normpath(joinpath(@__DIR__, ".."))
const MANIFEST = joinpath(REPO, "Manifest.toml")
const BACKUP = MANIFEST * ".bak"

const FRESH = "--fresh" in ARGS
const UPDATE = "--update" in ARGS

if FRESH && UPDATE
    error("--fresh and --update are mutually exclusive; pick one.")
end

# Headless plotting: StatsPlots/GR is the most fragile dependency on a cluster,
# and it must not try to open a display while precompiling.
ENV["GKSwstype"] = "nul"

println("="^76)
println("Setting up the non-conjugate ddCRP project environment")
println("="^76)
println("  project : $REPO")
println("  julia   : $(VERSION)  ($(Sys.BINDIR))")
println("  depot   : $(DEPOT_PATH[1])")
println("  mode    : ", FRESH ? "--fresh (re-resolve)" :
                        UPDATE ? "--update (upgrade)" : "instantiate (reproduce Manifest)")
println()

Pkg.activate(REPO)

# ============================================================================
# Resolve / install
# ============================================================================

"""
    fresh_resolve()

Re-resolve from Project.toml alone. `DistanceDependentCRP` is a registered
package in the General registry, so no special handling is needed here — but if
that ever changes (e.g. it moves to an unregistered fork), this is the call that
would fail, hence the explicit error guidance below.
"""
function fresh_resolve()
    isfile(MANIFEST) && mv(MANIFEST, BACKUP; force = true)
    isfile(BACKUP) && println("Existing Manifest.toml moved to Manifest.toml.bak\n")
    try
        Pkg.instantiate()          # no Manifest ⇒ resolves and writes a new one
    catch err
        println()
        @error "Fresh resolve failed — restoring the previous Manifest.toml" exception = err
        isfile(BACKUP) && mv(BACKUP, MANIFEST; force = true)
        println("""

        If the failure mentions DistanceDependentCRP, check that the General
        registry is reachable and up to date:

            julia --project=. -e 'using Pkg; Pkg.Registry.update()'

        The package is registered as DistanceDependentCRP (repo
        jmarsh96/DistanceDependentCRP.jl). Note this differs from the
        DDCRP.jl URL quoted in CLAUDE.md.
        """)
        rethrow()
    end
end

if FRESH
    fresh_resolve()
elseif UPDATE
    Pkg.update()
else
    if !isfile(MANIFEST)
        println("No Manifest.toml found — resolving from Project.toml instead.\n")
    end
    Pkg.instantiate()
end

println("\nPrecompiling (first run on a new machine takes several minutes)...")
Pkg.precompile()

println()
Pkg.status()

# ============================================================================
# Verify — loading the packages, not just installing them
# ============================================================================

println("\n" * "="^76)
println("Verifying the environment")
println("="^76)

# Every direct dependency. StatsPlots and JLD2 are the ones that actually tend
# to break on a cluster (binary artifacts and HDF5 respectively), so they are
# loaded rather than assumed.
const DEPS = [
    :CSV, :DataFrames, :DistanceDependentCRP, :Distances, :Distributed,
    :Distributions, :JLD2, :KernelDensity, :Printf, :Random,
    :SlurmClusterManager, :Statistics, :StatsBase, :StatsPlots,
]

failures = String[]
for dep in DEPS
    try
        @eval using $dep
        println("  ok    $dep")
    catch err
        println("  FAIL  $dep — $(first(sprint(showerror, err), 120))")
        push!(failures, String(dep))
    end
end

# The two locally-prototyped models. These exercise the parts of the package
# interface the analyses depend on, so a clean include is a real signal that the
# run scripts will start.
println()
for model in ["negbin_population_model.jl", "skewnormal_model.jl"]
    path = joinpath(REPO, "src", model)
    if !isfile(path)
        println("  skip  $model (not found)")
        continue
    end
    try
        Base.include(Main, path)
        println("  ok    $model")
    catch err
        println("  FAIL  $model — $(first(sprint(showerror, err), 120))")
        push!(failures, model)
    end
end

# The mapping step is R, not Julia, and is easy to forget until a job fails two
# hours in. Report it here rather than leaving it to be discovered late.
println()
r_ok = try
    success(pipeline(`Rscript -e 'stopifnot(requireNamespace("sf", quietly=TRUE),
                                            requireNamespace("ggplot2", quietly=TRUE),
                                            requireNamespace("dplyr", quietly=TRUE))'`,
                     devnull, devnull))
catch
    false
end
println(r_ok ? "  ok    R + sf/ggplot2/dplyr (needed by src/map_*.R)" :
               "  warn  R with sf/ggplot2/dplyr not available — the map_*.R steps will fail\n" *
               "        (the Julia run/analyse pipeline is unaffected)")

println("\n" * "="^76)
if isempty(failures)
    println("Environment OK. Next:")
    println("  julia --project=. src/run_dengue.jl --test")
    println("  julia --project=. src/run_aqs.jl --test --region ca")
    println("="^76)
else
    println("FAILED to load: ", join(failures, ", "))
    println("Try re-running with --fresh to re-resolve from Project.toml.")
    println("="^76)
    exit(1)
end

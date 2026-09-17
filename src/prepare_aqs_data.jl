# ============================================================================
# prepare_aqs_data.jl
#
# Builds the monitor-level PM2.5 dataset used by `run_aqs.jl`.
#
# Source: EPA AQS "pre-generated" daily summary file for parameter 88101
# (PM2.5 — Local Conditions), which is public and needs no API key:
#   https://aqs.epa.gov/aqsweb/airdata/daily_88101_<year>.zip
#
# One row of the daily file is one monitor-day. We keep continuous-monitor
# records (`Sample Duration == "1 HOUR"`, whose `Arithmetic Mean` is that day's
# mean of the 24 hourly values) with no exceptional-event flag, average across
# co-located instruments (POC) within a monitor-day, and require a site to have
# a reasonably complete year.
#
# Each site then contributes ONE response value, so that the ddCRP clusters
# monitoring sites over great-circle distance. Two candidate responses are
# written and `run_aqs.jl` picks one:
#
#   y_mean_log  mean of log daily PM2.5 over the year   (typical exposure)
#   y_p98       98th percentile of daily PM2.5          (the NAAQS design-value
#                                                        metric; right-skewed
#                                                        across sites, which is
#                                                        what the SkewNormal
#                                                        likelihood is for)
#
# 2023 is a deliberate choice: the June 2023 Canadian wildfire smoke episode
# produced a strong, spatially coherent anomaly over the Northeast and Midwest,
# so there is real spatial structure to find.
#
# Usage:
#   julia --project=. src/prepare_aqs_data.jl --year 2023 --region conus
#   julia --project=. src/prepare_aqs_data.jl --year 2023 --region west
# ============================================================================

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CSV, DataFrames, Downloads, Statistics, StatsBase, Printf

# Downloads (the 305 MB daily file, the Census zip) go to data/raw/, which is
# not committed; what the run and map scripts read goes to data/cleaned/.
const RAW_DIR  = joinpath(@__DIR__, "..", "data", "raw")
const DATA_DIR = joinpath(@__DIR__, "..", "data", "cleaned")

# Non-contiguous states and territories, excluded from every region so that
# great-circle distances describe one connected land mass.
const OFFSHORE = Set(["02", "15", "72", "78", "66", "69", "60", "80"])

# FIPS state codes by region.
const WEST = Set(["06", "41", "53", "04", "32", "16", "30", "56", "08", "49", "35"])
const REGIONS = Dict(
    "conus" => nothing,                 # everything except OFFSHORE
    "west"  => WEST,
    "east"  => nothing,                 # handled as "CONUS minus WEST"
    "ca"    => Set(["06"]),
)

function parse_args(args)
    year, region, force = 2023, "conus", false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--year";     year = parse(Int, args[i+1]); i += 2
        elseif a == "--region"; region = args[i+1]; i += 2
        elseif a == "--force";  force = true; i += 1
        else; error("unrecognised argument: $a")
        end
    end
    haskey(REGIONS, region) || error("region must be one of $(keys(REGIONS))")
    return (year = year, region = region, force = force)
end

# ============================================================================
# Download / extract
# ============================================================================

"""
    daily_file(year; force=false) -> path

Path to the extracted daily summary CSV, downloading and unzipping it on first
use. The archive is ~10 MB compressed and ~320 MB extracted, so both are cached.
"""
function daily_file(year::Int; force::Bool = false)
    csv = joinpath(RAW_DIR, "daily_88101_$(year).csv")
    (!force && isfile(csv)) && return csv

    zipf = joinpath(RAW_DIR, "daily_88101_$(year).zip")
    if force || !isfile(zipf)
        url = "https://aqs.epa.gov/aqsweb/airdata/daily_88101_$(year).zip"
        println("Downloading $url ...")
        Downloads.download(url, zipf; timeout = 600)
        @printf("  %.1f MB\n", filesize(zipf) / 1e6)
    end
    println("Extracting...")
    run(`unzip -o -q $zipf -d $RAW_DIR`)
    isfile(csv) || error("expected $csv inside the archive")
    return csv
end

"""
    state_boundaries(; force=false)

Cache the Census cartographic-boundary shapefile of US states, used only by
`src/map_aqs.R`. Failure here is non-fatal for the MCMC pipeline.
"""
function state_boundaries(; force::Bool = false)
    zipf = joinpath(RAW_DIR, "cb_2023_us_state_20m.zip")
    shp  = joinpath(DATA_DIR, "cb_2023_us_state_20m", "cb_2023_us_state_20m.shp")
    (!force && isfile(shp)) && (println("State boundaries already cached."); return shp)
    try
        mkpath(RAW_DIR); mkpath(DATA_DIR)
        println("Downloading US state boundaries...")
        Downloads.download("https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip",
                           zipf; timeout = 300)
        run(`unzip -o -q $zipf -d $(joinpath(DATA_DIR, "cb_2023_us_state_20m"))`)
        println("  saved $shp")
    catch err
        @warn "state boundary download failed — maps will fall back to plain points" err
    end
    return shp
end

# ============================================================================
# Site summaries
# ============================================================================

"""
    site_summaries(csv, region; min_days=300) -> DataFrame

Reduce the monitor-day file to one row per monitoring site.
"""
function site_summaries(csv::AbstractString, region::AbstractString; min_days::Int = 300)
    cols = ["State Code", "County Code", "Site Num", "POC", "Latitude", "Longitude",
            "Sample Duration", "Event Type", "Date Local", "Arithmetic Mean", "State Name"]
    println("Reading daily file (this takes a minute)...")
    df = CSV.read(csv, DataFrame;
                  select = cols,
                  types = Dict("State Code" => String, "County Code" => String,
                               "Site Num" => String))
    rename!(df, Dict("State Code" => :state, "County Code" => :county, "Site Num" => :site,
                     "POC" => :poc, "Latitude" => :lat, "Longitude" => :lon,
                     "Sample Duration" => :duration, "Event Type" => :event,
                     "Date Local" => :date, "Arithmetic Mean" => :pm25,
                     "State Name" => :state_name))
    @printf("  %d monitor-day records\n", nrow(df))

    # Continuous monitors only, no exceptional-event duplicates, valid values.
    df = df[(df.duration .== "1 HOUR") .& (df.event .== "None") .&
            .!ismissing.(df.pm25) .& (df.pm25 .> 0), :]
    @printf("  %d after keeping continuous-monitor, non-event, positive records\n", nrow(df))

    # Regional filter.
    df = df[.!in.(df.state, Ref(OFFSHORE)), :]
    if region == "west"
        df = df[in.(df.state, Ref(WEST)), :]
    elseif region == "east"
        df = df[.!in.(df.state, Ref(WEST)), :]
    elseif region != "conus"
        df = df[in.(df.state, Ref(REGIONS[region])), :]
    end
    @printf("  %d after the '%s' regional filter\n", nrow(df), region)

    # Average co-located instruments (POC) within a site-day, so each site-day
    # counts once regardless of how many instruments the site operates.
    df.site_id = string.(df.state, "-", df.county, "-", df.site)
    daily = combine(groupby(df, [:site_id, :date]), :pm25 => mean => :pm25)

    meta = combine(groupby(df, :site_id),
                   :lat => first => :lat, :lon => first => :lon,
                   :state => first => :state, :state_name => first => :state_name)

    per_site = combine(groupby(daily, :site_id),
                       :pm25 => length => :n_days,
                       :pm25 => (v -> mean(log.(v))) => :y_mean_log,
                       :pm25 => (v -> quantile(v, 0.98)) => :y_p98,
                       :pm25 => mean => :pm25_mean,
                       :pm25 => maximum => :pm25_max)

    out = innerjoin(per_site, meta, on = :site_id)
    n_before = nrow(out)
    out = out[out.n_days .>= min_days, :]
    @printf("  %d sites, %d with >= %d valid days\n", n_before, nrow(out), min_days)

    # Two sites at identical coordinates would give a zero distance and a
    # degenerate ddCRP link, so keep one per coordinate pair.
    out = combine(first, groupby(out, [:lat, :lon]))
    sort!(out, :site_id)
    return out
end

# ============================================================================
# Main
# ============================================================================

function main(args)
    opts = parse_args(args)
    mkpath(DATA_DIR); mkpath(RAW_DIR)

    outfile = joinpath(DATA_DIR, "aqs_pm25_$(opts.year)_$(opts.region).csv")
    if isfile(outfile) && !opts.force
        df = CSV.read(outfile, DataFrame)
        println("Dataset already cached: $outfile ($(nrow(df)) sites)")
        println("Pass --force to rebuild.")
        state_boundaries()
        summarise(df, opts)
        return
    end

    csv = daily_file(opts.year; force = opts.force)
    df = site_summaries(csv, opts.region)
    CSV.write(outfile, df)
    println("Wrote $outfile ($(nrow(df)) sites)")

    state_boundaries()
    summarise(df, opts)
end

function summarise(df::DataFrame, opts)
    println("\n--- EPA AQS PM2.5, $(opts.year), region '$(opts.region)' ---")
    @printf("  monitoring sites : %d\n", nrow(df))
    @printf("  states           : %d\n", length(unique(df.state)))
    @printf("  valid days/site  : min %d, median %.0f\n",
            minimum(df.n_days), median(df.n_days))
    println()
    # The SkewNormal likelihood only earns its keep if the response is genuinely
    # skewed ACROSS SITES — this is what decides which column run_aqs.jl uses.
    for (name, v) in [("y_mean_log (mean log daily PM2.5)", df.y_mean_log),
                      ("y_p98      (98th pct daily PM2.5)", df.y_p98),
                      ("log y_p98", log.(df.y_p98))]
        m, s = mean(v), std(v)
        g = mean(((x - m) / s)^3 for x in v)
        @printf("  %-36s mean %7.3f  sd %6.3f  skewness %6.3f\n", name, m, s, g)
    end
    println("\n  (SkewNormal's attainable skewness is |γ₁| < 0.995; a response whose")
    println("   skewness is near zero would make the skew parameter unidentifiable.)")
end

main(ARGS)

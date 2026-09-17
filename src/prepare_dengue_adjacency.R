# ============================================================================
# prepare_dengue_adjacency.R
#
# Build the municipal contiguity graph for one Brazilian state, as an edge list
# that `src/dengue_sensitivity.jl` reads. Julia cannot open the IBGE polygon
# mesh, and spatial work in this project uses R + sf, so the graph is built
# here and cached alongside the other derived datasets in `data/cleaned/`.
#
# Queen contiguity: two municipalities are adjacent if their boundaries touch
# at all, corner contacts included (`sf::st_touches`). Rook contiguity would
# drop the corner contacts, which for IBGE geometry are usually digitisation
# artefacts of a genuinely shared edge rather than real point contacts.
#
# The graph must be CONNECTED for hop distance to be finite everywhere, so any
# component left isolated by island or sliver geometry is joined to its nearest
# neighbouring component by centroid distance, and the added edges are reported.
#
# Writes data/cleaned/dengue_uf<UF>_adjacency.csv (geocode_i, geocode_j; each undirected
# edge once, i < j). Skips work if that file exists; pass --force to rebuild.
#
# Usage:
#   Rscript src/prepare_dengue_adjacency.R
#   Rscript src/prepare_dengue_adjacency.R --uf 41 --force
# ============================================================================

for (p in c("sf", "dplyr")) {
  if (!requireNamespace(p, quietly = TRUE)) {
    stop("Required R package '", p, "' is not installed. ",
         "Install it with install.packages('", p, "').")
  }
}
suppressPackageStartupMessages({
  library(sf); library(dplyr)
})

args  <- commandArgs(trailingOnly = TRUE)
force <- "--force" %in% args

arg_value <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[i + 1L]
}
uf <- as.integer(arg_value("--uf", "41"))

# Locate the repo root from the --file= path Rscript passes, falling back to the
# working directory, so the script runs from either the repo root or src/.
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
repo <- if (length(script_path) == 1) {
  normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)
} else {
  normalizePath(".")
}
if (!dir.exists(file.path(repo, "src"))) repo <- normalizePath(".")

out_file <- file.path(repo, "data", "cleaned", sprintf("dengue_uf%d_adjacency.csv", uf))
if (file.exists(out_file) && !force) {
  cat("Already present:", out_file, "\n")
  cat("Pass --force to rebuild.\n")
  quit(status = 0)
}

# ---------------------------------------------------------------------------
# Geometry, restricted to the municipalities the dengue dataset actually uses
# ---------------------------------------------------------------------------

mesh_file <- file.path(repo, "data", "cleaned", sprintf("malha_uf%d.geojson", uf))
if (!file.exists(mesh_file)) {
  stop("Missing ", mesh_file,
       "\nRun: julia --project=. src/prepare_dengue_data.jl --uf ", uf)
}

data_files <- list.files(file.path(repo, "data", "cleaned"),
                         pattern = sprintf("^dengue_uf%d_[0-9]+\\.csv$", uf),
                         full.names = TRUE)
if (length(data_files) == 0) {
  stop("Missing data/cleaned/dengue_uf", uf, "_<year>.csv",
       "\nRun: julia --project=. src/prepare_dengue_data.jl --uf ", uf)
}
obs <- read.csv(data_files[1])
cat(sprintf("Municipality set from %s: %d rows\n",
            basename(data_files[1]), nrow(obs)))

g <- st_read(mesh_file, quiet = TRUE)
# The IBGE mesh keys municipalities by the 7-digit geocode, exposed as
# `codarea` in the v3 API response.
key <- intersect(c("codarea", "CD_MUN", "code_muni"), names(g))[1]
if (is.na(key)) stop("No geocode column in ", mesh_file)
g$geocode <- as.numeric(as.character(g[[key]]))

# Match, and order, to the dengue CSV: the Julia side indexes 1..n in that
# order, so row i of the edge list must mean the same municipality there.
g <- g[match(obs$geocode, g$geocode), ]
missing_geom <- sum(is.na(st_dimension(g)))
if (any(is.na(g$geocode)) || missing_geom > 0) {
  stop(sum(is.na(g$geocode)), " municipalities have no polygon in the mesh; ",
       "the contiguity graph would be incomplete.")
}
g <- st_make_valid(g)
n <- nrow(g)

# ---------------------------------------------------------------------------
# Queen contiguity
# ---------------------------------------------------------------------------

cat("Computing queen contiguity...\n")
suppressMessages(nb <- st_touches(g, sparse = TRUE))

edges <- do.call(rbind, lapply(seq_len(n), function(i) {
  j <- nb[[i]]
  j <- j[j > i]
  if (length(j) == 0) NULL else data.frame(i = i, j = j)
}))
if (is.null(edges)) edges <- data.frame(i = integer(0), j = integer(0))

# ---------------------------------------------------------------------------
# Connectivity: hop distance must be finite between every pair
# ---------------------------------------------------------------------------

# Connected components by iterative label propagation over the edge list. Kept
# in base R rather than adding an igraph dependency for a dozen lines.
components <- function(n, edges) {
  lab <- seq_len(n)
  repeat {
    old <- lab
    if (nrow(edges) > 0) {
      m <- pmin(lab[edges$i], lab[edges$j])
      for (k in seq_len(nrow(edges))) {
        lab[edges$i[k]] <- min(lab[edges$i[k]], m[k])
        lab[edges$j[k]] <- min(lab[edges$j[k]], m[k])
      }
      lab <- lab[lab]  # path compression through the label map
    }
    if (identical(lab, old)) break
  }
  match(lab, sort(unique(lab)))
}

cent <- suppressWarnings(st_coordinates(st_centroid(st_geometry(g))))
comp <- components(n, edges)
n_comp0 <- max(comp)
n_added <- 0L

while (max(comp) > 1) {
  # Join the smallest component to whichever other component holds its nearest
  # municipality by centroid great-circle distance.
  sizes <- tabulate(comp)
  target <- which.min(sizes)
  inside  <- which(comp == target)
  outside <- which(comp != target)
  d <- st_distance(st_as_sf(as.data.frame(cent[inside, , drop = FALSE]),
                            coords = c("X", "Y"), crs = 4326),
                   st_as_sf(as.data.frame(cent[outside, , drop = FALSE]),
                            coords = c("X", "Y"), crs = 4326))
  hit <- arrayInd(which.min(d), dim(d))
  a <- inside[hit[1]]; b <- outside[hit[2]]
  edges <- rbind(edges, data.frame(i = min(a, b), j = max(a, b)))
  n_added <- n_added + 1L
  cat(sprintf("  bridged component %d (%d municipalities): %s -- %s (%.1f km)\n",
              target, length(inside), obs$name[a], obs$name[b],
              as.numeric(min(d)) / 1000))
  comp <- components(n, edges)
}

edges <- edges[order(edges$i, edges$j), ]
deg <- tabulate(c(edges$i, edges$j), nbins = n)

cat(sprintf("Graph: %d nodes, %d edges, mean degree %.2f (min %d, max %d), %d component%s%s\n",
            n, nrow(edges), 2 * nrow(edges) / n, min(deg), max(deg), n_comp0,
            if (n_comp0 == 1) "" else "s",
            if (n_added > 0) sprintf(", %d bridging edge%s added",
                                     n_added, if (n_added == 1) "" else "s") else ""))

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

out <- data.frame(geocode_i = obs$geocode[edges$i],
                  geocode_j = obs$geocode[edges$j])
write.csv(out, out_file, row.names = FALSE)
cat("Wrote", out_file, "\n")

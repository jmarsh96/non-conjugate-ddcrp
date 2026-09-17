# ============================================================================
# map_dengue.R
#
# Spatial maps for the Brazil dengue application. Run AFTER
# src/analyse_dengue.jl, which writes `cluster_assignments.csv`.
#
# Spatial work in this project uses R + sf (not Julia, not geopandas), so the
# maps live here rather than in the analyse script.
#
# Produces, in results/real_data/dengue/figures/:
#   map_clusters.png     point-estimate clustering, municipalities coloured by
#                        cluster and annotated with cluster-level incidence
#   map_uncertainty.png  how confidently each municipality is placed (mean
#                        posterior co-clustering probability with its cluster)
#   map_incidence.png    the raw 2024 incidence, for comparison
#
# Usage:
#   Rscript src/map_dengue.R
#   Rscript src/map_dengue.R --test
# ============================================================================

for (p in c("sf", "ggplot2", "dplyr")) {
  if (!requireNamespace(p, quietly = TRUE)) {
    stop("Required R package '", p, "' is not installed. ",
         "Install it with install.packages('", p, "').")
  }
}
suppressPackageStartupMessages({
  library(sf); library(ggplot2); library(dplyr)
})

args    <- commandArgs(trailingOnly = TRUE)
is_test <- "--test" %in% args

# Locate the repo root from the --file= path Rscript passes, falling back to the
# working directory, so the script runs from either the repo root or src/.
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
repo <- if (length(script_path) == 1) {
  normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)
} else {
  normalizePath(".")
}
if (!dir.exists(file.path(repo, "src"))) repo <- normalizePath(".")

outdir <- file.path(repo, "results", "real_data",
                    if (is_test) "dengue_test" else "dengue")
figdir <- file.path(outdir, "figures")
dir.create(figdir, recursive = TRUE, showWarnings = FALSE)

assign_file <- file.path(outdir, "cluster_assignments.csv")
if (!file.exists(assign_file)) {
  stop("Missing ", assign_file, "\nRun src/analyse_dengue.jl first.")
}
cl <- read.csv(assign_file)
cl$cluster <- factor(cl$cluster)
cat(sprintf("Loaded %d municipalities in %d clusters.\n",
            nrow(cl), nlevels(cl$cluster)))

# ---------------------------------------------------------------------------
# Geometry: IBGE municipal boundaries when available, otherwise centroids only.
# ---------------------------------------------------------------------------

uf_files <- list.files(file.path(repo, "data", "cleaned"), pattern = "^malha_uf[0-9]+\\.geojson$",
                       full.names = TRUE)
polys <- NULL
if (length(uf_files) > 0) {
  polys <- tryCatch({
    g <- st_read(uf_files[1], quiet = TRUE)
    # The IBGE mesh keys municipalities by the 7-digit geocode, exposed as
    # `codarea` in the v3 API response.
    key <- intersect(c("codarea", "CD_MUN", "code_muni"), names(g))[1]
    if (is.na(key)) NULL else { g$geocode <- as.numeric(as.character(g[[key]])); g }
  }, error = function(e) { message("Boundary read failed: ", conditionMessage(e)); NULL })
}

if (!is.null(polys)) {
  map_df <- left_join(polys, cl, by = "geocode")
  cat(sprintf("Using municipal polygons (%d matched).\n", sum(!is.na(map_df$cluster))))
  geom_layer <- function(fill_var)
    geom_sf(aes(fill = .data[[fill_var]]), colour = "grey30", linewidth = 0.08)
} else {
  message("No boundary mesh — falling back to centroid points.")
  map_df <- st_as_sf(cl, coords = c("lon", "lat"), crs = 4326)
  geom_layer <- function(fill_var)
    geom_sf(aes(colour = .data[[fill_var]]), size = 1.6)
}

base_theme <- theme_minimal(base_size = 11) +
  theme(axis.text = element_blank(), axis.title = element_blank(),
        panel.grid = element_blank())

# ---------------------------------------------------------------------------
# 1. Point-estimate clustering
# ---------------------------------------------------------------------------

# Order clusters by their fitted incidence so the palette reads low → high
# rather than in arbitrary label order.
ord <- cl %>% group_by(cluster) %>%
  summarise(rho = median(rho_mean, na.rm = TRUE), n = n(), .groups = "drop") %>%
  arrange(rho)
map_df$cluster <- factor(map_df$cluster, levels = as.character(ord$cluster))

p1 <- ggplot(map_df) + geom_layer("cluster") +
  scale_fill_viridis_d(option = "turbo", na.value = "white", name = "cluster") +
  scale_colour_viridis_d(option = "turbo", na.value = "white", name = "cluster") +
  labs(title = "ddCRP point-estimate clustering of 2024 dengue incidence",
       subtitle = sprintf("%d municipalities, %d clusters; Negative Binomial likelihood",
                          nrow(cl), nlevels(cl$cluster))) +
  base_theme
ggsave(file.path(figdir, "map_clusters.png"), p1, width = 9, height = 8, dpi = 150)

# ---------------------------------------------------------------------------
# 2. Clustering uncertainty
# ---------------------------------------------------------------------------

p2 <- ggplot(map_df) + geom_layer("cocluster_confidence") +
  scale_fill_viridis_c(option = "magma", limits = c(0, 1), na.value = "white",
                       name = "co-clustering\nprobability") +
  scale_colour_viridis_c(option = "magma", limits = c(0, 1), na.value = "white",
                         name = "co-clustering\nprobability") +
  labs(title = "Posterior clustering uncertainty",
       subtitle = "Mean posterior co-clustering probability with own cluster; low = unstable placement") +
  base_theme
ggsave(file.path(figdir, "map_uncertainty.png"), p2, width = 9, height = 8, dpi = 150)

# ---------------------------------------------------------------------------
# 3. Raw incidence, for comparison with the fitted clusters
# ---------------------------------------------------------------------------

map_df$log_incidence <- log10(pmax(map_df$incidence, 1))
p3 <- ggplot(map_df) + geom_layer("log_incidence") +
  scale_fill_viridis_c(option = "inferno", na.value = "white",
                       name = expression(log[10]~"incidence/100k")) +
  scale_colour_viridis_c(option = "inferno", na.value = "white",
                         name = expression(log[10]~"incidence/100k")) +
  labs(title = "Observed 2024 dengue incidence",
       subtitle = "Cases per 100,000 inhabitants") +
  base_theme
ggsave(file.path(figdir, "map_incidence.png"), p3, width = 9, height = 8, dpi = 150)

cat("\nCluster summary (ordered by fitted incidence):\n")
print(as.data.frame(ord %>% mutate(rho = round(rho, 1))))
cat("\nMaps written to ", figdir, "\n", sep = "")

# ============================================================================
# map_aqs.R
#
# Spatial maps for the EPA AQS PM2.5 application. Run AFTER src/analyse_aqs.jl,
# which writes `cluster_assignments.csv`.
#
# Spatial work in this project uses R + sf (not Julia, not geopandas).
#
# Produces, in results/real_data/aqs/figures/:
#   map_clusters.png     monitors coloured by point-estimate cluster
#   map_uncertainty.png  posterior co-clustering confidence per monitor
#   map_response.png     the observed 98th-percentile PM2.5, for comparison
#
# Usage:
#   Rscript src/map_aqs.R
#   Rscript src/map_aqs.R --test
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

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
repo <- if (length(script_path) == 1) {
  normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)
} else {
  normalizePath(".")
}
if (!dir.exists(file.path(repo, "src"))) repo <- normalizePath(".")

outdir <- file.path(repo, "results", "real_data", if (is_test) "aqs_test" else "aqs")
figdir <- file.path(outdir, "figures")
dir.create(figdir, recursive = TRUE, showWarnings = FALSE)

assign_file <- file.path(outdir, "cluster_assignments.csv")
if (!file.exists(assign_file)) {
  stop("Missing ", assign_file, "\nRun src/analyse_aqs.jl first.")
}
cl <- read.csv(assign_file)
cl$cluster <- factor(cl$cluster)
cat(sprintf("Loaded %d monitors in %d clusters.\n", nrow(cl), nlevels(cl$cluster)))

sites <- st_as_sf(cl, coords = c("lon", "lat"), crs = 4326)

# ---------------------------------------------------------------------------
# State outlines as a basemap, when the cached shapefile is present.
# ---------------------------------------------------------------------------

shp <- file.path(repo, "data", "cleaned", "cb_2023_us_state_20m", "cb_2023_us_state_20m.shp")
states <- if (file.exists(shp)) {
  tryCatch({
    g <- st_read(shp, quiet = TRUE)
    # Keep the contiguous states only, matching the analysis extent.
    g[!(g$STUSPS %in% c("AK", "HI", "PR", "VI", "GU", "MP", "AS")), ]
  }, error = function(e) { message("Basemap read failed: ", conditionMessage(e)); NULL })
} else {
  message("No cached state shapefile — plotting monitors without a basemap.")
  NULL
}

basemap <- if (is.null(states)) NULL else
  geom_sf(data = states, fill = "grey96", colour = "grey70", linewidth = 0.2)

# Crop the view to the monitors actually being analysed, so a regional run
# (e.g. --region west) does not render as a mostly-empty national map.
bb <- st_bbox(sites)
pad <- 1.5
coords_layer <- coord_sf(xlim = c(bb["xmin"] - pad, bb["xmax"] + pad),
                         ylim = c(bb["ymin"] - pad, bb["ymax"] + pad),
                         expand = FALSE)

base_theme <- theme_minimal(base_size = 11) +
  theme(panel.grid = element_blank(), axis.title = element_blank())

# ---------------------------------------------------------------------------
# 1. Point-estimate clustering
# ---------------------------------------------------------------------------

# Order clusters by fitted location so the palette reads low → high.
ord <- cl %>% group_by(cluster) %>%
  summarise(xi = median(xi_mean), n = n(), .groups = "drop") %>% arrange(xi)
sites$cluster <- factor(sites$cluster, levels = as.character(ord$cluster))

p1 <- ggplot() + basemap +
  geom_sf(data = sites, aes(colour = cluster), size = 1.9) +
  scale_colour_viridis_d(option = "turbo", name = "cluster") +
  coords_layer +
  labs(title = "ddCRP point-estimate clustering of PM2.5 monitoring sites",
       subtitle = sprintf("%d monitors, %d clusters; SkewNormal likelihood, 2023 98th-percentile PM2.5",
                          nrow(cl), nlevels(sites$cluster))) +
  base_theme
ggsave(file.path(figdir, "map_clusters.png"), p1, width = 11, height = 7.5, dpi = 150)

# ---------------------------------------------------------------------------
# 2. Clustering uncertainty
# ---------------------------------------------------------------------------

p2 <- ggplot() + basemap +
  geom_sf(data = sites, aes(colour = cocluster_confidence), size = 1.9) +
  scale_colour_viridis_c(option = "magma", limits = c(0, 1),
                         name = "co-clustering\nprobability") +
  coords_layer +
  labs(title = "Posterior clustering uncertainty",
       subtitle = "Mean posterior co-clustering probability with own cluster") +
  base_theme
ggsave(file.path(figdir, "map_uncertainty.png"), p2, width = 11, height = 7.5, dpi = 150)

# ---------------------------------------------------------------------------
# 3. Observed response
# ---------------------------------------------------------------------------

p3 <- ggplot() + basemap +
  geom_sf(data = sites, aes(colour = y), size = 1.9) +
  scale_colour_viridis_c(option = "inferno", name = "98th pct\nPM2.5 (µg/m³)") +
  coords_layer +
  labs(title = "Observed 2023 PM2.5 design-value metric",
       subtitle = "98th percentile of daily mean PM2.5 at each monitor") +
  base_theme
ggsave(file.path(figdir, "map_response.png"), p3, width = 11, height = 7.5, dpi = 150)

cat("\nCluster summary (ordered by fitted location ξ):\n")
print(as.data.frame(ord %>% mutate(xi = round(xi, 2))))
cat("\nMaps written to ", figdir, "\n", sep = "")

# ============================================================================
# map_dengue_mode.R
#
# Cluster map for the POSTERIOR-MODE (MAP-at-mode) dengue partition, from
# cluster_assignments_mode.csv written by src/dengue_mode_map.jl. Mirrors the
# cluster-map panel of map_dengue.R but reads the mode CSV and writes a separate
# figure (map_clusters_mode.png), so the existing map_clusters.png is untouched.
#
#   Rscript src/map_dengue_mode.R
# ============================================================================

for (p in c("sf", "ggplot2", "dplyr")) {
  if (!requireNamespace(p, quietly = TRUE)) {
    stop("Required R package '", p, "' is not installed. ",
         "Install it with install.packages('", p, "').")
  }
}
suppressPackageStartupMessages({ library(sf); library(ggplot2); library(dplyr) })

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
repo <- if (length(script_path) == 1) {
  normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)
} else {
  normalizePath(".")
}
if (!dir.exists(file.path(repo, "src"))) repo <- normalizePath(".")

outdir <- file.path(repo, "results", "real_data", "dengue")
figdir <- file.path(outdir, "figures")
dir.create(figdir, recursive = TRUE, showWarnings = FALSE)

assign_file <- file.path(outdir, "cluster_assignments_mode.csv")
if (!file.exists(assign_file)) {
  stop("Missing ", assign_file, "\nRun src/dengue_mode_map.jl first.")
}
cl <- read.csv(assign_file)
cl$cluster <- factor(cl$cluster)
cat(sprintf("Loaded %d municipalities in %d clusters.\n",
            nrow(cl), nlevels(cl$cluster)))

# Geometry: IBGE municipal boundaries when available, else centroid points.
uf_files <- list.files(file.path(repo, "data", "cleaned"), pattern = "^malha_uf[0-9]+\\.geojson$",
                       full.names = TRUE)
polys <- NULL
if (length(uf_files) > 0) {
  polys <- tryCatch({
    g <- st_read(uf_files[1], quiet = TRUE)
    key <- intersect(c("codarea", "CD_MUN", "code_muni"), names(g))[1]
    if (is.na(key)) NULL else { g$geocode <- as.numeric(as.character(g[[key]])); g }
  }, error = function(e) { message("Boundary read failed: ", conditionMessage(e)); NULL })
}

if (!is.null(polys)) {
  map_df <- left_join(polys, cl, by = "geocode")
  cat(sprintf("Using municipal polygons (%d matched).\n", sum(!is.na(map_df$cluster))))
  geom_layer <- function(v) geom_sf(aes(fill = .data[[v]]), colour = "grey30", linewidth = 0.08)
} else {
  message("No boundary mesh — falling back to centroid points.")
  map_df <- st_as_sf(cl, coords = c("lon", "lat"), crs = 4326)
  geom_layer <- function(v) geom_sf(aes(colour = .data[[v]]), size = 1.6)
}

base_theme <- theme_minimal(base_size = 11) +
  theme(axis.text = element_blank(), axis.title = element_blank(),
        panel.grid = element_blank())

# Order clusters by fitted incidence so the palette reads low -> high.
ord <- cl %>% group_by(cluster) %>%
  summarise(rho = median(rho_mean, na.rm = TRUE), n = n(), .groups = "drop") %>%
  arrange(rho)
map_df$cluster <- factor(map_df$cluster, levels = as.character(ord$cluster))

# Legend labels carry each cluster's member count: "id (n members)".
size_lab <- setNames(sprintf("%s  (%d)", ord$cluster, ord$n), as.character(ord$cluster))

p1 <- ggplot(map_df) + geom_layer("cluster") +
  scale_fill_viridis_d(option = "turbo", na.value = "white",
                       name = "cluster (members)", labels = size_lab) +
  scale_colour_viridis_d(option = "turbo", na.value = "white",
                         name = "cluster (members)", labels = size_lab) +
  labs(title = "ddCRP posterior-mode clustering of 2024 dengue incidence",
       subtitle = sprintf("%d municipalities, %d clusters (MAP at the posterior mode of K); Negative Binomial likelihood",
                          nrow(cl), nlevels(cl$cluster))) +
  base_theme
ggsave(file.path(figdir, "map_clusters_mode.png"), p1, width = 9, height = 8, dpi = 150)

cat("\nCluster summary (ordered by fitted incidence):\n")
print(as.data.frame(ord %>% mutate(rho = round(rho, 1))))
cat("\nWrote ", file.path(figdir, "map_clusters_mode.png"), "\n", sep = "")

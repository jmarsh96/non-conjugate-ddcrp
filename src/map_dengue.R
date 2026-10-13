# Maps for the dengue application. Run after analyse_dengue.jl.
# Writes dengue_maps.png (observed and posterior mean incidence on one scale, with the Dahl
# partition as boundaries) and dengue_map_uncertainty.png to results/real_data/dengue/figures/,
# and copies them to docs/ddcrp-paper/figures/ unless --test.
#
#   Rscript src/map_dengue.R [--test]

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

# Locate the repo root from the --file= path Rscript passes.
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
cl$log_incidence <- log10(pmax(cl$incidence, 1))
cat(sprintf("Loaded %d municipalities in %d clusters.\n",
            nrow(cl), nlevels(cl$cluster)))

# Geometry: IBGE municipal boundaries when available, otherwise centroids only.

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
  fill_layer <- function(v, border = "grey30")
    geom_sf(aes(fill = .data[[v]]), colour = border, linewidth = 0.08)
} else {
  message("No boundary mesh; falling back to centroid points.")
  map_df <- st_as_sf(cl, coords = c("lon", "lat"), crs = 4326)
  fill_layer <- function(v, border = NA) geom_sf(aes(colour = .data[[v]]), size = 1.6)
}

base_theme <- theme_minimal(base_size = 13) +
  theme(axis.text = element_blank(), axis.title = element_blank(),
        panel.grid = element_blank(), legend.position = "bottom",
        legend.key.width = unit(1.6, "cm"), legend.key.height = unit(0.35, "cm"),
        plot.title = element_text(size = 13, hjust = 0.5))

# 1. Observed and posterior mean incidence, on one shared scale

ord <- cl %>% group_by(cluster) %>%
  summarise(rho = median(rho_mean, na.rm = TRUE), n = n(), .groups = "drop") %>%
  arrange(rho)
map_df$log_rho <- log10(map_df$rho_mean)
lims <- range(c(map_df$log_incidence, map_df$log_rho), na.rm = TRUE)

inc_scale <- list(
  scale_fill_viridis_c(option = "inferno", na.value = "white", limits = lims,
                       name = expression(log[10]~"cases per 100,000")),
  scale_colour_viridis_c(option = "inferno", na.value = "white", limits = lims,
                         name = expression(log[10]~"cases per 100,000")))

p_obs <- ggplot(map_df) + fill_layer("log_incidence") + inc_scale +
  labs(title = "Observed incidence") + base_theme

p_fit <- ggplot(map_df) + fill_layer("log_rho", border = NA) + inc_scale +
  labs(title = "Posterior mean incidence")
if (!is.null(polys)) {
  bounds <- map_df %>% filter(!is.na(cluster)) %>% group_by(cluster) %>%
    summarise(.groups = "drop")
  p_fit <- p_fit + geom_sf(data = bounds, fill = NA, colour = "white", linewidth = 0.35)
}
p_fit <- p_fit + base_theme

# Both panels in one device with grid viewports.
maps_file <- file.path(figdir, "dengue_maps.png")
png(maps_file, width = 13, height = 6, units = "in", res = 200, bg = "white")
grid::grid.newpage()
grid::pushViewport(grid::viewport(layout = grid::grid.layout(1, 2)))
print(p_obs, vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 1))
print(p_fit, vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 2))
invisible(dev.off())

# 2. Clustering uncertainty

# Singletons have no other members and are drawn in grey.
p_unc <- ggplot(map_df) + fill_layer("cocluster_confidence") +
  scale_fill_viridis_c(option = "magma", limits = c(0, 1), na.value = "grey70",
                       name = "co-clustering probability",
                       guide = guide_colourbar(title.position = "top", title.hjust = 0.5)) +
  scale_colour_viridis_c(option = "magma", limits = c(0, 1), na.value = "grey70",
                         name = "co-clustering probability",
                       guide = guide_colourbar(title.position = "top", title.hjust = 0.5)) +
  base_theme
unc_file <- file.path(figdir, "dengue_map_uncertainty.png")
ggsave(unc_file, p_unc, width = 6.5, height = 6, dpi = 200, bg = "white")

cat("\nCluster summary (ordered by fitted incidence):\n")
print(as.data.frame(ord %>% mutate(rho = round(rho, 1))))

if (is_test) {
  cat("\n--test: maps left in ", figdir, " (docs/ddcrp-paper/ untouched)\n", sep = "")
} else if (!dir.exists(file.path(repo, "docs", "ddcrp-paper", "figures"))) {
  cat("\nNo docs/ddcrp-paper/figures (docs/ is git-ignored): maps left in ", figdir, "\n", sep = "")
} else {
  paper_figs <- file.path(repo, "docs", "ddcrp-paper", "figures")
  file.copy(c(maps_file, unc_file), paper_figs, overwrite = TRUE)
  cat("\nMaps written to ", figdir, " and copied to ", paper_figs, "\n", sep = "")
}

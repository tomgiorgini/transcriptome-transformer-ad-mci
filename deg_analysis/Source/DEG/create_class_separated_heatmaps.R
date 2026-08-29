options(stringsAsFactors = FALSE)

script_dir <- normalizePath(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), winslash = "/", mustWork = TRUE)
source(file.path(script_dir, "deg_utils.R"))

args <- parse_cli_args(commandArgs(trailingOnly = TRUE))
project_root <- get_project_root()

deg_dirs <- if (!is.null(args[["deg-dirs"]])) {
  strsplit(args[["deg-dirs"]], ",", fixed = TRUE)[[1]]
} else {
  file.path(
    project_root,
    "Results",
    "DEG",
    c("AD_vs_MCI_top50_by_abs_logFC", "AD_vs_MCI_top100_by_abs_logFC")
  )
}

deg_dirs <- normalizePath(deg_dirs, winslash = "/", mustWork = TRUE)
ensure_package("pheatmap")

write_class_separated_heatmap <- function(deg_dir) {
  matrix_path <- file.path(deg_dir, "matrix_DEG.txt")
  if (!file.exists(matrix_path)) {
    stop(sprintf("Missing matrix_DEG.txt in %s", deg_dir), call. = FALSE)
  }
  settings_path <- file.path(deg_dir, "settings.tsv")
  settings <- if (file.exists(settings_path)) {
    read.delim(settings_path, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
  } else {
    data.frame()
  }

  matrix_deg <- read.table(
    matrix_path,
    header = TRUE,
    sep = "\t",
    check.names = FALSE,
    row.names = 1,
    quote = "",
    comment.char = ""
  )

  sample_names <- colnames(matrix_deg)
  control_group <- if ("control_group" %in% colnames(settings)) settings$control_group[[1]] else "MCI"
  case_group <- if ("case_group" %in% colnames(settings)) settings$case_group[[1]] else "AD"
  control_label <- if ("control_label" %in% colnames(settings)) settings$control_label[[1]] else control_group
  case_label <- if ("case_label" %in% colnames(settings)) settings$case_label[[1]] else case_group

  control_samples <- sample_names[grepl(paste0("_", control_group, "$"), sample_names)]
  case_samples <- sample_names[grepl(paste0("_", case_group, "$"), sample_names)]
  other_samples <- setdiff(sample_names, c(control_samples, case_samples))
  ordered_samples <- c(control_samples, case_samples, other_samples)

  if (length(control_samples) == 0 || length(case_samples) == 0) {
    stop(sprintf("Could not infer %s/%s samples from column names in %s", control_group, case_group, matrix_path), call. = FALSE)
  }

  ordered_matrix <- matrix_deg[, ordered_samples, drop = FALSE]
  condition <- c(
    rep(control_group, length(control_samples)),
    rep(case_group, length(case_samples)),
    rep("Other", length(other_samples))
  )
  sample_annotation <- data.frame(
    condition = factor(condition, levels = c(control_group, case_group, "Other")),
    row.names = ordered_samples,
    stringsAsFactors = FALSE
  )
  annotation_colors <- list(
    condition = setNames(c("forestgreen", "darkorange", "grey60"), c(control_group, case_group, "Other"))
  )

  gap_positions <- length(control_samples)
  if (length(other_samples) > 0) {
    gap_positions <- c(gap_positions, length(control_samples) + length(case_samples))
  }

  for (ext in c("pdf", "png")) {
    pheatmap::pheatmap(
      ordered_matrix,
      scale = "row",
      border_color = NA,
      cluster_cols = FALSE,
      cluster_rows = TRUE,
      clustering_distance_rows = "correlation",
      clustering_method = "complete",
      annotation_col = sample_annotation,
      annotation_colors = annotation_colors,
      color = colorRampPalette(c("blue", "blue3", "black", "yellow3", "yellow"))(100),
      show_rownames = FALSE,
      show_colnames = FALSE,
      gaps_col = gap_positions,
      width = 12,
      height = 10,
      filename = file.path(deg_dir, paste0("heatmap_class_separated.", ext))
    )
  }

  data.frame(
    deg_dir = deg_dir,
    n_genes = nrow(ordered_matrix),
    control_group = control_group,
    case_group = case_group,
    control_label = control_label,
    case_label = case_label,
    n_control = length(control_samples),
    n_case = length(case_samples),
    n_other = length(other_samples),
    pdf = file.path(deg_dir, "heatmap_class_separated.pdf"),
    png = file.path(deg_dir, "heatmap_class_separated.png"),
    stringsAsFactors = FALSE
  )
}

summary <- do.call(rbind, lapply(deg_dirs, write_class_separated_heatmap))
write.table(
  summary,
  file = file.path(dirname(deg_dirs[[1]]), "class_separated_heatmaps_summary.tsv"),
  row.names = FALSE,
  sep = "\t",
  quote = FALSE
)
print(summary)

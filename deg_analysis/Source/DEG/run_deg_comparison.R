options(stringsAsFactors = FALSE)

script_dir <- normalizePath(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), winslash = "/", mustWork = TRUE)
source(file.path(script_dir, "deg_utils.R"))

args <- parse_cli_args(commandArgs(trailingOnly = TRUE))
project_root <- get_project_root()
shared_data_root <- get_shared_alzheimer_data_dir()

comparison_id <- args$comparison
if (is.null(comparison_id)) {
  stop("Missing required argument --comparison", call. = FALSE)
}

manifest_path <- if (!is.null(args$config)) {
  normalizePath(args$config, winslash = "/", mustWork = TRUE)
} else {
  file.path(project_root, "Source", "DEG", "comparisons.tsv")
}

matrix_path <- if (!is.null(args$matrix)) {
  normalizePath(args$matrix, winslash = "/", mustWork = TRUE)
} else {
  file.path(shared_data_root, "matrix.txt")
}

groups_dir <- if (!is.null(args[["groups-dir"]])) {
  normalizePath(args[["groups-dir"]], winslash = "/", mustWork = TRUE)
} else {
  file.path(shared_data_root, "groups")
}

results_root <- if (!is.null(args[["results-dir"]])) {
  normalizePath(args[["results-dir"]], winslash = "/", mustWork = FALSE)
} else {
  file.path(project_root, "Results", "DEG")
}

skip_plots <- isTRUE(args[["skip-plots"]])

comparisons <- load_comparisons_manifest(manifest_path)
selected <- comparisons[comparisons$comparison_id == comparison_id, , drop = FALSE]

if (nrow(selected) != 1) {
  stop(sprintf("Comparison '%s' not found in manifest %s", comparison_id, manifest_path), call. = FALSE)
}

ensure_dir(results_root)

summary_table <- run_deg_analysis(
  comparison_row = selected,
  matrix_path = matrix_path,
  groups_dir = groups_dir,
  results_root = results_root,
  skip_plots = skip_plots
)

message(sprintf(
  "Completed %s: %d DEG (%d UP, %d DOWN)",
  summary_table$comparison_id,
  summary_table$n_deg_total,
  summary_table$n_deg_up,
  summary_table$n_deg_down
))


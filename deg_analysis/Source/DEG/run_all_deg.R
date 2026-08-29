options(stringsAsFactors = FALSE)

script_dir <- normalizePath(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), winslash = "/", mustWork = TRUE)
source(file.path(script_dir, "deg_utils.R"))

args <- parse_cli_args(commandArgs(trailingOnly = TRUE))
project_root <- get_project_root()
shared_data_root <- get_shared_alzheimer_data_dir()

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
dry_run <- isTRUE(args[["dry-run"]])

comparisons <- load_comparisons_manifest(manifest_path)

if (!is.null(args$comparisons)) {
  requested_ids <- trimws(strsplit(args$comparisons, ",", fixed = TRUE)[[1]])
  comparisons <- comparisons[comparisons$comparison_id %in% requested_ids, , drop = FALSE]

  missing_ids <- setdiff(requested_ids, comparisons$comparison_id)
  if (length(missing_ids) > 0) {
    stop(sprintf("Comparison ids not found in manifest: %s", paste(missing_ids, collapse = ", ")), call. = FALSE)
  }
}

if (nrow(comparisons) == 0) {
  stop("No comparisons selected.", call. = FALSE)
}

if (dry_run) {
  message(sprintf("Matrix path: %s [%s]", matrix_path, if (file.exists(matrix_path)) "OK" else "MISSING"))
  message(sprintf("Groups dir: %s [%s]", groups_dir, if (dir.exists(groups_dir)) "OK" else "MISSING"))
  message("Comparisons to run:")
  for (i in seq_len(nrow(comparisons))) {
    message(sprintf(
      "- %s (%s vs %s)",
      comparisons$comparison_id[[i]],
      comparisons$case_group[[i]],
      comparisons$control_group[[i]]
    ))
  }
  quit(save = "no", status = 0)
}

ensure_dir(results_root)

summaries <- vector("list", nrow(comparisons))

for (i in seq_len(nrow(comparisons))) {
  message(sprintf(
    "[%d/%d] Running %s",
    i,
    nrow(comparisons),
    comparisons$comparison_id[[i]]
  ))

  summaries[[i]] <- run_deg_analysis(
    comparison_row = comparisons[i, , drop = FALSE],
    matrix_path = matrix_path,
    groups_dir = groups_dir,
    results_root = results_root,
    skip_plots = skip_plots
  )
}

combined_summary <- do.call(rbind, summaries)
write.table(
  combined_summary,
  file = file.path(results_root, "deg_run_summary.tsv"),
  row.names = FALSE,
  sep = "\t",
  quote = FALSE
)

message("All DEG comparisons completed.")

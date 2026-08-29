options(stringsAsFactors = FALSE)

script_dir <- normalizePath(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), winslash = "/", mustWork = TRUE)
source(file.path(script_dir, "deg_utils.R"))
source(file.path(script_dir, "enrichment_utils.R"))

args <- parse_cli_args(commandArgs(trailingOnly = TRUE))
project_root <- get_project_root()

comparison_id <- args$comparison
if (is.null(comparison_id)) {
  stop("Missing required argument --comparison", call. = FALSE)
}

results_root <- if (!is.null(args[["results-dir"]])) {
  normalizePath(args[["results-dir"]], winslash = "/", mustWork = FALSE)
} else {
  file.path(project_root, "Results", "DEG")
}

comparison_dir <- file.path(results_root, comparison_id)
if (!dir.exists(comparison_dir)) {
  stop(sprintf("Comparison results directory not found: %s", comparison_dir), call. = FALSE)
}

top_term <- if (!is.null(args[["top-term"]])) as.integer(args[["top-term"]]) else 10L
thr_pval <- if (!is.null(args[["pval-threshold"]])) as.numeric(args[["pval-threshold"]]) else 0.05

summary_table <- run_single_comparison_enrichment(
  comparison_dir = comparison_dir,
  top_term = top_term,
  thr_pval = thr_pval
)

message(sprintf("Completed enrichment for %s (%d rows in summary)", comparison_id, nrow(summary_table)))


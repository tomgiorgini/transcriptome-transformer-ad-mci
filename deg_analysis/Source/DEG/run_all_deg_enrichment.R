options(stringsAsFactors = FALSE)

script_dir <- normalizePath(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), winslash = "/", mustWork = TRUE)
source(file.path(script_dir, "deg_utils.R"))
source(file.path(script_dir, "enrichment_utils.R"))

args <- parse_cli_args(commandArgs(trailingOnly = TRUE))
project_root <- get_project_root()

manifest_path <- if (!is.null(args$config)) {
  normalizePath(args$config, winslash = "/", mustWork = TRUE)
} else {
  file.path(project_root, "Source", "DEG", "comparisons.tsv")
}

results_root <- if (!is.null(args[["results-dir"]])) {
  normalizePath(args[["results-dir"]], winslash = "/", mustWork = FALSE)
} else {
  file.path(project_root, "Results", "DEG")
}

top_term <- if (!is.null(args[["top-term"]])) as.integer(args[["top-term"]]) else 10L
thr_pval <- if (!is.null(args[["pval-threshold"]])) as.numeric(args[["pval-threshold"]]) else 0.05

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
  stop("No comparisons selected for enrichment.", call. = FALSE)
}

all_summaries <- vector("list", nrow(comparisons))

for (i in seq_len(nrow(comparisons))) {
  comparison_id <- comparisons$comparison_id[[i]]
  message(sprintf("[%d/%d] Running enrichment for %s", i, nrow(comparisons), comparison_id))
  all_summaries[[i]] <- run_single_comparison_enrichment(
    comparison_dir = file.path(results_root, comparison_id),
    top_term = top_term,
    thr_pval = thr_pval
  )
  all_summaries[[i]]$comparison_id <- comparison_id
}

summary_table <- do.call(rbind, all_summaries)
summary_table <- summary_table[, c("comparison_id", "direction", "database", "n_terms", "plotted")]

write.table(
  summary_table,
  file = file.path(results_root, "deg_enrichment_summary.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

message("All DEG enrichment runs completed.")

options(stringsAsFactors = FALSE)

script_dir <- normalizePath(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), winslash = "/", mustWork = TRUE)
source(file.path(script_dir, "deg_utils.R"))

args <- parse_cli_args(commandArgs(trailingOnly = TRUE))
project_root <- get_project_root()

source_dir <- if (!is.null(args[["source-dir"]])) {
  normalizePath(args[["source-dir"]], winslash = "/", mustWork = TRUE)
} else {
  file.path(project_root, "Results", "DEG", "AD_vs_MCI")
}

output_root <- if (!is.null(args[["output-root"]])) {
  normalizePath(args[["output-root"]], winslash = "/", mustWork = FALSE)
} else {
  file.path(project_root, "Results", "DEG")
}

top_n_values <- if (!is.null(args[["top-n"]])) {
  as.integer(strsplit(args[["top-n"]], ",", fixed = TRUE)[[1]])
} else {
  c(50L, 100L)
}

prefix <- if (!is.null(args[["prefix"]])) args[["prefix"]] else "AD_vs_MCI_top"
skip_plots <- isTRUE(args[["skip-plots"]])

ensure_package("pheatmap")

deg_path <- file.path(source_dir, "DEG.txt")
matrix_path <- file.path(source_dir, "matrix_DEG.txt")
settings_path <- file.path(source_dir, "settings.tsv")
summary_path <- file.path(source_dir, "summary.tsv")

if (!file.exists(deg_path)) stop(sprintf("Missing DEG file: %s", deg_path), call. = FALSE)
if (!file.exists(matrix_path)) stop(sprintf("Missing DEG matrix: %s", matrix_path), call. = FALSE)
if (!file.exists(settings_path)) stop(sprintf("Missing settings file: %s", settings_path), call. = FALSE)

deg_table <- read.delim(deg_path, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
matrix_deg <- read.table(
  matrix_path,
  header = TRUE,
  sep = "\t",
  check.names = FALSE,
  row.names = 1,
  quote = "",
  comment.char = ""
)
settings <- read.delim(settings_path, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
source_summary <- if (file.exists(summary_path)) {
  read.delim(summary_path, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
} else {
  data.frame()
}
source_comparison_id <- if ("comparison_id" %in% colnames(settings)) {
  settings$comparison_id[[1]]
} else {
  basename(source_dir)
}

required_columns <- c("GeneSymbol", "ensembl_id", "pval", "adj_pval", "logFC", "direction")
missing_columns <- setdiff(required_columns, colnames(deg_table))
if (length(missing_columns) > 0) {
  stop(sprintf("DEG.txt missing columns: %s", paste(missing_columns, collapse = ", ")), call. = FALSE)
}

gene_keys <- ifelse(
  is.na(deg_table$ensembl_id) | !nzchar(deg_table$ensembl_id),
  deg_table$GeneSymbol,
  paste0(deg_table$GeneSymbol, "|", deg_table$ensembl_id)
)
matrix_gene_keys <- rownames(matrix_deg)

for (top_n in top_n_values) {
  if (is.na(top_n) || top_n <= 0) {
    stop(sprintf("Invalid top-n value: %s", top_n), call. = FALSE)
  }

  selected_n <- min(top_n, nrow(deg_table))
  selected_idx <- head(order(abs(deg_table$logFC), decreasing = TRUE), selected_n)
  selected_table <- deg_table[selected_idx, , drop = FALSE]
  selected_gene_keys <- gene_keys[selected_idx]

  missing_matrix_genes <- setdiff(selected_gene_keys, matrix_gene_keys)
  if (length(missing_matrix_genes) > 0) {
    stop(
      sprintf("Selected genes missing from matrix_DEG.txt. Example: %s", paste(head(missing_matrix_genes, 10), collapse = ", ")),
      call. = FALSE
    )
  }

  selected_matrix <- matrix_deg[selected_gene_keys, , drop = FALSE]
  rownames(selected_matrix) <- selected_gene_keys

  out_dir <- file.path(output_root, paste0(prefix, selected_n, "_by_abs_logFC"))
  ensure_dir(out_dir)

  write.table(
    selected_table,
    file = file.path(out_dir, "DEG.txt"),
    row.names = FALSE,
    sep = "\t",
    quote = FALSE
  )
  write.table(
    selected_table$GeneSymbol[selected_table$direction == "UP"],
    file = file.path(out_dir, "genes_UP.txt"),
    row.names = FALSE,
    col.names = FALSE,
    sep = "\t",
    quote = FALSE
  )
  write.table(
    selected_table$GeneSymbol[selected_table$direction == "DOWN"],
    file = file.path(out_dir, "genes_DOWN.txt"),
    row.names = FALSE,
    col.names = FALSE,
    sep = "\t",
    quote = FALSE
  )
  write.table(
    selected_matrix,
    file = file.path(out_dir, "matrix_DEG.txt"),
    row.names = TRUE,
    col.names = NA,
    sep = "\t",
    quote = FALSE
  )

  subset_settings <- settings
  subset_settings$source_dir <- source_dir
  subset_settings$selection_method <- "top_abs_logFC_from_existing_DEG"
  subset_settings$requested_top_n <- top_n
  subset_settings$selected_top_n <- selected_n
  write.table(
    subset_settings,
    file = file.path(out_dir, "settings.tsv"),
    row.names = FALSE,
    sep = "\t",
    quote = FALSE
  )

  heatmap_written <- FALSE
  boxplot_written <- FALSE
  if (!skip_plots) {
    n_control <- NA_integer_
    n_case <- NA_integer_
    if ("n_control_samples" %in% colnames(settings)) {
      n_control <- as.integer(settings$n_control_samples[[1]])
    }
    if ("n_case_samples" %in% colnames(settings)) {
      n_case <- as.integer(settings$n_case_samples[[1]])
    }
    if (is.na(n_control) || is.na(n_case)) {
      if (nrow(source_summary) > 0 && all(c("n_control_samples", "n_case_samples") %in% colnames(source_summary))) {
        n_control <- as.integer(source_summary$n_control_samples[[1]])
        n_case <- as.integer(source_summary$n_case_samples[[1]])
      }
    }
    if (is.na(n_control) || is.na(n_case) || n_control + n_case != ncol(selected_matrix)) {
      stop("Could not infer control/case sample counts for heatmap annotation.", call. = FALSE)
    }

    control_samples <- colnames(selected_matrix)[seq_len(n_control)]
    case_samples <- colnames(selected_matrix)[seq(from = n_control + 1, to = n_control + n_case)]
    sample_annotation <- build_sample_annotation(control_samples, case_samples)
    heatmap_written <- write_heatmap(
      filtered_data = selected_matrix,
      sample_annotation = sample_annotation,
      output_file = file.path(out_dir, "heatmap.pdf")
    )
    boxplot_written <- write_boxplots(
      filtered_data_control = selected_matrix[, control_samples, drop = FALSE],
      filtered_data_case = selected_matrix[, case_samples, drop = FALSE],
      result_table = selected_table,
      output_file = file.path(out_dir, "boxplot.pdf"),
      control_label = settings$control_label[[1]],
      case_label = settings$case_label[[1]],
      top_n = min(10L, selected_n)
    )
  }

  summary <- data.frame(
    comparison_id = paste0(source_comparison_id, "_top", selected_n, "_by_abs_logFC"),
    source_comparison_id = source_comparison_id,
    selection_method = "top_abs_logFC_from_existing_DEG",
    requested_top_n = top_n,
    selected_top_n = selected_n,
    n_deg_total = nrow(selected_table),
    n_deg_up = sum(selected_table$direction == "UP"),
    n_deg_down = sum(selected_table$direction == "DOWN"),
    min_abs_logFC = min(abs(selected_table$logFC)),
    max_abs_logFC = max(abs(selected_table$logFC)),
    heatmap_written = heatmap_written,
    boxplot_written = boxplot_written,
    stringsAsFactors = FALSE
  )
  write.table(
    summary,
    file = file.path(out_dir, "summary.tsv"),
    row.names = FALSE,
    sep = "\t",
    quote = FALSE
  )

  message(sprintf("Wrote %s with %d genes.", out_dir, selected_n))
}

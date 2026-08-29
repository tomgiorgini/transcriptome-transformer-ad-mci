options(stringsAsFactors = FALSE)

# Rebuild DEG tables from the ComBat-corrected 3-dataset matrix:
# GSE63060 + GSE63061 + GSE140829.
#
# Input matrix orientation: samples x genes.
# limma expects genes x samples, so the script transposes internally.

find_thesis_dir <- function() {
  candidates <- unique(normalizePath(c(getwd(), dirname(getwd())), winslash = "/", mustWork = FALSE))
  for (path in candidates) {
    if (dir.exists(file.path(path, "pretraining_dataset", "ad_mci_combat_dataset"))) {
      return(path)
    }
  }
  stop("Could not find thesis directory containing pretraining_dataset/ad_mci_combat_dataset.",
       call. = FALSE)
}

thesis_dir <- find_thesis_dir()

cfg <- list(
  input_dir = file.path(thesis_dir, "pretraining_dataset", "ad_mci_combat_dataset"),
  output_dir = file.path(thesis_dir, "pretraining_dataset", "ad_mci_combat_dataset",
                         "DEG_from_combat_3datasets"),
  matrix_file = file.path(thesis_dir, "pretraining_dataset", "ad_mci_combat_dataset",
                          "ad_mci_ctl_combat_matrix_samples_x_genes.csv"),
  metadata_file = file.path(thesis_dir, "pretraining_dataset", "ad_mci_combat_dataset",
                            "ad_mci_ctl_metadata.csv"),
  adj_pval_threshold = 0.05,
  logfc_threshold = 0,
  iqr_quantile = 0.10,
  heatmap_top_n = 100
)

comparisons <- data.frame(
  comparison_id = c("AD_vs_MCI", "AD_vs_CTL", "MCI_vs_CTL"),
  case_status = c("AD", "AD", "MCI"),
  control_status = c("MCI", "CTL", "CTL"),
  stringsAsFactors = FALSE
)

ensure_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Package could not be installed/loaded: ", pkg, call. = FALSE)
  }
}

clean_text <- function(x) {
  x <- enc2utf8(as.character(x))
  x[is.na(x)] <- ""
  x <- trimws(gsub("\\s+", " ", x))
  x <- gsub('^["\']|["\']$', "", x)
  x
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
}

write_tsv <- function(x, path) {
  write.table(x, path, sep = "\t", row.names = FALSE, quote = FALSE)
}

write_deg_heatmaps <- function(expr_all_genes, deg, metadata, comparison_id, out_dir) {
  if (nrow(deg) == 0) {
    return(FALSE)
  }

  ensure_package("pheatmap")

  n_heatmap <- min(cfg$heatmap_top_n, nrow(deg))
  ord <- order(deg$adj_pval, -abs(deg$logFC), deg$GeneSymbol)
  selected_genes <- deg$GeneSymbol[ord][seq_len(n_heatmap)]
  heatmap_matrix <- expr_all_genes[selected_genes, metadata$sample_id, drop = FALSE]

  annotation_col <- data.frame(
    status = metadata$status,
    batch = metadata$batch,
    row.names = metadata$sample_id,
    stringsAsFactors = FALSE
  )

  title <- paste0(comparison_id, " - top ", n_heatmap, " DEG")
  pdf_file <- file.path(out_dir, paste0(comparison_id, "_heatmap_top", n_heatmap, ".pdf"))
  png_file <- file.path(out_dir, paste0(comparison_id, "_heatmap_top", n_heatmap, ".png"))

  pheatmap::pheatmap(
    heatmap_matrix,
    scale = "row",
    show_rownames = n_heatmap <= 100,
    show_colnames = FALSE,
    cluster_rows = TRUE,
    cluster_cols = TRUE,
    annotation_col = annotation_col,
    border_color = NA,
    fontsize_row = ifelse(n_heatmap <= 50, 7, 5),
    main = title,
    filename = pdf_file,
    width = 11,
    height = 10
  )

  pheatmap::pheatmap(
    heatmap_matrix,
    scale = "row",
    show_rownames = n_heatmap <= 100,
    show_colnames = FALSE,
    cluster_rows = TRUE,
    cluster_cols = TRUE,
    annotation_col = annotation_col,
    border_color = NA,
    fontsize_row = ifelse(n_heatmap <= 50, 7, 5),
    main = title,
    filename = png_file,
    width = 11,
    height = 10
  )

  TRUE
}

read_matrix_samples_x_genes <- function(path) {
  message("Reading matrix: ", path)
  mat_df <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  if (!("sample_id" %in% colnames(mat_df))) {
    stop("Matrix CSV must contain a sample_id column.", call. = FALSE)
  }
  sample_id <- clean_text(mat_df$sample_id)
  mat_df$sample_id <- NULL
  mat <- as.matrix(mat_df)
  storage.mode(mat) <- "numeric"
  rownames(mat) <- sample_id
  colnames(mat) <- clean_text(colnames(mat))
  mat
}

read_metadata <- function(path) {
  metadata <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  required <- c("sample_id", "status", "batch")
  missing <- setdiff(required, colnames(metadata))
  if (length(missing) > 0) {
    stop("Metadata missing columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  metadata$sample_id <- clean_text(metadata$sample_id)
  metadata$status <- clean_text(metadata$status)
  metadata$batch <- clean_text(metadata$batch)
  metadata
}

prepare_expression <- function(matrix_samples_x_genes, metadata) {
  missing <- setdiff(metadata$sample_id, rownames(matrix_samples_x_genes))
  if (length(missing) > 0) {
    stop("Metadata samples missing from matrix. Examples: ",
         paste(head(missing, 10), collapse = ", "), call. = FALSE)
  }
  mat <- matrix_samples_x_genes[metadata$sample_id, , drop = FALSE]
  if (anyDuplicated(colnames(mat)) > 0) {
    stop("Duplicated gene names in matrix.", call. = FALSE)
  }
  if (any(!is.finite(mat))) {
    stop("Matrix contains NA/Inf values.", call. = FALSE)
  }
  t(mat)
}

filter_low_iqr <- function(expr_genes_x_samples) {
  variation <- apply(expr_genes_x_samples, 1, IQR, na.rm = TRUE)
  threshold <- unname(stats::quantile(variation, cfg$iqr_quantile, na.rm = TRUE))
  keep <- is.finite(variation) & variation > threshold
  list(
    expr = expr_genes_x_samples[keep, , drop = FALSE],
    threshold = threshold,
    n_removed = sum(!keep),
    n_after = sum(keep)
  )
}

run_one_comparison <- function(expr_filtered, expr_all_genes, metadata, comparison_id,
                               case_status, control_status) {
  out_dir <- file.path(cfg$output_dir, comparison_id)
  ensure_dir(out_dir)

  keep_samples <- metadata$status %in% c(case_status, control_status)
  comp_md <- metadata[keep_samples, , drop = FALSE]
  comp_expr <- expr_filtered[, comp_md$sample_id, drop = FALSE]

  comp_md$status <- factor(comp_md$status, levels = c(control_status, case_status))
  design <- stats::model.matrix(~ 0 + status, data = comp_md)
  colnames(design) <- sub("^status", "", colnames(design))

  fit <- limma::lmFit(comp_expr, design)
  contrast <- limma::makeContrasts(contrasts = paste0(case_status, "-", control_status),
                                  levels = design)
  fit2 <- limma::contrasts.fit(fit, contrast)
  fit2 <- limma::eBayes(fit2)

  all_results <- limma::topTable(fit2, number = Inf, adjust.method = "BH", sort.by = "P")
  all_results$GeneSymbol <- rownames(all_results)
  all_results <- all_results[, c("GeneSymbol", "P.Value", "adj.P.Val", "logFC",
                                 "AveExpr", "t", "B"), drop = FALSE]
  colnames(all_results) <- c("GeneSymbol", "pval", "adj_pval", "logFC",
                             "AveExpr", "t", "B")
  all_results$direction <- ifelse(all_results$logFC > 0, "UP", "DOWN")

  deg <- all_results[
    all_results$adj_pval < cfg$adj_pval_threshold &
      abs(all_results$logFC) >= cfg$logfc_threshold,
    ,
    drop = FALSE
  ]
  deg <- deg[order(deg$logFC, decreasing = TRUE), , drop = FALSE]
  deg$ensembl_id <- ""
  deg <- deg[, c("GeneSymbol", "ensembl_id", "pval", "adj_pval", "logFC",
                 "direction", "AveExpr", "t", "B"), drop = FALSE]

  write_tsv(all_results, file.path(out_dir, "all_genes_limma_results.tsv"))
  write_tsv(deg, file.path(out_dir, "DEG.txt"))
  write.table(deg$GeneSymbol[deg$direction == "UP"], file.path(out_dir, "genes_UP.txt"),
              row.names = FALSE, col.names = FALSE, sep = "\t", quote = FALSE)
  write.table(deg$GeneSymbol[deg$direction == "DOWN"], file.path(out_dir, "genes_DOWN.txt"),
              row.names = FALSE, col.names = FALSE, sep = "\t", quote = FALSE)

  if (nrow(deg) > 0) {
    matrix_deg <- expr_all_genes[deg$GeneSymbol, comp_md$sample_id, drop = FALSE]
  } else {
    matrix_deg <- expr_all_genes[FALSE, comp_md$sample_id, drop = FALSE]
  }
  write.table(matrix_deg, file.path(out_dir, "matrix_DEG.txt"),
              row.names = TRUE, col.names = NA, sep = "\t", quote = FALSE)

  heatmap_written <- write_deg_heatmaps(
    expr_all_genes = expr_all_genes,
    deg = deg,
    metadata = comp_md,
    comparison_id = comparison_id,
    out_dir = out_dir
  )

  settings <- data.frame(
    comparison_id = comparison_id,
    case_status = case_status,
    control_status = control_status,
    matrix_file = cfg$matrix_file,
    metadata_file = cfg$metadata_file,
    method = "limma on ComBat-corrected expression",
    pval_adjustment = "BH/FDR",
    adj_pval_threshold = cfg$adj_pval_threshold,
    logfc_threshold = cfg$logfc_threshold,
    heatmap_top_n = cfg$heatmap_top_n,
    iqr_quantile = cfg$iqr_quantile,
    stringsAsFactors = FALSE
  )
  write_tsv(settings, file.path(out_dir, "settings.tsv"))

  summary <- data.frame(
    comparison_id = comparison_id,
    case_status = case_status,
    control_status = control_status,
    n_case_samples = sum(comp_md$status == case_status),
    n_control_samples = sum(comp_md$status == control_status),
    n_input_genes = nrow(expr_all_genes),
    iqr_quantile = cfg$iqr_quantile,
    n_after_iqr = nrow(expr_filtered),
    adj_pval_threshold = cfg$adj_pval_threshold,
    logfc_threshold = cfg$logfc_threshold,
    n_deg_total = nrow(deg),
    n_deg_up = sum(deg$direction == "UP"),
    n_deg_down = sum(deg$direction == "DOWN"),
    heatmap_written = heatmap_written,
    stringsAsFactors = FALSE
  )
  write_tsv(summary, file.path(out_dir, "summary.tsv"))
  summary
}

main <- function() {
  required <- c(cfg$matrix_file, cfg$metadata_file)
  missing <- required[!file.exists(required)]
  if (length(missing) > 0) {
    stop("Missing required files:\n", paste(missing, collapse = "\n"), call. = FALSE)
  }

  ensure_package("limma")
  ensure_package("pheatmap")
  ensure_dir(cfg$output_dir)

  metadata <- read_metadata(cfg$metadata_file)
  matrix_samples_x_genes <- read_matrix_samples_x_genes(cfg$matrix_file)
  expr <- prepare_expression(matrix_samples_x_genes, metadata)
  filtered <- filter_low_iqr(expr)

  summaries <- lapply(seq_len(nrow(comparisons)), function(i) {
    run_one_comparison(
      expr_filtered = filtered$expr,
      expr_all_genes = expr,
      metadata = metadata,
      comparison_id = comparisons$comparison_id[i],
      case_status = comparisons$case_status[i],
      control_status = comparisons$control_status[i]
    )
  })
  run_summary <- do.call(rbind, summaries)
  run_summary$n_iqr_removed <- filtered$n_removed
  run_summary$iqr_threshold <- filtered$threshold
  write_tsv(run_summary, file.path(cfg$output_dir, "deg_run_summary.tsv"))

  dataset_summary <- data.frame(
    metric = c(
      "matrix_file",
      "metadata_file",
      "samples",
      "genes",
      "statuses",
      "batches",
      "method",
      "common_gene_count"
    ),
    value = c(
      normalizePath(cfg$matrix_file, winslash = "/", mustWork = FALSE),
      normalizePath(cfg$metadata_file, winslash = "/", mustWork = FALSE),
      ncol(expr),
      nrow(expr),
      paste(sort(unique(metadata$status)), collapse = ", "),
      paste(sort(unique(metadata$batch)), collapse = ", "),
      "limma after sva::ComBat batch correction",
      nrow(expr)
    ),
    stringsAsFactors = FALSE
  )
  write_tsv(dataset_summary, file.path(cfg$output_dir, "dataset_summary.tsv"))

  message("\nDone.")
  message("Genes in ComBat dataset: ", nrow(expr))
  message("Samples in ComBat dataset: ", ncol(expr))
  message("DEG output: ", normalizePath(cfg$output_dir, winslash = "/", mustWork = FALSE))
}

main()

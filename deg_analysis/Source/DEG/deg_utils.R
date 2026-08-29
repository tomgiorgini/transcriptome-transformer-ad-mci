options(stringsAsFactors = FALSE)

ensure_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(
      sprintf("Missing R package '%s'. Install it with install.packages('%s').", pkg, pkg),
      call. = FALSE
    )
  }
}

parse_cli_args <- function(args) {
  parsed <- list()
  i <- 1
  while (i <= length(args)) {
    arg <- args[[i]]
    if (!startsWith(arg, "--")) {
      stop(sprintf("Unexpected argument: %s", arg), call. = FALSE)
    }

    key <- sub("^--", "", arg)
    value <- TRUE

    if (grepl("=", key, fixed = TRUE)) {
      parts <- strsplit(key, "=", fixed = TRUE)[[1]]
      key <- parts[[1]]
      value <- paste(parts[-1], collapse = "=")
    } else if (i < length(args) && !startsWith(args[[i + 1]], "--")) {
      value <- args[[i + 1]]
      i <- i + 1
    }

    parsed[[key]] <- value
    i <- i + 1
  }

  parsed
}

get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd(), winslash = "/", mustWork = TRUE))
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = TRUE)
}

get_project_root <- function() {
  script_path <- get_script_path()
  normalizePath(file.path(dirname(script_path), "..", ".."), winslash = "/", mustWork = TRUE)
}

get_monorepo_root <- function() {
  normalizePath(file.path(get_project_root(), ".."), winslash = "/", mustWork = TRUE)
}

get_shared_alzheimer_data_dir <- function() {
  file.path(get_monorepo_root(), "Data", "alzheimer")
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
}

read_group_list <- function(path) {
  values <- readLines(path, warn = FALSE)
  values <- trimws(gsub("\r", "", values, fixed = TRUE))
  values <- values[nzchar(values)]
  unique(values)
}

read_expression_matrix <- function(path) {
  tmp <- read.table(
    path,
    header = TRUE,
    sep = "\t",
    check.names = FALSE,
    row.names = 1,
    quote = "",
    comment.char = ""
  )

  classes <- sapply(tmp, class)

  read.table(
    path,
    header = TRUE,
    sep = "\t",
    check.names = FALSE,
    row.names = 1,
    quote = "",
    comment.char = "",
    colClasses = classes
  )
}

split_gene_identifiers <- function(genes) {
  parsed <- strsplit(genes, "\\|")
  gene_symbol <- vapply(parsed, function(x) if (length(x) >= 1) x[[1]] else "", character(1))
  ensembl_id <- vapply(parsed, function(x) if (length(x) >= 2) x[[2]] else "", character(1))
  data.frame(GeneSymbol = gene_symbol, ensembl_id = ensembl_id, stringsAsFactors = FALSE)
}

safe_t_test_pvalue <- function(control_values, case_values, paired = FALSE) {
  control_values <- as.numeric(control_values)
  case_values <- as.numeric(case_values)

  if (all(is.na(control_values)) || all(is.na(case_values))) {
    return(1)
  }

  result <- tryCatch(
    stats::t.test(control_values, case_values, paired = paired),
    error = function(e) NULL,
    warning = function(w) suppressWarnings(stats::t.test(control_values, case_values, paired = paired))
  )

  if (is.null(result) || is.null(result$p.value) || is.na(result$p.value)) {
    return(1)
  }

  result$p.value
}

resolve_logfc_threshold <- function(comparison_row) {
  logfc_threshold <- suppressWarnings(as.numeric(comparison_row$logfc_threshold[[1]]))
  if (is.na(logfc_threshold)) {
    stop("Missing or invalid logfc_threshold in the manifest.", call. = FALSE)
  }
  logfc_threshold
}

write_volcano_plot <- function(logFC, adj_pval, output_file, title, thr_logfc, thr_pval) {
  pdf(output_file)

  finite_idx <- is.finite(logFC) & is.finite(adj_pval) & adj_pval > 0

  if (!any(finite_idx)) {
    plot.new()
    title(main = title)
    text(0.5, 0.5, "No points available")
  } else {
    plot(
      logFC[finite_idx],
      -log10(adj_pval[finite_idx]),
      main = title,
      xlab = "log2 Fold Change (FC)",
      ylab = "-log10 adjusted p-value"
    )
    abline(h = -log10(thr_pval), lty = 2, lwd = 2, col = "blue")
    abline(v = c(-thr_logfc, thr_logfc), lty = 2, lwd = 2, col = "red")
  }

  dev.off()
}

write_boxplots <- function(filtered_data_control,
                           filtered_data_case,
                           result_table,
                           output_file,
                           control_label,
                           case_label,
                           top_n) {
  total_genes <- nrow(result_table)
  if (total_genes == 0) {
    return(FALSE)
  }

  top_n <- min(top_n, total_genes)
  positive_idx <- head(order(result_table$logFC, decreasing = TRUE), top_n)
  negative_idx <- head(order(result_table$logFC, decreasing = FALSE), top_n)
  selected_idx <- unique(c(positive_idx, negative_idx))
  selected_idx <- selected_idx[seq_len(min(length(selected_idx), max(1, 2 * top_n)))]

  pdf(output_file)
  on.exit(dev.off(), add = TRUE)

  n_panels <- length(selected_idx)
  n_cols <- min(3, n_panels)
  n_rows <- ceiling(n_panels / n_cols)
  par(mfrow = c(n_rows, n_cols), mar = c(4, 4, 3, 1))

  for (idx in selected_idx) {
    boxplot(
      as.numeric(filtered_data_control[idx, ]),
      as.numeric(filtered_data_case[idx, ]),
      main = paste0(
        result_table$GeneSymbol[idx],
        "\nadj p-val = ",
        format(result_table$adj_pval[idx], digits = 3)
      ),
      notch = FALSE,
      ylab = "Gene expression value",
      names = c(control_label, case_label),
      col = c("forestgreen", "darkorange"),
      pars = list(boxwex = 0.3, staplewex = 0.6),
      cex.lab = 1.0,
      cex.axis = 0.9
    )
  }

  TRUE
}

write_heatmap <- function(filtered_data, sample_annotation, output_file) {
  if (nrow(filtered_data) == 0 || ncol(filtered_data) == 0) {
    return(FALSE)
  }

  annotation_colors <- list(
    condition = c(
      "control" = "forestgreen",
      "case" = "darkorange"
    )
  )

  pheatmap::pheatmap(
    filtered_data,
    scale = "row",
    border_color = NA,
    cluster_cols = TRUE,
    cluster_rows = TRUE,
    clustering_distance_rows = "correlation",
    clustering_distance_cols = "correlation",
    clustering_method = "complete",
    annotation_col = sample_annotation,
    annotation_colors = annotation_colors,
    color = colorRampPalette(c("blue", "blue3", "black", "yellow3", "yellow"))(100),
    show_rownames = FALSE,
    show_colnames = FALSE,
    cutree_rows = 2,
    cutree_cols = 2,
    width = 10,
    height = 10,
    filename = output_file
  )

  TRUE
}

build_sample_annotation <- function(control_samples, case_samples) {
  annotation <- data.frame(
    condition = c(
      rep("control", length(control_samples)),
      rep("case", length(case_samples))
    ),
    row.names = c(control_samples, case_samples),
    stringsAsFactors = FALSE
  )

  annotation
}

run_deg_analysis <- function(comparison_row,
                             matrix_path,
                             groups_dir,
                             results_root,
                             skip_plots = FALSE) {
  ensure_package("pheatmap")

  comparison_id <- comparison_row$comparison_id[[1]]
  case_group <- comparison_row$case_group[[1]]
  control_group <- comparison_row$control_group[[1]]
  case_label <- comparison_row$case_label[[1]]
  control_label <- comparison_row$control_label[[1]]
  thr_logfc <- resolve_logfc_threshold(comparison_row)
  thr_pval <- as.numeric(comparison_row$pval_threshold[[1]])
  iqr_quantile <- as.numeric(comparison_row$iqr_quantile[[1]])
  boxplot_top_n <- as.integer(comparison_row$boxplot_top_n[[1]])

  out_dir <- file.path(results_root, comparison_id)
  ensure_dir(out_dir)

  case_samples <- read_group_list(file.path(groups_dir, paste0(case_group, ".txt")))
  control_samples <- read_group_list(file.path(groups_dir, paste0(control_group, ".txt")))

  expression_matrix <- read_expression_matrix(matrix_path)

  missing_case <- setdiff(case_samples, colnames(expression_matrix))
  missing_control <- setdiff(control_samples, colnames(expression_matrix))

  if (length(missing_case) > 0 || length(missing_control) > 0) {
    stop(
      sprintf(
        "Missing samples in matrix for comparison %s. Missing case: %s. Missing control: %s.",
        comparison_id,
        paste(missing_case, collapse = ", "),
        paste(missing_control, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  data_control <- expression_matrix[, control_samples, drop = FALSE]
  data_case <- expression_matrix[, case_samples, drop = FALSE]
  data_all <- cbind(data_control, data_case)
  genes <- rownames(data_all)

  n_input_genes <- nrow(data_all)
  n_control_samples <- ncol(data_control)
  n_case_samples <- ncol(data_case)

  overall_mean <- rowMeans(data_all, na.rm = TRUE)
  zero_mean_idx <- which(is.na(overall_mean) | overall_mean == 0)

  if (length(zero_mean_idx) > 0) {
    data_control <- data_control[-zero_mean_idx, , drop = FALSE]
    data_case <- data_case[-zero_mean_idx, , drop = FALSE]
    data_all <- data_all[-zero_mean_idx, , drop = FALSE]
    genes <- genes[-zero_mean_idx]
  }

  data_control <- log2(data_control + 1)
  data_case <- log2(data_case + 1)
  data_all <- log2(data_all + 1)

  variation <- apply(data_all, 1, IQR, na.rm = TRUE)
  iqr_threshold <- unname(stats::quantile(variation, iqr_quantile, na.rm = TRUE))
  low_iqr_idx <- which(is.na(variation) | variation <= iqr_threshold)

  if (length(low_iqr_idx) > 0) {
    data_control <- data_control[-low_iqr_idx, , drop = FALSE]
    data_case <- data_case[-low_iqr_idx, , drop = FALSE]
    data_all <- data_all[-low_iqr_idx, , drop = FALSE]
    genes <- genes[-low_iqr_idx]
    variation <- variation[-low_iqr_idx]
  }

  logFC <- rowMeans(data_case, na.rm = TRUE) - rowMeans(data_control, na.rm = TRUE)
  pval <- vapply(
    seq_len(nrow(data_all)),
    function(i) safe_t_test_pvalue(data_control[i, ], data_case[i, ], paired = FALSE),
    numeric(1)
  )
  adj_pval <- p.adjust(pval, method = "fdr")

  write_volcano_plot(
    logFC = logFC,
    adj_pval = adj_pval,
    output_file = file.path(out_dir, paste0(comparison_id, "_VOLCANO_PRE_FILTERING.pdf")),
    title = paste(comparison_id, "Volcano plot before filtering"),
    thr_logfc = thr_logfc,
    thr_pval = thr_pval
  )

  deg_idx <- which(adj_pval < thr_pval & abs(logFC) >= thr_logfc)

  filtered_data_control <- data_control[deg_idx, , drop = FALSE]
  filtered_data_case <- data_case[deg_idx, , drop = FALSE]
  filtered_data_all <- data_all[deg_idx, , drop = FALSE]
  filtered_genes <- genes[deg_idx]
  filtered_logFC <- logFC[deg_idx]
  filtered_pval <- pval[deg_idx]
  filtered_adj_pval <- adj_pval[deg_idx]

  write_volcano_plot(
    logFC = filtered_logFC,
    adj_pval = filtered_adj_pval,
    output_file = file.path(out_dir, paste0(comparison_id, "_VOLCANO_POST_FILTERING.pdf")),
    title = paste(comparison_id, "Volcano plot after filtering"),
    thr_logfc = thr_logfc,
    thr_pval = thr_pval
  )

  deg_info <- split_gene_identifiers(filtered_genes)
  direction <- ifelse(filtered_logFC > 0, "UP", "DOWN")
  result_table <- data.frame(
    GeneSymbol = deg_info$GeneSymbol,
    ensembl_id = deg_info$ensembl_id,
    pval = filtered_pval,
    adj_pval = filtered_adj_pval,
    logFC = filtered_logFC,
    direction = direction,
    stringsAsFactors = FALSE
  )

  if (nrow(result_table) > 0) {
    order_idx <- order(result_table$logFC, decreasing = TRUE)
    result_table <- result_table[order_idx, , drop = FALSE]
    filtered_data_control <- filtered_data_control[order_idx, , drop = FALSE]
    filtered_data_case <- filtered_data_case[order_idx, , drop = FALSE]
    filtered_data_all <- filtered_data_all[order_idx, , drop = FALSE]
  }

  write.table(
    result_table,
    file = file.path(out_dir, "DEG.txt"),
    row.names = FALSE,
    sep = "\t",
    quote = FALSE
  )

  write.table(
    result_table$GeneSymbol[result_table$direction == "UP"],
    file = file.path(out_dir, "genes_UP.txt"),
    row.names = FALSE,
    col.names = FALSE,
    sep = "\t",
    quote = FALSE
  )

  write.table(
    result_table$GeneSymbol[result_table$direction == "DOWN"],
    file = file.path(out_dir, "genes_DOWN.txt"),
    row.names = FALSE,
    col.names = FALSE,
    sep = "\t",
    quote = FALSE
  )

  write.table(
    filtered_data_all,
    file = file.path(out_dir, "matrix_DEG.txt"),
    row.names = TRUE,
    col.names = NA,
    sep = "\t",
    quote = FALSE
  )

  settings_table <- data.frame(
    comparison_id = comparison_id,
    case_group = case_group,
    control_group = control_group,
    case_label = case_label,
    control_label = control_label,
    logfc_threshold = thr_logfc,
    pval_threshold = thr_pval,
    iqr_quantile = iqr_quantile,
    boxplot_top_n = boxplot_top_n,
    matrix_path = matrix_path,
    groups_dir = groups_dir,
    stringsAsFactors = FALSE
  )

  write.table(
    settings_table,
    file = file.path(out_dir, "settings.tsv"),
    row.names = FALSE,
    sep = "\t",
    quote = FALSE
  )

  plots_written <- FALSE
  heatmap_written <- FALSE
  boxplot_written <- FALSE

  if (!skip_plots) {
    plots_written <- TRUE
    sample_annotation <- build_sample_annotation(control_samples, case_samples)
    heatmap_written <- write_heatmap(
      filtered_data = filtered_data_all,
      sample_annotation = sample_annotation,
      output_file = file.path(out_dir, "heatmap.pdf")
    )
    boxplot_written <- write_boxplots(
      filtered_data_control = filtered_data_control,
      filtered_data_case = filtered_data_case,
      result_table = result_table,
      output_file = file.path(out_dir, "boxplot.pdf"),
      control_label = control_label,
      case_label = case_label,
      top_n = boxplot_top_n
    )
  }

  summary_table <- data.frame(
    comparison_id = comparison_id,
    case_group = case_group,
    control_group = control_group,
    n_case_samples = n_case_samples,
    n_control_samples = n_control_samples,
    n_input_genes = n_input_genes,
    n_zero_mean_removed = length(zero_mean_idx),
    n_after_zero_mean = n_input_genes - length(zero_mean_idx),
    iqr_quantile = iqr_quantile,
    iqr_threshold = iqr_threshold,
    n_iqr_removed = length(low_iqr_idx),
    n_after_iqr = nrow(data_all),
    logfc_threshold = thr_logfc,
    pval_threshold = thr_pval,
    n_deg_total = nrow(result_table),
    n_deg_up = sum(result_table$direction == "UP"),
    n_deg_down = sum(result_table$direction == "DOWN"),
    plots_requested = plots_written,
    heatmap_written = heatmap_written,
    boxplot_written = boxplot_written,
    stringsAsFactors = FALSE
  )

  write.table(
    summary_table,
    file = file.path(out_dir, "summary.tsv"),
    row.names = FALSE,
    sep = "\t",
    quote = FALSE
  )

  summary_table
}

load_comparisons_manifest <- function(path) {
  read.delim(path, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
}

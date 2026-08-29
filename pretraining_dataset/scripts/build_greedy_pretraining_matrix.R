options(stringsAsFactors = FALSE)

# Build a pretraining matrix from the greedy dataset selection.
# Output matrix orientation: samples x genes.

find_thesis_dir <- function() {
  candidates <- unique(normalizePath(c(getwd(), dirname(getwd())), winslash = "/", mustWork = FALSE))
  for (path in candidates) {
    if (dir.exists(file.path(path, "pretraining_dataset", "geo_downloads"))) {
      return(path)
    }
    if (basename(path) == "pretraining_dataset" && dir.exists(file.path(path, "geo_downloads"))) {
      return(dirname(path))
    }
  }
  stop("Could not find thesis directory containing pretraining_dataset/geo_downloads.", call. = FALSE)
}

thesis_dir <- find_thesis_dir()
pretraining_dir <- file.path(thesis_dir, "pretraining_dataset")
pretraining_scripts_dir <- file.path(pretraining_dir, "scripts")

cfg <- list(
  thesis_dir = thesis_dir,
  pretraining_dir = pretraining_dir,
  pretraining_scripts_dir = pretraining_scripts_dir,
  geo_dir = file.path(pretraining_dir, "geo_downloads"),
  reference_matrix = file.path(thesis_dir, "task_dataset", "matrix.txt"),
  reference_genes = file.path(pretraining_scripts_dir, "reference_genes.txt"),
  greedy_script = file.path(pretraining_scripts_dir, "greedy_reference_intersection_selection.R"),
  greedy_report = file.path(pretraining_dir, "geo_downloads", "qc_reports",
                            "greedy_reference_intersection_iterations_clean.tsv"),
  gene_symbol_summary = file.path(pretraining_dir, "geo_downloads",
                                  "gene_symbol_extraction_summary.tsv"),
  out_dir = file.path(pretraining_dir, "geo_downloads", "merged_pretraining"),
  refresh_greedy_selection = TRUE,
  include_reference_matrix = TRUE
)

clean_text <- function(x) {
  x <- enc2utf8(as.character(x))
  x[is.na(x)] <- ""
  x <- trimws(gsub("\\s+", " ", x))
  x <- gsub('^["\']|["\']$', "", x)
  x
}

read_gene_list <- function(path) {
  x <- clean_text(readLines(path, warn = FALSE))
  unique(x[nzchar(x)])
}

read_tsv <- function(path) {
  read.delim(path, sep = "\t", check.names = FALSE, quote = "",
             comment.char = "", stringsAsFactors = FALSE)
}

write_tsv <- function(x, path) {
  write.table(x, path, sep = "\t", row.names = FALSE, quote = FALSE)
}

parse_stop_iteration <- function(max_iteration) {
  args <- commandArgs(trailingOnly = TRUE)
  default_stop_iteration <- 21L
  hit <- grep("^--stop[_-]iteration=", args, value = TRUE)
  if (length(hit) > 0) {
    value <- as.integer(sub("^--stop[_-]iteration=", "", hit[1]))
  } else {
    cat("\nGreedy iterations available: 0-", max_iteration, "\n", sep = "")
    cat("Iteration 0 is the reference only; selected GEO datasets start from iteration 1.\n")
    raw_value <- readline(paste0("Stop at greedy iteration [", default_stop_iteration, "]: "))
    raw_value <- trimws(raw_value)
    value <- if (nzchar(raw_value)) as.integer(raw_value) else default_stop_iteration
  }

  if (is.na(value) || value < 1L || value > max_iteration) {
    stop("Invalid stop iteration. Choose an integer between 1 and ", max_iteration, ".", call. = FALSE)
  }
  value
}

parse_bool_arg <- function(name, default) {
  args <- commandArgs(trailingOnly = TRUE)
  names <- unique(c(name, gsub("-", "_", name)))
  pattern <- paste0("^--(", paste(names, collapse = "|"), ")=")
  hit <- grep(pattern, args, value = TRUE)
  if (length(hit) == 0) {
    return(default)
  }

  value <- tolower(trimws(sub(pattern, "", hit[1])))
  if (value %in% c("true", "t", "1", "yes", "y", "on")) {
    return(TRUE)
  }
  if (value %in% c("false", "f", "0", "no", "n", "off")) {
    return(FALSE)
  }
  stop("Invalid boolean value for --", name, ": ", value,
       ". Use true or false.", call. = FALSE)
}

parse_char_arg <- function(name, default = "") {
  args <- commandArgs(trailingOnly = TRUE)
  names <- unique(c(name, gsub("-", "_", name)))
  pattern <- paste0("^--(", paste(names, collapse = "|"), ")=")
  hit <- grep(pattern, args, value = TRUE)
  if (length(hit) == 0) {
    return(default)
  }
  clean_text(sub(pattern, "", hit[1]))
}

first_existing_column <- function(df, candidates, default = NA) {
  for (candidate in candidates) {
    if (candidate %in% colnames(df)) {
      return(df[[candidate]])
    }
  }
  rep(default, nrow(df))
}

read_filter_gene_list <- function(path, column_name = "") {
  if (!nzchar(path)) {
    return(character())
  }
  if (!file.exists(path)) {
    stop("Gene filter file not found: ", path, call. = FALSE)
  }

  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("tsv", "txt", "csv")) {
    sep <- if (ext == "csv") "," else "\t"
    df <- read.delim(path, sep = sep, check.names = FALSE, quote = "",
                     comment.char = "", stringsAsFactors = FALSE)
    if (ncol(df) == 0) {
      return(character())
    }
    if (nzchar(column_name) && column_name %in% colnames(df)) {
      genes <- df[[column_name]]
    } else if ("GeneSymbol" %in% colnames(df)) {
      genes <- df[["GeneSymbol"]]
    } else if ("gene_symbol" %in% colnames(df)) {
      genes <- df[["gene_symbol"]]
    } else {
      genes <- df[[1]]
    }
    return(unique(clean_text(genes[nzchar(clean_text(genes))])))
  }

  read_gene_list(path)
}

resolve_local_mapping_file <- function(gse_id, eset_dir, mapping_path) {
  mapping_path <- clean_text(mapping_path)
  candidates <- character()
  if (nzchar(mapping_path)) {
    candidates <- c(candidates, mapping_path)
    candidates <- c(candidates, file.path(cfg$geo_dir, gse_id, eset_dir, basename(mapping_path)))

    for (dataset_marker in c("pretraining_dataset/", "Dataset_R/")) {
      marker_pos <- regexpr(dataset_marker, mapping_path, fixed = TRUE)
      if (marker_pos[1] > 0) {
        relative <- substr(mapping_path, marker_pos[1] + nchar(dataset_marker), nchar(mapping_path))
        candidates <- c(candidates, file.path(cfg$pretraining_dir, relative))
      }
    }
  }
  candidates <- c(candidates, file.path(cfg$geo_dir, gse_id, eset_dir, "probe_to_gene_symbol.tsv"))

  candidates <- unique(normalizePath(candidates, winslash = "/", mustWork = FALSE))
  for (candidate in candidates) {
    if (file.exists(candidate)) {
      return(candidate)
    }
  }
  stop("Could not resolve probe-to-gene mapping file for ", gse_id, " / ", eset_dir,
       ". Tried: ", paste(candidates, collapse = " | "), call. = FALSE)
}

refresh_greedy_report <- function() {
  if (!isTRUE(cfg$refresh_greedy_selection)) {
    return(invisible(FALSE))
  }
  if (!file.exists(cfg$greedy_script)) {
    stop("Greedy script not found: ", cfg$greedy_script, call. = FALSE)
  }

  message("Refreshing greedy selection report...")
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(cfg$thesis_dir)
  greedy_env <- new.env(parent = globalenv())
  sys.source(cfg$greedy_script, envir = greedy_env)
  invisible(TRUE)
}

select_greedy_datasets <- function(stop_iteration) {
  greedy <- read_tsv(cfg$greedy_report)
  selected <- greedy[greedy$iteration >= 1L & greedy$iteration <= stop_iteration, , drop = FALSE]
  if (nrow(selected) == 0) {
    stop("No GEO dataset selected. Use stop_iteration >= 1.", call. = FALSE)
  }
  selected
}

load_dataset_index <- function(selected) {
  summary <- read_tsv(cfg$gene_symbol_summary)
  summary <- summary[summary$error == "" | is.na(summary$error), , drop = FALSE]

  selected$key <- paste(selected$gse_id, selected$platform_id, sep = "__")
  summary$key <- paste(summary$gse_id, summary$platform_id, sep = "__")
  matched <- summary[match(selected$key, summary$key), , drop = FALSE]

  missing <- selected$key[is.na(matched$gse_id)]
  if (length(missing) > 0) {
    stop("Missing gene-symbol mapping summary for: ", paste(missing, collapse = ", "), call. = FALSE)
  }

  mapping_column <- first_existing_column(matched, c("matrix_row_mapping_file", "mapping_file"))
  matrix_row_mapping_file <- mapply(
    resolve_local_mapping_file,
    matched$gse_id,
    matched$eset_dir,
    mapping_column,
    USE.NAMES = FALSE
  )

  data.frame(
    selected[, c("iteration", "gse_id", "platform_id", "conditions", "added_samples",
                 "cumulative_samples", "genes_after_intersection",
                 "deg_ad_vs_mci_intersection")],
    eset_dir = matched$eset_dir,
    matrix_rows = first_existing_column(matched, c("matrix_rows", "input_probe_ids")),
    matrix_rows_with_symbol = first_existing_column(matched, c("matrix_rows_with_symbol", "probes_with_symbol")),
    unique_primary_gene_symbols = first_existing_column(matched, c("unique_primary_gene_symbols", "unique_gene_symbols")),
    matrix_row_mapping_file = matrix_row_mapping_file,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

find_expression_matrix <- function(gse_id, eset_dir) {
  path <- file.path(cfg$geo_dir, gse_id, eset_dir)
  files <- list.files(path, pattern = "_expr_matrix\\.tsv(\\.gz)?$", full.names = TRUE)
  files <- sort(files)
  if (length(files) == 0) {
    stop("Expression matrix not found in: ", path, call. = FALSE)
  }
  files[1]
}

read_expression_matrix <- function(path) {
  con <- if (grepl("\\.gz$", path, ignore.case = TRUE)) gzfile(path, open = "rt") else file(path, open = "rt")
  on.exit(close(con), add = TRUE)
  mat <- read.delim(con, sep = "\t", row.names = 1, check.names = FALSE,
                    quote = "", comment.char = "", stringsAsFactors = FALSE)
  mat <- as.matrix(mat)
  storage.mode(mat) <- "numeric"
  rownames(mat) <- clean_text(rownames(mat))
  colnames(mat) <- clean_text(colnames(mat))
  mat
}

read_reference_matrix <- function(path) {
  mat <- read.delim(path, sep = "\t", row.names = 1, check.names = FALSE,
                    quote = "", comment.char = "", stringsAsFactors = FALSE)
  mat <- as.matrix(mat)
  storage.mode(mat) <- "numeric"
  rownames(mat) <- clean_text(rownames(mat))
  colnames(mat) <- clean_text(colnames(mat))
  mat
}

read_row_map <- function(path) {
  row_map <- read_tsv(path)
  if (!"feature_id" %in% colnames(row_map) && "probe_id" %in% colnames(row_map)) {
    row_map$feature_id <- row_map$probe_id
  }
  if (!"keep_for_gene_symbol_matrix" %in% colnames(row_map)) {
    row_map$keep_for_gene_symbol_matrix <- TRUE
  }

  required <- c("feature_id", "gene_symbol", "keep_for_gene_symbol_matrix")
  missing <- setdiff(required, colnames(row_map))
  if (length(missing) > 0) {
    stop("Mapping file missing columns: ", paste(missing, collapse = ", "), "\n", path, call. = FALSE)
  }
  row_map$feature_id <- clean_text(row_map$feature_id)
  row_map$gene_symbol <- clean_text(row_map$gene_symbol)
  row_map$keep_for_gene_symbol_matrix <- as.logical(row_map$keep_for_gene_symbol_matrix) & nzchar(row_map$gene_symbol)
  row_map
}

genes_from_row_map <- function(path) {
  row_map <- read_row_map(path)
  genes <- row_map$gene_symbol[row_map$keep_for_gene_symbol_matrix & nzchar(row_map$gene_symbol)]
  unique(genes)
}

collapse_to_gene_symbols <- function(mat, row_map, common_genes) {
  idx <- match(clean_text(row_map$feature_id), clean_text(rownames(mat)))
  keep <- !is.na(idx) & row_map$keep_for_gene_symbol_matrix & row_map$gene_symbol %in% common_genes

  mat <- mat[idx[keep], , drop = FALSE]
  gene_symbol <- row_map$gene_symbol[keep]
  if (nrow(mat) == 0) {
    stop("No matrix rows remain after gene-symbol mapping and common-gene filtering.", call. = FALSE)
  }

  mat_zero <- mat
  mat_zero[is.na(mat_zero)] <- 0
  summed <- rowsum(mat_zero, group = gene_symbol, reorder = FALSE)
  counts <- rowsum((!is.na(mat)) * 1, group = gene_symbol, reorder = FALSE)
  collapsed <- summed / counts
  collapsed[counts == 0] <- NA_real_
  aligned <- matrix(
    NA_real_,
    nrow = length(common_genes),
    ncol = ncol(collapsed),
    dimnames = list(common_genes, colnames(collapsed))
  )
  present <- intersect(common_genes, rownames(collapsed))
  aligned[present, ] <- collapsed[present, , drop = FALSE]
  aligned
}

standardize_gene_columns <- function(mat) {
  means <- colMeans(mat, na.rm = TRUE)
  sds <- apply(mat, 2, sd, na.rm = TRUE)
  sds[is.na(sds) | sds == 0] <- 1
  scaled <- sweep(mat, 2, means, "-")
  scaled <- sweep(scaled, 2, sds, "/")
  scaled[is.na(scaled)] <- 0
  scaled
}

process_one_dataset <- function(dataset_row, common_genes) {
  gse_id <- dataset_row$gse_id
  platform_id <- dataset_row$platform_id
  eset_dir <- dataset_row$eset_dir

  message("Processing ", gse_id, " / ", platform_id, "...")
  matrix_file <- find_expression_matrix(gse_id, eset_dir)
  row_map <- read_row_map(dataset_row$matrix_row_mapping_file)
  mat <- read_expression_matrix(matrix_file)
  collapsed <- collapse_to_gene_symbols(mat, row_map, common_genes)
  sample_matrix <- t(collapsed)

  original_sample_id <- rownames(sample_matrix)
  unique_sample_id <- paste(gse_id, platform_id, original_sample_id, sep = "__")
  rownames(sample_matrix) <- unique_sample_id

  metadata <- data.frame(
    sample_id = unique_sample_id,
    original_sample_id = original_sample_id,
    gse_id = gse_id,
    platform_id = platform_id,
    eset_dir = eset_dir,
    greedy_iteration = dataset_row$iteration,
    status = "",
    conditions = dataset_row$conditions,
    stringsAsFactors = FALSE
  )

  list(matrix = sample_matrix, metadata = metadata)
}

process_reference_dataset <- function(common_genes) {
  if (!file.exists(cfg$reference_matrix)) {
    stop("Reference matrix not found: ", cfg$reference_matrix, call. = FALSE)
  }

  message("Processing reference GSE63060 + GSE63061...")
  mat <- read_reference_matrix(cfg$reference_matrix)
  mat <- mat[rownames(mat) %in% common_genes, , drop = FALSE]
  if (nrow(mat) == 0) {
    stop("No reference genes remain after common-gene filtering.", call. = FALSE)
  }

  if (anyDuplicated(rownames(mat)) > 0) {
    mat_zero <- mat
    mat_zero[is.na(mat_zero)] <- 0
    summed <- rowsum(mat_zero, group = rownames(mat), reorder = FALSE)
    counts <- rowsum((!is.na(mat)) * 1, group = rownames(mat), reorder = FALSE)
    mat <- summed / counts
    mat[counts == 0] <- NA_real_
  }

  mat <- mat[common_genes, , drop = FALSE]
  sample_matrix <- t(mat)

  original_sample_id <- rownames(sample_matrix)
  unique_sample_id <- paste("GSE63060_GSE63061", "reference", original_sample_id, sep = "__")
  rownames(sample_matrix) <- unique_sample_id

  status <- sub("^.*_", "", original_sample_id)
  status[status == original_sample_id] <- ""

  metadata <- data.frame(
    sample_id = unique_sample_id,
    original_sample_id = original_sample_id,
    gse_id = "GSE63060_GSE63061",
    platform_id = "GPL6947_GPL10558",
    eset_dir = "task_dataset/matrix.txt",
    greedy_iteration = 0L,
    status = status,
    conditions = "Reference Alzheimer / Mild cognitive impairment / Control cohort from GSE63060 and GSE63061",
    stringsAsFactors = FALSE
  )

  list(matrix = sample_matrix, metadata = metadata)
}

main <- function() {
  dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

  refresh_greedy_report()
  if (!file.exists(cfg$greedy_report)) {
    stop("Greedy report not found: ", cfg$greedy_report, call. = FALSE)
  }
  if (!file.exists(cfg$gene_symbol_summary)) {
    stop("Gene-symbol summary not found. Run extract_gene_symbols_all_datasets.R first.", call. = FALSE)
  }

  greedy <- read_tsv(cfg$greedy_report)
  stop_iteration <- parse_stop_iteration(max(greedy$iteration, na.rm = TRUE))
  include_reference_matrix <- parse_bool_arg("include-reference", cfg$include_reference_matrix)
  gene_list_file <- parse_char_arg("gene-list-file", "")
  gene_list_column <- parse_char_arg("gene-list-column", "")
  gene_set_name <- parse_char_arg("gene-set-name", "")
  selected <- select_greedy_datasets(stop_iteration)
  dataset_index <- load_dataset_index(selected)

  reference_genes <- read_gene_list(cfg$reference_genes)
  common_genes <- reference_genes
  for (i in seq_len(nrow(dataset_index))) {
    dataset_genes <- genes_from_row_map(dataset_index$matrix_row_mapping_file[i])
    common_genes <- common_genes[common_genes %in% dataset_genes]
  }
  if (length(common_genes) == 0) {
    stop("The selected datasets have no genes in common with the reference.", call. = FALSE)
  }

  filter_genes <- read_filter_gene_list(gene_list_file, gene_list_column)
  filter_genes <- filter_genes[nzchar(filter_genes)]
  if (length(filter_genes) > 0) {
    before_filter <- length(common_genes)
    common_genes <- common_genes[common_genes %in% filter_genes]
    if (length(common_genes) == 0) {
      stop("The selected datasets have no genes in common with the supplied gene list.", call. = FALSE)
    }
    message("Gene filter file: ", gene_list_file)
    message("Gene filter retained: ", length(common_genes), " / ", before_filter,
            " common genes (", length(filter_genes), " requested genes)")
  }

  message("Selected GEO datasets: ", nrow(dataset_index))
  message("Common genes retained: ", length(common_genes))
  message("Reference matrix included: ", include_reference_matrix)

  processed <- lapply(seq_len(nrow(dataset_index)), function(i) {
    process_one_dataset(dataset_index[i, , drop = FALSE], common_genes)
  })
  if (isTRUE(include_reference_matrix)) {
    processed <- c(list(process_reference_dataset(common_genes)), processed)
  }

  merged_matrix_raw <- do.call(rbind, lapply(processed, `[[`, "matrix"))
  merged_matrix <- standardize_gene_columns(merged_matrix_raw)
  metadata <- do.call(rbind, lapply(processed, `[[`, "metadata"))

  prefix <- paste0(
    "pretraining_greedy_iter_",
    stop_iteration,
    ifelse(isTRUE(include_reference_matrix), "_with_reference", "_no_reference"),
    ifelse(nzchar(gene_set_name), paste0("_", gene_set_name), "")
  )
  matrix_out <- file.path(cfg$out_dir, paste0(prefix, "_global_zscore_matrix_samples_x_genes.csv"))
  metadata_out <- file.path(cfg$out_dir, paste0(prefix, "_metadata.csv"))
  genes_out <- file.path(cfg$out_dir, paste0(prefix, "_common_genes.txt"))
  selected_out <- file.path(cfg$out_dir, paste0(prefix, "_selected_datasets.tsv"))
  summary_out <- file.path(cfg$out_dir, paste0(prefix, "_global_zscore_merge_summary.tsv"))

  matrix_df <- data.frame(sample_id = rownames(merged_matrix), merged_matrix,
                          check.names = FALSE)
  write.csv(matrix_df, matrix_out, row.names = FALSE, quote = FALSE)
  write.csv(metadata, metadata_out, row.names = FALSE, quote = FALSE)
  writeLines(common_genes, genes_out, useBytes = TRUE)
  selected_out_df <- dataset_index
  reference_samples <- 0L
  if (isTRUE(include_reference_matrix)) {
    reference_samples <- nrow(processed[[1]]$matrix)
    reference_row <- data.frame(
      iteration = 0L,
      gse_id = "GSE63060_GSE63061",
      platform_id = "GPL6947_GPL10558",
      conditions = "Reference Alzheimer / Mild cognitive impairment / Control cohort from GSE63060 and GSE63061",
      added_samples = reference_samples,
      cumulative_samples = reference_samples,
      genes_after_intersection = length(common_genes),
      deg_ad_vs_mci_intersection = NA_integer_,
      eset_dir = "task_dataset/matrix.txt",
      matrix_rows = ncol(processed[[1]]$matrix),
      matrix_rows_with_symbol = ncol(processed[[1]]$matrix),
      unique_primary_gene_symbols = ncol(processed[[1]]$matrix),
      matrix_row_mapping_file = "",
      stringsAsFactors = FALSE
    )
    selected_out_df <- rbind(reference_row, selected_out_df)
  }
  write_tsv(selected_out_df, selected_out)

  merge_summary <- data.frame(
    stop_iteration = stop_iteration,
    selected_geo_datasets = nrow(dataset_index),
    reference_samples = reference_samples,
    samples = nrow(merged_matrix),
    common_genes = ncol(merged_matrix),
    gene_filter_file = ifelse(nzchar(gene_list_file), normalizePath(gene_list_file, winslash = "/", mustWork = FALSE), ""),
    gene_filter_column = gene_list_column,
    gene_set_name = gene_set_name,
    standardization = "global per gene: z-score across all merged samples after duplicate probes are averaged and datasets are merged",
    duplicate_probe_strategy = "mean expression per gene symbol within each dataset",
    reference_matrix_included = include_reference_matrix,
    matrix_file = normalizePath(matrix_out, winslash = "/", mustWork = FALSE),
    metadata_file = normalizePath(metadata_out, winslash = "/", mustWork = FALSE),
    common_genes_file = normalizePath(genes_out, winslash = "/", mustWork = FALSE),
    selected_datasets_file = normalizePath(selected_out, winslash = "/", mustWork = FALSE),
    stringsAsFactors = FALSE
  )
  write_tsv(merge_summary, summary_out)

  message("\nDone.")
  message("Matrix: ", normalizePath(matrix_out, winslash = "/", mustWork = FALSE))
  message("Metadata: ", normalizePath(metadata_out, winslash = "/", mustWork = FALSE))
  message("Common genes: ", normalizePath(genes_out, winslash = "/", mustWork = FALSE))
  message("Selected datasets: ", normalizePath(selected_out, winslash = "/", mustWork = FALSE))
  message("Summary: ", normalizePath(summary_out, winslash = "/", mustWork = FALSE))
}

main()

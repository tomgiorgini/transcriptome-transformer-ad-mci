options(stringsAsFactors = FALSE)

# Build expression datasets from GSE63060 + GSE63061 + GSE140829, using
# each dataset's GEO matrix and matrix_row_to_gene_symbol.tsv.
# Outputs:
# - AD/MCI-only dataset
# - AD/MCI/CTL dataset
# Both are ComBat-corrected by dataset/platform.

find_thesis_dir <- function() {
  candidates <- unique(normalizePath(c(getwd(), dirname(getwd())), winslash = "/", mustWork = FALSE))
  for (path in candidates) {
    if (file.exists(file.path(path, "task_dataset", "matrix.txt")) &&
        dir.exists(file.path(path, "pretraining_dataset", "geo_downloads"))) {
      return(path)
    }
  }
  stop("Could not find thesis directory containing task_dataset/matrix.txt and pretraining_dataset/geo_downloads.",
       call. = FALSE)
}

thesis_dir <- find_thesis_dir()

cfg <- list(
  thesis_dir = thesis_dir,
  geo_dir = file.path(thesis_dir, "pretraining_dataset", "geo_downloads"),
  out_dir = file.path(thesis_dir, "pretraining_dataset", "ad_mci_combat_dataset")
)

dataset_cfg <- list(
  GSE63060 = list(
    gse_id = "GSE63060",
    platform_id = "GPL6947",
    eset_dir = "eset_1_GPL6947",
    status_col = "status:ch1",
    status_map = c(AD = "AD", MCI = "MCI", CTL = "CTL")
  ),
  GSE63061 = list(
    gse_id = "GSE63061",
    platform_id = "GPL10558",
    eset_dir = "eset_1_GPL10558",
    status_col = "status:ch1",
    status_map = c(AD = "AD", MCI = "MCI", CTL = "CTL")
  ),
  GSE140829 = list(
    gse_id = "GSE140829",
    platform_id = "GPL15988",
    eset_dir = "eset_1_GPL15988",
    status_col = "diagnosis:ch1",
    status_map = c(AD = "AD", MCI = "MCI", Control = "CTL")
  )
)

legacy_cfg <- list(
  reference_matrix = file.path(thesis_dir, "task_dataset", "matrix.txt"),
  gse140829_dir = file.path(thesis_dir, "pretraining_dataset", "geo_downloads",
                            "GSE140829", "eset_1_GPL15988")
)

legacy_cfg$gse140829_matrix <- file.path(legacy_cfg$gse140829_dir, "GSE140829_expr_matrix.tsv.gz")
legacy_cfg$gse140829_row_map <- file.path(legacy_cfg$gse140829_dir, "matrix_row_to_gene_symbol.tsv")
legacy_cfg$gse140829_metadata <- file.path(legacy_cfg$gse140829_dir, "GSE140829_sample_metadata.tsv.gz")
legacy_cfg$gse63060_metadata <- file.path(cfg$geo_dir, "GSE63060", "eset_1_GPL6947",
                                          "GSE63060_sample_metadata.tsv.gz")
legacy_cfg$gse63061_metadata <- file.path(cfg$geo_dir, "GSE63061", "eset_1_GPL10558",
                                          "GSE63061_sample_metadata.tsv.gz")

clean_text <- function(x) {
  x <- enc2utf8(as.character(x))
  x[is.na(x)] <- ""
  x <- trimws(gsub("\\s+", " ", x))
  x <- gsub('^["\']|["\']$', "", x)
  x
}

clean_colnames <- function(x) {
  clean_text(colnames(x))
}

ensure_package <- function(pkg, bioc = FALSE) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    return(invisible(TRUE))
  }
  if (bioc) {
    if (!requireNamespace("BiocManager", quietly = TRUE)) {
      install.packages("BiocManager")
    }
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  } else {
    install.packages(pkg)
  }
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Package could not be installed/loaded: ", pkg, call. = FALSE)
  }
  invisible(TRUE)
}

read_tsv <- function(path) {
  read.delim(path, sep = "\t", check.names = FALSE, quote = "",
             comment.char = "", stringsAsFactors = FALSE)
}

write_tsv <- function(x, path) {
  write.table(x, path, sep = "\t", row.names = FALSE, quote = FALSE)
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

read_metadata <- function(path) {
  md <- read_tsv(path)
  colnames(md) <- clean_colnames(md)
  for (col in colnames(md)) {
    if (is.character(md[[col]])) {
      md[[col]] <- clean_text(md[[col]])
    }
  }
  md
}

get_sample_id_column <- function(md) {
  if ("sample_id" %in% colnames(md)) {
    return("sample_id")
  }
  if ("geo_accession" %in% colnames(md)) {
    return("geo_accession")
  }
  stop("Metadata has no sample_id or geo_accession column.", call. = FALSE)
}

dataset_paths <- function(ds) {
  base <- file.path(cfg$geo_dir, ds$gse_id, ds$eset_dir)
  list(
    base = base,
    matrix = file.path(base, paste0(ds$gse_id, "_expr_matrix.tsv.gz")),
    row_map = file.path(base, "matrix_row_to_gene_symbol.tsv"),
    metadata = file.path(base, paste0(ds$gse_id, "_sample_metadata.tsv.gz"))
  )
}

build_reference_metadata <- function() {
  md60 <- read_metadata(legacy_cfg$gse63060_metadata)
  md61 <- read_metadata(legacy_cfg$gse63061_metadata)

  id60 <- get_sample_id_column(md60)
  id61 <- get_sample_id_column(md61)

  out60 <- data.frame(
    original_sample_id = clean_text(md60[[id60]]),
    gsm_id = sub("_.*$", "", clean_text(md60[[id60]])),
    gse_id = "GSE63060",
    platform_id = "GPL6947",
    status = clean_text(md60[["status:ch1"]]),
    batch = "GSE63060",
    stringsAsFactors = FALSE
  )

  out61 <- data.frame(
    original_sample_id = clean_text(md61[[id61]]),
    gsm_id = sub("_.*$", "", clean_text(md61[[id61]])),
    gse_id = "GSE63061",
    platform_id = "GPL10558",
    status = clean_text(md61[["status:ch1"]]),
    batch = "GSE63061",
    stringsAsFactors = FALSE
  )

  rbind(out60, out61)
}

load_reference_ad_mci <- function() {
  message("Loading reference matrix: GSE63060 + GSE63061...")
  mat <- read_expression_matrix(legacy_cfg$reference_matrix)
  ref_md_all <- build_reference_metadata()

  sample_ids <- clean_text(colnames(mat))
  gsm_ids <- sub("_.*$", "", sample_ids)
  idx <- match(gsm_ids, ref_md_all$gsm_id)
  if (any(is.na(idx))) {
    missing <- sample_ids[is.na(idx)]
    stop("Could not match reference matrix samples to GSE63060/GSE63061 metadata. Examples: ",
         paste(head(missing, 10), collapse = ", "), call. = FALSE)
  }

  md <- ref_md_all[idx, , drop = FALSE]
  md$original_sample_id <- sample_ids
  md$status_from_matrix_id <- sub("^.*_", "", sample_ids)

  # task_dataset/matrix.txt encodes exact status in column names; keep only exact AD/MCI.
  keep <- md$status_from_matrix_id %in% c("AD", "MCI") & md$status %in% c("AD", "MCI")
  mat <- mat[, keep, drop = FALSE]
  md <- md[keep, , drop = FALSE]

  md$sample_id <- paste(md$gse_id, md$platform_id, md$original_sample_id, sep = "__")
  rownames(md) <- md$sample_id
  colnames(mat) <- md$sample_id

  list(matrix = mat, metadata = md)
}

read_row_map <- function(path) {
  row_map <- read_tsv(path)
  required <- c("feature_id", "gene_symbol", "keep_for_gene_symbol_matrix")
  missing <- setdiff(required, colnames(row_map))
  if (length(missing) > 0) {
    stop("Row map missing columns: ", paste(missing, collapse = ", "), "\n", path, call. = FALSE)
  }
  row_map$feature_id <- clean_text(row_map$feature_id)
  row_map$gene_symbol <- clean_text(row_map$gene_symbol)
  row_map
}

collapse_to_gene_symbols <- function(mat, row_map) {
  idx <- match(rownames(mat), row_map$feature_id)
  keep <- !is.na(idx) &
    as.logical(row_map$keep_for_gene_symbol_matrix[idx]) &
    nzchar(row_map$gene_symbol[idx])

  mat <- mat[keep, , drop = FALSE]
  gene_symbol <- row_map$gene_symbol[idx[keep]]
  if (nrow(mat) == 0) {
    stop("No GSE140829 rows remain after gene-symbol mapping.", call. = FALSE)
  }

  mat_zero <- mat
  mat_zero[is.na(mat_zero)] <- 0
  summed <- rowsum(mat_zero, group = gene_symbol, reorder = FALSE)
  counts <- rowsum((!is.na(mat)) * 1, group = gene_symbol, reorder = FALSE)
  collapsed <- summed / counts
  collapsed[counts == 0] <- NA_real_
  collapsed
}

read_gse140829_row_map <- function() {
  read_row_map(legacy_cfg$gse140829_row_map)
}

load_gse140829_ad_mci <- function() {
  message("Loading GSE140829...")
  mat <- read_expression_matrix(legacy_cfg$gse140829_matrix)
  row_map <- read_gse140829_row_map()
  mat <- collapse_to_gene_symbols(mat, row_map)

  md <- read_metadata(legacy_cfg$gse140829_metadata)
  id_col <- get_sample_id_column(md)
  md$original_sample_id <- clean_text(md[[id_col]])
  md$status <- clean_text(md[["diagnosis:ch1"]])
  md$gse_id <- "GSE140829"
  md$platform_id <- "GPL15988"
  md$batch <- "GSE140829"

  idx <- match(colnames(mat), md$original_sample_id)
  if (any(is.na(idx))) {
    missing <- colnames(mat)[is.na(idx)]
    stop("Could not match GSE140829 matrix samples to metadata. Examples: ",
         paste(head(missing, 10), collapse = ", "), call. = FALSE)
  }
  md <- md[idx, , drop = FALSE]

  keep <- md$status %in% c("AD", "MCI")
  mat <- mat[, keep, drop = FALSE]
  md <- md[keep, , drop = FALSE]

  md$sample_id <- paste(md$gse_id, md$platform_id, md$original_sample_id, sep = "__")
  rownames(md) <- md$sample_id
  colnames(mat) <- md$sample_id

  md <- md[, c("sample_id", "original_sample_id", "gse_id", "platform_id", "status", "batch"), drop = FALSE]
  list(matrix = mat, metadata = md)
}

align_to_common_genes <- function(reference, gse140829) {
  common_genes <- rownames(reference$matrix)
  common_genes <- common_genes[common_genes %in% rownames(gse140829$matrix)]
  if (length(common_genes) == 0) {
    stop("No common genes between reference matrix and GSE140829.", call. = FALSE)
  }

  reference$matrix <- reference$matrix[common_genes, , drop = FALSE]
  gse140829$matrix <- gse140829$matrix[common_genes, , drop = FALSE]

  list(reference = reference, gse140829 = gse140829, common_genes = common_genes)
}

make_output_matrix <- function(mat_genes_x_samples) {
  out <- t(mat_genes_x_samples)
  data.frame(sample_id = rownames(out), out, check.names = FALSE)
}

validate_outputs <- function(raw_mat, combat_mat, metadata, allowed_statuses) {
  if (!identical(dim(raw_mat), dim(combat_mat))) {
    stop("Raw and ComBat matrices have different dimensions.", call. = FALSE)
  }
  if (!identical(colnames(raw_mat), metadata$sample_id)) {
    stop("Metadata sample_id does not match matrix sample order.", call. = FALSE)
  }
  if (!all(metadata$status %in% allowed_statuses)) {
    stop("Metadata contains statuses outside allowed set: ",
         paste(allowed_statuses, collapse = ", "), call. = FALSE)
  }
  if (anyDuplicated(rownames(raw_mat)) > 0) {
    stop("Duplicated gene symbols in final matrix.", call. = FALSE)
  }
  if (any(!is.finite(raw_mat))) {
    stop("Raw matrix contains NA/Inf after filtering.", call. = FALSE)
  }
  if (any(!is.finite(combat_mat))) {
    stop("ComBat matrix contains NA/Inf after correction.", call. = FALSE)
  }

  batch_status <- table(metadata$batch, metadata$status)
  if (any(batch_status[, "AD"] == 0) || any(batch_status[, "MCI"] == 0)) {
    stop("Each batch must contain both AD and MCI for model-preserving ComBat.", call. = FALSE)
  }

  invisible(TRUE)
}

make_dataset_summary <- function(reference, gse140829, raw_mat, combat_mat, metadata, common_genes) {
  counts <- as.data.frame(table(metadata$batch, metadata$status), stringsAsFactors = FALSE)
  colnames(counts) <- c("batch", "status", "samples")

  data.frame(
    metric = c(
      "reference_samples_ad_mci",
      "gse140829_samples_ad_mci",
      "total_samples",
      "common_genes",
      "raw_matrix_rows_genes",
      "raw_matrix_columns_samples",
      "combat_matrix_rows_genes",
      "combat_matrix_columns_samples",
      "batch_correction_method",
      "batch_variable",
      "combat_model"
    ),
    value = c(
      ncol(reference$matrix),
      ncol(gse140829$matrix),
      ncol(raw_mat),
      length(common_genes),
      nrow(raw_mat),
      ncol(raw_mat),
      nrow(combat_mat),
      ncol(combat_mat),
      "sva::ComBat",
      "dataset/platform: GSE63060, GSE63061, GSE140829",
      "~ status"
    ),
    stringsAsFactors = FALSE
  )
}

load_geo_dataset <- function(ds) {
  paths <- dataset_paths(ds)
  message("Loading ", ds$gse_id, " / ", ds$platform_id, "...")
  mat <- read_expression_matrix(paths$matrix)
  row_map <- read_row_map(paths$row_map)
  mat <- collapse_to_gene_symbols(mat, row_map)

  md <- read_metadata(paths$metadata)
  id_col <- get_sample_id_column(md)
  raw_status <- clean_text(md[[ds$status_col]])
  mapped_status <- unname(ds$status_map[raw_status])

  md_out <- data.frame(
    original_sample_id = clean_text(md[[id_col]]),
    gse_id = ds$gse_id,
    platform_id = ds$platform_id,
    status_raw = raw_status,
    status = mapped_status,
    batch = ds$gse_id,
    stringsAsFactors = FALSE
  )

  idx <- match(colnames(mat), md_out$original_sample_id)
  if (any(is.na(idx))) {
    missing <- colnames(mat)[is.na(idx)]
    stop("Could not match matrix samples to metadata for ", ds$gse_id, ". Examples: ",
         paste(head(missing, 10), collapse = ", "), call. = FALSE)
  }
  md_out <- md_out[idx, , drop = FALSE]
  md_out$sample_id <- paste(md_out$gse_id, md_out$platform_id, md_out$original_sample_id, sep = "__")
  rownames(md_out) <- md_out$sample_id
  colnames(mat) <- md_out$sample_id

  list(matrix = mat, metadata = md_out)
}

build_combat_dataset <- function(datasets, allowed_statuses, output_prefix, status_model = TRUE) {
  selected <- lapply(datasets, load_geo_dataset)

  for (i in seq_along(selected)) {
    keep <- !is.na(selected[[i]]$metadata$status) &
      selected[[i]]$metadata$status %in% allowed_statuses
    selected[[i]]$matrix <- selected[[i]]$matrix[, keep, drop = FALSE]
    selected[[i]]$metadata <- selected[[i]]$metadata[keep, , drop = FALSE]
  }

  common_genes <- rownames(selected[[1]]$matrix)
  for (i in seq_along(selected)[-1]) {
    common_genes <- common_genes[common_genes %in% rownames(selected[[i]]$matrix)]
  }
  if (length(common_genes) == 0) {
    stop("No common genes for output prefix: ", output_prefix, call. = FALSE)
  }

  for (i in seq_along(selected)) {
    selected[[i]]$matrix <- selected[[i]]$matrix[common_genes, , drop = FALSE]
  }

  raw_mat <- do.call(cbind, lapply(selected, `[[`, "matrix"))
  metadata <- do.call(rbind, lapply(selected, `[[`, "metadata"))
  metadata <- metadata[, c("sample_id", "original_sample_id", "gse_id", "platform_id",
                           "status", "status_raw", "batch"), drop = FALSE]
  raw_mat <- raw_mat[, metadata$sample_id, drop = FALSE]
  raw_mat[is.na(raw_mat)] <- 0

  message("Applying ComBat for ", output_prefix, "...")
  mod <- if (isTRUE(status_model)) model.matrix(~ status, data = metadata) else NULL
  combat_mat <- sva::ComBat(dat = raw_mat, batch = metadata$batch, mod = mod,
                            par.prior = TRUE, prior.plots = FALSE)

  validate_outputs(raw_mat, combat_mat, metadata, allowed_statuses)

  raw_out <- file.path(cfg$out_dir, paste0(output_prefix, "_raw_intersection_matrix_samples_x_genes.csv"))
  combat_out <- file.path(cfg$out_dir, paste0(output_prefix, "_combat_matrix_samples_x_genes.csv"))
  metadata_out <- file.path(cfg$out_dir, paste0(output_prefix, "_metadata.csv"))
  genes_out <- file.path(cfg$out_dir, paste0(output_prefix, "_common_genes.txt"))
  summary_out <- file.path(cfg$out_dir, paste0(output_prefix, "_dataset_summary.tsv"))
  counts_out <- file.path(cfg$out_dir, paste0(output_prefix, "_sample_counts_by_batch_status.tsv"))

  write.csv(make_output_matrix(raw_mat), raw_out, row.names = FALSE, quote = FALSE)
  write.csv(make_output_matrix(combat_mat), combat_out, row.names = FALSE, quote = FALSE)
  write.csv(metadata, metadata_out, row.names = FALSE, quote = FALSE)
  writeLines(common_genes, genes_out, useBytes = TRUE)

  counts <- as.data.frame(table(metadata$batch, metadata$status), stringsAsFactors = FALSE)
  colnames(counts) <- c("batch", "status", "samples")
  write_tsv(counts, counts_out)

  summary <- data.frame(
    metric = c(
      "datasets",
      "statuses_included",
      "total_samples",
      "common_genes",
      "raw_matrix_rows_genes",
      "raw_matrix_columns_samples",
      "combat_matrix_rows_genes",
      "combat_matrix_columns_samples",
      "batch_correction_method",
      "batch_variable",
      "combat_model",
      "ambiguous_statuses_excluded"
    ),
    value = c(
      paste(vapply(datasets, `[[`, character(1), "gse_id"), collapse = ", "),
      paste(allowed_statuses, collapse = ", "),
      ncol(raw_mat),
      length(common_genes),
      nrow(raw_mat),
      ncol(raw_mat),
      nrow(combat_mat),
      ncol(combat_mat),
      "sva::ComBat",
      "dataset/platform: GSE63060, GSE63061, GSE140829",
      ifelse(isTRUE(status_model), "~ status", "NULL"),
      "GSE63061 borderline MCI, CTL to AD, MCI to CTL, OTHER"
    ),
    stringsAsFactors = FALSE
  )
  write_tsv(summary, summary_out)

  list(
    raw = raw_out,
    combat = combat_out,
    metadata = metadata_out,
    genes = genes_out,
    summary = summary_out,
    counts = counts_out,
    n_samples = ncol(raw_mat),
    n_genes = nrow(raw_mat)
  )
}

main <- function() {
  required_files <- c(
    unlist(lapply(dataset_cfg, function(ds) {
      paths <- dataset_paths(ds)
      c(paths$matrix, paths$row_map, paths$metadata)
    }), use.names = FALSE)
  )
  missing <- required_files[!file.exists(required_files)]
  if (length(missing) > 0) {
    stop("Missing required files:\n", paste(missing, collapse = "\n"), call. = FALSE)
  }

  dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)
  ensure_package("sva", bioc = TRUE)

  ad_mci <- build_combat_dataset(
    datasets = dataset_cfg,
    allowed_statuses = c("AD", "MCI"),
    output_prefix = "ad_mci"
  )
  all_with_ctl <- build_combat_dataset(
    datasets = dataset_cfg,
    allowed_statuses = c("AD", "MCI", "CTL"),
    output_prefix = "ad_mci_ctl"
  )

  message("\nDone.")
  message("AD/MCI samples: ", ad_mci$n_samples, " | genes: ", ad_mci$n_genes)
  message("AD/MCI/CTL samples: ", all_with_ctl$n_samples, " | genes: ", all_with_ctl$n_genes)
  message("Output dir: ", normalizePath(cfg$out_dir, winslash = "/", mustWork = FALSE))
}

main()

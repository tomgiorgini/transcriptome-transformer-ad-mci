#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

parse_args <- function() {
  cfg <- list(output_dir = "task_dataset", combat = FALSE, install = FALSE)
  for (arg in commandArgs(trailingOnly = TRUE)) {
    if (startsWith(arg, "--output-dir=")) {
      cfg$output_dir <- sub("^--output-dir=", "", arg)
    } else if (identical(arg, "--combat")) {
      cfg$combat <- TRUE
    } else if (identical(arg, "--install")) {
      cfg$install <- TRUE
    } else {
      stop("Unknown argument: ", arg, call. = FALSE)
    }
  }
  cfg
}

ensure_packages <- function(use_combat, install) {
  required <- c("GEOquery", "Biobase")
  if (use_combat) required <- c(required, "sva")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) == 0) return(invisible(TRUE))
  if (!install) {
    stop(
      "Missing R package(s): ", paste(missing, collapse = ", "),
      ". Install them with BiocManager or rerun with --install.",
      call. = FALSE
    )
  }
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  }
  BiocManager::install(missing, ask = FALSE, update = FALSE)
}

select_eset <- function(gse_id) {
  series <- GEOquery::getGEO(gse_id, GSEMatrix = TRUE, getGPL = TRUE)
  if (!is.list(series)) series <- list(series)
  sizes <- vapply(series, function(eset) ncol(Biobase::exprs(eset)), numeric(1))
  series[[which.max(sizes)]]
}

find_column <- function(frame, patterns, description) {
  names_lower <- tolower(colnames(frame))
  for (pattern in patterns) {
    hits <- grep(pattern, names_lower, perl = TRUE)
    if (length(hits) > 0) return(colnames(frame)[hits[1]])
  }
  stop("Could not identify ", description, " column. Available columns: ",
       paste(colnames(frame), collapse = ", "), call. = FALSE)
}

clean_symbol <- function(x) {
  x <- trimws(as.character(x))
  x <- sub("[[:space:]]*///.*$", "", x)
  x <- sub("[[:space:]]*//.*$", "", x)
  x[x %in% c("", "---", "NA", "na")] <- NA_character_
  x
}

extract_cohort <- function(gse_id) {
  message("Downloading and preparing ", gse_id, " ...")
  eset <- select_eset(gse_id)
  expression <- Biobase::exprs(eset)
  phenotype <- data.frame(Biobase::pData(eset), check.names = FALSE)
  features <- data.frame(Biobase::fData(eset), check.names = FALSE)

  status_col <- find_column(
    phenotype,
    c("^status:ch1$", "status.*ch1", "diagnos", "disease.*state"),
    "diagnostic status"
  )
  accession_col <- find_column(phenotype, c("^geo_accession$", "accession"), "GEO accession")
  symbol_col <- find_column(
    features,
    c("^gene[._ ]?symbol$", "gene.*symbol", "^symbol$", "ilmn.*gene"),
    "gene symbol"
  )

  status <- toupper(trimws(as.character(phenotype[[status_col]])))
  status[status %in% c("CONTROL", "CN", "NC")] <- "CTL"
  keep <- status %in% c("AD", "MCI", "CTL")
  expression <- expression[, keep, drop = FALSE]
  phenotype <- phenotype[keep, , drop = FALSE]
  status <- status[keep]
  sample_ids <- trimws(as.character(phenotype[[accession_col]]))
  colnames(expression) <- sample_ids

  symbols <- clean_symbol(features[[symbol_col]])
  mapped <- !is.na(symbols)
  expression <- expression[mapped, , drop = FALSE]
  symbols <- symbols[mapped]
  aggregated <- rowsum(expression, group = symbols, reorder = TRUE) /
    as.vector(table(factor(symbols, levels = unique(sort(symbols)))))

  list(expression = aggregated, sample_ids = sample_ids, status = status)
}

write_ids <- function(ids, path) {
  writeLines(ids, con = path, useBytes = TRUE)
}

cfg <- parse_args()
ensure_packages(cfg$combat, cfg$install)
dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)

cohort_60 <- extract_cohort("GSE63060")
cohort_61 <- extract_cohort("GSE63061")
shared_genes <- intersect(rownames(cohort_60$expression), rownames(cohort_61$expression))
shared_genes <- sort(shared_genes)
matrix <- cbind(
  cohort_60$expression[shared_genes, , drop = FALSE],
  cohort_61$expression[shared_genes, , drop = FALSE]
)

if (cfg$combat) {
  batch <- c(rep("GSE63060", length(cohort_60$sample_ids)),
             rep("GSE63061", length(cohort_61$sample_ids)))
  matrix <- sva::ComBat(dat = matrix, batch = batch, mod = NULL, par.prior = TRUE)
}

all_ids <- c(cohort_60$sample_ids, cohort_61$sample_ids)
all_status <- c(cohort_60$status, cohort_61$status)
matrix <- matrix[is.finite(rowSums(matrix)), , drop = FALSE]
matrix <- matrix[rowMeans(matrix) != 0, , drop = FALSE]

utils::write.table(
  matrix,
  file = file.path(cfg$output_dir, "matrix.txt"),
  sep = "\t",
  quote = FALSE,
  col.names = NA
)
write_ids(all_ids[all_status == "AD"], file.path(cfg$output_dir, "AD.txt"))
write_ids(all_ids[all_status == "MCI"], file.path(cfg$output_dir, "MCI.txt"))
write_ids(all_ids[all_status == "CTL"], file.path(cfg$output_dir, "CTL.txt"))

manifest <- data.frame(
  dataset = c("GSE63060", "GSE63061", "combined"),
  samples = c(length(cohort_60$sample_ids), length(cohort_61$sample_ids), length(all_ids)),
  genes = c(nrow(cohort_60$expression), nrow(cohort_61$expression), nrow(matrix)),
  combat = c(FALSE, FALSE, cfg$combat)
)
utils::write.table(
  manifest,
  file = file.path(cfg$output_dir, "geo_preparation_manifest.tsv"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

message("Wrote ", nrow(matrix), " genes x ", ncol(matrix), " samples to ",
        file.path(cfg$output_dir, "matrix.txt"))

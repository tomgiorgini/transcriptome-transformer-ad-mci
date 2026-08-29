#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  cfg <- list(
    matrix = file.path("task_dataset", "matrix.txt"),
    out = file.path("pretraining_dataset", "scripts", "reference_genes.txt")
  )

  for (arg in args) {
    if (startsWith(arg, "--matrix=")) {
      cfg$matrix <- sub("^--matrix=", "", arg)
    } else if (startsWith(arg, "--out=")) {
      cfg$out <- sub("^--out=", "", arg)
    } else {
      stop("Unknown argument: ", arg, call. = FALSE)
    }
  }

  cfg
}

clean_text <- function(x) {
  x <- enc2utf8(as.character(x))
  x[is.na(x)] <- ""
  trimws(gsub("\\s+", " ", x))
}

read_reference_genes <- function(path) {
  header <- readLines(path, n = 1L, warn = FALSE)
  if (length(header) == 0) {
    stop("Reference matrix is empty: ", path, call. = FALSE)
  }

  n_cols <- length(strsplit(header, "\t", fixed = TRUE)[[1]])
  first_col <- read.delim(
    path,
    sep = "\t",
    header = TRUE,
    check.names = FALSE,
    quote = "",
    comment.char = "",
    stringsAsFactors = FALSE,
    colClasses = c("character", rep("NULL", n_cols - 1L))
  )[[1]]

  genes <- clean_text(first_col)
  genes <- genes[nzchar(genes)]
  unique(genes)
}

main <- function() {
  cfg <- parse_args()

  if (!file.exists(cfg$matrix)) {
    stop("Matrix file not found: ", cfg$matrix, call. = FALSE)
  }

  genes <- read_reference_genes(cfg$matrix)
  dir.create(dirname(cfg$out), recursive = TRUE, showWarnings = FALSE)
  writeLines(genes, con = cfg$out, useBytes = TRUE)

  message("Genes written: ", length(genes))
  message("Output: ", normalizePath(cfg$out, winslash = "/", mustWork = FALSE))
}

if (sys.nframe() == 0) {
  main()
}

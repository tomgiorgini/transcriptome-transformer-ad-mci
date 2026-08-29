#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  cfg <- list(
    root = file.path("pretraining_dataset", "geo_downloads"),
    output_name = "genes.txt",
    use_symbols = FALSE,
    install_annotation = FALSE
  )

  for (arg in args) {
    if (startsWith(arg, "--root=")) {
      cfg$root <- sub("^--root=", "", arg)
    } else if (startsWith(arg, "--output-name=")) {
      cfg$output_name <- sub("^--output-name=", "", arg)
    } else if (identical(arg, "--symbols")) {
      cfg$use_symbols <- TRUE
    } else if (identical(arg, "--install-annotation")) {
      cfg$install_annotation <- TRUE
    } else {
      stop("Unknown argument: ", arg, call. = FALSE)
    }
  }

  cfg
}

clean_text <- function(x) {
  x <- enc2utf8(as.character(x))
  x[is.na(x)] <- ""
  x <- gsub("\u00a0", " ", x, fixed = TRUE)
  trimws(gsub("\\s+", " ", x))
}

normalize_key <- function(x) {
  toupper(gsub("[^A-Za-z0-9]+", "", clean_text(x)))
}

ensure_package <- function(pkg, install_pkg = FALSE, bioc = FALSE) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    return(TRUE)
  }

  if (!install_pkg) {
    return(FALSE)
  }

  if (bioc) {
    if (!requireNamespace("BiocManager", quietly = TRUE)) {
      install.packages("BiocManager", repos = "https://cloud.r-project.org")
    }
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  } else {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }

  requireNamespace(pkg, quietly = TRUE)
}

pick_feature_column <- function(df) {
  keys <- normalize_key(colnames(df))
  priority <- c("FEATUREID", "ID", "IDREF", "PROBEID", "ILMNID", "SPOTID", "NAME")
  hits <- match(priority, keys)
  hits <- hits[!is.na(hits)]

  if (length(hits) > 0) {
    return(colnames(df)[hits[[1]]])
  }

  colnames(df)[[1]]
}

pick_symbol_columns <- function(df) {
  keys <- normalize_key(colnames(df))
  priority <- c(
    "GENESYMBOL", "OFFICIALGENESYMBOL", "SYMBOL", "GENESYMBOLS",
    "ILMNGENE", "GENE", "GENENAME", "GENEASSIGNMENT"
  )
  hits <- match(priority, keys)
  hits <- hits[!is.na(hits)]
  regex_hits <- grep("GENESYMBOL|SYMBOL|GENEASSIGNMENT|GENE", keys)
  idx <- unique(c(hits, regex_hits))

  if (length(idx) == 0) {
    return(character())
  }

  colnames(df)[idx]
}

extract_primary_symbol <- function(x) {
  x <- clean_text(x)
  x[!nzchar(x)] <- NA_character_
  x <- gsub("^\"|\"$", "", x)
  x <- gsub("\\s*///\\s*", "|", x)
  x <- gsub("\\s*//\\s*", "|", x)
  x <- gsub("\\s*;\\s*", "|", x)
  x <- gsub("\\s*,\\s*", "|", x)

  out <- vapply(strsplit(x, "\\|", fixed = FALSE), function(parts) {
    parts <- clean_text(parts)
    parts <- parts[nzchar(parts)]
    if (length(parts) == 0) {
      return(NA_character_)
    }

    candidate <- parts[[1]]
    candidate <- sub("\\s+\\[.*$", "", candidate)
    candidate <- sub("\\s+chromosome.*$", "", candidate, ignore.case = TRUE)
    candidate <- sub("\\s+gene.*$", "", candidate, ignore.case = TRUE)
    candidate <- clean_text(candidate)

    if (!nzchar(candidate) || candidate %in% c("---", "NA", "NULL", "N/A")) {
      return(NA_character_)
    }

    toupper(candidate)
  }, character(1))

  out[!nzchar(out)] <- NA_character_
  out
}

build_symbol_map <- function(df) {
  if (is.null(df) || nrow(df) == 0 || ncol(df) == 0) {
    return(character())
  }

  feature_col <- pick_feature_column(df)
  symbol_cols <- pick_symbol_columns(df)

  if (length(symbol_cols) == 0) {
    return(character())
  }

  feature_ids <- clean_text(df[[feature_col]])
  symbols <- rep(NA_character_, length(feature_ids))

  for (col in symbol_cols) {
    candidate <- extract_primary_symbol(df[[col]])
    fill <- is.na(symbols) & !is.na(candidate)
    symbols[fill] <- candidate[fill]
  }

  keep <- nzchar(feature_ids) & !is.na(symbols)
  if (!any(keep)) {
    return(character())
  }

  feature_ids <- feature_ids[keep]
  symbols <- symbols[keep]
  dup <- duplicated(feature_ids)
  stats::setNames(symbols[!dup], feature_ids[!dup])
}

read_feature_ids <- function(feature_path, expr_path) {
  if (file.exists(feature_path)) {
    feature_meta <- read.delim(
      gzfile(feature_path),
      sep = "\t",
      header = TRUE,
      check.names = FALSE,
      quote = "\"",
      comment.char = "",
      stringsAsFactors = FALSE
    )

    if (nrow(feature_meta) > 0) {
      feature_col <- pick_feature_column(feature_meta)
      feature_ids <- clean_text(feature_meta[[feature_col]])
      feature_ids <- feature_ids[nzchar(feature_ids)]
      return(list(feature_ids = feature_ids, feature_meta = feature_meta))
    }
  }

  expr_head <- read.delim(
    gzfile(expr_path),
    sep = "\t",
    header = TRUE,
    check.names = FALSE,
    quote = "\"",
    comment.char = "",
    stringsAsFactors = FALSE
  )

  if (!"feature_id" %in% colnames(expr_head)) {
    stop("No feature_id column found in ", expr_path, call. = FALSE)
  }

  feature_ids <- clean_text(expr_head$feature_id)
  feature_ids <- feature_ids[nzchar(feature_ids)]
  list(feature_ids = feature_ids, feature_meta = NULL)
}

read_platform_id <- function(sample_meta_path) {
  if (!file.exists(sample_meta_path)) {
    return("")
  }

  meta <- read.delim(
    gzfile(sample_meta_path),
    sep = "\t",
    header = TRUE,
    check.names = FALSE,
    quote = "\"",
    comment.char = "",
    stringsAsFactors = FALSE
  )

  if (!"platform_id" %in% colnames(meta)) {
    return("")
  }

  vals <- unique(clean_text(meta$platform_id))
  vals <- vals[nzchar(vals)]

  if (length(vals) == 0) {
    return("")
  }

  vals[[1]]
}

load_gpl_symbol_map <- function(platform_id, cache_dir, install_annotation = FALSE) {
  if (!nzchar(platform_id)) {
    return(character())
  }

  cache_file <- file.path(cache_dir, paste0(platform_id, "_symbol_map.rds"))
  if (file.exists(cache_file)) {
    return(readRDS(cache_file))
  }

  if (!ensure_package("GEOquery", install_pkg = install_annotation, bioc = TRUE)) {
    warning("GEOquery is unavailable; cannot load GPL annotation for ", platform_id)
    return(character())
  }

  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

  gpl <- tryCatch(
    GEOquery::getGEO(platform_id, AnnotGPL = TRUE, destdir = cache_dir),
    error = function(e) NULL
  )

  if (is.null(gpl)) {
    warning("Could not download GPL annotation for ", platform_id)
    return(character())
  }

  gpl_table <- tryCatch(GEOquery::Table(gpl), error = function(e) NULL)
  symbol_map <- build_symbol_map(gpl_table)
  saveRDS(symbol_map, cache_file)
  symbol_map
}

resolve_symbols <- function(feature_ids, feature_meta, sample_meta_path, root, install_annotation = FALSE) {
  symbol_map <- build_symbol_map(feature_meta)

  if (length(symbol_map) == 0) {
    platform_id <- read_platform_id(sample_meta_path)
    symbol_map <- load_gpl_symbol_map(
      platform_id = platform_id,
      cache_dir = file.path(root, "gpl_cache"),
      install_annotation = install_annotation
    )
  }

  symbols <- unname(symbol_map[feature_ids])
  symbols <- extract_primary_symbol(symbols)
  symbols <- symbols[!is.na(symbols) & nzchar(symbols)]
  unique(symbols)
}

write_genes <- function(genes, path) {
  genes <- unique(clean_text(genes))
  genes <- genes[nzchar(genes)]
  writeLines(genes, con = path, useBytes = TRUE)
}

process_one_expr <- function(expr_path, root, output_name, use_symbols = FALSE, install_annotation = FALSE) {
  eset_dir <- dirname(expr_path)
  dataset <- sub("_expr_matrix\\.tsv\\.gz$", "", basename(expr_path))
  feature_path <- file.path(eset_dir, paste0(dataset, "_feature_metadata.tsv.gz"))
  sample_meta_path <- file.path(eset_dir, paste0(dataset, "_sample_metadata.tsv.gz"))

  feature_info <- read_feature_ids(feature_path, expr_path)
  genes <- feature_info$feature_ids

  if (isTRUE(use_symbols)) {
    symbol_genes <- resolve_symbols(
      feature_ids = feature_info$feature_ids,
      feature_meta = feature_info$feature_meta,
      sample_meta_path = sample_meta_path,
      root = root,
      install_annotation = install_annotation
    )

    if (length(symbol_genes) > 0) {
      genes <- symbol_genes
    } else {
      warning("No symbols resolved for ", dataset, "; writing feature IDs instead.")
    }
  }

  out_path <- file.path(eset_dir, output_name)
  write_genes(genes, out_path)

  data.frame(
    dataset = dataset,
    eset_dir = normalizePath(eset_dir, winslash = "/", mustWork = FALSE),
    output = normalizePath(out_path, winslash = "/", mustWork = FALSE),
    n_genes = length(unique(clean_text(genes[nzchar(clean_text(genes))]))),
    mode = if (isTRUE(use_symbols)) "symbols" else "feature_ids",
    stringsAsFactors = FALSE
  )
}

main <- function() {
  cfg <- parse_args()
  expr_files <- list.files(
    cfg$root,
    pattern = "_expr_matrix\\.tsv\\.gz$",
    recursive = TRUE,
    full.names = TRUE
  )

  if (length(expr_files) == 0) {
    stop("No *_expr_matrix.tsv.gz files found under ", cfg$root, call. = FALSE)
  }

  rows <- lapply(sort(expr_files), function(path) {
    message("Extracting genes from ", path)
    process_one_expr(
      expr_path = path,
      root = cfg$root,
      output_name = cfg$output_name,
      use_symbols = cfg$use_symbols,
      install_annotation = cfg$install_annotation
    )
  })

  summary_df <- do.call(rbind, rows)
  summary_path <- file.path(cfg$root, "gene_extraction_summary.tsv")
  utils::write.table(summary_df, file = summary_path, sep = "\t", row.names = FALSE, quote = FALSE)

  message("Done. Datasets processed: ", nrow(summary_df))
  message("Summary: ", normalizePath(summary_path, winslash = "/", mustWork = FALSE))
}

if (sys.nframe() == 0) {
  main()
}

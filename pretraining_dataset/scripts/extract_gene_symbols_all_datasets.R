options(stringsAsFactors = FALSE)

cfg <- list(
  geo_dir = file.path("pretraining_dataset", "geo_downloads"),
  summary_file = file.path("pretraining_dataset", "geo_downloads", "download_summary.tsv"),
  platform_cache_dir = file.path("pretraining_dataset", "geo_downloads", "platform_annotations"),
  report_file = file.path("pretraining_dataset", "geo_downloads", "gene_symbol_extraction_summary.tsv"),
  exclude_gse = c("GSE63060", "GSE63061")
)

.platform_table_cache <- new.env(parent = emptyenv())

clean_text <- function(x) {
  x <- enc2utf8(as.character(x))
  x[is.na(x)] <- ""
  x <- trimws(gsub("\\s+", " ", x))
  x <- gsub('^["\']|["\']$', "", x)
  x[toupper(x) %in% c("NA", "N/A", "NULL", "NONE", "---", "--", "-")] <- ""
  x
}

read_gene_list <- function(path) {
  x <- clean_text(readLines(path, warn = FALSE))
  unique(x[nzchar(x)])
}

read_matrix_feature_ids <- function(eset_dir) {
  matrix_files <- list.files(
    eset_dir,
    pattern = "_expr_matrix\\.tsv(\\.gz)?$",
    full.names = TRUE
  )
  matrix_files <- sort(matrix_files)
  if (length(matrix_files) == 0) {
    return(list(feature_ids = character(0), matrix_file = ""))
  }

  matrix_file <- matrix_files[1]
  con <- if (grepl("\\.gz$", matrix_file, ignore.case = TRUE)) {
    gzfile(matrix_file, open = "rt")
  } else {
    file(matrix_file, open = "rt")
  }
  on.exit(close(con), add = TRUE)

  lines <- readLines(con, warn = FALSE)
  if (length(lines) <= 1L) {
    return(list(feature_ids = character(0), matrix_file = matrix_file))
  }

  feature_ids <- sub("\t.*$", "", lines[-1L])
  feature_ids <- clean_text(feature_ids)
  list(
    feature_ids = feature_ids[nzchar(feature_ids)],
    matrix_file = matrix_file
  )
}

platform_prefix <- function(gpl_id) {
  paste0(substr(gpl_id, 1, nchar(gpl_id) - 3), "nnn")
}

platform_data_url <- function(gpl_id) {
  paste0(
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=",
    gpl_id,
    "&targ=self&form=text&view=data"
  )
}

download_platform_table <- function(gpl_id, out_path) {
  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  message("Downloading platform table: ", gpl_id)
  download.file(platform_data_url(gpl_id), out_path, mode = "wb", quiet = FALSE)
}

read_platform_table <- function(gpl_id, cache_dir) {
  if (exists(gpl_id, envir = .platform_table_cache, inherits = FALSE)) {
    return(get(gpl_id, envir = .platform_table_cache, inherits = FALSE))
  }

  path <- file.path(cache_dir, paste0(gpl_id, "_platform_data.txt"))
  if (!file.exists(path)) {
    download_platform_table(gpl_id, path)
  }

  lines <- readLines(path, warn = FALSE)
  start <- grep("^!platform_table_begin", lines)[1]
  end <- grep("^!platform_table_end", lines)[1]
  if (is.na(start) || is.na(end) || end <= start + 1L) {
    stop("Could not find !platform_table in ", path, call. = FALSE)
  }

  tab <- read.delim(
    text = paste(lines[(start + 1L):(end - 1L)], collapse = "\n"),
    sep = "\t", check.names = FALSE, quote = "", comment.char = "",
    stringsAsFactors = FALSE
  )
  assign(gpl_id, tab, envir = .platform_table_cache)
  tab
}

split_symbols <- function(x) {
  x <- clean_text(x)
  x <- x[nzchar(x)]
  if (length(x) == 0) {
    return(character(0))
  }

  x <- unlist(
    strsplit(x, "\\s*///\\s*|\\s*;\\s*|\\s*,\\s*", perl = TRUE),
    use.names = FALSE
  )
  x <- clean_text(x)
  unique(x[nzchar(x)])
}

extract_symbols_from_annotation <- function(x, column_name) {
  x <- clean_text(x)
  x <- x[nzchar(x)]
  if (length(x) == 0) {
    return(character(0))
  }

  if (tolower(column_name) == "gene_assignment") {
    records <- unlist(strsplit(x, "\\s*///\\s*", perl = TRUE), use.names = FALSE)
    fields <- strsplit(records, "\\s*//\\s*", perl = TRUE)
    symbols <- vapply(fields, function(parts) {
      if (length(parts) >= 2L) parts[2L] else ""
    }, character(1))
    symbols <- clean_text(symbols)
    return(unique(symbols[nzchar(symbols)]))
  }

  split_symbols(x)
}

nonempty_count_for_column <- function(tab, col_name, probe_ids) {
  idx <- match(probe_ids, clean_text(tab$ID))
  vals <- clean_text(tab[[col_name]][idx[!is.na(idx)]])
  sum(nzchar(vals))
}

choose_annotation_column <- function(tab, platform_title, probe_ids) {
  cols <- colnames(tab)
  lower_cols <- tolower(cols)
  is_illumina <- grepl("illumina|humanht|humanref|beadchip", platform_title, ignore.case = TRUE)

  preferred <- if (is_illumina) {
    c("ILMN_Gene", "Symbol", "Gene Symbol", "Gene symbol", "GENE_SYMBOL", "GeneName", "GENE")
  } else {
    c("Gene Symbol", "Gene symbol", "GENE_SYMBOL", "Symbol", "GeneSymbol", "GENE_SYMBOLS",
      "ILMN_Gene", "Gene Name", "GENE", "gene_assignment")
  }

  preferred <- preferred[preferred %in% cols]
  if (length(preferred) > 0) {
    counts <- vapply(preferred, nonempty_count_for_column, integer(1), tab = tab, probe_ids = probe_ids)
    if (max(counts) > 0) {
      return(names(which.max(counts)))
    }
  }

  pattern_hits <- cols[
    grepl("(^|_ |\\b)(gene|symbol|ilmn)", lower_cols, perl = TRUE) &
      !grepl("title|description|ontology|sequence|chromosome|cytoband|protein|product|synonym", lower_cols)
  ]
  pattern_hits <- setdiff(pattern_hits, c("ID", "GB_ACC"))
  if (length(pattern_hits) > 0) {
    counts <- vapply(pattern_hits, nonempty_count_for_column, integer(1), tab = tab, probe_ids = probe_ids)
    if (max(counts) > 0) {
      return(names(which.max(counts)))
    }
  }

  NA_character_
}

convert_one_eset <- function(eset_dir, summary) {
  folder_name <- basename(eset_dir)
  parent_gse <- basename(dirname(eset_dir))
  platform_id <- sub("^eset_[0-9]+_", "", folder_name)
  genes_path <- file.path(eset_dir, "genes.txt")

  if (!file.exists(genes_path)) {
    stop("Missing genes.txt: ", genes_path, call. = FALSE)
  }

  row_idx <- which(summary$gse_id == parent_gse & summary$platform_id == platform_id)
  platform_title <- if (length(row_idx) > 0) summary$platform_title[row_idx[1]] else ""

  matrix_features <- read_matrix_feature_ids(eset_dir)
  matrix_feature_ids <- matrix_features$feature_ids
  if (length(matrix_feature_ids) == 0) {
    matrix_feature_ids <- readLines(genes_path, warn = FALSE)
    matrix_feature_ids <- clean_text(matrix_feature_ids)
    matrix_feature_ids <- matrix_feature_ids[nzchar(matrix_feature_ids)]
  }
  probes <- unique(matrix_feature_ids)
  tab <- read_platform_table(platform_id, cfg$platform_cache_dir)
  if (!("ID" %in% colnames(tab))) {
    stop(platform_id, " platform table has no ID column", call. = FALSE)
  }

  tab$ID <- clean_text(tab$ID)
  tab <- tab[tab$ID %in% probes, , drop = FALSE]
  chosen_col <- choose_annotation_column(tab, platform_title, probes)

  if (is.na(chosen_col)) {
    mapped_unique <- data.frame(probe_id = character(0), gene_symbol = character(0))
    row_map <- data.frame(
      matrix_row_index = seq_along(matrix_feature_ids),
      feature_id = matrix_feature_ids,
      gene_symbol = "",
      all_gene_symbols = "",
      n_gene_symbols = 0L,
      mapping_status = "unmapped",
      keep_for_gene_symbol_matrix = FALSE,
      stringsAsFactors = FALSE
    )
    symbols <- character(0)
  } else {
    raw_map <- data.frame(
      probe_id = tab$ID,
      gene_symbol_raw = clean_text(tab[[chosen_col]]),
      stringsAsFactors = FALSE
    )
    raw_map <- raw_map[nzchar(raw_map$gene_symbol_raw), , drop = FALSE]
    if (tolower(chosen_col) == "gene_assignment") {
      split_values <- lapply(raw_map$gene_symbol_raw, extract_symbols_from_annotation, column_name = chosen_col)
    } else {
      split_values <- strsplit(raw_map$gene_symbol_raw, "\\s*///\\s*|\\s*;\\s*|\\s*,\\s*", perl = TRUE)
    }
    split_lengths <- lengths(split_values)
    mapped_unique <- data.frame(
      probe_id = rep(raw_map$probe_id, split_lengths),
      gene_symbol = clean_text(unlist(split_values, use.names = FALSE)),
      stringsAsFactors = FALSE
    )
    mapped_unique <- mapped_unique[nzchar(mapped_unique$gene_symbol), , drop = FALSE]
    mapped_unique <- unique(mapped_unique)
    symbols <- unique(mapped_unique$gene_symbol)

    symbols_by_probe <- split(mapped_unique$gene_symbol, mapped_unique$probe_id)
    symbols_by_probe <- lapply(symbols_by_probe, unique)
    probe_symbol_counts <- lengths(symbols_by_probe)
    all_symbols <- vapply(matrix_feature_ids, function(feature_id) {
      values <- symbols_by_probe[[feature_id]]
      if (is.null(values) || length(values) == 0) "" else paste(values, collapse = ";")
    }, character(1))
    primary_symbols <- sub(";.*$", "", all_symbols)
    n_symbols <- as.integer(probe_symbol_counts[matrix_feature_ids])
    n_symbols[is.na(n_symbols)] <- 0L
    row_map <- data.frame(
      matrix_row_index = seq_along(matrix_feature_ids),
      feature_id = matrix_feature_ids,
      gene_symbol = primary_symbols,
      all_gene_symbols = all_symbols,
      n_gene_symbols = n_symbols,
      mapping_status = ifelse(n_symbols == 0L, "unmapped",
                              ifelse(n_symbols > 1L, "multiple_symbols", "mapped")),
      keep_for_gene_symbol_matrix = n_symbols > 0L,
      stringsAsFactors = FALSE
    )
  }

  duplicated_symbol_rows <- row_map[row_map$keep_for_gene_symbol_matrix, , drop = FALSE]
  duplicated_symbol_rows <- duplicated_symbol_rows[nzchar(duplicated_symbol_rows$gene_symbol), , drop = FALSE]
  if (nrow(duplicated_symbol_rows) > 0) {
    symbol_table <- table(duplicated_symbol_rows$gene_symbol)
    duplicated_symbols <- names(symbol_table)[symbol_table > 1L]
  } else {
    duplicated_symbols <- character(0)
  }

  if (length(duplicated_symbols) > 0) {
    duplicated_symbol_rows <- duplicated_symbol_rows[
      duplicated_symbol_rows$gene_symbol %in% duplicated_symbols,
      ,
      drop = FALSE
    ]
    rows_by_symbol <- split(duplicated_symbol_rows, duplicated_symbol_rows$gene_symbol)
    duplicate_groups <- do.call(rbind, lapply(names(rows_by_symbol), function(symbol) {
      rows <- rows_by_symbol[[symbol]]
      data.frame(
        gene_symbol = symbol,
        n_matrix_rows = nrow(rows),
        matrix_row_indices = paste(rows$matrix_row_index, collapse = ";"),
        feature_ids = paste(unique(rows$feature_id), collapse = ";"),
        stringsAsFactors = FALSE
      )
    }))
    duplicate_groups <- duplicate_groups[order(-duplicate_groups$n_matrix_rows, duplicate_groups$gene_symbol), ]
  } else {
    duplicate_groups <- data.frame(
      gene_symbol = character(0),
      n_matrix_rows = integer(0),
      matrix_row_indices = character(0),
      feature_ids = character(0),
      stringsAsFactors = FALSE
    )
  }

  symbols_out <- file.path(eset_dir, "gene_symbols.txt")
  symbols_alias_out <- file.path(eset_dir, "gene_simbols.txt")
  mapping_out <- file.path(eset_dir, "probe_to_gene_symbol.tsv")
  row_mapping_out <- file.path(eset_dir, "matrix_row_to_gene_symbol.tsv")
  duplicate_groups_out <- file.path(eset_dir, "duplicate_gene_symbol_groups.tsv")
  writeLines(symbols, symbols_out, useBytes = TRUE)
  writeLines(symbols, symbols_alias_out, useBytes = TRUE)
  write.table(mapped_unique, mapping_out, sep = "\t", row.names = FALSE, quote = FALSE)
  write.table(row_map, row_mapping_out, sep = "\t", row.names = FALSE, quote = FALSE)
  write.table(duplicate_groups, duplicate_groups_out, sep = "\t", row.names = FALSE, quote = FALSE)

  data.frame(
    gse_id = parent_gse,
    eset_dir = folder_name,
    platform_id = platform_id,
    platform_title = platform_title,
    matrix_rows = length(matrix_feature_ids),
    input_probe_ids = length(probes),
    probes_found_in_platform_table = length(unique(tab$ID)),
    annotation_column_used = ifelse(is.na(chosen_col), "", chosen_col),
    probes_with_symbol = length(unique(mapped_unique$probe_id)),
    unique_gene_symbols = length(symbols),
    matrix_rows_with_symbol = sum(row_map$keep_for_gene_symbol_matrix),
    unique_primary_gene_symbols = length(unique(row_map$gene_symbol[nzchar(row_map$gene_symbol)])),
    duplicated_primary_gene_symbols = length(duplicated_symbols),
    unmapped_probe_ids = length(setdiff(probes, unique(mapped_unique$probe_id))),
    gene_symbols_file = normalizePath(symbols_out, winslash = "/", mustWork = FALSE),
    mapping_file = normalizePath(mapping_out, winslash = "/", mustWork = FALSE),
    matrix_row_mapping_file = normalizePath(row_mapping_out, winslash = "/", mustWork = FALSE),
    duplicate_groups_file = normalizePath(duplicate_groups_out, winslash = "/", mustWork = FALSE),
    error = "",
    stringsAsFactors = FALSE
  )
}

main <- function() {
  if (!file.exists(cfg$summary_file)) {
    stop("Summary file not found: ", cfg$summary_file, call. = FALSE)
  }
  dir.create(cfg$platform_cache_dir, recursive = TRUE, showWarnings = FALSE)

  summary <- read.delim(cfg$summary_file, sep = "\t", check.names = FALSE, quote = "",
                        comment.char = "", stringsAsFactors = FALSE)
  required <- c("gse_id", "platform_id", "platform_title")
  missing <- setdiff(required, colnames(summary))
  if (length(missing) > 0) {
    stop("download_summary.tsv missing columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }

  eset_dirs <- list.dirs(cfg$geo_dir, recursive = TRUE, full.names = TRUE)
  eset_dirs <- eset_dirs[grepl("[/\\\\]eset_[0-9]+_", eset_dirs)]
  eset_dirs <- sort(eset_dirs[file.exists(file.path(eset_dirs, "genes.txt"))])
  eset_dirs <- eset_dirs[!(basename(dirname(eset_dirs)) %in% cfg$exclude_gse)]

  reports <- list()
  for (i in seq_along(eset_dirs)) {
    message("[", i, "/", length(eset_dirs), "] ", eset_dirs[i])
    reports[[i]] <- tryCatch(
      convert_one_eset(eset_dirs[i], summary),
      error = function(e) {
        data.frame(
          gse_id = basename(dirname(eset_dirs[i])),
          eset_dir = basename(eset_dirs[i]),
          platform_id = sub("^eset_[0-9]+_", "", basename(eset_dirs[i])),
          platform_title = "",
          matrix_rows = NA_integer_,
          input_probe_ids = NA_integer_,
          probes_found_in_platform_table = NA_integer_,
          annotation_column_used = "",
          probes_with_symbol = NA_integer_,
          unique_gene_symbols = NA_integer_,
          matrix_rows_with_symbol = NA_integer_,
          unique_primary_gene_symbols = NA_integer_,
          duplicated_primary_gene_symbols = NA_integer_,
          unmapped_probe_ids = NA_integer_,
          gene_symbols_file = "",
          mapping_file = "",
          matrix_row_mapping_file = "",
          duplicate_groups_file = "",
          error = conditionMessage(e),
          stringsAsFactors = FALSE
        )
      }
    )
    partial_report <- do.call(rbind, reports)
    write.table(partial_report, cfg$report_file, sep = "\t", row.names = FALSE, quote = FALSE)
  }

  report <- do.call(rbind, reports)
  if (!("error" %in% colnames(report))) {
    report$error <- ""
  }
  write.table(report, cfg$report_file, sep = "\t", row.names = FALSE, quote = FALSE)
  print(report[, c("gse_id", "platform_id", "annotation_column_used",
                   "matrix_rows", "matrix_rows_with_symbol",
                   "unique_primary_gene_symbols",
                   "duplicated_primary_gene_symbols", "error")], row.names = FALSE)
  message("Report: ", normalizePath(cfg$report_file, winslash = "/", mustWork = FALSE))
}

main()

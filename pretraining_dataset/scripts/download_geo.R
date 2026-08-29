#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

default_gse_ids <- c(
  "GSE833", "GSE1297", "GSE3790", "GSE5281", "GSE6575",
  "GSE6613", "GSE7621", "GSE13162", "GSE13732", "GSE17048",
  "GSE18123", "GSE20292", "GSE20333", "GSE21942", "GSE22491",
  "GSE23832", "GSE26927", "GSE28146", "GSE29378", "GSE32915",
  "GSE36980", "GSE37772", "GSE41848", "GSE48350", "GSE52946",
  "GSE57475", "GSE61405", "GSE63060", "GSE63061", "GSE64810",
  "GSE68605", "GSE68719", "GSE72267", "GSE95587", "GSE97760",
  "GSE99039", "GSE122063", "GSE122649", "GSE126427", "GSE135589",
  "GSE138614", "GSE140829", "GSE153104", "GSE153960", "GSE165082",
  "GSE174332", "GSE213897", "GSE261050", "GSE270454", "GSE272626"
)

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  cfg <- list(
    out_dir = file.path("pretraining_dataset", "geo_downloads"),
    gse_ids = default_gse_ids,
    install_pkgs = FALSE,
    skip_existing = TRUE,
    get_gpl = FALSE
  )

  for (arg in args) {
    if (startsWith(arg, "--out=")) {
      cfg$out_dir <- sub("^--out=", "", arg)
    } else if (startsWith(arg, "--gse=")) {
      raw_ids <- sub("^--gse=", "", arg)
      ids <- trimws(unlist(strsplit(raw_ids, ",")))
      cfg$gse_ids <- unique(ids[nzchar(ids)])
    } else if (identical(arg, "--install")) {
      cfg$install_pkgs <- TRUE
    } else if (identical(arg, "--overwrite")) {
      cfg$skip_existing <- FALSE
    } else if (identical(arg, "--getGPL")) {
      cfg$get_gpl <- TRUE
    } else {
      stop("Unknown argument: ", arg, call. = FALSE)
    }
  }

  cfg
}

ensure_packages <- function(install_pkgs = FALSE) {
  needed <- c("GEOquery", "Biobase")
  missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]

  if (length(missing) == 0) {
    return(invisible(TRUE))
  }

  if (!install_pkgs) {
    stop(
      "Missing required package(s): ", paste(missing, collapse = ", "),
      ". Re-run with --install or install them manually.",
      call. = FALSE
    )
  }

  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  }

  for (pkg in missing) {
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  }

  still_missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
  if (length(still_missing) > 0) {
    stop("Package installation failed for: ", paste(still_missing, collapse = ", "), call. = FALSE)
  }
}

safe_name <- function(x) {
  x <- as.character(x)
  x[is.na(x) | !nzchar(x)] <- "unknown"
  x <- gsub("[^A-Za-z0-9._-]+", "_", x)
  x <- gsub("_+", "_", x)
  gsub("^_|_$", "", x)
}

write_table_gz <- function(df, path) {
  con <- gzfile(path, open = "wt")
  on.exit(close(con), add = TRUE)
  utils::write.table(
    df,
    file = con,
    sep = "\t",
    row.names = FALSE,
    quote = TRUE,
    na = ""
  )
}

make_expr_df <- function(expr_mat) {
  if (is.null(expr_mat) || nrow(expr_mat) == 0) {
    return(data.frame(feature_id = character(), stringsAsFactors = FALSE))
  }

  data.frame(
    feature_id = rownames(expr_mat),
    expr_mat,
    check.names = FALSE
  )
}

make_sample_df <- function(eset) {
  pd <- data.frame(Biobase::pData(eset), check.names = FALSE)
  sample_ids <- rownames(pd)
  if (is.null(sample_ids) || length(sample_ids) == 0) {
    sample_ids <- colnames(Biobase::exprs(eset))
  }

  if (!"sample_id" %in% colnames(pd)) {
    pd <- data.frame(sample_id = sample_ids, pd, check.names = FALSE)
  }

  pd
}

make_feature_df <- function(eset) {
  fd <- data.frame(Biobase::fData(eset), check.names = FALSE)
  feature_ids <- rownames(fd)
  if (is.null(feature_ids) || length(feature_ids) == 0) {
    feature_ids <- rownames(Biobase::exprs(eset))
  }

  if (is.null(fd) || nrow(fd) == 0) {
    fd <- data.frame(feature_id = feature_ids, stringsAsFactors = FALSE)
  } else if (!"feature_id" %in% colnames(fd)) {
    fd <- data.frame(feature_id = feature_ids, fd, check.names = FALSE)
  }

  fd
}

save_one_eset <- function(eset, gse_id, idx, out_dir) {
  platform <- safe_name(Biobase::annotation(eset))
  if (!nzchar(platform)) {
    platform <- "unknown_platform"
  }

  eset_dir <- file.path(out_dir, gse_id, paste0("eset_", idx, "_", platform))
  dir.create(eset_dir, recursive = TRUE, showWarnings = FALSE)

  expr_mat <- Biobase::exprs(eset)
  expr_df <- make_expr_df(expr_mat)
  sample_df <- make_sample_df(eset)
  feature_df <- make_feature_df(eset)

  saveRDS(eset, file.path(eset_dir, paste0(gse_id, "_eset.rds")))
  write_table_gz(expr_df, file.path(eset_dir, paste0(gse_id, "_expr_matrix.tsv.gz")))
  write_table_gz(sample_df, file.path(eset_dir, paste0(gse_id, "_sample_metadata.tsv.gz")))
  write_table_gz(feature_df, file.path(eset_dir, paste0(gse_id, "_feature_metadata.tsv.gz")))

  data.frame(
    gse_id = gse_id,
    eset_index = idx,
    platform = platform,
    n_features = nrow(expr_mat),
    n_samples = ncol(expr_mat),
    out_dir = normalizePath(eset_dir, winslash = "/", mustWork = FALSE),
    stringsAsFactors = FALSE
  )
}

download_one_gse <- function(gse_id, out_dir, skip_existing = TRUE, get_gpl = FALSE) {
  gse_dir <- file.path(out_dir, gse_id)
  done_file <- file.path(gse_dir, "_DOWNLOAD_COMPLETE")

  if (skip_existing && file.exists(done_file)) {
    message("Skipping existing complete dataset: ", gse_id)
    return(data.frame(
      gse_id = gse_id,
      status = "skipped_existing",
      eset_count = NA_integer_,
      n_features_total = NA_integer_,
      n_samples_total = NA_integer_,
      stringsAsFactors = FALSE
    ))
  }

  dir.create(gse_dir, recursive = TRUE, showWarnings = FALSE)
  gse_dir <- normalizePath(gse_dir, winslash = "/", mustWork = TRUE)
  message("Downloading ", gse_id)

  eset_list <- GEOquery::getGEO(
    gse_id,
    GSEMatrix = TRUE,
    getGPL = get_gpl,
    destdir = gse_dir
  )

  if (!is.list(eset_list)) {
    eset_list <- list(eset_list)
  }

  saveRDS(eset_list, file.path(gse_dir, paste0(gse_id, "_eset_list.rds")))

  bundle_rows <- lapply(seq_along(eset_list), function(i) {
    save_one_eset(eset_list[[i]], gse_id = gse_id, idx = i, out_dir = out_dir)
  })
  bundle_summary <- do.call(rbind, bundle_rows)

  utils::write.table(
    bundle_summary,
    file = file.path(gse_dir, paste0(gse_id, "_download_summary.tsv")),
    sep = "\t",
    row.names = FALSE,
    quote = FALSE
  )

  writeLines(as.character(Sys.time()), done_file, useBytes = TRUE)

  data.frame(
    gse_id = gse_id,
    status = "downloaded",
    eset_count = length(eset_list),
    n_features_total = sum(bundle_summary$n_features),
    n_samples_total = sum(bundle_summary$n_samples),
    stringsAsFactors = FALSE
  )
}

main <- function() {
  cfg <- parse_args()
  ensure_packages(cfg$install_pkgs)
  dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)
  cfg$out_dir <- normalizePath(cfg$out_dir, winslash = "/", mustWork = TRUE)

  summaries <- list()
  errors <- list()

  for (gse_id in cfg$gse_ids) {
    result <- tryCatch(
      download_one_gse(
        gse_id = gse_id,
        out_dir = cfg$out_dir,
        skip_existing = cfg$skip_existing,
        get_gpl = cfg$get_gpl
      ),
      error = function(e) {
        message("Failed ", gse_id, ": ", conditionMessage(e))
        errors[[length(errors) + 1]] <<- data.frame(
          gse_id = gse_id,
          error = conditionMessage(e),
          stringsAsFactors = FALSE
        )
        NULL
      }
    )

    if (!is.null(result)) {
      summaries[[length(summaries) + 1]] <- result
    }
  }

  if (length(summaries) > 0) {
    summary_df <- do.call(rbind, summaries)
    summary_path <- file.path(cfg$out_dir, "download_summary.tsv")
    if (file.exists(summary_path)) {
      existing_summary <- read.delim(summary_path, sep = "\t", header = TRUE, check.names = FALSE)
      existing_summary <- existing_summary[!existing_summary$gse_id %in% summary_df$gse_id, , drop = FALSE]
      summary_df <- rbind(existing_summary, summary_df)
      summary_df <- summary_df[order(summary_df$gse_id), , drop = FALSE]
    }
    utils::write.table(
      summary_df,
      file = summary_path,
      sep = "\t",
      row.names = FALSE,
      quote = FALSE
    )
  }

  if (length(errors) > 0) {
    error_df <- do.call(rbind, errors)
    error_path <- file.path(cfg$out_dir, "download_errors.tsv")
    if (file.exists(error_path)) {
      existing_errors <- read.delim(error_path, sep = "\t", header = TRUE, check.names = FALSE)
      existing_errors <- existing_errors[!existing_errors$gse_id %in% error_df$gse_id, , drop = FALSE]
      error_df <- rbind(existing_errors, error_df)
      error_df <- error_df[order(error_df$gse_id), , drop = FALSE]
    }
    utils::write.table(
      error_df,
      file = error_path,
      sep = "\t",
      row.names = FALSE,
      quote = FALSE
    )
  }

  message("Done. Requested datasets: ", length(cfg$gse_ids), "; errors: ", length(errors))
}

if (sys.nframe() == 0) {
  main()
}

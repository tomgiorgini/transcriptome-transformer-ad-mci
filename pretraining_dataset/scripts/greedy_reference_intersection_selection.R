options(stringsAsFactors = FALSE)

cfg <- list(
  geo_dir = file.path("pretraining_dataset", "geo_downloads"),
  reference_genes = file.path("pretraining_dataset", "scripts", "reference_genes.txt"),
  deg_ad_vs_mci = file.path("deg_analysis", "Results", "DEG", "AD_vs_MCI", "DEG.txt"),
  summary_file = file.path("pretraining_dataset", "geo_downloads", "download_summary.tsv"),
  out_dir = file.path("pretraining_dataset", "geo_downloads", "qc_reports"),
  exclude_gse = c("GSE63060", "GSE63061"),
  exclude_dataset_id = c("GSE3790__eset_2_GPL97"),
  sample_weight = 0
)

clean_text <- function(x) {
  x <- enc2utf8(as.character(x))
  x[is.na(x)] <- ""
  replacements <- c(
    "â€™" = "'",
    "â€˜" = "'",
    "â€œ" = "\"",
    "â€\u009d" = "\"",
    "â€“" = "-",
    "â€”" = "-",
    "Â " = " ",
    "Â" = "",
    "Ã¯" = "i"
  )
  for (pattern in names(replacements)) {
    x <- gsub(pattern, replacements[[pattern]], x, fixed = TRUE)
  }
  trimws(gsub("\\s+", " ", x))
}

read_gene_list <- function(path) {
  x <- clean_text(readLines(path, warn = FALSE))
  unique(x[nzchar(x)])
}

read_deg_genes <- function(path) {
  deg <- read.delim(path, sep = "\t", check.names = FALSE, quote = "",
                    comment.char = "", stringsAsFactors = FALSE)
  if (!("GeneSymbol" %in% colnames(deg))) {
    stop("DEG file has no GeneSymbol column: ", path, call. = FALSE)
  }
  genes <- clean_text(deg$GeneSymbol)
  unique(genes[nzchar(genes)])
}

condition_overview <- function(gse_id, platform_id, summary) {
  if (gse_id == "reference") {
    return("Reference Alzheimer / Mild cognitive impairment / Control cohort from GSE63060 and GSE63061")
  }

  high_level_conditions <- c(
    GSE122063 = "Vascular dementia / Alzheimer / Control",
    GSE1297 = "Alzheimer severity / Control",
    GSE13162 = "Frontotemporal dementia / Control",
    GSE135589 = "Huntington progression / Control",
    GSE13732 = "Clinically isolated syndrome / Multiple sclerosis risk / Control",
    GSE140829 = "Alzheimer / Mild cognitive impairment / Control / Frontotemporal dementia spectrum",
    GSE17048 = "Multiple sclerosis subtypes / Control",
    GSE20292 = "Parkinson / Control",
    GSE21942 = "Multiple sclerosis / Control",
    GSE22491 = "Parkinson / Control",
    GSE23832 = "Multiple sclerosis / Control",
    GSE26927 = "Mixed neurodegenerative diseases / Control",
    GSE28146 = "Alzheimer severity / Control",
    GSE29378 = "Alzheimer / Control",
    GSE32915 = "Multiple sclerosis lesion tissue / Control tissue",
    GSE36980 = "Alzheimer / Non-Alzheimer comparison",
    GSE37772 = "Autism spectrum disorder family cohort",
    GSE3790 = "Huntington progression / Control",
    GSE41848 = "Multiple sclerosis treatment response",
    GSE48350 = "Alzheimer / Brain aging / Control",
    GSE5281 = "Alzheimer / Brain-region controls",
    GSE57475 = "Parkinson / Control",
    GSE63060 = "Alzheimer / Mild cognitive impairment / Control",
    GSE63061 = "Alzheimer / Mild cognitive impairment / Control",
    GSE6575 = "Autism spectrum disorder / Control",
    GSE6613 = "Parkinson / Neurological disease control / Healthy control",
    GSE68605 = "Amyotrophic lateral sclerosis with C9ORF72 mutation / Control",
    GSE72267 = "Early Parkinson / Control",
    GSE7621 = "Parkinson / Control",
    GSE97760 = "Advanced Alzheimer / Control",
    GSE99039 = "Parkinson / Related neurological disorders / Control"
  )

  if (gse_id %in% names(high_level_conditions)) {
    return(unname(high_level_conditions[[gse_id]]))
  }

  row_idx <- which(summary$gse_id == gse_id & summary$platform_id == platform_id)
  if (length(row_idx) == 0) {
    return("")
  }

  disease <- if ("disease_focus" %in% colnames(summary)) clean_text(summary$disease_focus[row_idx[1]]) else ""
  conditions <- if ("conditions" %in% colnames(summary)) clean_text(summary$conditions[row_idx[1]]) else ""
  tissue <- if ("tissue_or_source" %in% colnames(summary)) clean_text(summary$tissue_or_source[row_idx[1]]) else ""

  parts <- character(0)
  if (nzchar(disease)) {
    parts <- c(parts, disease)
  }
  if (nzchar(conditions)) {
    simplified <- gsub("\\s*\\|\\s*", " vs ", conditions)
    simplified <- gsub(";.*$", "", simplified)
    parts <- c(parts, simplified)
  }
  if (nzchar(tissue)) {
    simplified_tissue <- gsub("\\s*\\|\\s*", ", ", tissue)
    parts <- c(parts, paste0("source: ", simplified_tissue))
  }

  paste(unique(parts), collapse = " - ")
}

write_xlsx_report <- function(report, path) {
  if (!requireNamespace("openxlsx", quietly = TRUE)) {
    warning("Package openxlsx is not installed; only TSV was written.", call. = FALSE)
    return(invisible(FALSE))
  }

  wb <- openxlsx::createWorkbook()
  openxlsx::addWorksheet(wb, "greedy_iterations")
  openxlsx::writeDataTable(wb, "greedy_iterations", report, tableStyle = "TableStyleMedium2")
  openxlsx::freezePane(wb, "greedy_iterations", firstRow = TRUE)
  openxlsx::setColWidths(wb, "greedy_iterations", cols = 1:ncol(report), widths = "auto")
  if ("conditions" %in% colnames(report)) {
    openxlsx::setColWidths(wb, "greedy_iterations", cols = which(colnames(report) == "conditions"), widths = 52)
  }
  openxlsx::saveWorkbook(wb, path, overwrite = TRUE)
  invisible(TRUE)
}

count_matrix_samples <- function(eset_dir) {
  matrix_files <- list.files(eset_dir, pattern = "_expr_matrix\\.tsv\\.gz$", full.names = TRUE)
  if (length(matrix_files) == 0) {
    return(NA_integer_)
  }

  header <- readLines(gzfile(matrix_files[1]), n = 1L, warn = FALSE)
  if (length(header) == 0) {
    return(NA_integer_)
  }

  max(length(strsplit(header, "\t", fixed = TRUE)[[1]]) - 1L, 0L)
}

count_reference_samples <- function(path) {
  header <- readLines(path, n = 1L, warn = FALSE)
  if (length(header) == 0) {
    return(0L)
  }
  max(length(strsplit(header, "\t", fixed = TRUE)[[1]]) - 1L, 0L)
}

collect_datasets <- function() {
  eset_dirs <- list.dirs(cfg$geo_dir, recursive = TRUE, full.names = TRUE)
  eset_dirs <- eset_dirs[grepl("[/\\\\]eset_[0-9]+_", eset_dirs)]
  eset_dirs <- sort(eset_dirs)

  rows <- lapply(eset_dirs, function(eset_dir) {
    gse_id <- basename(dirname(eset_dir))
    platform_id <- sub("^eset_[0-9]+_", "", basename(eset_dir))
    gene_file <- file.path(eset_dir, "gene_symbols.txt")
    if (!file.exists(gene_file)) {
      gene_file <- file.path(eset_dir, "gene_simbols.txt")
    }
        dataset_id <- paste(gse_id, basename(eset_dir), sep = "__")
        if (!file.exists(gene_file) || gse_id %in% cfg$exclude_gse ||
            dataset_id %in% cfg$exclude_dataset_id) {
      return(NULL)
    }

    genes <- read_gene_list(gene_file)
    data.frame(
      dataset_id = dataset_id,
      gse_id = gse_id,
      eset_dir = basename(eset_dir),
      platform_id = platform_id,
      sample_count = count_matrix_samples(eset_dir),
      gene_count = length(genes),
      gene_file = normalizePath(gene_file, winslash = "/", mustWork = FALSE),
      stringsAsFactors = FALSE
    )
  })

  out <- do.call(rbind, rows[!vapply(rows, is.null, logical(1))])
  rownames(out) <- NULL
  out
}

run_greedy <- function(reference, reference_sample_count, datasets, deg_genes, summary) {
  gene_sets <- setNames(lapply(datasets$gene_file, read_gene_list), datasets$dataset_id)
  remaining <- datasets$dataset_id
  current_genes <- reference
  cumulative_samples <- reference_sample_count
  reports <- list(data.frame(
    iteration = 0L,
    gse_id = "reference",
    platform_id = "",
    conditions = condition_overview("reference", "", summary),
    dataset_gene_symbols = length(reference),
    added_samples = reference_sample_count,
    cumulative_samples = reference_sample_count,
    genes_before_intersection = length(reference),
    genes_after_intersection = length(reference),
    genes_lost_this_step = 0L,
    deg_ad_vs_mci_intersection = sum(reference %in% deg_genes),
    stringsAsFactors = FALSE
  ))

  iteration <- 1L
  while (length(remaining) > 0 && length(current_genes) > 0) {
    scores <- do.call(rbind, lapply(remaining, function(dataset_id) {
      idx <- match(dataset_id, datasets$dataset_id)
      after_count <- sum(current_genes %in% gene_sets[[dataset_id]])
      added_samples <- ifelse(is.na(datasets$sample_count[idx]), 0L, datasets$sample_count[idx])
      cumulative_samples_after_candidate <- cumulative_samples + added_samples
      data.frame(
        dataset_id = dataset_id,
        candidate_intersection_genes = after_count,
        sample_count = added_samples,
        candidate_cumulative_samples = cumulative_samples_after_candidate,
        selection_score = after_count + cfg$sample_weight * cumulative_samples_after_candidate,
        gene_count = datasets$gene_count[idx],
        stringsAsFactors = FALSE
      )
    }))

    # Primary objective: configured score. Tie-breakers: keep more genes, then add more samples.
    scores <- scores[order(-scores$selection_score,
                           -scores$candidate_intersection_genes,
                           -scores$sample_count,
                           scores$dataset_id), ]
    chosen <- scores[1, ]
    idx <- match(chosen$dataset_id, datasets$dataset_id)

    before_genes <- length(current_genes)
    selected_genes <- gene_sets[[chosen$dataset_id]]
    current_genes <- current_genes[current_genes %in% selected_genes]
    after_genes <- length(current_genes)
    added_samples <- ifelse(is.na(datasets$sample_count[idx]), 0L, datasets$sample_count[idx])
    cumulative_samples <- cumulative_samples + added_samples

    reports[[length(reports) + 1L]] <- data.frame(
      iteration = iteration,
      gse_id = datasets$gse_id[idx],
      platform_id = datasets$platform_id[idx],
      conditions = condition_overview(datasets$gse_id[idx], datasets$platform_id[idx], summary),
      dataset_gene_symbols = datasets$gene_count[idx],
      added_samples = added_samples,
      cumulative_samples = cumulative_samples,
      genes_before_intersection = before_genes,
      genes_after_intersection = after_genes,
      genes_lost_this_step = before_genes - after_genes,
      deg_ad_vs_mci_intersection = sum(current_genes %in% deg_genes),
      stringsAsFactors = FALSE
    )

    remaining <- setdiff(remaining, chosen$dataset_id)
    iteration <- iteration + 1L
  }

  list(report = do.call(rbind, reports), final_genes = current_genes)
}

main <- function() {
  if (!file.exists(cfg$reference_genes)) {
    stop("Reference genes file not found: ", cfg$reference_genes, call. = FALSE)
  }
  if (!file.exists(cfg$deg_ad_vs_mci)) {
    stop("DEG AD_vs_MCI file not found: ", cfg$deg_ad_vs_mci, call. = FALSE)
  }
  dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

  reference <- read_gene_list(cfg$reference_genes)
  reference_sample_count <- count_reference_samples(file.path("task_dataset", "matrix.txt"))
  deg_genes <- read_deg_genes(cfg$deg_ad_vs_mci)
  datasets <- collect_datasets()
  if (nrow(datasets) == 0) {
    stop("No candidate datasets found with gene_symbols.txt", call. = FALSE)
  }

  summary <- read.delim(cfg$summary_file, sep = "\t", check.names = FALSE, quote = "",
                        comment.char = "", stringsAsFactors = FALSE)
  result <- run_greedy(reference, reference_sample_count, datasets, deg_genes, summary)

  report_out <- file.path(cfg$out_dir, "greedy_reference_intersection_iterations_clean.tsv")
  xlsx_out <- file.path(cfg$out_dir, "greedy_reference_intersection_iterations_clean.xlsx")

  write.table(result$report, report_out, sep = "\t", row.names = FALSE, quote = FALSE)
  write_xlsx_report(result$report, xlsx_out)

  print(result$report[, c(
    "iteration", "gse_id", "platform_id", "added_samples",
    "cumulative_samples", "genes_after_intersection", "deg_ad_vs_mci_intersection"
  )], row.names = FALSE)
  message("Iterations: ", normalizePath(report_out, winslash = "/", mustWork = FALSE))
  message("Excel: ", normalizePath(xlsx_out, winslash = "/", mustWork = FALSE))
}

main()

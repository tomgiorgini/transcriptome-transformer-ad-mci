options(stringsAsFactors = FALSE)

ensure_enrichment_packages <- function() {
  required <- c("enrichR", "ggplot2", "forcats", "stringr")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(sprintf("Missing R packages: %s", paste(missing, collapse = ", ")), call. = FALSE)
  }

  suppressPackageStartupMessages(library(enrichR))
  invisible(enrichR::listEnrichrSites())
  options(
    enrichR.sites.base.address = "https://maayanlab.cloud/",
    enrichR.sites = c("Enrichr", "FlyEnrichr", "WormEnrichr", "YeastEnrichr", "FishEnrichr", "OxEnrichr"),
    enrichR.base.address = "https://maayanlab.cloud/Enrichr/",
    enrichR.live = TRUE
  )
}

default_enrichr_databases <- function() {
  c(
    "DisGeNET",
    "GO_Molecular_Function_2025",
    "GO_Biological_Process_2025",
    "KEGG_2021_Human"
  )
}

safe_enrichr <- function(genes, dbs) {
  genes <- unique(stats::na.omit(genes))
  genes <- genes[nzchar(genes)]
  if (length(genes) == 0) {
    return(NULL)
  }

  tryCatch(
    enrichR::enrichr(genes, dbs),
    error = function(e) stop(sprintf("enrichR failed: %s", conditionMessage(e)), call. = FALSE)
  )
}

compute_gene_count <- function(gene_string) {
  if (is.na(gene_string) || !nzchar(gene_string)) {
    return(0)
  }
  length(strsplit(gene_string, ";", fixed = TRUE)[[1]])
}

compute_gene_ratio <- function(overlap_string) {
  if (is.na(overlap_string) || !nzchar(overlap_string)) {
    return(NA_real_)
  }
  parts <- suppressWarnings(as.numeric(strsplit(overlap_string, "/", fixed = TRUE)[[1]]))
  if (length(parts) != 2 || any(is.na(parts)) || parts[[2]] == 0) {
    return(NA_real_)
  }
  parts[[1]] / parts[[2]]
}

prepare_enrichment_table <- function(annotation, thr_pval) {
  if (is.null(annotation) || nrow(annotation) == 0) {
    return(annotation)
  }

  annotation <- annotation[annotation$Adjusted.P.value < thr_pval, , drop = FALSE]
  annotation <- annotation[order(annotation$Adjusted.P.value), , drop = FALSE]

  if (nrow(annotation) == 0) {
    return(annotation)
  }

  annotation$Gene_count <- vapply(annotation$Genes, compute_gene_count, numeric(1))
  annotation$Gene_ratio <- vapply(annotation$Overlap, compute_gene_ratio, numeric(1))
  annotation
}

write_empty_enrichment_table <- function(out_file) {
  empty <- data.frame(
    Term = character(),
    Overlap = character(),
    P.value = numeric(),
    Adjusted.P.value = numeric(),
    Gene_count = numeric(),
    Gene_ratio = numeric(),
    Genes = character(),
    stringsAsFactors = FALSE
  )
  write.table(empty, out_file, sep = "\t", quote = FALSE, row.names = FALSE)
}

write_enrichment_plots <- function(annotation, type, top_term, dir_out) {
  annotation_top <- if (nrow(annotation) > top_term) annotation[seq_len(top_term), , drop = FALSE] else annotation
  if (nrow(annotation_top) == 0) {
    return(FALSE)
  }

  barplot_path <- file.path(dir_out, paste0(type, "_barplot.pdf"))
  dotplot_path <- file.path(dir_out, paste0(type, "_dotplot.pdf"))

  g1 <- ggplot2::ggplot(
    annotation_top,
    ggplot2::aes(
      x = Gene_count,
      y = forcats::fct_reorder(Term, Gene_count),
      fill = Adjusted.P.value
    )
  ) +
    ggplot2::geom_bar(stat = "identity") +
    ggplot2::scale_fill_continuous(
      low = "red",
      high = "blue",
      name = "Adjusted.P.value",
      guide = ggplot2::guide_colorbar(reverse = TRUE)
    ) +
    ggplot2::scale_y_discrete(labels = function(x) stringr::str_wrap(x, width = 40)) +
    ggplot2::theme_bw(base_size = 10) +
    ggplot2::ylab(NULL)

  grDevices::pdf(barplot_path)
  print(g1)
  grDevices::dev.off()

  color_limits <- range(annotation_top$Adjusted.P.value, na.rm = TRUE)
  if (!all(is.finite(color_limits)) || color_limits[[1]] == color_limits[[2]]) {
    color_limits <- c(0, max(annotation_top$Adjusted.P.value, na.rm = TRUE) + 1e-12)
  }

  g2 <- ggplot2::ggplot(
    annotation_top,
    ggplot2::aes(
      x = Gene_count,
      y = forcats::fct_reorder(Term, Gene_count)
    )
  ) +
    ggplot2::geom_point(ggplot2::aes(size = Gene_ratio, color = Adjusted.P.value)) +
    ggplot2::scale_colour_gradient(limits = color_limits, low = "red", high = "blue") +
    ggplot2::theme_bw(base_size = 10) +
    ggplot2::scale_y_discrete(labels = function(x) stringr::str_wrap(x, width = 40)) +
    ggplot2::ylab(NULL)

  grDevices::pdf(dotplot_path)
  print(g2)
  grDevices::dev.off()

  TRUE
}

write_single_enrichment_result <- function(annotation, type, top_term, thr_pval, dir_out) {
  ensure_dir(dir_out)
  prepared <- prepare_enrichment_table(annotation, thr_pval)
  table_path <- file.path(dir_out, paste0(type, "_adj_pval_", thr_pval, ".txt"))

  if (is.null(prepared) || nrow(prepared) == 0) {
    write_empty_enrichment_table(table_path)
    return(list(n_terms = 0L, plotted = FALSE))
  }

  write.table(
    prepared[, c("Term", "Overlap", "P.value", "Adjusted.P.value", "Gene_count", "Gene_ratio", "Genes")],
    table_path,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
  )

  plotted <- write_enrichment_plots(prepared, type, top_term, dir_out)
  list(n_terms = nrow(prepared), plotted = plotted)
}

run_single_comparison_enrichment <- function(comparison_dir,
                                             top_term = 10,
                                             thr_pval = 0.05,
                                             dbs = default_enrichr_databases()) {
  ensure_enrichment_packages()

  deg_path <- file.path(comparison_dir, "DEG.txt")
  if (!file.exists(deg_path)) {
    stop(sprintf("Missing DEG file: %s", deg_path), call. = FALSE)
  }

  deg_table <- read.delim(deg_path, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
  enrichment_root <- file.path(comparison_dir, "Functional_Enrichment")
  ensure_dir(enrichment_root)

  direction_groups <- split(deg_table$GeneSymbol, deg_table$direction)
  direction_groups <- direction_groups[c("UP", "DOWN")]

  all_results <- list()

  for (direction in names(direction_groups)) {
    direction_dir <- file.path(enrichment_root, direction)
    ensure_dir(direction_dir)
    enrichment_objects <- safe_enrichr(direction_groups[[direction]], dbs)

    if (is.null(enrichment_objects)) {
      for (type in c("DisGeNET", "GO_BP", "GO_MF", "KEGG")) {
        write_empty_enrichment_table(file.path(direction_dir, paste0(type, "_adj_pval_", thr_pval, ".txt")))
      }
      all_results[[direction]] <- data.frame(
        direction = direction,
        database = c("DisGeNET", "GO_BP", "GO_MF", "KEGG"),
        n_terms = 0L,
        plotted = FALSE,
        stringsAsFactors = FALSE
      )
      next
    }

    db_map <- list(
      DisGeNET = enrichment_objects$DisGeNET,
      GO_BP = enrichment_objects$GO_Biological_Process_2025,
      GO_MF = enrichment_objects$GO_Molecular_Function_2025,
      KEGG = enrichment_objects$KEGG_2021_Human
    )

    rows <- lapply(names(db_map), function(type) {
      outcome <- write_single_enrichment_result(
        annotation = db_map[[type]],
        type = type,
        top_term = top_term,
        thr_pval = thr_pval,
        dir_out = direction_dir
      )

      data.frame(
        direction = direction,
        database = type,
        n_terms = outcome$n_terms,
        plotted = outcome$plotted,
        stringsAsFactors = FALSE
      )
    })

    all_results[[direction]] <- do.call(rbind, rows)
  }

  summary_table <- do.call(rbind, all_results)
  write.table(
    summary_table,
    file = file.path(enrichment_root, "enrichment_summary.tsv"),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
  )

  summary_table
}

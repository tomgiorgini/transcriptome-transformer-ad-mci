options(stringsAsFactors = FALSE)

parse_args <- function(args) {
  parsed <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument: %s", key), call. = FALSE)
    }
    key <- sub("^--", "", key)
    if (i == length(args) || startsWith(args[[i + 1]], "--")) {
      parsed[[key]] <- TRUE
    } else {
      parsed[[key]] <- args[[i + 1]]
      i <- i + 1
    }
    i <- i + 1
  }
  parsed
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
required <- c("x-file", "y-file", "out-dir", "adj-p-threshold")
missing <- required[!required %in% names(args)]
if (length(missing) > 0) {
  stop(sprintf("Missing required args: %s", paste(missing, collapse = ", ")), call. = FALSE)
}

if (!requireNamespace("limma", quietly = TRUE)) {
  stop("Missing R package 'limma'.", call. = FALSE)
}

x_file <- normalizePath(args[["x-file"]], winslash = "/", mustWork = TRUE)
y_file <- normalizePath(args[["y-file"]], winslash = "/", mustWork = TRUE)
out_dir <- args[["out-dir"]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
threshold <- as.numeric(args[["adj-p-threshold"]])
p_value_threshold <- if ("p-value-threshold" %in% names(args)) as.numeric(args[["p-value-threshold"]]) else NA_real_

x <- read.csv(x_file, row.names = 1, check.names = FALSE)
y <- read.csv(y_file, row.names = 1, check.names = FALSE)
if (!"label" %in% colnames(y)) {
  stop("y-file must contain a 'label' column.", call. = FALSE)
}

common <- intersect(rownames(x), rownames(y))
if (length(common) < 2) {
  stop("X and y have too few overlapping samples.", call. = FALSE)
}
x <- x[common, , drop = FALSE]
y <- y[common, , drop = FALSE]

group <- factor(ifelse(as.integer(y$label) == 1, "AD", "MCI"), levels = c("MCI", "AD"))
expr <- t(as.matrix(x))
mode(expr) <- "numeric"

design <- stats::model.matrix(~ 0 + group)
colnames(design) <- levels(group)
fit <- limma::lmFit(expr, design)
contrast <- limma::makeContrasts(AD_vs_MCI = AD - MCI, levels = design)
fit2 <- limma::contrasts.fit(fit, contrast)
fit2 <- limma::eBayes(fit2)
table <- limma::topTable(fit2, coef = "AD_vs_MCI", number = Inf, adjust.method = "BH", sort.by = "P")
table$gene <- rownames(table)
table <- table[, c("gene", setdiff(colnames(table), "gene"))]
selected <- table[table$adj.P.Val < threshold, , drop = FALSE]
selection_rule <- "adj.P.Val"
effective_threshold <- threshold
if (nrow(selected) == 0 && !is.na(p_value_threshold)) {
  selected <- table[table$P.Value < p_value_threshold, , drop = FALSE]
  selection_rule <- "P.Value"
  effective_threshold <- p_value_threshold
}

write.csv(table, file.path(out_dir, "dgs_table.csv"), row.names = FALSE)
writeLines(as.character(selected$gene), file.path(out_dir, "selected_genes.txt"))
manifest <- data.frame(
  dgs_method = "limma",
  contrast = "AD_vs_MCI",
  adj_p_threshold = threshold,
  p_value_threshold = ifelse(is.na(p_value_threshold), NA, p_value_threshold),
  selection_rule = selection_rule,
  effective_threshold = effective_threshold,
  n_samples = nrow(x),
  n_input_genes = ncol(x),
  n_selected_genes = nrow(selected)
)
write.csv(manifest, file.path(out_dir, "dgs_manifest.csv"), row.names = FALSE)

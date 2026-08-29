suppressPackageStartupMessages(library(limma))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript deg_limma.R <x_train_csv> <y_train_csv> <out_csv>")
}

x_path <- args[[1]]
y_path <- args[[2]]
out_path <- args[[3]]

x <- read.csv(x_path, row.names = 1, check.names = FALSE)
y <- read.csv(y_path, row.names = 1, check.names = FALSE)
labels <- factor(y$label, levels = c(0, 1), labels = c("MCI", "AD"))

expr <- t(as.matrix(x))
design <- model.matrix(~ labels)
fit <- lmFit(expr, design)
fit <- eBayes(fit)
tab <- topTable(fit, coef = "labelsAD", number = Inf, sort.by = "P")
tab$gene <- rownames(tab)
tab <- tab[, c("gene", setdiff(colnames(tab), "gene"))]
write.csv(tab, out_path, row.names = FALSE)

options(stringsAsFactors = FALSE)

qc <- file.path("pretraining_dataset", "geo_downloads", "qc_reports")
tsv <- file.path(qc, "greedy_reference_intersection_iterations_clean.tsv")
png_out <- file.path(qc, "greedy_reference_intersection_plot.png")
pdf_out <- file.path(qc, "greedy_reference_intersection_plot.pdf")
genes_png_out <- file.path(qc, "greedy_reference_total_genes_plot.png")
genes_pdf_out <- file.path(qc, "greedy_reference_total_genes_plot.pdf")
deg_png_out <- file.path(qc, "greedy_reference_deg_ad_vs_mci_plot.png")
deg_pdf_out <- file.path(qc, "greedy_reference_deg_ad_vs_mci_plot.pdf")

df <- read.delim(tsv, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)

make_single_plot <- function(y_col, y_label, title, line_label, line_col, png_path, pdf_path, device = c("png", "pdf")) {
  device <- match.arg(device)
  if (device == "png") {
    png(png_path, width = 1800, height = 1050, res = 160)
  } else {
    pdf(pdf_path, width = 12, height = 7)
  }
  on.exit(dev.off())

  sample_col <- "#2f7d4f"
  grid_col <- "#dedbd0"

  par(bg = "white", mar = c(5.0, 7.4, 4.8, 6.4), family = "sans")
  x <- df$iteration
  y_main <- df[[y_col]]
  y_samples <- df$cumulative_samples

  y_left_step <- if (max(y_main, na.rm = TRUE) > 5000) 1000 else 100
  y_left_max <- ceiling(max(y_main, na.rm = TRUE) / y_left_step) * y_left_step
  y_right_max <- ceiling(max(y_samples, na.rm = TRUE) / 500) * 500

  plot(
    x, y_main,
    type = "n", ylim = c(0, y_left_max), xaxt = "n", yaxt = "n",
    xlab = "Iterazione greedy", ylab = "",
    main = title
  )
  rect(par("usr")[1], par("usr")[3], par("usr")[2], par("usr")[4], col = "#fbfbf8", border = NA)
  grid(nx = NA, ny = NULL, col = grid_col, lty = 1, lwd = 0.8)
  axis(1, at = x, labels = x, las = 1, cex.axis = 0.82)
  axis(2, las = 1, col.axis = line_col, col = line_col)
  mtext(y_label, side = 2, line = 5.1, col = line_col)
  box(bty = "l")

  lines(x, y_main, type = "l", lwd = 2.8, col = line_col)
  points(x, y_main, pch = 16, cex = 0.72, col = line_col)

  par(new = TRUE)
  plot(
    x, y_samples,
    type = "n", ylim = c(0, y_right_max), xaxt = "n", yaxt = "n",
    xlab = "", ylab = ""
  )
  lines(x, y_samples, type = "l", lwd = 2.6, lty = 2, col = sample_col)
  points(x, y_samples, pch = 17, cex = 0.7, col = sample_col)
  axis(4, las = 1, col.axis = sample_col, col = sample_col)
  mtext("Numero di sample cumulativi", side = 4, line = 3.5, col = sample_col)

  legend(
    "top", inset = -0.02, horiz = TRUE, bty = "n", xpd = TRUE,
    legend = c(line_label, "Sample cumulativi"),
    col = c(line_col, sample_col), lwd = c(2.8, 2.6),
    lty = c(1, 2), pch = c(16, 17), cex = 0.95
  )

  drop_idx <- which.max(df$genes_lost_this_step)
  if (y_col == "genes_after_intersection" && length(drop_idx) == 1 && df$genes_lost_this_step[drop_idx] > 0) {
    abline(v = df$iteration[drop_idx], col = "#777777", lty = 3, lwd = 1.2)
    text(
      df$iteration[drop_idx] + 2.3,
      df$genes_after_intersection[drop_idx] + 2800,
      labels = paste0("drop massimo: -", df$genes_lost_this_step[drop_idx], " geni\n", df$gse_id[drop_idx]),
      cex = 0.82, col = "#444444", adj = 0
    )
    arrows(
      df$iteration[drop_idx] + 1.9,
      df$genes_after_intersection[drop_idx] + 2300,
      df$iteration[drop_idx],
      df$genes_after_intersection[drop_idx],
      length = 0.08, col = "#666666", lwd = 1
    )
  }
}

make_single_plot(
  y_col = "genes_after_intersection",
  y_label = "Numero di geni rimasti",
  title = "Geni rimasti e sample cumulativi",
  line_label = "Geni rimasti totali",
  line_col = "#1f5a85",
  png_path = genes_png_out,
  pdf_path = genes_pdf_out,
  device = "png"
)
make_single_plot(
  y_col = "genes_after_intersection",
  y_label = "Numero di geni rimasti",
  title = "Geni rimasti e sample cumulativi",
  line_label = "Geni rimasti totali",
  line_col = "#1f5a85",
  png_path = genes_png_out,
  pdf_path = genes_pdf_out,
  device = "pdf"
)
make_single_plot(
  y_col = "deg_ad_vs_mci_intersection",
  y_label = "Numero di geni DEG AD_vs_MCI rimasti",
  title = "DEG AD_vs_MCI rimasti e sample cumulativi",
  line_label = "Overlap con DEG AD_vs_MCI",
  line_col = "#c45a2a",
  png_path = deg_png_out,
  pdf_path = deg_pdf_out,
  device = "png"
)
make_single_plot(
  y_col = "deg_ad_vs_mci_intersection",
  y_label = "Numero di geni DEG AD_vs_MCI rimasti",
  title = "DEG AD_vs_MCI rimasti e sample cumulativi",
  line_label = "Overlap con DEG AD_vs_MCI",
  line_col = "#c45a2a",
  png_path = deg_png_out,
  pdf_path = deg_pdf_out,
  device = "pdf"
)

cat("TOTAL_GENES_PNG\t", normalizePath(genes_png_out, winslash = "/", mustWork = FALSE), "\n", sep = "")
cat("TOTAL_GENES_PDF\t", normalizePath(genes_pdf_out, winslash = "/", mustWork = FALSE), "\n", sep = "")
cat("DEG_PNG\t", normalizePath(deg_png_out, winslash = "/", mustWork = FALSE), "\n", sep = "")
cat("DEG_PDF\t", normalizePath(deg_pdf_out, winslash = "/", mustWork = FALSE), "\n", sep = "")

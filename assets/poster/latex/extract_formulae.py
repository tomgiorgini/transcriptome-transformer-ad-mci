from __future__ import annotations

import base64
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
PDF = HERE / "build" / "TxT_formulae.pdf"


def content_rect(page: fitz.Page) -> fitz.Rect:
    rects: list[fitz.Rect] = []

    data = page.get_text("dict")
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    rects.append(fitz.Rect(span["bbox"]))

    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is not None and not fitz.Rect(rect).is_empty:
            rects.append(fitz.Rect(rect))

    if not rects:
        raise RuntimeError(f"No visible formula content found on page {page.number + 1}")

    union = rects[0]
    for rect in rects[1:]:
        union |= rect

    return fitz.Rect(union.x0 - 5, union.y0 - 5, union.x1 + 5, union.y1 + 5)


def main() -> None:
    doc = fitz.open(PDF)
    names = ["formula_bias", "formula_attention", "formula_pooling"]

    for page, name in zip(doc, names, strict=True):
        crop = content_rect(page)
        page.set_cropbox(crop)
        svg = page.get_svg_image(text_as_path=True)
        (HERE / f"{name}.svg").write_text(svg, encoding="utf-8")
        encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        (HERE / f"{name}.b64").write_text(encoded, encoding="ascii")


if __name__ == "__main__":
    main()

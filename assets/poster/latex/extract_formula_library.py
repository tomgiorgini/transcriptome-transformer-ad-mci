from __future__ import annotations

from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
PDF = HERE / "library_build" / "TxT_formula_library.pdf"
NAMES = [
    "library_01_inputs_ppi",
    "library_02_content_qkv",
    "library_03_tupe_full",
    "library_04_attention_full",
    "library_HERO_tupe_complete",
    "library_05_multihead",
    "library_06_postnorm",
    "library_07_pooling",
    "library_08_heads_masks",
    "library_09_loss",
    "library_A_tupe_compact",
    "library_B_model_compact",
    "library_C_loss_compact",
]


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
        raise RuntimeError(f"No visible content on page {page.number + 1}")
    union = rects[0]
    for rect in rects[1:]:
        union |= rect
    return fitz.Rect(union.x0 - 5, union.y0 - 5, union.x1 + 5, union.y1 + 5)


def main() -> None:
    doc = fitz.open(PDF)
    if len(doc) != len(NAMES):
        raise RuntimeError(f"Expected {len(NAMES)} pages, found {len(doc)}")
    for page, name in zip(doc, NAMES, strict=True):
        page.set_cropbox(content_rect(page))
        svg = page.get_svg_image(text_as_path=True)
        (HERE / f"{name}.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()

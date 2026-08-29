from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "TxT_method_editorial.svg.in"
OUTPUT = HERE / "TxT_method_editorial.svg"
LATEX = HERE / "latex"


def main() -> None:
    svg = TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "__FORMULA_BIAS__": (LATEX / "formula_bias.b64").read_text(encoding="ascii").strip(),
        "__FORMULA_ATTENTION__": (LATEX / "formula_attention.b64").read_text(encoding="ascii").strip(),
        "__FORMULA_POOLING__": (LATEX / "formula_pooling.b64").read_text(encoding="ascii").strip(),
    }
    for placeholder, value in replacements.items():
        svg = svg.replace(placeholder, value)
    OUTPUT.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()

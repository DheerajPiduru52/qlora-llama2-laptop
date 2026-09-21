"""Shared look for every README figure: GitHub Primer palette (light + dark), GitHub's system font stack,
flat square-cornered marks, light horizontal gridlines, left-aligned titles, muted source footers.

Colour roles (never use the warning red for a method):
    qlora  - QLoRA / my tuned model (purple accent)
    lora   - bf16 LoRA
    full   - full fine-tuning / the untuned base model
    warn   - limits and known weaknesses only (VRAM line, invented slots)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES: dict[str, dict[str, str]] = {
    "light": {"bg": "#ffffff", "text": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0",
              "qlora": "#8250df", "lora": "#0969da", "full": "#818b98", "warn": "#cf222e"},
    "dark": {"bg": "#0d1117", "text": "#e6edf3", "muted": "#9198a1", "grid": "#3d444d",
             "qlora": "#ab7df8", "lora": "#4493f8", "full": "#656c76", "warn": "#f85149"},
}  # fmt: skip

FONT_STACK = ["Segoe UI", "-apple-system", "Helvetica", "Arial", "sans-serif"]
LABEL_PT, TITLE_PT, FOOTER_PT = 11, 13, 8.5
HATCH = "////"
PNG_DPI = 200  # 2x a 100-dpi base size
MIN_CONTRAST = 3.0


def apply(theme: str) -> dict[str, str]:
    """Set rcParams for one theme and return its colours."""
    c = THEMES[theme]
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": FONT_STACK, "font.size": LABEL_PT, "svg.fonttype": "none",
        "text.color": c["text"], "axes.labelcolor": c["muted"], "axes.edgecolor": c["grid"],
        "axes.titlesize": TITLE_PT, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlecolor": c["text"], "axes.titlepad": 12,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": c["grid"], "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.facecolor": c["bg"], "figure.facecolor": c["bg"],
        "savefig.facecolor": c["bg"], "savefig.edgecolor": c["bg"],
        "xtick.color": c["muted"], "ytick.color": c["muted"], "xtick.labelcolor": c["muted"],
        "ytick.labelcolor": c["muted"], "xtick.labelsize": LABEL_PT, "ytick.labelsize": LABEL_PT,
        "legend.frameon": False, "legend.fontsize": LABEL_PT, "legend.labelcolor": c["text"],
        "hatch.linewidth": 1.1, "patch.linewidth": 0,
    })  # fmt: skip
    return c


def title(fig, ax, text: str, sub: str | None = None) -> None:
    """Left-aligned semibold title (the takeaway) with an optional muted subtitle, both at the figure's left edge."""
    c = plt.rcParams
    fig.text(0.012, 0.985, text, ha="left", va="top", fontsize=TITLE_PT, fontweight="semibold", color=c["text.color"])
    if sub:
        fig.text(0.012, 0.985 - 0.055 * (7.0 / fig.get_figheight()), sub, ha="left", va="top", fontsize=LABEL_PT,
                 color=c["axes.labelcolor"])  # fmt: skip


def footer(fig, text: str) -> None:
    fig.text(0.012, 0.012, text, ha="left", va="bottom", fontsize=FOOTER_PT, color=plt.rcParams["axes.labelcolor"])


def estimated_bar(ax, x, h, color: str, bottom=0.0, width=0.6, label=None, horizontal=False):
    """ESTIMATED values: outlined + hatched, never solid like measured ones."""
    kw = {"facecolor": plt.rcParams["axes.facecolor"], "edgecolor": color, "hatch": HATCH, "linewidth": 1.4,
          "label": label}  # fmt: skip
    if horizontal:
        return ax.barh(x, h, height=width, left=bottom, **kw)
    return ax.bar(x, h, width=width, bottom=bottom, **kw)


def save(fig, out_dir: Path, name: str, theme: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}-{theme}.svg", out_dir / f"{name}-{theme}.png"]
    fig.savefig(paths[0], format="svg")
    fig.savefig(paths[1], format="png", dpi=PNG_DPI)
    plt.close(fig)
    return paths


# ------------------------------------------------------------------ WCAG contrast (marks vs background)


def _lum(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    rgb = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def contrast_report() -> list[tuple[str, str, float, bool]]:
    rows = []
    for theme, c in THEMES.items():
        for role in ("text", "muted", "qlora", "lora", "full", "warn"):
            r = contrast(c[role], c["bg"])
            rows.append((theme, role, round(r, 2), r >= MIN_CONTRAST))
    return rows

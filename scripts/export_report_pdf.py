"""Export report/assignment3_report.md to report/assignment3_report.pdf.

Offline markdown-to-PDF rendering with matplotlib (pandoc/reportlab/browser
tools are not installed on this machine). Headings, wrapped body text,
typeset tables (bold header, rule lines, no raw markdown pipes), page
numbers in the footer, and the PNG figures from report/figures/ are all
placed on A4 pages. The script verifies the result (size, page count,
page numbers, key text).

Usage:
    python scripts/export_report_pdf.py
"""

import re
import sys
import textwrap
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.use14corefonts"] = True  # text stays extractable
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = REPO_ROOT / "report" / "assignment3_report.md"
REPORT_PDF = REPO_ROOT / "report" / "assignment3_report.pdf"

PAGE_W, PAGE_H = 8.27, 11.69  # A4 inches
TOP, BOTTOM, LEFT = 0.6, 0.6, 0.6
CONTENT_W = PAGE_W - 2 * LEFT
LINE_H = 0.15
IMAGE_DPI = 150.0

IMAGE_RE = re.compile(r"^!\[[^\]]*\]\(([^)]+)\)\s*$")
SEPARATOR_RE = re.compile(r"^[\s|:-]+$")

KEY_TEXT = ["Assignment 3 Report", "Ackley", "CVRP", "Rush Hour", "GP", "GEP",
            "23.0063", "2.9466", "5.5139", "advanced local-search", "2-opt*",
            "blocker_depth", "without A*", "11/14", "13/14", "120736",
            "Iterated Local Search", "exploration", "exploitation",
            "CPU time", "1.5219", "4.9072", "5.463", "References",
            "Submitted by", "Page 2"]


def strip_markup(text):
    """Remove inline markdown markers (bold, code ticks) for clean PDF text."""
    return text.replace("**", "").replace("`", "")


def typeset_table(block):
    """Turn consecutive '|' lines into padded text rows without pipes.

    Returns (header_line, data_lines): markdown separator rows (---) are
    dropped, cells are stripped of inline markup and padded so columns line
    up in the monospace table font. The caller draws a rule under the
    header instead of the markdown separator row.
    """
    rows = []
    for line in block:
        cells = [strip_markup(cell.strip())
                 for cell in line.strip().strip("|").split("|")]
        if SEPARATOR_RE.match("".join(cells)) and any("-" in c for c in cells):
            continue  # markdown header separator row
        rows.append(cells)
    if not rows:
        return None, []
    widths = [max(len(row[i]) if i < len(row) else 0 for row in rows)
              for i in range(max(len(r) for r in rows))]
    def pad(row):
        return "  ".join((row[i] if i < len(row) else "").ljust(widths[i])
                         for i in range(len(widths))).rstrip()
    return pad(rows[0]), [pad(row) for row in rows[1:]]


def parse_items():
    """Turn the markdown into a list of (kind, payload) render items."""
    lines = REPORT_MD.read_text(encoding="utf-8").splitlines()
    items = []
    i = 0
    while i < len(lines):
        line = lines[i]
        image = IMAGE_RE.match(line)
        if image:
            items.append(("image", image.group(1)))
        elif line.startswith("# "):
            items.append(("h1", strip_markup(line[2:])))
        elif line.startswith("## "):
            items.append(("h2", strip_markup(line[3:])))
        elif line.startswith("### "):
            items.append(("h3", strip_markup(line[4:])))
        elif line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            header, data_rows = typeset_table(block)
            if header is not None:
                items.append(("table_header", header))
                items.append(("table_rule", len(header)))
                for row in data_rows:
                    for part in textwrap.wrap(row, width=126,
                                              subsequent_indent="    ") or [""]:
                        items.append(("table", part))
                items.append(("table_gap", ""))
            continue
        else:
            for part in textwrap.wrap(strip_markup(line), width=98,
                                      subsequent_indent="  ") or [""]:
                items.append(("body", part))
        i += 1
    return items


STYLE = {  # kind -> (fontsize, bold, spacing factor)
    "h1": (15, True, 2.4),
    "h2": (12.5, True, 2.0),
    "h3": (10.5, True, 1.6),
    "body": (8.5, False, 1.0),
    "table": (6.5, False, 0.85),
    "table_header": (6.5, True, 0.85),
    "table_gap": (6.5, False, 0.5),
}


def export():
    items = parse_items()
    with PdfPages(REPORT_PDF) as pdf:
        page_number = 1
        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        y = PAGE_H - TOP

        def finish_page(figure, number):
            """Stamp the footer page number and write the page out."""
            figure.text(0.5, (BOTTOM / 2) / PAGE_H, f"Page {number}",
                        fontsize=7, family="monospace", ha="center",
                        va="center")
            pdf.savefig(figure)
            plt.close(figure)

        def new_page(figure):
            nonlocal page_number
            finish_page(figure, page_number)
            page_number += 1
            return plt.figure(figsize=(PAGE_W, PAGE_H)), PAGE_H - TOP

        for kind, payload in items:
            if kind == "image":
                image_path = (REPORT_MD.parent / payload).resolve()
                if not image_path.exists():
                    raise SystemExit(f"missing figure referenced in report: {payload}")
                array = mpimg.imread(image_path)
                height_px, width_px = array.shape[0], array.shape[1]
                width_in = min(width_px / IMAGE_DPI, CONTENT_W)
                height_in = width_in * height_px / width_px
                if height_in > PAGE_H - TOP - BOTTOM:
                    height_in = PAGE_H - TOP - BOTTOM - 0.2
                    width_in = height_in * width_px / height_px
                if y - height_in - 0.15 < BOTTOM:
                    fig, y = new_page(fig)
                x0 = LEFT + (CONTENT_W - width_in) / 2
                ax = fig.add_axes([x0 / PAGE_W, (y - height_in) / PAGE_H,
                                   width_in / PAGE_W, height_in / PAGE_H])
                ax.imshow(array)
                ax.axis("off")
                y -= height_in + 0.25
                continue

            if kind == "table_rule":
                # thin rule under the table header, sized to the header text
                rule_w = min(payload * 0.054, CONTENT_W)  # ~6.5pt mono width
                fig.add_artist(plt.Line2D(
                    [LEFT / PAGE_W, (LEFT + rule_w) / PAGE_W],
                    [(y + 0.02) / PAGE_H, (y + 0.02) / PAGE_H],
                    linewidth=0.6, color="black"))
                continue

            size, bold, spacing = STYLE[kind]
            need = LINE_H * spacing
            if kind in ("h1", "h2") and y < BOTTOM + 1.2:
                fig, y = new_page(fig)  # do not leave a heading at page bottom
            if kind == "table_header" and y < BOTTOM + 0.6:
                fig, y = new_page(fig)  # keep the header with its table
            if y - need < BOTTOM:
                fig, y = new_page(fig)
            if payload:
                fig.text(LEFT / PAGE_W, y / PAGE_H, payload, fontsize=size,
                         family="monospace",
                         fontweight="bold" if bold else "normal",
                         va="top", ha="left")
            y -= need
        finish_page(fig, page_number)


def verify():
    data = REPORT_PDF.read_bytes()
    size = len(data)
    pages = data.count(b"/Type /Page") - data.count(b"/Type /Pages")
    extracted = b""
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        try:
            extracted += zlib.decompress(match.group(1))
        except zlib.error:
            extracted += match.group(1)
    missing = [key for key in KEY_TEXT if key.encode() not in extracted]
    raw_table_artifacts = b"| ---" in extracted or b"| --- |" in extracted
    print(f"pdf: {REPORT_PDF}")
    print(f"size: {size} bytes")
    print(f"pages: {pages}")
    print(f"key text missing: {missing or 'none'}")
    print(f"raw markdown table artifacts: {'FOUND' if raw_table_artifacts else 'none'}")
    ok = (size > 100_000 and pages >= 10 and not missing
          and not raw_table_artifacts)
    print(f"verify: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    if not REPORT_MD.exists():
        raise SystemExit(f"report not found: {REPORT_MD}")
    export()
    sys.exit(0 if verify() else 1)


if __name__ == "__main__":
    main()

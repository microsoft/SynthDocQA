# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Markdown → PDF renderer.

Three backends available:
  - ReportLabRenderer  (DEFAULT) — uses ReportLab Platypus directly.
      Tables are rendered with Table + Paragraph cells, giving pixel-perfect
      column widths and word-wrapping.  ReportLab ships with xhtml2pdf.
  - XHtml2PdfRenderer  — legacy, kept for reference.
  - WeasyPrintRenderer — requires GTK/GLib system libraries.
"""
from __future__ import annotations

import abc
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


def _bg_from_style(style: str) -> str | None:
    """Extract the first #rrggbb / #rgb colour from a CSS background-color declaration."""
    m = re.search(r"background-color\s*:\s*(#[0-9a-fA-F]{3,8})", style)
    return m.group(1) if m else None


# Mapping: light highlight background colour → dark foreground for ReportLab rendering.
# ReportLab Paragraph cannot render inline background colour, so we substitute a
# contrasting dark foreground that conveys the same colour family visually.
_MARK_BG_TO_FG: dict[str, str] = {
    "#fff176": "#7c6200",   # yellow  → dark amber
    "#bfdbfe": "#1e40af",   # blue    → dark blue
    "#bbf7d0": "#14532d",   # green   → dark green
    "#fed7aa": "#c2410c",   # orange  → dark orange
    "#fce7f3": "#9d174d",   # pink    → dark pink
    "#ddd6fe": "#5b21b6",   # purple  → dark purple
    "#fecaca": "#991b1b",   # red     → dark red
    "#cffafe": "#0e7490",   # cyan    → dark cyan
}
_LIGHT_FG = {"#ffffff", "#fff", "white", "#fffffe", "#fefefe"}


def _mark_bg_to_fg(bg_hex: str) -> str:
    """Return a readable dark foreground colour for the given highlight background."""
    key = bg_hex.lower().strip()
    if key in _MARK_BG_TO_FG:
        return _MARK_BG_TO_FG[key]
    # Generic fallback: darken by ~45 %
    h = key.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return "#b45309"
    r = max(0, int(int(h[0:2], 16) * 0.45))
    g = max(0, int(int(h[2:4], 16) * 0.45))
    b = max(0, int(int(h[4:6], 16) * 0.45))
    return f"#{r:02x}{g:02x}{b:02x}"


def _span_font_from_style(style_str: str) -> str | None:
    """Return a ReportLab-registered font name from a CSS font-family property.

    Extracts the first token from ``font-family: Name, fallback;`` and verifies
    that it has been registered with ReportLab's pdfmetrics.  Returns None if the
    font is not registered (caller falls back to default document font).
    """
    m = re.search(r"font-family\s*:\s*([^;]+)", style_str)
    if not m:
        return None
    first = m.group(1).split(",")[0].strip().strip("'\"")
    if not first:
        return None
    try:
        from reportlab.pdfbase import pdfmetrics
        pdfmetrics.getFont(first)   # raises KeyError if not registered
        return first
    except Exception:
        return None


def _span_fg_from_style(style_str: str) -> str | None:
    """Derive a ReportLab-usable foreground colour from an inline CSS style string.

    Priority:
      1. ``color: X`` where X is not white/near-white → use directly.
      2. ``color: white`` + ``background[-color]: X`` → use X as fg (badge pattern).
      3. Only ``background[-color]: X`` present → darken X for fg.
    Returns None when no useful colour is found.
    """
    color_m = re.search(
        r"(?:^|;)\s*color\s*:\s*(#[0-9a-fA-F]{3,8})", style_str
    )
    bg_m = re.search(
        r"background(?:-color)?\s*:\s*(#[0-9a-fA-F]{3,8})", style_str
    )
    fg_raw = color_m.group(1) if color_m else None
    bg_raw = bg_m.group(1) if bg_m else None

    fg_is_light = fg_raw is not None and fg_raw.lower() in _LIGHT_FG

    if fg_raw and not fg_is_light:
        return fg_raw          # Non-white fg: use directly
    if fg_is_light and bg_raw:
        return bg_raw          # White-on-colour badge: use bg as fg
    if bg_raw:
        return _mark_bg_to_fg(bg_raw)  # Only bg: derive dark fg
    if fg_raw:
        return fg_raw          # Light fg with no bg: use as-is
    return None


# ---------------------------------------------------------------------------
# Abstraction
# ---------------------------------------------------------------------------

class BaseRenderer(abc.ABC):
    """Abstract base for all rendering backends."""

    @abc.abstractmethod
    def render(self, md_path: Path, pdf_path: Path) -> dict[str, Any]:
        """Render *md_path* Markdown file to *pdf_path* PDF.

        Returns a dict with:
            - tool_versions: dict of tool name → version string
            - render_time_seconds: float
            - page_count: int
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable renderer name."""


# ---------------------------------------------------------------------------
# ReportLab renderer  (DEFAULT)
# ---------------------------------------------------------------------------

def _rl_styles() -> dict[str, Any]:
    """Build a set of ReportLab ParagraphStyles used throughout the document."""
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors

    return {
        "h1": ParagraphStyle(
            "h1", fontName="Helvetica-Bold", fontSize=18, leading=22,
            spaceBefore=12, spaceAfter=6,
            textColor=colors.HexColor("#1a1a2e"),
        ),
        "h2": ParagraphStyle(
            "h2", fontName="Helvetica-Bold", fontSize=14, leading=18,
            spaceBefore=10, spaceAfter=4,
        ),
        "h3": ParagraphStyle(
            "h3", fontName="Helvetica-Bold", fontSize=12, leading=16,
            spaceBefore=8, spaceAfter=4,
        ),
        "normal": ParagraphStyle(
            "normal", fontName="Helvetica", fontSize=10, leading=14,
            spaceAfter=4,
        ),
        "cell": ParagraphStyle(
            "cell", fontName="Helvetica", fontSize=8, leading=11,
        ),
        "cell_hdr": ParagraphStyle(
            "cell_hdr", fontName="Helvetica-Bold", fontSize=8, leading=11,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName="Helvetica", fontSize=10, leading=14,
            leftIndent=15, firstLineIndent=-10, spaceAfter=2,
        ),
        "blockquote": ParagraphStyle(
            "blockquote", fontName="Helvetica-Oblique", fontSize=10, leading=14,
            leftIndent=20, spaceAfter=4,
            textColor=colors.HexColor("#555555"),
        ),
    }


class _PlatypusBuilder(HTMLParser):
    """Parse HTML (output of Python markdown library) into ReportLab Platypus flowables.

    Handles: h1-h3, p, hr, ul/ol/li, blockquote, table/tr/td/th, img,
             and inline tags b/strong, i/em, code.
    """

    def __init__(self, page_width: float) -> None:
        super().__init__(convert_charrefs=True)
        self._pw = page_width
        self._st = _rl_styles()
        self.flowables: list = []

        # Block-level state
        self._buf: list[str] = []      # text accumulator for current block
        self._blk: str | None = None   # current block tag name

        # Ordered list tracking
        self._in_ol: bool = False
        self._ol_counter: int = 0

        # Table state — each cell tuple: (content, is_hdr, colspan, rowspan, bg_hex|None)
        self._in_tbl = False
        self._tbl_rows: list = []
        self._cur_row: list = []
        self._cell_buf: list[str] = []
        self._cell_hdr = False
        self._in_cell = False
        self._cell_colspan = 1
        self._cell_rowspan = 1
        self._cell_bg: str | None = None

        # Nested-table support: stack of saved outer-table states
        self._tbl_stack: list[dict] = []
        # When an inner </table> is built, the result is parked here until </td>
        self._nested_tbl: Any = None
        # Width fraction of the current containing cell (set from width="N%" attribute)
        self._cell_width_pct: float | None = None

        # Inline span tracking: each entry is the close-fragment to emit on </span>
        self._span_stack: list[str] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit(self, text: str) -> None:
        """Route text to the active buffer (cell or block).

        Text that arrives while inside a table but outside a cell (i.e. whitespace
        between tags) is silently dropped to avoid polluting the block buffer.
        """
        if self._in_cell:
            self._cell_buf.append(text)
        elif not self._in_tbl:
            self._buf.append(text)

    def _flush(self, tag: str | None = None) -> None:
        """Flush the block text buffer as a Paragraph flowable."""
        from reportlab.platypus import Paragraph, Spacer

        t = tag or self._blk
        text = "".join(self._buf).strip()
        self._buf = []
        if not text:
            return

        s = self._st
        if t == "h1":
            self.flowables.append(Paragraph(text, s["h1"]))
        elif t == "h2":
            self.flowables.append(Paragraph(text, s["h2"]))
        elif t == "h3":
            self.flowables.append(Paragraph(text, s["h3"]))
        elif t == "blockquote":
            self.flowables.append(Paragraph(text, s["blockquote"]))
        elif t == "li":
            if self._in_ol:
                prefix = f"{self._ol_counter}.&nbsp;"
            else:
                prefix = "&bull;&nbsp;"
            self.flowables.append(Paragraph(f"{prefix}{text}", s["bullet"]))
        else:
            self.flowables.append(Paragraph(text, s["normal"]))

        if t not in ("h1", "h2", "h3"):
            self.flowables.append(Spacer(1, 2))

    @staticmethod
    def _resolve_grid(raw_rows: list) -> tuple:
        """Expand colspan/rowspan into a rectangular grid.

        Args:
            raw_rows: list of rows, each row a list of
                      (content, is_hdr, colspan, rowspan, bg_hex|None)
                      — legacy 2-tuples (content, is_hdr) are also accepted.

        Returns:
            grid       – dict[(row, col)] → (content, is_hdr, bg|None) | None (placeholder)
            n_rows     – int
            n_cols     – int
            span_cmds  – list[((col0, row0), (col1, row1))]  in ReportLab (col, row) order
        """
        grid: dict = {}
        occupied: set = set()
        span_cmds: list = []
        max_col = 0

        for r_idx, raw_row in enumerate(raw_rows):
            c_idx = 0
            for item in raw_row:
                content = item[0]
                is_hdr  = item[1] if len(item) > 1 else False
                cs      = int(item[2]) if len(item) > 2 else 1
                rs      = int(item[3]) if len(item) > 3 else 1
                bg      = item[4] if len(item) > 4 else None

                # Skip columns already occupied by a rowspan from a previous row
                while (r_idx, c_idx) in occupied:
                    c_idx += 1

                grid[(r_idx, c_idx)] = (content, is_hdr, bg)

                # Mark all cells covered by this span as occupied (placeholder)
                for dr in range(rs):
                    for dc in range(cs):
                        if dr > 0 or dc > 0:
                            key = (r_idx + dr, c_idx + dc)
                            occupied.add(key)
                            grid[key] = None

                if cs > 1 or rs > 1:
                    span_cmds.append(
                        ((c_idx, r_idx), (c_idx + cs - 1, r_idx + rs - 1))
                    )

                max_col = max(max_col, c_idx + cs)
                c_idx += cs

        n_rows = (max(r for r, _ in grid) + 1) if grid else len(raw_rows)
        return grid, n_rows, max_col, span_cmds

    def _build_table_internal(self) -> Any:
        """Build and return a ReportLab Table flowable (does NOT append to flowables)."""
        from reportlab.platypus import Table, TableStyle, Paragraph
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle

        if not self._tbl_rows:
            return None

        grid, n_rows, n_cols, span_cmds = self._resolve_grid(self._tbl_rows)
        if n_cols == 0:
            return None

        col_w = self._pw / n_cols

        # For wide tables with narrow columns, scale down font and padding so that
        # cell content wraps less and rows stay compact enough to fit on the page.
        _MIN_COL = 42  # pt: below this threshold we start scaling
        if col_w < _MIN_COL:
            _scale = max(0.55, col_w / _MIN_COL)
            _fsize = max(5, round(8 * _scale))
            _lead  = max(7, round(11 * _scale))
            _pad   = 3
            _c_sty: ParagraphStyle | None = ParagraphStyle(
                f"_ct{_fsize}", fontName="Helvetica", fontSize=_fsize, leading=_lead
            )
            _h_sty: ParagraphStyle | None = ParagraphStyle(
                f"_ht{_fsize}", fontName="Helvetica-Bold", fontSize=_fsize, leading=_lead
            )
        else:
            _fsize = 8; _lead = 11; _pad = 6
            _c_sty = None; _h_sty = None

        # Build the data matrix; every cell must be a Paragraph or flowable
        data: list[list] = []
        is_hdr_row: list[bool] = []
        for r in range(n_rows):
            cells: list = []
            row_has_hdr = False
            for c in range(n_cols):
                cell = grid.get((r, c))
                if cell is None:
                    # Placeholder for a spanned cell — empty Paragraph
                    cells.append(Paragraph("", self._st["cell"]))
                else:
                    content, is_hdr, _ = cell
                    if _c_sty is not None and isinstance(content, Paragraph):
                        content = Paragraph(content.text, _h_sty if is_hdr else _c_sty)
                    cells.append(content)
                    if is_hdr:
                        row_has_hdr = True
            data.append(cells)
            is_hdr_row.append(row_has_hdr)

        n_hdr_rows = sum(1 for h in is_hdr_row if h)
        tbl = Table(
            data,
            colWidths=[col_w] * n_cols,
            repeatRows=max(1, n_hdr_rows),
            hAlign="LEFT",
        )

        cmds = [
            ("FONTNAME",       (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",       (0, 0), (-1, -1), _fsize),
            ("LEADING",        (0, 0), (-1, -1), _lead),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), _pad),
            ("RIGHTPADDING",   (0, 0), (-1, -1), _pad),
            ("VALIGN",         (0, 0), (-1, -1), "TOP"),
            ("ALIGN",          (0, 0), (-1, -1), "LEFT"),
            ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.HexColor("#f4f8ff"), colors.white]),
        ]

        # Header-row styling (bold, blue background)
        for i, is_h in enumerate(is_hdr_row):
            if is_h:
                cmds += [
                    ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#dce6f1")),
                    ("FONTNAME",   (0, i), (-1, i), "Helvetica-Bold"),
                ]

        # Build a span-extent lookup so BACKGROUND fills the full merged area.
        # span_cmds entries are ((start_col, start_row), (end_col, end_row)).
        span_extent: dict[tuple[int, int], tuple[int, int]] = {}
        for (sc, sr), (ec, er) in span_cmds:
            span_extent[(sr, sc)] = (ec, er)  # keyed by (row, col) of origin cell

        # Per-cell background overrides from inline CSS (parsed from HTML).
        # Extend each command to cover the full colspan/rowspan so that merged
        # cells (e.g. ragged-colspan section headers, multi-tier headers) are
        # entirely filled with the intended colour.
        for (r, c), cell in grid.items():
            if cell is not None:
                _, is_hdr, bg = cell
                if bg:
                    try:
                        ec, er = span_extent.get((r, c), (c, r))
                        cmds.append(("BACKGROUND", (c, r), (ec, er), colors.HexColor(bg)))
                        # White text for dark backgrounds
                        lum = int(bg.lstrip("#"), 16)
                        r_ch = (lum >> 16) & 0xFF
                        g_ch = (lum >> 8)  & 0xFF
                        b_ch =  lum        & 0xFF
                        if (0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch) < 128:
                            cmds.append(("TEXTCOLOR", (c, r), (ec, er), colors.white))
                            cmds.append(("FONTNAME",  (c, r), (ec, er), "Helvetica-Bold"))
                    except Exception:
                        pass

        # Span commands — must come after all BACKGROUND commands
        for start_pos, end_pos in span_cmds:
            cmds.append(("SPAN", start_pos, end_pos))

        tbl.setStyle(TableStyle(cmds))
        return tbl

    def _build_table(self) -> None:
        """Build the current table and append it to self.flowables."""
        from reportlab.platypus import Spacer
        tbl = self._build_table_internal()
        self._tbl_rows = []
        if tbl is not None:
            self.flowables.append(tbl)
            self.flowables.append(Spacer(1, 8))

    # ------------------------------------------------------------------
    # HTMLParser callbacks
    # ------------------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list) -> None:
        a = dict(attrs)

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush()
            self._blk = tag if tag in ("h1", "h2", "h3") else "h3"

        elif tag == "p":
            self._flush()
            self._blk = "p"

        elif tag == "ol":
            self._in_ol = True
            self._ol_counter = 0
        elif tag == "li":
            self._flush()
            if self._in_ol:
                self._ol_counter += 1
            self._blk = "li"

        elif tag == "blockquote":
            self._flush()
            self._blk = "blockquote"

        elif tag == "hr":
            from reportlab.platypus import HRFlowable, Spacer
            from reportlab.lib import colors
            self._flush()
            self.flowables.append(
                HRFlowable(width="100%", color=colors.HexColor("#cccccc"), thickness=0.5)
            )
            self.flowables.append(Spacer(1, 4))

        elif tag == "div":
            cls = dict(attrs).get("class", "")
            if cls == "synthpdf-page-break" and self.flowables:
                from reportlab.platypus import PageBreak
                self.flowables.append(PageBreak())
                return
            # Treat other divs as block containers (flush any pending content, start new block)
            self._flush()
            self._blk = "p"

        elif tag == "table":
            if self._in_tbl:
                # Estimate the available width for the inner table.
                # Use the outer cell's explicit width% if present; otherwise divide
                # the outer page width equally among the outer table's columns.
                outer_pw = self._pw
                if self._cell_width_pct is not None:
                    inner_pw = outer_pw * self._cell_width_pct
                else:
                    max_cells = max((len(r) for r in self._tbl_rows), default=0)
                    cur_cells = len(self._cur_row) + 1  # +1 for current containing cell
                    outer_n_cols = max(max_cells, cur_cells, 1)
                    inner_pw = outer_pw * self._cell_colspan / outer_n_cols

                # Push the outer table context so the inner table can be parsed cleanly
                self._tbl_stack.append({
                    "tbl_rows":       self._tbl_rows,
                    "cur_row":        self._cur_row,
                    "in_cell":        self._in_cell,
                    "cell_buf":       self._cell_buf[:],
                    "cell_hdr":       self._cell_hdr,
                    "cell_colspan":   self._cell_colspan,
                    "cell_rowspan":   self._cell_rowspan,
                    "cell_bg":        self._cell_bg,
                    "cell_width_pct": self._cell_width_pct,
                    "pw":             outer_pw,
                })
                # Switch to the estimated inner-table width
                self._pw = inner_pw
                self._cell_width_pct = None
                # Temporarily leave cell context so inner-table data routes correctly
                self._in_cell = False
                self._cell_buf = []
            else:
                self._flush()
            self._in_tbl = True
            self._tbl_rows = []
            self._cur_row = []

        elif tag == "tr":
            self._cur_row = []

        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_buf = []
            self._cell_hdr = tag == "th"
            attrs_d = dict(attrs)
            self._cell_colspan = max(1, int(attrs_d.get("colspan", 1) or 1))
            self._cell_rowspan = max(1, int(attrs_d.get("rowspan", 1) or 1))
            self._cell_bg = (
                _bg_from_style(attrs_d.get("style", ""))
                or attrs_d.get("bgcolor") or None
            )
            # Track cell width (used to constrain nested inner-table widths)
            w_attr = attrs_d.get("width", "")
            if w_attr.endswith("%"):
                try:
                    self._cell_width_pct = float(w_attr.rstrip("%")) / 100.0
                except ValueError:
                    self._cell_width_pct = None
            else:
                self._cell_width_pct = None

        elif tag == "img":
            from reportlab.platypus import Image, Spacer
            self._flush()
            src = a.get("src", "")
            p = Path(src)
            if not p.is_absolute():
                p = Path.cwd() / p
            if p.exists():
                w = self._pw * 0.85
                try:
                    # Preserve the image's natural aspect ratio.
                    # Scanned pages are portrait A4 (~1.41 h/w); charts are landscape.
                    # Hardcoding 0.6 distorts portrait images — read actual dimensions.
                    try:
                        from PIL import Image as _PILImg
                        with _PILImg.open(str(p)) as _pi:
                            _iw, _ih = _pi.size
                        h = w * (_ih / _iw) if _iw > 0 else w * 0.6
                    except Exception:
                        h = w * 0.6  # fallback for unreadable images
                    self.flowables.append(Image(str(p), width=w, height=h))
                    self.flowables.append(Spacer(1, 6))
                except Exception:
                    pass

        elif tag in ("strong", "b"):
            self._emit("<b>")
        elif tag in ("em", "i"):
            self._emit("<i>")
        elif tag == "code":
            self._emit('<font name="Courier" size="8">')
        elif tag == "br":
            self._emit("<br/>")
        elif tag == "sup":
            self._emit("<super>")
        elif tag == "sub":
            self._emit("<sub>")
        elif tag in ("strike", "s", "del"):
            self._emit("<strike>")
        elif tag == "mark":
            # Highlight: read background-color from style (set by annotation generators)
            # and derive a contrasting dark foreground — ReportLab cannot render inline
            # background colour on Paragraph text, so we simulate it with coloured fg.
            style_str = dict(attrs).get("style", "") if attrs else ""
            bg_m = re.search(r"background-color\s*:\s*(#[0-9a-fA-F]{3,8})", style_str)
            fg = _mark_bg_to_fg(bg_m.group(1)) if bg_m else "#b45309"
            self._emit(f'<b><font color="{fg}">')
        elif tag == "span":
            # Inline style support: coloured text, badge spans, strikethrough,
            # and registered custom fonts (e.g. handwriting TTF fonts in forms).
            # Pushes the matching close-fragment onto _span_stack so </span> can
            # emit the correct closing tag regardless of nesting depth.
            style_str = dict(attrs).get("style", "") if attrs else ""
            if "line-through" in style_str:
                self._emit("<strike>")
                self._span_stack.append("</strike>")
            else:
                fg   = _span_fg_from_style(style_str)
                face = _span_font_from_style(style_str)
                if fg or face:
                    attr_xml = ""
                    if face:
                        attr_xml += f' face="{face}"'
                    if fg:
                        attr_xml += f' color="{fg}"'
                    self._emit(f'<font{attr_xml}>')
                    self._span_stack.append("</font>")
                else:
                    self._span_stack.append("")  # no-op

    def handle_endtag(self, tag: str) -> None:
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush(tag if tag in ("h1", "h2", "h3") else "h3")
            self._blk = None

        elif tag in ("p", "li", "blockquote"):
            self._flush(tag)
            self._blk = None

        elif tag in ("td", "th"):
            from reportlab.platypus import Paragraph
            if self._nested_tbl is not None:
                # Cell contains a nested ReportLab Table built from an inner <table>
                content = self._nested_tbl
                self._nested_tbl = None
            else:
                cell_text = "".join(self._cell_buf).strip() or " "
                style = self._st["cell_hdr" if self._cell_hdr else "cell"]
                content = Paragraph(cell_text, style)

            self._cur_row.append((
                content,
                self._cell_hdr,
                self._cell_colspan,
                self._cell_rowspan,
                self._cell_bg,
            ))
            self._in_cell = False
            self._cell_buf = []
            self._cell_colspan = 1
            self._cell_rowspan = 1
            self._cell_bg = None

        elif tag == "tr":
            if self._cur_row:
                self._tbl_rows.append(self._cur_row)
            self._cur_row = []

        elif tag == "table":
            self._in_tbl = False
            inner_tbl = self._build_table_internal()
            self._tbl_rows = []

            if self._tbl_stack:
                # Restore the outer table context
                saved = self._tbl_stack.pop()
                self._tbl_rows       = saved["tbl_rows"]
                self._cur_row        = saved["cur_row"]
                self._in_cell        = saved["in_cell"]
                self._cell_buf       = saved["cell_buf"]
                self._cell_hdr       = saved["cell_hdr"]
                self._cell_colspan   = saved["cell_colspan"]
                self._cell_rowspan   = saved["cell_rowspan"]
                self._cell_bg        = saved["cell_bg"]
                self._cell_width_pct = saved.get("cell_width_pct")
                self._pw             = saved["pw"]
                self._in_tbl = True
                # Park the inner table; it will be picked up by the next </td>
                self._nested_tbl = inner_tbl
            else:
                # Outermost table — append directly to flowables
                if inner_tbl is not None:
                    from reportlab.platypus import Spacer
                    self.flowables.append(inner_tbl)
                    self.flowables.append(Spacer(1, 8))

        elif tag in ("strong", "b"):
            self._emit("</b>")
        elif tag in ("em", "i"):
            self._emit("</i>")
        elif tag == "code":
            self._emit("</font>")
        elif tag == "sup":
            self._emit("</super>")
        elif tag == "sub":
            self._emit("</sub>")
        elif tag in ("strike", "s", "del"):
            self._emit("</strike>")
        elif tag == "mark":
            self._emit("</font></b>")
        elif tag == "div":
            self._flush("p")
            self._blk = None
        elif tag == "span":
            close = self._span_stack.pop() if self._span_stack else ""
            self._emit(close)

        elif tag == "ol":
            self._in_ol = False
            self._ol_counter = 0
            from reportlab.platypus import Spacer
            self.flowables.append(Spacer(1, 4))
        elif tag == "ul":
            from reportlab.platypus import Spacer
            self.flowables.append(Spacer(1, 4))

    def handle_data(self, data: str) -> None:
        # Re-escape raw text for ReportLab's XML paragraph parser.
        # (convert_charrefs=True already decoded HTML entities to Unicode chars.)
        safe = data.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._emit(safe)

    def close(self) -> None:
        super().close()
        self._flush()  # flush any trailing buffered text


class ReportLabRenderer(BaseRenderer):
    """Renders Markdown → ReportLab Platypus → PDF.

    Pipeline:
        1. Markdown  →  HTML  (via Python markdown library)
        2. HTML      →  Platypus flowables  (via _PlatypusBuilder)
        3. Flowables →  PDF  (via SimpleDocTemplate)

    Tables are rendered with ReportLab's Table class, using Paragraph objects
    inside every cell so text wraps naturally within each column.
    ReportLab is already installed as an xhtml2pdf dependency — no extra deps.
    """

    name = "reportlab"

    def render(
        self,
        md_path: Path,
        pdf_path: Path,
        branding_spec: Any = None,
    ) -> dict[str, Any]:
        import markdown as md_lib
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Spacer
        from reportlab import __version__ as rl_version

        start = time.perf_counter()

        md_text = md_path.read_text(encoding="utf-8")
        html = md_lib.markdown(
            md_text,
            extensions=["tables", "fenced_code", "footnotes"],
        )

        # A4 content width = page width minus left + right margins (2 cm each)
        # Subtract 12pt for SimpleDocTemplate's default Frame left+right padding (6pt each)
        # so table colWidths match the actual available frame width (prevents "too wide" errors).
        page_w = A4[0] - 4 * cm - 12
        builder = _PlatypusBuilder(page_w)
        builder.feed(html)
        builder.close()

        flowables = builder.flowables or [Spacer(1, 1)]

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        # Wire up logo branding callbacks if a spec is provided
        build_kwargs: dict[str, Any] = {}
        if branding_spec is not None:
            try:
                from synthpdf.render.branding import make_rl_callbacks
                on_first, on_later = make_rl_callbacks(branding_spec)
                build_kwargs["onFirstPage"] = on_first
                build_kwargs["onLaterPages"] = on_later
            except Exception:
                pass  # never block rendering because of branding errors

        doc.build(flowables, **build_kwargs)

        elapsed = time.perf_counter() - start

        page_count = -1
        try:
            from pypdf import PdfReader
            page_count = len(PdfReader(str(pdf_path)).pages)
        except Exception:
            pass

        return {
            "tool_versions": {
                "markdown": md_lib.__version__,
                "reportlab": rl_version,
            },
            "render_time_seconds": round(elapsed, 3),
            "page_count": page_count,
        }


# ---------------------------------------------------------------------------
# xhtml2pdf implementation (kept as fallback option)
# ---------------------------------------------------------------------------

_DEFAULT_CSS = """
body { font-family: Helvetica, Arial, sans-serif; font-size: 11pt;
       line-height: 1.5; color: #222; margin: 2cm; }
h1 { font-size: 18pt; } h2 { font-size: 14pt; } h3 { font-size: 12pt; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; }
th, td { border: 1px solid #999; padding: 4px 6px; text-align: left; }
th { background-color: #dce6f1; font-weight: bold; }
img { max-width: 100%; height: auto; }
hr { border: none; border-top: 1px solid #ddd; margin: 1em 0; }
.synthpdf-page-break { page-break-before: always; break-before: page; }
"""


class XHtml2PdfRenderer(BaseRenderer):
    """Renders Markdown → HTML → PDF using xhtml2pdf (fallback)."""

    name = "xhtml2pdf"

    def __init__(self, css: str | None = None, css_file: Path | None = None) -> None:
        self._css = css or _DEFAULT_CSS
        if css_file and css_file.exists():
            self._css = css_file.read_text(encoding="utf-8")

    def render(
        self,
        md_path: Path,
        pdf_path: Path,
        branding_spec: Any = None,
    ) -> dict[str, Any]:
        import markdown as md_lib
        import xhtml2pdf.pisa as pisa
        import xhtml2pdf

        start = time.perf_counter()
        md_text = md_path.read_text(encoding="utf-8")
        html_body = md_lib.markdown(md_text, extensions=["tables", "fenced_code", "footnotes"])

        branding_html = ""
        if branding_spec is not None:
            try:
                from synthpdf.render.branding import make_html_logo_snippet
                branding_html = make_html_logo_snippet(branding_spec)
            except Exception:
                pass

        full_html = (
            "<!DOCTYPE html><html><head>"
            f"<meta charset='utf-8'><style>{self._css}</style>"
            f"</head><body>{branding_html}{html_body}</body></html>"
        )
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        def _link_callback(uri: str, rel: str) -> str:
            if uri.startswith(("http://", "https://", "data:", "file:")):
                return uri
            p = Path(uri)
            if p.is_absolute():
                return str(p)
            from_cwd = Path.cwd() / p
            if from_cwd.exists():
                return str(from_cwd.resolve())
            from_md = md_path.parent / p
            if from_md.exists():
                return str(from_md.resolve())
            return str(from_cwd)

        with open(pdf_path, "wb") as fh:
            result = pisa.CreatePDF(full_html, dest=fh, encoding="utf-8",
                                    link_callback=_link_callback)
        if result.err:
            raise RuntimeError(f"xhtml2pdf reported {result.err} error(s)")

        elapsed = time.perf_counter() - start
        page_count = -1
        try:
            from pypdf import PdfReader
            page_count = len(PdfReader(str(pdf_path)).pages)
        except Exception:
            pass
        return {
            "tool_versions": {"markdown": md_lib.__version__,
                               "xhtml2pdf": getattr(xhtml2pdf, "__version__", "unknown")},
            "render_time_seconds": round(elapsed, 3),
            "page_count": page_count,
        }


# ---------------------------------------------------------------------------
# WeasyPrint implementation (optional — requires GTK/GLib)
# ---------------------------------------------------------------------------

class WeasyPrintRenderer(BaseRenderer):
    name = "weasyprint"

    def __init__(self, css: str | None = None, css_file: Path | None = None) -> None:
        self._css = css or _DEFAULT_CSS
        if css_file and css_file.exists():
            self._css = css_file.read_text(encoding="utf-8")

    def render(
        self,
        md_path: Path,
        pdf_path: Path,
        branding_spec: Any = None,
    ) -> dict[str, Any]:
        import markdown as md_lib
        import weasyprint
        start = time.perf_counter()
        md_text = md_path.read_text(encoding="utf-8")
        html_body = md_lib.markdown(md_text, extensions=["tables", "fenced_code", "footnotes"])

        branding_html = ""
        if branding_spec is not None:
            try:
                from synthpdf.render.branding import make_html_logo_snippet
                branding_html = make_html_logo_snippet(branding_spec)
            except Exception:
                pass

        full_html = (f"<!DOCTYPE html><html><head><style>{self._css}</style></head>"
                     f"<body>{branding_html}{html_body}</body></html>")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        weasyprint.HTML(string=full_html, base_url=str(md_path.parent)).write_pdf(str(pdf_path))
        elapsed = time.perf_counter() - start
        page_count = -1
        try:
            from pypdf import PdfReader
            page_count = len(PdfReader(str(pdf_path)).pages)
        except Exception:
            pass
        return {
            "tool_versions": {"markdown": md_lib.__version__,
                               "weasyprint": weasyprint.__version__},
            "render_time_seconds": round(elapsed, 3),
            "page_count": page_count,
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Playwright implementation (best quality — requires playwright + chromium)
# ---------------------------------------------------------------------------

class PlaywrightMarkdownRenderer(BaseRenderer):
    """Renders Markdown -> HTML -> PDF using headless Chromium via Playwright.

    Produces the highest-quality output because Playwright supports full CSS3
    (Grid, Flexbox, gradients, shadows) and handles multi-page documents
    automatically via Chromium's print pipeline.

    Falls back to WeasyPrint then xhtml2pdf if Playwright is unavailable.
    """

    name = "playwright"

    def __init__(self, css: str | None = None, css_file: Path | None = None) -> None:
        self._css = css or _DEFAULT_CSS
        if css_file and css_file.exists():
            self._css = css_file.read_text(encoding="utf-8")

    def render(
        self,
        md_path: Path,
        pdf_path: Path,
        branding_spec: Any = None,
    ) -> dict[str, Any]:
        import markdown as md_lib
        import tempfile

        start = time.perf_counter()
        md_text = md_path.read_text(encoding="utf-8")
        html_body = md_lib.markdown(md_text, extensions=["tables", "fenced_code", "footnotes"])

        branding_html = ""
        if branding_spec is not None:
            try:
                from synthpdf.render.branding import make_html_logo_snippet
                branding_html = make_html_logo_snippet(branding_spec)
            except Exception:
                pass

        # Playwright's print pipeline handles multi-page layout automatically
        # when @page is set and the body flows freely (no fixed height).
        full_html = (
            "<!DOCTYPE html><html><head>"
            "<meta charset='utf-8'>"
            "<style>"
            "@page { size: A4; margin: 2cm; }"
            f"{self._css}"
            "</style>"
            f"</head><body>{branding_html}{html_body}</body></html>"
        )

        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        # Write HTML to a temp file so Playwright can load it via file:// URI
        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="w", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write(full_html)
            tmp_path = Path(tmp.name)

        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                try:
                    page = browser.new_page()
                    page.goto(tmp_path.resolve().as_uri(), wait_until="networkidle", timeout=60_000)
                    page.pdf(
                        path=str(pdf_path),
                        format="A4",
                        print_background=True,
                        margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                        prefer_css_page_size=True,
                    )
                finally:
                    browser.close()
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

        elapsed = time.perf_counter() - start
        page_count = -1
        try:
            from pypdf import PdfReader
            page_count = len(PdfReader(str(pdf_path)).pages)
        except Exception:
            pass
        return {
            "tool_versions": {"markdown": md_lib.__version__, "backend": "playwright"},
            "render_time_seconds": round(elapsed, 3),
            "page_count": page_count,
        }


def render_markdown_to_pdf(
    md_path: Path,
    pdf_path: Path,
    backend: str = "reportlab",
    css_file: Path | None = None,
    branding_spec: Any = None,
) -> dict[str, Any]:
    """Render *md_path* to *pdf_path*.

    Backends:
        "reportlab"  — default. Uses ReportLab Platypus directly for clean tables.
        "playwright" — best quality; headless Chromium. Falls back gracefully.
        "xhtml2pdf"  — HTML/CSS renderer, limited table support.
        "weasyprint" — requires GTK/GLib system libraries.

    Args:
        branding_spec: Optional :class:`~synthpdf.render.branding.BrandingSpec`
            instance.  When provided, a company logo is drawn in the header/footer
            on the specified pages.
    """
    if backend == "reportlab":
        return ReportLabRenderer().render(md_path, pdf_path, branding_spec=branding_spec)
    if backend == "playwright":
        return PlaywrightMarkdownRenderer(css_file=css_file).render(md_path, pdf_path, branding_spec=branding_spec)
    if backend == "xhtml2pdf":
        return XHtml2PdfRenderer(css_file=css_file).render(md_path, pdf_path, branding_spec=branding_spec)
    if backend == "weasyprint":
        return WeasyPrintRenderer(css_file=css_file).render(md_path, pdf_path, branding_spec=branding_spec)
    raise ValueError(f"Unknown rendering backend: {backend!r}")

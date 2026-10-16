# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Infographic variant generators — pure Pillow (PIL) implementation.

Eight rich visual sub-types:
  1. stat_dashboard   — KPI cards, bar chart, donut, callouts
  2. timeline         — horizontal milestone spine
  3. comparison       — feature comparison table
  4. process_steps    — step-by-step process boxes
  5. hub_spoke        — hub-and-spoke / org chart
  6. magazine         — hero-number magazine-style layout
  7. pictograph       — icon-array / pictograph
  8. floor_plan       — technical floor plan schematic

Public API (backwards-compatible):
    generate_flow_variants(rng, count, assets_dir)         -> list[Variant]
    generate_org_chart_variants(rng, count, assets_dir)    -> list[Variant]
    generate_plotly_variants(rng, count, assets_dir)       -> list[Variant]
    generate_dashboard_variants(rng, count, assets_dir)    -> list[Variant]
    generate_floor_plan_variants(rng, count, assets_dir)   -> list[Variant]
    generate_timeline_variants(rng, count, assets_dir)     -> list[Variant]
    generate_comparison_variants(rng, count, assets_dir)   -> list[Variant]
    generate_magazine_variants(rng, count, assets_dir)     -> list[Variant]
    generate_pictograph_variants(rng, count, assets_dir)   -> list[Variant]
    generate_infographic_pool(rng, count, assets_dir)      -> list[Variant]
"""
from __future__ import annotations

import io
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from synthpdf.rng import fork_rng
from synthpdf.types import Variant


# ===========================================================================
# SECTION 1: Color Theme System
# ===========================================================================

@dataclass(frozen=True)
class _Theme:
    name: str
    primary: tuple
    accent: tuple
    secondary: tuple
    bg: tuple
    surface: tuple
    border: tuple
    text: tuple
    subtext: tuple
    positive: tuple
    negative: tuple
    neutral: tuple


def _rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _darken(c: tuple, f: float = 0.2) -> tuple:
    return (
        max(0, int(c[0] * (1 - f))),
        max(0, int(c[1] * (1 - f))),
        max(0, int(c[2] * (1 - f))),
    )


def _lighten(c: tuple, f: float = 0.2) -> tuple:
    return (
        min(255, int(c[0] + (255 - c[0]) * f)),
        min(255, int(c[1] + (255 - c[1]) * f)),
        min(255, int(c[2] + (255 - c[2]) * f)),
    )


def _blend(c1: tuple, c2: tuple, t: float) -> tuple:
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


_THEMES: dict[str, _Theme] = {
    "corporate_blue": _Theme(
        name="corporate_blue",
        primary=_rgb("#1E3A5F"), accent=_rgb("#2196F3"), secondary=_rgb("#90CAF9"),
        bg=_rgb("#FFFFFF"), surface=_rgb("#F4F6F9"), border=_rgb("#E2E8F0"),
        text=_rgb("#1A1A2E"), subtext=_rgb("#64748B"),
        positive=_rgb("#10B981"), negative=_rgb("#EF4444"), neutral=_rgb("#94A3B8"),
    ),
    "emerald_finance": _Theme(
        name="emerald_finance",
        primary=_rgb("#064E3B"), accent=_rgb("#10B981"), secondary=_rgb("#6EE7B7"),
        bg=_rgb("#FFFFFF"), surface=_rgb("#F0FDF9"), border=_rgb("#D1FAE5"),
        text=_rgb("#1A2E1E"), subtext=_rgb("#6B7280"),
        positive=_rgb("#059669"), negative=_rgb("#EF4444"), neutral=_rgb("#9CA3AF"),
    ),
    "slate_tech": _Theme(
        name="slate_tech",
        primary=_rgb("#1E293B"), accent=_rgb("#6366F1"), secondary=_rgb("#A5B4FC"),
        bg=_rgb("#F8FAFC"), surface=_rgb("#FFFFFF"), border=_rgb("#E2E8F0"),
        text=_rgb("#0F172A"), subtext=_rgb("#475569"),
        positive=_rgb("#22C55E"), negative=_rgb("#F43F5E"), neutral=_rgb("#94A3B8"),
    ),
    "warm_amber": _Theme(
        name="warm_amber",
        primary=_rgb("#78350F"), accent=_rgb("#F59E0B"), secondary=_rgb("#FCD34D"),
        bg=_rgb("#FFFBEB"), surface=_rgb("#FEF3C7"), border=_rgb("#FDE68A"),
        text=_rgb("#1C1917"), subtext=_rgb("#78716C"),
        positive=_rgb("#16A34A"), negative=_rgb("#DC2626"), neutral=_rgb("#A8A29E"),
    ),
    "healthcare_teal": _Theme(
        name="healthcare_teal",
        primary=_rgb("#134E4A"), accent=_rgb("#0D9488"), secondary=_rgb("#5EEAD4"),
        bg=_rgb("#F0FDFA"), surface=_rgb("#CCFBF1"), border=_rgb("#99F6E4"),
        text=_rgb("#0F2522"), subtext=_rgb("#4B5563"),
        positive=_rgb("#059669"), negative=_rgb("#EF4444"), neutral=_rgb("#6B7280"),
    ),
    "burgundy_legal": _Theme(
        name="burgundy_legal",
        primary=_rgb("#4C0519"), accent=_rgb("#9F1239"), secondary=_rgb("#FDA4AF"),
        bg=_rgb("#FFFFFF"), surface=_rgb("#FFF1F2"), border=_rgb("#FFE4E6"),
        text=_rgb("#1C0A0D"), subtext=_rgb("#6B7280"),
        positive=_rgb("#16A34A"), negative=_rgb("#DC2626"), neutral=_rgb("#9CA3AF"),
    ),
    "midnight_dark": _Theme(
        name="midnight_dark",
        primary=_rgb("#020617"), accent=_rgb("#38BDF8"), secondary=_rgb("#7DD3FC"),
        bg=_rgb("#0F172A"), surface=_rgb("#1E293B"), border=_rgb("#334155"),
        text=_rgb("#F1F5F9"), subtext=_rgb("#94A3B8"),
        positive=_rgb("#4ADE80"), negative=_rgb("#FB7185"), neutral=_rgb("#64748B"),
    ),
    "sand_neutral": _Theme(
        name="sand_neutral",
        primary=_rgb("#44403C"), accent=_rgb("#A16207"), secondary=_rgb("#CA8A04"),
        bg=_rgb("#FAFAF9"), surface=_rgb("#F5F5F4"), border=_rgb("#E7E5E4"),
        text=_rgb("#1C1917"), subtext=_rgb("#78716C"),
        positive=_rgb("#15803D"), negative=_rgb("#B91C1C"), neutral=_rgb("#A8A29E"),
    ),
}

_THEME_KEYS = list(_THEMES.keys())


# ===========================================================================
# SECTION 2: Font Loading
# ===========================================================================

_FONT_CACHE: dict[tuple, Any] = {}


def _load_font(size: int, bold: bool = False) -> Any:
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    if bold:
        candidates = [
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/Arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]

    font = None
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            break
        except (OSError, IOError):
            continue

    if font is None:
        font = ImageFont.load_default()

    _FONT_CACHE[key] = font
    return font


# ===========================================================================
# SECTION 3: Drawing Helpers
# ===========================================================================

def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _text_size(draw: Any, text: str, font: Any) -> tuple:
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        return (bb[2] - bb[0], bb[3] - bb[1])
    except AttributeError:
        pass
    try:
        return font.getsize(text)
    except Exception:
        return (len(text) * 7, 14)


def _draw_text_center(draw: Any, cx: int, cy: int, text: str, font: Any, fill: tuple) -> None:
    try:
        draw.text((cx, cy), text, font=font, fill=fill, anchor="mm")
    except TypeError:
        w, h = _text_size(draw, text, font)
        draw.text((cx - w // 2, cy - h // 2), text, font=font, fill=fill)


def _wrap_text(draw: Any, text: str, font: Any, max_w: int) -> list:
    words = text.split()
    lines = []
    cur = ""
    for word in words:
        test = (cur + " " + word).strip() if cur else word
        w, _ = _text_size(draw, test, font)
        if w <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines if lines else [text]


def _rounded_rect(
    draw: Any, x0: int, y0: int, x1: int, y1: int, r: int,
    fill: Any = None, outline: Any = None, outline_w: int = 1,
) -> None:
    r = max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    try:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill,
                                outline=outline, width=outline_w)
    except AttributeError:
        # Fallback for PIL < 9.2
        if fill:
            draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
            draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)
            draw.ellipse([x0, y0, x0 + 2*r, y0 + 2*r], fill=fill)
            draw.ellipse([x1 - 2*r, y0, x1, y0 + 2*r], fill=fill)
            draw.ellipse([x0, y1 - 2*r, x0 + 2*r, y1], fill=fill)
            draw.ellipse([x1 - 2*r, y1 - 2*r, x1, y1], fill=fill)
        if outline:
            draw.arc([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=outline, width=outline_w)
            draw.arc([x1 - 2*r, y0, x1, y0 + 2*r], 270, 360, fill=outline, width=outline_w)
            draw.arc([x0, y1 - 2*r, x0 + 2*r, y1], 90, 180, fill=outline, width=outline_w)
            draw.arc([x1 - 2*r, y1 - 2*r, x1, y1], 0, 90, fill=outline, width=outline_w)
            draw.line([x0 + r, y0, x1 - r, y0], fill=outline, width=outline_w)
            draw.line([x0 + r, y1, x1 - r, y1], fill=outline, width=outline_w)
            draw.line([x0, y0 + r, x0, y1 - r], fill=outline, width=outline_w)
            draw.line([x1, y0 + r, x1, y1 - r], fill=outline, width=outline_w)


def _gradient_rect(
    img: Image.Image, x: int, y: int, w: int, h: int,
    c1: tuple, c2: tuple, vertical: bool = True,
) -> None:
    if w <= 0 or h <= 0:
        return
    n = h if vertical else w
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(n):
        t = i / max(n - 1, 1)
        color = _blend(c1, c2, t)
        if vertical:
            arr[i, :] = color
        else:
            arr[:, i] = color
    patch = Image.fromarray(arr, "RGB")
    img.paste(patch, (x, y))


def _shadow_card(
    img: Image.Image, draw: Any, x: int, y: int, w: int, h: int,
    fill: tuple, theme: _Theme, radius: int = 8,
) -> None:
    # Shadow
    shadow_color = _darken(theme.bg, 0.08)
    _rounded_rect(draw, x + 3, y + 3, x + w + 3, y + h + 3, radius, fill=shadow_color)
    # Main card
    _rounded_rect(draw, x, y, x + w, y + h, radius, fill=fill, outline=theme.border, outline_w=1)


# ===========================================================================
# SECTION 4: Chart Primitives
# ===========================================================================

def _sparkline(
    values: list, w: int, h: int, color: tuple, bg: tuple
) -> Image.Image:
    img = Image.new("RGB", (w, h), bg)
    if not values or len(values) < 2:
        return img
    draw = ImageDraw.Draw(img)
    mn = min(values)
    mx = max(values)
    rng = mx - mn if mx != mn else 1
    pad = 4

    def _px(i: int, v: float) -> tuple:
        px = pad + int((i / (len(values) - 1)) * (w - 2 * pad))
        py = h - pad - int(((v - mn) / rng) * (h - 2 * pad))
        return (px, py)

    pts = [_px(i, v) for i, v in enumerate(values)]
    area_color = _lighten(color, 0.5)
    # Area fill polygon
    poly = [pts[0]] + pts + [(pts[-1][0], h), (pts[0][0], h)]
    draw.polygon(poly, fill=area_color)
    # Line
    draw.line(pts, fill=color, width=2)
    # Terminal dot
    lx, ly = pts[-1]
    r = 3
    draw.ellipse([lx - r, ly - r, lx + r, ly + r], fill=color)
    return img


def _bar_chart(
    labels: list, series: list, w: int, h: int, theme: _Theme
) -> Image.Image:
    img = Image.new("RGB", (w, h), theme.bg)
    draw = ImageDraw.Draw(img)

    series_colors = [
        theme.accent, theme.secondary, theme.neutral,
        _darken(theme.accent), _lighten(theme.accent),
    ]
    font_sm = _load_font(9)
    font_tiny = _load_font(8)

    margin_l = 45
    margin_r = 15
    margin_t = 30
    margin_b = 40

    chart_w = w - margin_l - margin_r
    chart_h = h - margin_t - margin_b

    # Compute max value
    all_vals = []
    for s in series:
        all_vals.extend(s.get("values", []))
    if not all_vals:
        return img
    max_val = max(all_vals) * 1.15 if max(all_vals) > 0 else 1

    n_cats = len(labels)
    n_series = len(series)
    if n_cats == 0:
        return img

    cat_w = chart_w // n_cats
    bar_w = max(4, (cat_w - 8) // max(n_series, 1))

    # Grid lines
    n_grid = 5
    for gi in range(n_grid + 1):
        gy = margin_t + chart_h - int((gi / n_grid) * chart_h)
        draw.line([(margin_l, gy), (margin_l + chart_w, gy)],
                  fill=theme.border, width=1)
        val_label = f"{(gi / n_grid) * max_val:.1f}"
        tw, _ = _text_size(draw, val_label, font_tiny)
        draw.text((margin_l - tw - 4, gy), val_label, font=font_tiny, fill=theme.subtext)

    # Bars
    for ci, label in enumerate(labels):
        cat_x = margin_l + ci * cat_w
        group_w = n_series * bar_w + (n_series - 1) * 2
        group_x = cat_x + (cat_w - group_w) // 2

        for si, s in enumerate(series):
            vals = s.get("values", [])
            color = s.get("color") or series_colors[si % len(series_colors)]
            if ci < len(vals):
                v = vals[ci]
                bar_h = int((v / max_val) * chart_h)
                bx = group_x + si * (bar_w + 2)
                by = margin_t + chart_h - bar_h
                draw.rectangle([bx, by, bx + bar_w, margin_t + chart_h], fill=color)
                # Value label above bar
                val_str = str(v) if isinstance(v, int) else f"{v:.1f}"
                tw, _ = _text_size(draw, val_str, font_tiny)
                if bar_h > 12:
                    draw.text((bx + bar_w // 2 - tw // 2, by - 12),
                              val_str, font=font_tiny, fill=theme.text)

        # X-axis label
        lw, _ = _text_size(draw, label, font_sm)
        lx = cat_x + cat_w // 2 - lw // 2
        draw.text((lx, margin_t + chart_h + 5), label, font=font_sm, fill=theme.text)

    # Legend
    if n_series > 1:
        lx = margin_l
        ly = margin_t - 18
        for si, s in enumerate(series):
            color = s.get("color") or series_colors[si % len(series_colors)]
            draw.rectangle([lx, ly + 3, lx + 10, ly + 11], fill=color)
            nm = s.get("name", f"Series {si+1}")
            draw.text((lx + 13, ly), nm, font=font_tiny, fill=theme.text)
            nw, _ = _text_size(draw, nm, font_tiny)
            lx += nw + 30

    return img


def _donut_chart(
    segments: list, w: int, h: int, theme: _Theme, hole_color: Any = None
) -> Image.Image:
    img = Image.new("RGB", (w, h), theme.bg)
    draw = ImageDraw.Draw(img)

    if not segments:
        return img

    hc = hole_color or theme.surface
    total = sum(s.get("value", 0) for s in segments)
    if total <= 0:
        return img

    # Chart area: left portion
    legend_w = 120
    chart_area_w = w - legend_w - 10
    cx = chart_area_w // 2
    cy = h // 2
    outer_r = min(cx, cy) - 10
    inner_r = int(outer_r * 0.55)

    angle = -90.0
    gap = 1.5
    for s in segments:
        pct = s["value"] / total
        sweep = pct * 360 - gap
        color = s.get("color") or theme.accent
        draw.pieslice(
            [cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r],
            start=angle, end=angle + sweep, fill=color,
        )
        angle += pct * 360

    # Inner hole
    draw.ellipse(
        [cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
        fill=hc,
    )

    # Largest segment % in center
    largest = max(segments, key=lambda s: s["value"])
    pct_str = f"{int(round(largest['value'] / total * 100))}%"
    font_big = _load_font(16, bold=True)
    font_sm = _load_font(9)
    _draw_text_center(draw, cx, cy - 6, pct_str, font_big, theme.text)
    _draw_text_center(draw, cx, cy + 12, largest.get("label", ""), font_sm, theme.subtext)

    # Legend
    font_leg = _load_font(9)
    lx = chart_area_w + 8
    ly = max(10, cy - len(segments) * 14)
    for s in segments:
        color = s.get("color") or theme.accent
        draw.rectangle([lx, ly, lx + 10, ly + 10], fill=color)
        pct_v = int(round(s["value"] / total * 100))
        label_str = f"{s.get('label', '')} {pct_v}%"
        draw.text((lx + 14, ly), label_str, font=font_leg, fill=theme.text)
        ly += 16

    return img


def _lollipop_chart(
    labels: list, values: list, w: int, h: int, theme: _Theme,
    max_val: Any = None,
) -> Image.Image:
    img = Image.new("RGB", (w, h), theme.bg)
    draw = ImageDraw.Draw(img)

    if not labels or not values:
        return img

    font_sm = _load_font(10)
    font_val = _load_font(10, bold=True)

    n = len(labels)
    margin_l = 140
    margin_r = 60
    margin_t = 16
    margin_b = 16
    row_h = max(18, (h - margin_t - margin_b) // n)

    mv = max_val if max_val is not None else (max(values) * 1.15 if values else 1)
    if mv <= 0:
        mv = 1

    chart_w = w - margin_l - margin_r

    for i, (label, val) in enumerate(zip(labels, values)):
        cy = margin_t + i * row_h + row_h // 2
        bar_len = int((val / mv) * chart_w)
        ex = margin_l + bar_len

        # Stem
        draw.line([(margin_l, cy), (ex, cy)], fill=theme.border, width=2)
        # Circle
        r = 7
        draw.ellipse([ex - r, cy - r, ex + r, cy + r], fill=theme.accent)

        # Label left
        lw, lh = _text_size(draw, label, font_sm)
        draw.text((margin_l - lw - 8, cy - lh // 2), label, font=font_sm, fill=theme.text)

        # Value right
        val_str = str(val) if isinstance(val, int) else f"{val:.1f}"
        draw.text((ex + r + 4, cy - 6), val_str, font=font_val, fill=theme.accent)

    return img


def _progress_bar(
    value: float, max_val: float, w: int, h: int,
    fill_color: tuple, bg_color: tuple,
) -> Image.Image:
    img = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(img)

    r = h // 2
    # Track
    _rounded_rect(draw, 0, 0, w, h, r, fill=_lighten(fill_color, 0.6))
    # Fill
    pct = min(1.0, value / max_val if max_val > 0 else 0)
    fill_w = max(r * 2, int(pct * w))
    _rounded_rect(draw, 0, 0, fill_w, h, r, fill=fill_color)
    return img


def _gauge_semicircle(
    value: float, min_val: float, max_val: float,
    w: int, h: int, theme: _Theme, label: str = "",
) -> Image.Image:
    img = Image.new("RGB", (w, h), theme.bg)
    draw = ImageDraw.Draw(img)

    cx = w // 2
    cy = int(h * 0.62)
    r = min(cx - 20, cy - 10)
    thickness = max(12, r // 5)

    # Track arc (180 to 360)
    bbox = [cx - r, cy - r, cx + r, cy + r]
    draw.arc(bbox, start=180, end=360, fill=theme.border, width=thickness)

    # Value arc
    pct = (value - min_val) / (max_val - min_val) if max_val != min_val else 0
    pct = max(0.0, min(1.0, pct))
    end_angle = 180 + pct * 180
    if pct > 0:
        draw.arc(bbox, start=180, end=end_angle, fill=theme.accent, width=thickness)

    # Value text
    font_big = _load_font(18, bold=True)
    font_sm = _load_font(10)
    val_str = f"{value:.1f}" if isinstance(value, float) else str(value)
    _draw_text_center(draw, cx, cy + 10, val_str, font_big, theme.text)
    if label:
        _draw_text_center(draw, cx, cy + 30, label, font_sm, theme.subtext)

    return img


# ===========================================================================
# SECTION 5: Data Pools
# ===========================================================================

_STAT_DASHBOARD_DATA: list[dict] = [
    {
        "title": "Q3 2024 Sales Performance",
        "subtitle": "Revenue & Pipeline Metrics",
        "period_badge": "Q3 2024",
        "theme": "corporate_blue",
        "kpis": [
            {"label": "Revenue", "value": "$4.2M", "delta": "+23%", "trend": "up",
             "sparkline": [3.1, 3.4, 3.6, 3.9, 4.2]},
            {"label": "Gross Margin", "value": "61.4%", "delta": "+2pp", "trend": "up",
             "sparkline": [58, 59, 60, 61, 61.4]},
            {"label": "New Customers", "value": "847", "delta": "+12%", "trend": "up",
             "sparkline": [720, 755, 780, 810, 847]},
            {"label": "Avg Deal", "value": "$4.9K", "delta": "-3%", "trend": "down",
             "sparkline": [5.2, 5.1, 5.0, 4.9, 4.9]},
        ],
        "chart": {
            "type": "bar", "title": "Revenue by Region",
            "labels": ["EMEA", "AMER", "APAC", "ROW"],
            "series": [
                {"name": "Q3 2024", "values": [1.8, 1.6, 0.6, 0.2]},
                {"name": "Q3 2023", "values": [1.5, 1.3, 0.5, 0.1]},
            ],
        },
        "donut": {
            "title": "Revenue Mix",
            "segments": [
                {"label": "Enterprise", "value": 52, "color": (33, 150, 243)},
                {"label": "Mid-Market", "value": 31, "color": (100, 181, 246)},
                {"label": "SMB",        "value": 17, "color": (187, 222, 251)},
            ],
        },
        "callouts": ["NRR: 118%", "EMEA leads growth", "3 new markets"],
    },
    {
        "title": "IT Operations Dashboard",
        "subtitle": "Infrastructure Health Metrics",
        "period_badge": "Week 44",
        "theme": "slate_tech",
        "kpis": [
            {"label": "Uptime", "value": "99.94%", "delta": "+0.1pp", "trend": "up",
             "sparkline": [99.7, 99.8, 99.9, 99.9, 99.94]},
            {"label": "Incidents", "value": "3", "delta": "-40%", "trend": "down",
             "sparkline": [8, 6, 5, 4, 3]},
            {"label": "MTTR", "value": "2.1h", "delta": "-18%", "trend": "down",
             "sparkline": [3.2, 2.8, 2.5, 2.2, 2.1]},
            {"label": "Change Success", "value": "97%", "delta": "+2pp", "trend": "up",
             "sparkline": [93, 94, 95, 96, 97]},
        ],
        "chart": {
            "type": "bar", "title": "Incident Count by Category",
            "labels": ["Network", "App", "Security", "DB", "Other"],
            "series": [
                {"name": "This Month", "values": [1, 0, 1, 0, 1]},
                {"name": "Last Month", "values": [2, 1, 1, 1, 0]},
            ],
        },
        "donut": {
            "title": "Issue Mix",
            "segments": [
                {"label": "Network",     "value": 35, "color": (99, 102, 241)},
                {"label": "Application", "value": 28, "color": (165, 180, 252)},
                {"label": "Security",    "value": 22, "color": (224, 231, 255)},
                {"label": "Other",       "value": 15, "color": (238, 242, 255)},
            ],
        },
        "callouts": ["P1 incidents: 0", "SLA met: 100%", "ITIL compliant"],
    },
    {
        "title": "HR Analytics",
        "subtitle": "Workforce Metrics Q3 2024",
        "period_badge": "Q3 2024",
        "theme": "healthcare_teal",
        "kpis": [
            {"label": "Headcount", "value": "342", "delta": "+8%", "trend": "up",
             "sparkline": [290, 305, 318, 330, 342]},
            {"label": "Attrition", "value": "8.2%", "delta": "-1.5pp", "trend": "down",
             "sparkline": [11.0, 10.2, 9.5, 8.8, 8.2]},
            {"label": "eNPS", "value": "+44", "delta": "+6pts", "trend": "up",
             "sparkline": [30, 35, 38, 41, 44]},
            {"label": "Time to Hire", "value": "28d", "delta": "-5d", "trend": "down",
             "sparkline": [38, 35, 33, 30, 28]},
        ],
        "chart": {
            "type": "bar", "title": "Headcount by Dept",
            "labels": ["Eng", "Sales", "Ops", "Finance", "HR"],
            "series": [
                {"name": "Current", "values": [95, 75, 68, 42, 22]},
                {"name": "Target",  "values": [100, 90, 70, 45, 25]},
            ],
        },
        "donut": {
            "title": "Workforce Mix",
            "segments": [
                {"label": "Perm",      "value": 68, "color": (13, 148, 136)},
                {"label": "Contract",  "value": 22, "color": (20, 184, 166)},
                {"label": "Part-time", "value": 10, "color": (94, 234, 212)},
            ],
        },
        "callouts": ["Diversity: 48% women", "Avg tenure: 3.2yr", "L&D spend $1.2M"],
    },
    {
        "title": "Financial Summary",
        "subtitle": "YTD Performance Metrics",
        "period_badge": "Q3 2024",
        "theme": "burgundy_legal",
        "kpis": [
            {"label": "Net Revenue", "value": "$18.4M", "delta": "+11%", "trend": "up",
             "sparkline": [15.2, 16.0, 16.8, 17.6, 18.4]},
            {"label": "EBITDA", "value": "$4.1M", "delta": "+18%", "trend": "up",
             "sparkline": [2.8, 3.2, 3.5, 3.8, 4.1]},
            {"label": "Cash", "value": "$7.2M", "delta": "+$0.8M", "trend": "up",
             "sparkline": [5.8, 6.1, 6.5, 6.9, 7.2]},
            {"label": "Debt/EBITDA", "value": "1.8x", "delta": "-0.3x", "trend": "down",
             "sparkline": [2.4, 2.3, 2.1, 2.0, 1.8]},
        ],
        "chart": {
            "type": "bar", "title": "Revenue by Quarter",
            "labels": ["Q3 23", "Q4 23", "Q1 24", "Q2 24", "Q3 24"],
            "series": [
                {"name": "Revenue", "values": [15.2, 16.0, 16.8, 17.6, 18.4]},
                {"name": "Target",  "values": [14.5, 15.5, 16.5, 17.0, 18.0]},
            ],
        },
        "donut": {
            "title": "Revenue by Segment",
            "segments": [
                {"label": "Advisory",     "value": 45, "color": (159, 18, 57)},
                {"label": "Transactions", "value": 35, "color": (253, 164, 175)},
                {"label": "Other",        "value": 20, "color": (255, 228, 230)},
            ],
        },
        "callouts": ["Dividend: $0.42/share", "Buyback: $2M completed", "S&P rating: BBB+"],
    },
    {
        "title": "Supply Chain",
        "subtitle": "Operations Scorecard",
        "period_badge": "Nov 2024",
        "theme": "warm_amber",
        "kpis": [
            {"label": "On-Time Del.", "value": "94.2%", "delta": "+2pp", "trend": "up",
             "sparkline": [88, 90, 91, 93, 94.2]},
            {"label": "Inventory", "value": "$12.1M", "delta": "-8%", "trend": "down",
             "sparkline": [14.2, 13.8, 13.1, 12.5, 12.1]},
            {"label": "Fill Rate", "value": "97.8%", "delta": "+0.5pp", "trend": "up",
             "sparkline": [96, 96.5, 97, 97.5, 97.8]},
            {"label": "Lead Time", "value": "14d", "delta": "-3d", "trend": "down",
             "sparkline": [19, 18, 17, 15, 14]},
        ],
        "chart": {
            "type": "bar", "title": "Delivery Performance by Region",
            "labels": ["North", "South", "East", "West", "Intl"],
            "series": [
                {"name": "On-Time", "values": [96, 93, 91, 95, 92]},
                {"name": "Target",  "values": [95, 95, 95, 95, 90]},
            ],
        },
        "donut": {
            "title": "Inventory by Category",
            "segments": [
                {"label": "Raw Mat",  "value": 42, "color": (245, 158, 11)},
                {"label": "WIP",      "value": 28, "color": (252, 211, 77)},
                {"label": "Finished", "value": 30, "color": (254, 243, 199)},
            ],
        },
        "callouts": ["Supplier OTD: 91%", "DIFOT: 91.2%", "3 new logistics partners"],
    },
]

_TIMELINE_DATA: list[dict] = [
    {
        "title": "Product Roadmap 2023–2025",
        "subtitle": "Key milestones and delivery targets",
        "theme": "slate_tech",
        "milestones": [
            {"date": "Q1 2023", "title": "Foundation",       "desc": "Core platform and API build",          "status": "done"},
            {"date": "Q3 2023", "title": "Beta Launch",      "desc": "50 design partners onboarded",         "status": "done"},
            {"date": "Q1 2024", "title": "GA Release",       "desc": "General availability — 200 customers", "status": "done"},
            {"date": "Q4 2024", "title": "v2.0 — Mobile",   "desc": "iOS and Android apps",                  "status": "current"},
            {"date": "Q2 2025", "title": "Enterprise Tier",  "desc": "SSO SAML audit logs",                  "status": "future"},
            {"date": "Q4 2025", "title": "AI Features",      "desc": "Predictive analytics module",          "status": "future"},
        ],
        "kpi_callouts": [
            {"label": "Customers", "value": "1200+"},
            {"label": "NPS",       "value": "+62"},
            {"label": "Uptime",    "value": "99.9%"},
        ],
    },
    {
        "title": "M&A Transaction Timeline",
        "subtitle": "Target Co. acquisition process",
        "theme": "corporate_blue",
        "milestones": [
            {"date": "Jan 2024", "title": "LOI Signed",           "desc": "Letter of intent with Target Co",         "status": "done"},
            {"date": "Mar 2024", "title": "Due Diligence",        "desc": "Financial legal and tech review",         "status": "done"},
            {"date": "Jun 2024", "title": "Financing Secured",    "desc": "$85M debt facility arranged",             "status": "done"},
            {"date": "Sep 2024", "title": "Regulatory Approval",  "desc": "Competition authority clearance",         "status": "current"},
            {"date": "Nov 2024", "title": "Closing",              "desc": "Transaction expected to complete",        "status": "future"},
            {"date": "Q1 2025",  "title": "Integration",          "desc": "100-day integration plan begins",         "status": "future"},
        ],
        "kpi_callouts": [
            {"label": "Deal Value",      "value": "$120M"},
            {"label": "Target Revenue",  "value": "$22M"},
            {"label": "EV/EBITDA",       "value": "8.5x"},
        ],
    },
    {
        "title": "Construction Project",
        "subtitle": "Commercial development schedule",
        "theme": "warm_amber",
        "milestones": [
            {"date": "Mar 2023", "title": "Site Prep",       "desc": "Demolition and groundwork",       "status": "done"},
            {"date": "Aug 2023", "title": "Foundation",      "desc": "Steel frame and foundation poured","status": "done"},
            {"date": "Feb 2024", "title": "Structure",       "desc": "Floors 1-12 complete",            "status": "done"},
            {"date": "Jul 2024", "title": "Fit-Out",         "desc": "MEP and interior works",          "status": "current"},
            {"date": "Oct 2024", "title": "Commissioning",   "desc": "Systems testing and sign-off",    "status": "future"},
            {"date": "Dec 2024", "title": "Handover",        "desc": "Practical completion",            "status": "future"},
        ],
        "kpi_callouts": [
            {"label": "Budget",     "value": "$48M"},
            {"label": "Completion", "value": "94%"},
            {"label": "Units",      "value": "144"},
        ],
    },
    {
        "title": "Regulatory Milestones",
        "subtitle": "ISO 27001 Certification Programme",
        "theme": "burgundy_legal",
        "milestones": [
            {"date": "Q4 2023", "title": "Gap Assessment",   "desc": "ISO 27001 readiness audit",        "status": "done"},
            {"date": "Q1 2024", "title": "Policy Update",    "desc": "28 new policies published",        "status": "done"},
            {"date": "Q2 2024", "title": "Staff Training",   "desc": "850 employees certified",          "status": "done"},
            {"date": "Q3 2024", "title": "Stage 1 Audit",    "desc": "Documentation review passed",      "status": "current"},
            {"date": "Q4 2024", "title": "Stage 2 Audit",    "desc": "On-site assessment",               "status": "future"},
            {"date": "Q1 2025", "title": "Certification",    "desc": "ISO 27001 certificate issued",     "status": "future"},
        ],
        "kpi_callouts": [
            {"label": "Controls Remediated", "value": "34/34"},
            {"label": "Staff Trained",        "value": "850"},
            {"label": "Cert Target",          "value": "Q1 2025"},
        ],
    },
]

_COMPARISON_DATA: list[dict] = [
    {
        "title": "Cloud Storage Solutions",
        "subtitle": "Feature and pricing comparison",
        "theme": "corporate_blue",
        "options": [
            {"name": "StorageX Basic",  "subtitle": "$49/mo",  "recommended": False},
            {"name": "CloudVault Pro",  "subtitle": "$149/mo", "recommended": True},
            {"name": "DataStore Ent",   "subtitle": "$399/mo", "recommended": False},
        ],
        "features": [
            {"name": "Storage",     "values": ["100 GB", "1 TB", "Unlimited"]},
            {"name": "Users",       "values": ["5", "50", "Unlimited"]},
            {"name": "Uptime SLA",  "values": ["99%", "99.9%", "99.99%"]},
            {"name": "Support",     "values": ["Email only", "Chat + Email", "24/7 Phone"]},
            {"name": "API Access",  "values": [True, True, True]},
            {"name": "SSO/SAML",    "values": [False, True, True]},
            {"name": "Audit Logs",  "values": [False, True, True]},
            {"name": "Data Region", "values": ["US only", "US/EU", "Global"]},
        ],
        "scores": [5, 8, 9],
    },
    {
        "title": "CRM Platform Evaluation",
        "subtitle": "Sales tool comparison for mid-market",
        "theme": "slate_tech",
        "options": [
            {"name": "SalesForce Lite", "subtitle": "$89/mo",  "recommended": False},
            {"name": "PipelinePro",     "subtitle": "$195/mo", "recommended": True},
            {"name": "CRM Enterprise",  "subtitle": "$450/mo", "recommended": False},
        ],
        "features": [
            {"name": "Contacts",        "values": ["5,000", "Unlimited", "Unlimited"]},
            {"name": "Pipelines",       "values": ["1", "5", "Unlimited"]},
            {"name": "Email Sync",      "values": [True, True, True]},
            {"name": "AI Insights",     "values": [False, True, True]},
            {"name": "Mobile App",      "values": [True, True, True]},
            {"name": "Custom Reports",  "values": [False, True, True]},
            {"name": "API/Webhooks",    "values": [False, True, True]},
            {"name": "Onboarding",      "values": ["Self-serve", "Guided", "Dedicated CSM"]},
        ],
        "scores": [5, 9, 8],
    },
    {
        "title": "Investment Fund Options",
        "subtitle": "Risk and return profile comparison",
        "theme": "emerald_finance",
        "options": [
            {"name": "Balanced Growth",  "subtitle": "Moderate risk", "recommended": False},
            {"name": "Global Equity",    "subtitle": "Higher risk",   "recommended": True},
            {"name": "Capital Preserve", "subtitle": "Low risk",      "recommended": False},
        ],
        "features": [
            {"name": "10yr Return",      "values": ["7.2%", "11.4%", "4.8%"]},
            {"name": "Volatility",       "values": ["Medium", "High", "Low"]},
            {"name": "Min. Investment",  "values": ["$10,000", "$50,000", "$5,000"]},
            {"name": "Management Fee",   "values": ["0.65%", "0.95%", "0.35%"]},
            {"name": "ESG Rated",        "values": [True, True, False]},
            {"name": "Liquidity",        "values": ["Daily", "Daily", "Weekly"]},
            {"name": "Currency Hedge",   "values": [False, True, False]},
            {"name": "Distribution",     "values": ["Accumulation", "Accumulation", "Income"]},
        ],
        "scores": [7, 8, 6],
    },
    {
        "title": "HR Software Comparison",
        "subtitle": "HCM platform evaluation",
        "theme": "healthcare_teal",
        "options": [
            {"name": "CoreHR Basic",   "subtitle": "$12/emp/mo", "recommended": False},
            {"name": "PeopleHub Pro",  "subtitle": "$28/emp/mo", "recommended": True},
            {"name": "HCM Enterprise", "subtitle": "$55/emp/mo", "recommended": False},
        ],
        "features": [
            {"name": "Payroll",           "values": [True, True, True]},
            {"name": "Recruitment",       "values": [False, True, True]},
            {"name": "Performance Mgmt",  "values": [False, True, True]},
            {"name": "Learning (LMS)",    "values": [False, True, True]},
            {"name": "Analytics",         "values": [False, True, True]},
            {"name": "Mobile App",        "values": [True, True, True]},
            {"name": "Integrations",      "values": ["10", "50+", "Unlimited"]},
            {"name": "Support",           "values": ["Email", "Chat + Email", "Dedicated"]},
        ],
        "scores": [5, 9, 8],
    },
]

_PROCESS_STEPS_DATA: list[dict] = [
    {
        "title": "Software Release Pipeline",
        "subtitle": "CI/CD process from planning to production",
        "theme": "corporate_blue",
        "steps": [
            {"number": "01", "title": "Plan",    "desc": "Define scope requirements and acceptance criteria", "duration": "2 wks",  "icon_char": "P"},
            {"number": "02", "title": "Develop", "desc": "Feature implementation with unit tests",           "duration": "4 wks",  "icon_char": "D"},
            {"number": "03", "title": "Test",    "desc": "Integration regression and UAT testing",           "duration": "2 wks",  "icon_char": "T"},
            {"number": "04", "title": "Review",  "desc": "Code review security scan and sign-off",          "duration": "1 wk",   "icon_char": "R"},
            {"number": "05", "title": "Deploy",  "desc": "Staged rollout with feature flags",               "duration": "3 days", "icon_char": "X"},
        ],
        "progress": 0.4,
        "progress_label": "Steps 1-2 complete (40%)",
    },
    {
        "title": "Purchase Order Approval",
        "subtitle": "Enterprise procurement workflow",
        "theme": "warm_amber",
        "steps": [
            {"number": "01", "title": "Request",    "desc": "Requester submits PO with justification",         "duration": "1 day",   "icon_char": "R"},
            {"number": "02", "title": "Budget Check","desc": "Finance confirms budget availability",            "duration": "1 day",   "icon_char": "B"},
            {"number": "03", "title": "Mgr Approval","desc": "Line manager reviews and approves",              "duration": "2 days",  "icon_char": "M"},
            {"number": "04", "title": "Procurement", "desc": "Procurement reviews vendor and terms",           "duration": "3 days",  "icon_char": "P"},
            {"number": "05", "title": "CFO Sign-off","desc": "Finance director approves >= $25K",             "duration": "1 day",   "icon_char": "C"},
            {"number": "06", "title": "Issue PO",    "desc": "ERP system generates and sends PO",             "duration": "1 hour",  "icon_char": "I"},
        ],
        "progress": 0.5,
        "progress_label": "Steps 1-3 complete (50%)",
    },
    {
        "title": "Employee Onboarding",
        "subtitle": "New hire activation journey",
        "theme": "healthcare_teal",
        "steps": [
            {"number": "01", "title": "Offer Accepted", "desc": "Contract signed and start date confirmed", "duration": "Day 0",    "icon_char": "O"},
            {"number": "02", "title": "Pre-boarding",   "desc": "IT setup payroll badge access",           "duration": "Week -1",  "icon_char": "P"},
            {"number": "03", "title": "Day 1",          "desc": "Orientation induction and team intro",    "duration": "Day 1",    "icon_char": "D"},
            {"number": "04", "title": "30-Day Plan",    "desc": "Role-specific training and shadowing",    "duration": "Week 1-4", "icon_char": "T"},
            {"number": "05", "title": "90-Day Review",  "desc": "Check-in goals and feedback",            "duration": "Day 90",   "icon_char": "R"},
        ],
        "progress": 0.6,
        "progress_label": "Steps 1-3 complete (60%)",
    },
    {
        "title": "Incident Response",
        "subtitle": "IT security incident lifecycle",
        "theme": "slate_tech",
        "steps": [
            {"number": "01", "title": "Detect",     "desc": "Alert triggered via monitoring",          "duration": "0-5 min",    "icon_char": "A"},
            {"number": "02", "title": "Triage",     "desc": "On-call assesses severity P1-P4",        "duration": "5-15 min",   "icon_char": "T"},
            {"number": "03", "title": "Escalate",   "desc": "Page incident commander if P1-P2",       "duration": "15 min",     "icon_char": "E"},
            {"number": "04", "title": "Remediate",  "desc": "Apply fix or failover",                  "duration": "15-120 min", "icon_char": "R"},
            {"number": "05", "title": "Restore",    "desc": "Verify full service restoration",        "duration": "+30 min",    "icon_char": "S"},
            {"number": "06", "title": "Post-mortem","desc": "Blameless review within 48h",            "duration": "48h",        "icon_char": "P"},
        ],
        "progress": 0.33,
        "progress_label": "Detect and Triage complete",
    },
    {
        "title": "Compliance Audit Process",
        "subtitle": "Internal audit lifecycle",
        "theme": "burgundy_legal",
        "steps": [
            {"number": "01", "title": "Scoping",            "desc": "Define audit scope objectives and timeline",      "duration": "2 wks", "icon_char": "S"},
            {"number": "02", "title": "Evidence Collection","desc": "Gather policies controls and artefacts",          "duration": "3 wks", "icon_char": "E"},
            {"number": "03", "title": "Fieldwork",          "desc": "Auditors test controls and interview staff",      "duration": "2 wks", "icon_char": "F"},
            {"number": "04", "title": "Draft Report",       "desc": "Findings and recommendations issued",            "duration": "1 wk",  "icon_char": "D"},
            {"number": "05", "title": "Mgmt Response",      "desc": "Auditee responds to findings",                   "duration": "2 wks", "icon_char": "M"},
            {"number": "06", "title": "Final Report",       "desc": "Sign-off and distribution to Board",             "duration": "1 wk",  "icon_char": "R"},
        ],
        "progress": 0.5,
        "progress_label": "Steps 1-3 complete (50%)",
    },
]

_HUB_SPOKE_DATA: list[dict] = [
    {
        "title": "Business Strategy 2024",
        "subtitle": "Strategic initiatives and key metrics",
        "theme": "corporate_blue",
        "center": {"title": "Strategic\nGrowth", "value": "+28%", "label": "YoY Revenue"},
        "spokes": [
            {"title": "Product",    "icon_char": "P", "stats": [{"label": "Features", "value": "47"},    {"label": "NPS",      "value": "+62"}]},
            {"title": "Revenue",    "icon_char": "R", "stats": [{"label": "ARR",      "value": "$4.2M"}, {"label": "Growth",   "value": "+23%"}]},
            {"title": "Customers",  "icon_char": "C", "stats": [{"label": "Active",   "value": "1204"},  {"label": "Churn",    "value": "1.2%"}]},
            {"title": "People",     "icon_char": "H", "stats": [{"label": "Headcount","value": "87"},    {"label": "eNPS",     "value": "+44"}]},
            {"title": "Technology", "icon_char": "T", "stats": [{"label": "Uptime",   "value": "99.9%"}, {"label": "Incidents","value": "2"}]},
            {"title": "Market",     "icon_char": "M", "stats": [{"label": "Share",    "value": "12.4%"}, {"label": "Rank",     "value": "#3"}]},
        ],
    },
    {
        "title": "Market Segmentation",
        "subtitle": "Total addressable market breakdown",
        "theme": "emerald_finance",
        "center": {"title": "Total\nAddressable", "value": "$4.8B", "label": "Market Size 2024"},
        "spokes": [
            {"title": "Enterprise",    "icon_char": "E", "stats": [{"label": "Segment",   "value": "$2.1B"}, {"label": "Our Share", "value": "2.1%"}]},
            {"title": "Mid-Market",    "icon_char": "M", "stats": [{"label": "Segment",   "value": "$1.4B"}, {"label": "Our Share", "value": "5.8%"}]},
            {"title": "SMB",           "icon_char": "S", "stats": [{"label": "Segment",   "value": "$0.8B"}, {"label": "Our Share", "value": "8.2%"}]},
            {"title": "Public Sector", "icon_char": "G", "stats": [{"label": "Segment",   "value": "$0.3B"}, {"label": "Our Share", "value": "3.1%"}]},
            {"title": "Healthcare",    "icon_char": "H", "stats": [{"label": "Segment",   "value": "$0.1B"}, {"label": "Our Share", "value": "14.2%"}]},
            {"title": "Education",     "icon_char": "A", "stats": [{"label": "Segment",   "value": "$0.1B"}, {"label": "Our Share", "value": "22.0%"}]},
        ],
    },
    {
        "title": "IT Architecture",
        "subtitle": "Cloud platform capabilities",
        "theme": "slate_tech",
        "center": {"title": "Cloud\nPlatform", "value": "99.95%", "label": "Availability SLA"},
        "spokes": [
            {"title": "Security",     "icon_char": "S", "stats": [{"label": "Endpoints",  "value": "2840"},   {"label": "Incidents",  "value": "3"}]},
            {"title": "Network",      "icon_char": "N", "stats": [{"label": "Latency",    "value": "8ms"},    {"label": "Bandwidth",  "value": "10Gbps"}]},
            {"title": "Compute",      "icon_char": "C", "stats": [{"label": "vCPUs",      "value": "3200"},   {"label": "Utilisation","value": "62%"}]},
            {"title": "Storage",      "icon_char": "D", "stats": [{"label": "Capacity",   "value": "2.4PB"},  {"label": "Used",       "value": "68%"}]},
            {"title": "Applications", "icon_char": "A", "stats": [{"label": "Services",   "value": "142"},    {"label": "P99 RT",     "value": "210ms"}]},
            {"title": "Monitoring",   "icon_char": "M", "stats": [{"label": "Alerts",     "value": "1204/d"}, {"label": "MTTD",       "value": "4min"}]},
        ],
    },
    {
        "title": "Risk Management",
        "subtitle": "Enterprise risk register overview",
        "theme": "burgundy_legal",
        "center": {"title": "Risk\nRegister", "value": "14", "label": "Open Risks"},
        "spokes": [
            {"title": "Strategic",   "icon_char": "S", "stats": [{"label": "Critical", "value": "2"}, {"label": "High", "value": "4"}]},
            {"title": "Operational", "icon_char": "O", "stats": [{"label": "Critical", "value": "1"}, {"label": "High", "value": "3"}]},
            {"title": "Financial",   "icon_char": "F", "stats": [{"label": "Critical", "value": "0"}, {"label": "High", "value": "2"}]},
            {"title": "Compliance",  "icon_char": "C", "stats": [{"label": "Critical", "value": "1"}, {"label": "High", "value": "2"}]},
            {"title": "Technology",  "icon_char": "T", "stats": [{"label": "Critical", "value": "2"}, {"label": "High", "value": "3"}]},
            {"title": "People",      "icon_char": "P", "stats": [{"label": "Critical", "value": "0"}, {"label": "High", "value": "1"}]},
        ],
    },
]

_MAGAZINE_DATA: list[dict] = [
    {
        "title": "Q3 Revenue Report",
        "theme": "corporate_blue",
        "hero_number": "$4.2M",
        "hero_label": "Total Revenue Q3 2024",
        "hero_context": (
            "Revenue exceeded analyst consensus by 8% driven by strong EMEA performance "
            "and an accelerating enterprise pipeline. Three new marquee logos contributed "
            "$420K in first-quarter ACV."
        ),
        "stats": [
            {"label": "Year-over-year growth", "value": 23,  "max": 40,  "suffix": "%"},
            {"label": "Gross margin",           "value": 61.4,"max": 100, "suffix": "%"},
            {"label": "Net retention",          "value": 118, "max": 150, "suffix": "%"},
            {"label": "New logos",              "value": 47,  "max": 80,  "suffix": ""},
        ],
        "callout": "Management raised FY2024 outlook by 5% to $16.8M, citing continued momentum in enterprise and APAC.",
    },
    {
        "title": "Patient Safety",
        "theme": "healthcare_teal",
        "hero_number": "99.2%",
        "hero_label": "Patient Safety Score",
        "hero_context": (
            "Ward C achieved its highest safety score since monitoring began in 2021, "
            "with zero serious adverse events for the third consecutive quarter. "
            "Infection rates fell to a record low of 0.8 per 1,000 patient-days."
        ),
        "stats": [
            {"label": "Zero harm days",           "value": 287,  "max": 365, "suffix": "d"},
            {"label": "Hand hygiene compliance",  "value": 97.4, "max": 100, "suffix": "%"},
            {"label": "Readmission rate",         "value": 4.2,  "max": 20,  "suffix": "%"},
            {"label": "Patient satisfaction",     "value": 92,   "max": 100, "suffix": "%"},
        ],
        "callout": "Implementation of the new medication reconciliation protocol contributed to a 34% reduction in prescription errors.",
    },
    {
        "title": "Infrastructure Uptime",
        "theme": "slate_tech",
        "hero_number": "99.95%",
        "hero_label": "System Availability",
        "hero_context": (
            "The platform recorded its best availability quarter following the completion "
            "of the N+2 redundancy upgrade across all primary data centres. Total downtime "
            "for the quarter was 22 minutes — all planned maintenance."
        ),
        "stats": [
            {"label": "Planned downtime",      "value": 22,  "max": 60,  "suffix": "min"},
            {"label": "Unplanned incidents",   "value": 0,   "max": 10,  "suffix": ""},
            {"label": "MTTR (all incidents)",  "value": 2.1, "max": 10,  "suffix": "hrs"},
            {"label": "Change success rate",   "value": 97,  "max": 100, "suffix": "%"},
        ],
        "callout": "The infrastructure team delivered the DR validation exercise 3 weeks ahead of schedule, enabling early risk reduction.",
    },
    {
        "title": "Market Analysis",
        "theme": "emerald_finance",
        "hero_number": "12.4%",
        "hero_label": "Market Share",
        "hero_context": (
            "The company gained 1.8 percentage points of market share in Q3, the largest "
            "quarterly gain in five years, driven by competitive displacement in the "
            "mid-market segment. Analyst consensus now places the company third in the category."
        ),
        "stats": [
            {"label": "YoY share gain",      "value": 1.8, "max": 5,  "suffix": "pp"},
            {"label": "Competitive wins",    "value": 34,  "max": 50, "suffix": ""},
            {"label": "Avg deal premium",    "value": 18,  "max": 50, "suffix": "%"},
            {"label": "Pipeline coverage",   "value": 3.2, "max": 5,  "suffix": "x"},
        ],
        "callout": "Two top-five competitors lost share this quarter; positioning ahead of their contract renewal cycles is a priority for Q4.",
    },
]

_PICTOGRAPH_DATA: list[dict] = [
    {
        "title": "Customer Satisfaction",
        "subtitle": "Responses: 1,204 customers  |  Net Promoter Score: +56",
        "theme": "emerald_finance",
        "unit_label": "Each icon = 2% of respondents",
        "rows": [
            {"label": "Promoters (9-10)", "value": 74, "icon_color": (16, 185, 129)},
            {"label": "Passives (7-8)",   "value": 18, "icon_color": (251, 191, 36)},
            {"label": "Detractors (0-6)", "value": 8,  "icon_color": (239, 68, 68)},
        ],
        "summary": {"label": "NPS Score", "value": 56, "min": -100, "max": 100, "suffix": ""},
    },
    {
        "title": "Budget Allocation",
        "subtitle": "FY2024 Total Budget: $24.8M  |  Last approved: Oct 2024",
        "theme": "corporate_blue",
        "unit_label": "Each icon = 2% of total budget",
        "rows": [
            {"label": "Engineering & Product", "value": 38, "icon_color": (33, 150, 243)},
            {"label": "Sales & Marketing",     "value": 27, "icon_color": (100, 181, 246)},
            {"label": "Operations",            "value": 18, "icon_color": (187, 222, 251)},
            {"label": "Finance & Legal",       "value": 10, "icon_color": (226, 232, 240)},
            {"label": "HR & Admin",            "value": 7,  "icon_color": (248, 250, 252)},
        ],
        "summary": {"label": "Budget Utilised", "value": 91, "min": 0, "max": 100, "suffix": "%"},
    },
    {
        "title": "Headcount Distribution",
        "subtitle": "Total: 342 employees  |  As of Q3 2024",
        "theme": "healthcare_teal",
        "unit_label": "Each icon = 2% of workforce",
        "rows": [
            {"label": "Engineering",      "value": 28, "icon_color": (13, 148, 136)},
            {"label": "Sales",            "value": 22, "icon_color": (20, 184, 166)},
            {"label": "Operations",       "value": 20, "icon_color": (94, 234, 212)},
            {"label": "Customer Success", "value": 15, "icon_color": (153, 246, 228)},
            {"label": "Finance",          "value": 10, "icon_color": (204, 251, 241)},
            {"label": "HR & Other",       "value": 5,  "icon_color": (240, 253, 250)},
        ],
        "summary": {"label": "Billable Utilisation", "value": 78, "min": 0, "max": 100, "suffix": "%"},
    },
    {
        "title": "Market Share",
        "subtitle": "Global Market: $4.8B  |  Source: Industry Report Q3 2024",
        "theme": "warm_amber",
        "unit_label": "Each icon = 2% market share",
        "rows": [
            {"label": "Leader Co.",    "value": 31, "icon_color": (245, 158, 11)},
            {"label": "Our Company",   "value": 12, "icon_color": (252, 211, 77)},
            {"label": "Competitor B",  "value": 11, "icon_color": (253, 230, 138)},
            {"label": "Competitor C",  "value": 9,  "icon_color": (254, 243, 199)},
            {"label": "Other",         "value": 37, "icon_color": (254, 252, 232)},
        ],
        "summary": {"label": "YoY Gain", "value": 1.8, "min": 0, "max": 10, "suffix": "pp"},
    },
]

_FLOOR_PLANS: list[dict] = [
    {
        "title": "NexGen Systems — Level 3 Office Floor",
        "scale": "1:100", "grid_w": 12, "grid_h": 8,
        "rooms": [
            ("Reception",          0, 0, 3, 2, "reception"),
            ("Meeting Room A",     0, 2, 2, 2, "meeting"),
            ("Meeting Room B",     0, 4, 2, 2, "meeting"),
            ("Open Plan Office",   3, 0, 6, 4, "open_office"),
            ("Executive Suite",    3, 4, 3, 2, "executive"),
            ("Server Room",        6, 4, 2, 2, "server"),
            ("Breakout / Kitchen", 9, 0, 3, 3, "kitchen"),
            ("Conference Room",    9, 3, 3, 3, "meeting"),
            ("Toilets (M)",        2, 6, 2, 2, "toilet"),
            ("Toilets (F)",        4, 6, 2, 2, "toilet"),
            ("Storage",            6, 6, 2, 2, "storage"),
            ("Corridor",           2, 2, 1, 4, "corridor"),
        ],
        "description": "Open-plan corporate office with meeting rooms and support spaces",
    },
    {
        "title": "Apex Medical Center — Ward C Floorplan",
        "scale": "1:100", "grid_w": 14, "grid_h": 7,
        "rooms": [
            ("Patient Room 1",     0, 0, 3, 3, "patient_room"),
            ("Patient Room 2",     0, 4, 3, 3, "patient_room"),
            ("Patient Room 3",     3, 0, 3, 3, "patient_room"),
            ("Patient Room 4",     3, 4, 3, 3, "patient_room"),
            ("Nurse Station",      6, 2, 3, 3, "nurse_station"),
            ("Medication Room",    6, 0, 3, 2, "clinical"),
            ("Treatment Room",     6, 5, 3, 2, "clinical"),
            ("Waiting Area",       9, 0, 3, 3, "waiting"),
            ("Consultation",       9, 3, 2, 2, "meeting"),
            ("Utility / Sluice",   9, 5, 2, 2, "storage"),
            ("Toilets",           11, 0, 3, 3, "toilet"),
            ("Staff Room",        11, 3, 3, 4, "staff"),
            ("Corridor",           0, 3, 6, 1, "corridor"),
        ],
        "description": "Hospital ward with patient rooms, nurse station, and clinical spaces",
    },
    {
        "title": "Continental Freight — Distribution Warehouse",
        "scale": "1:200", "grid_w": 16, "grid_h": 8,
        "rooms": [
            ("Receiving Dock",     0, 0, 4, 3, "dock"),
            ("Inbound Staging",    0, 3, 4, 3, "staging"),
            ("Bulk Storage A",     4, 0, 4, 4, "storage"),
            ("Bulk Storage B",     4, 4, 4, 4, "storage"),
            ("Rack Storage",       8, 0, 4, 6, "storage"),
            ("Pick & Pack Area",   8, 6, 4, 2, "ops"),
            ("Dispatch Staging",  12, 0, 4, 3, "staging"),
            ("Dispatch Dock",     12, 3, 4, 3, "dock"),
            ("Office",            12, 6, 2, 2, "office"),
            ("Break Room",        14, 6, 2, 2, "kitchen"),
        ],
        "description": "Industrial distribution warehouse with receiving, storage, and dispatch zones",
    },
    {
        "title": "Harbor Properties — Unit 4B Apartment",
        "scale": "1:50", "grid_w": 10, "grid_h": 8,
        "rooms": [
            ("Living Room",        0, 0, 5, 4, "living"),
            ("Dining Room",        5, 0, 3, 3, "living"),
            ("Kitchen",            5, 3, 3, 2, "kitchen"),
            ("Master Bedroom",     0, 4, 4, 4, "bedroom"),
            ("Bedroom 2",          4, 4, 3, 3, "bedroom"),
            ("Bedroom 3",          7, 3, 3, 3, "bedroom"),
            ("Ensuite",            4, 7, 2, 1, "toilet"),
            ("Main Bathroom",      6, 7, 2, 1, "toilet"),
            ("Utility Room",       8, 0, 2, 3, "storage"),
            ("Hallway",            8, 3, 2, 4, "corridor"),
        ],
        "description": "Three-bedroom residential apartment floor plan",
    },
    {
        "title": "Pacific Bistro — Restaurant Floor Plan",
        "scale": "1:75", "grid_w": 12, "grid_h": 8,
        "rooms": [
            ("Main Dining",        0, 0, 6, 6, "dining"),
            ("Private Dining",     0, 6, 3, 2, "dining"),
            ("Bar & Lounge",       6, 0, 4, 4, "bar"),
            ("Host Station",       6, 4, 2, 2, "reception"),
            ("Main Kitchen",       8, 4, 4, 4, "kitchen"),
            ("Prep Kitchen",      10, 0, 2, 4, "kitchen"),
            ("Cold Storage",       8, 0, 2, 4, "storage"),
            ("Toilets (M)",        3, 6, 2, 2, "toilet"),
            ("Toilets (F)",        5, 6, 2, 2, "toilet"),
            ("Staff Area",         7, 6, 2, 2, "staff"),
        ],
        "description": "Restaurant layout with dining, bar, kitchen, and support areas",
    },
]

_ROOM_COLORS: dict = {
    "reception":    (180, 215, 245),
    "meeting":      (200, 230, 200),
    "open_office":  (240, 245, 252),
    "executive":    (220, 200, 245),
    "server":       (200, 200, 210),
    "kitchen":      (252, 235, 200),
    "toilet":       (200, 220, 220),
    "storage":      (220, 220, 215),
    "corridor":     (245, 245, 240),
    "patient_room": (210, 235, 210),
    "nurse_station":(180, 215, 245),
    "clinical":     (235, 215, 215),
    "waiting":      (248, 238, 210),
    "staff":        (230, 220, 240),
    "dock":         (220, 215, 200),
    "staging":      (230, 225, 210),
    "ops":          (210, 230, 210),
    "office":       (240, 245, 252),
    "living":       (240, 245, 252),
    "bedroom":      (210, 235, 210),
    "bar":          (248, 225, 200),
    "dining":       (252, 242, 220),
}


# ===========================================================================
# SECTION 6: Template Renderers
# ===========================================================================

def _render_stat_dashboard(data: dict, theme: _Theme, rng: Any) -> Image.Image:
    try:
        W, H = 1200, 780
        img = Image.new("RGB", (W, H), theme.bg)
        draw = ImageDraw.Draw(img)

        font_title   = _load_font(20, bold=True)
        font_sub     = _load_font(11)
        font_badge   = _load_font(11, bold=True)
        font_kpi_val = _load_font(26, bold=True)
        font_kpi_lab = _load_font(11)
        font_delta   = _load_font(12)
        font_sect    = _load_font(13, bold=True)
        font_body    = _load_font(11)

        # Header gradient
        _gradient_rect(img, 0, 0, W, 72, theme.primary, _lighten(theme.primary, 0.2))

        # Title
        draw.text((20, 14), data["title"], font=font_title, fill=(255, 255, 255))
        draw.text((20, 44), data["subtitle"], font=font_sub, fill=_lighten(theme.primary, 0.7))

        # Period badge
        badge_text = data.get("period_badge", "")
        bw, bh = _text_size(draw, badge_text, font_badge)
        bx = W - bw - 32
        _rounded_rect(draw, bx - 8, 22, bx + bw + 8, 22 + bh + 10, 6,
                      fill=theme.accent)
        draw.text((bx, 27), badge_text, font=font_badge, fill=(255, 255, 255))

        # 4 KPI cards
        kpis = data.get("kpis", [])
        card_y = 84
        card_h = 142
        card_w = (W - 80) // 4 - 10
        for ki, kpi in enumerate(kpis[:4]):
            cx = 20 + ki * (card_w + 10)
            _shadow_card(img, draw, cx, card_y, card_w, card_h, theme.surface, theme, 6)
            # Left accent strip
            draw.rectangle([cx, card_y, cx + 4, card_y + card_h], fill=theme.accent)
            # Label
            draw.text((cx + 14, card_y + 10), kpi["label"], font=font_kpi_lab, fill=theme.subtext)
            # Value
            draw.text((cx + 14, card_y + 28), kpi["value"], font=font_kpi_val, fill=theme.text)
            # Delta
            trend = kpi.get("trend", "up")
            delta_color = theme.positive if trend == "up" else theme.negative
            arrow = "+" if trend == "up" else ""
            draw.text((cx + 14, card_y + 76), f"{arrow}{kpi['delta']}", font=font_delta, fill=delta_color)
            # Sparkline
            spark_vals = kpi.get("sparkline", [])
            if spark_vals:
                spark_img = _sparkline(spark_vals, 80, 30, theme.accent, theme.surface)
                img.paste(spark_img, (cx + card_w - 88, card_y + card_h - 38))

        # Left big card: bar chart
        lc_x, lc_y, lc_w, lc_h = 20, 244, 720, 510
        _shadow_card(img, draw, lc_x, lc_y, lc_w, lc_h, theme.surface, theme, 8)
        chart_data = data.get("chart", {})
        draw.text((lc_x + 14, lc_y + 10), chart_data.get("title", "Chart"), font=font_sect, fill=theme.text)
        bar_img = _bar_chart(
            chart_data.get("labels", []),
            chart_data.get("series", []),
            lc_w - 24, lc_h - 44,
            theme,
        )
        img.paste(bar_img, (lc_x + 12, lc_y + 36))

        # Right top: donut
        rt_x, rt_y, rt_w, rt_h = 752, 244, 428, 248
        _shadow_card(img, draw, rt_x, rt_y, rt_w, rt_h, theme.surface, theme, 8)
        donut_data = data.get("donut", {})
        draw.text((rt_x + 14, rt_y + 10), donut_data.get("title", ""), font=font_sect, fill=theme.text)
        donut_img = _donut_chart(
            donut_data.get("segments", []),
            rt_w - 24, rt_h - 40,
            theme, hole_color=theme.surface,
        )
        img.paste(donut_img, (rt_x + 12, rt_y + 32))

        # Right bottom: callouts
        rb_x, rb_y, rb_w, rb_h = 752, 504, 428, 250
        _shadow_card(img, draw, rb_x, rb_y, rb_w, rb_h, theme.surface, theme, 8)
        draw.text((rb_x + 14, rb_y + 10), "Key Callouts", font=font_sect, fill=theme.text)
        callouts = data.get("callouts", [])
        for ci, callout in enumerate(callouts[:6]):
            cy2 = rb_y + 38 + ci * 34
            draw.ellipse([rb_x + 14, cy2 + 4, rb_x + 22, cy2 + 12], fill=theme.accent)
            lines = _wrap_text(draw, callout, font_body, rb_w - 42)
            for li, line in enumerate(lines[:2]):
                draw.text((rb_x + 30, cy2 + li * 14), line, font=font_body, fill=theme.text)

        return img
    except Exception:
        img = Image.new("RGB", (800, 600), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "stat_dashboard render error", fill=(100, 100, 100))
        return img


def _render_timeline(data: dict, theme: _Theme, rng: Any) -> Image.Image:
    try:
        W, H = 1200, 600
        img = Image.new("RGB", (W, H), theme.bg)
        draw = ImageDraw.Draw(img)

        font_title = _load_font(18, bold=True)
        font_sub   = _load_font(11)
        font_date  = _load_font(10)
        font_bold  = _load_font(12, bold=True)
        font_desc  = _load_font(10)
        font_kpi_v = _load_font(18, bold=True)
        font_kpi_l = _load_font(10)

        # Header
        _gradient_rect(img, 0, 0, W, 64, theme.primary, _lighten(theme.primary, 0.2))
        draw.text((20, 12), data["title"], font=font_title, fill=(255, 255, 255))
        draw.text((20, 40), data["subtitle"], font=font_sub, fill=_lighten(theme.primary, 0.7))

        # Spine
        spine_y = 280
        spine_x0, spine_x1 = 90, 1110
        draw.line([(spine_x0, spine_y), (spine_x1, spine_y)],
                  fill=theme.border, width=3)

        milestones = data.get("milestones", [])
        n = len(milestones)
        if n == 0:
            return img

        spacing = (spine_x1 - spine_x0) / max(n - 1, 1)

        for i, ms in enumerate(milestones):
            mx = int(spine_x0 + i * spacing)
            status = ms.get("status", "future")

            # Circle at spine
            r = 12
            if status == "done":
                draw.ellipse([mx - r, spine_y - r, mx + r, spine_y + r],
                             fill=theme.accent)
            elif status == "current":
                draw.ellipse([mx - r, spine_y - r, mx + r, spine_y + r],
                             fill=theme.bg, outline=theme.accent, width=3)
                draw.ellipse([mx - 5, spine_y - 5, mx + 5, spine_y + 5],
                             fill=theme.accent)
            else:
                draw.ellipse([mx - r, spine_y - r, mx + r, spine_y + r],
                             fill=theme.bg, outline=theme.border, width=2)

            above = (i % 2 == 0)
            if above:
                # Labels above spine
                date_y, title_y, desc_y = 185, 205, 223
                tick_y0, tick_y1 = spine_y - r, date_y + 14
                draw.line([(mx, tick_y0), (mx, tick_y1)], fill=theme.border, width=1)
            else:
                # Labels below spine
                date_y, title_y, desc_y = 305, 325, 345
                tick_y0, tick_y1 = spine_y + r, date_y - 2
                draw.line([(mx, tick_y0), (mx, tick_y1)], fill=theme.border, width=1)

            date_str = ms.get("date", "")
            tw, _ = _text_size(draw, date_str, font_date)
            draw.text((mx - tw // 2, date_y), date_str, font=font_date, fill=theme.subtext)

            title_str = ms.get("title", "")
            tw2, _ = _text_size(draw, title_str, font_bold)
            draw.text((mx - tw2 // 2, title_y), title_str, font=font_bold, fill=theme.text)

            desc_str = ms.get("desc", "")
            max_desc_w = max(80, int(spacing) - 10)
            desc_lines = _wrap_text(draw, desc_str, font_desc, max_desc_w)
            for di, dl in enumerate(desc_lines[:2]):
                dw, _ = _text_size(draw, dl, font_desc)
                draw.text((mx - dw // 2, desc_y + di * 13), dl, font=font_desc, fill=theme.subtext)

        # KPI callout row
        kpi_callouts = data.get("kpi_callouts", [])
        if kpi_callouts:
            n_kpi = len(kpi_callouts)
            kpi_card_w = (W - 60) // n_kpi - 10
            kpi_y = 430
            kpi_h = 130
            for ki, kpi in enumerate(kpi_callouts):
                kx = 30 + ki * (kpi_card_w + 10)
                _shadow_card(img, draw, kx, kpi_y, kpi_card_w, kpi_h, theme.surface, theme, 6)
                _draw_text_center(draw, kx + kpi_card_w // 2, kpi_y + 42,
                                  kpi["value"], font_kpi_v, theme.accent)
                _draw_text_center(draw, kx + kpi_card_w // 2, kpi_y + 72,
                                  kpi["label"], font_kpi_l, theme.subtext)

        return img
    except Exception:
        img = Image.new("RGB", (800, 600), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "timeline render error", fill=(100, 100, 100))
        return img


def _render_comparison(data: dict, theme: _Theme, rng: Any) -> Image.Image:
    try:
        W, H = 1100, 740
        img = Image.new("RGB", (W, H), theme.bg)
        draw = ImageDraw.Draw(img)

        font_title  = _load_font(18, bold=True)
        font_sub    = _load_font(11)
        font_bold   = _load_font(13, bold=True)
        font_med    = _load_font(12)
        font_sm     = _load_font(10)
        font_check  = _load_font(13, bold=True)

        # Header
        _gradient_rect(img, 0, 0, W, 64, theme.primary, _lighten(theme.primary, 0.2))
        draw.text((20, 12), data["title"], font=font_title, fill=(255, 255, 255))
        draw.text((20, 42), data["subtitle"], font=font_sub, fill=_lighten(theme.primary, 0.7))

        options = data.get("options", [])
        features = data.get("features", [])
        scores = data.get("scores", [])
        n_opt = len(options)
        if n_opt == 0:
            return img

        label_col_w = 200
        opt_col_w = (W - label_col_w) // n_opt
        row_h = 36
        header_h = 118
        header_y = 72

        # Column headers
        for oi, opt in enumerate(options):
            ox = label_col_w + oi * opt_col_w
            is_rec = opt.get("recommended", False)
            card_fill = theme.accent if is_rec else theme.surface
            card_text = (255, 255, 255) if is_rec else theme.text
            _shadow_card(img, draw, ox + 4, header_y + 4, opt_col_w - 8, header_h - 8,
                         card_fill, theme, 8)
            # Name
            name_font = _load_font(13, bold=True)
            tw, _ = _text_size(draw, opt["name"], name_font)
            draw.text((ox + (opt_col_w - tw) // 2, header_y + 14),
                      opt["name"], font=name_font, fill=card_text)
            # Subtitle
            sub_str = opt.get("subtitle", "")
            sw, _ = _text_size(draw, sub_str, font_sm)
            draw.text((ox + (opt_col_w - sw) // 2, header_y + 34),
                      sub_str, font=font_sm, fill=_lighten(card_text, 0.2) if is_rec else theme.subtext)
            # Score dots
            score = scores[oi] if oi < len(scores) else 5
            dot_y = header_y + 60
            dot_x = ox + (opt_col_w - score * 10) // 2
            for di in range(10):
                dot_fill = theme.positive if di < score else theme.border
                draw.ellipse([dot_x + di * 10, dot_y, dot_x + di * 10 + 7, dot_y + 7],
                             fill=dot_fill)
            # Recommended star
            if is_rec:
                star_str = "Recommended"
                sw2, _ = _text_size(draw, star_str, font_sm)
                draw.text((ox + (opt_col_w - sw2) // 2, header_y + 88),
                          star_str, font=font_sm, fill=(255, 255, 200))

        # Feature rows
        feat_y = header_y + header_h
        for fi, feat in enumerate(features):
            fy = feat_y + fi * row_h
            row_bg = theme.surface if fi % 2 == 0 else theme.bg
            draw.rectangle([0, fy, W, fy + row_h], fill=row_bg)
            # Feature name
            draw.text((14, fy + (row_h - 12) // 2), feat["name"],
                      font=font_med, fill=theme.text)
            # Values
            vals = feat.get("values", [])
            for oi in range(n_opt):
                ox = label_col_w + oi * opt_col_w
                val = vals[oi] if oi < len(vals) else ""
                if isinstance(val, bool):
                    check_str = "+" if val else "-"
                    check_color = theme.positive if val else theme.neutral
                    cw, _ = _text_size(draw, check_str, font_check)
                    draw.text((ox + (opt_col_w - cw) // 2, fy + (row_h - 14) // 2),
                              check_str, font=font_check, fill=check_color)
                else:
                    val_str = str(val)
                    vw, _ = _text_size(draw, val_str, font_sm)
                    draw.text((ox + (opt_col_w - vw) // 2, fy + (row_h - 10) // 2),
                              val_str, font=font_sm, fill=theme.text)
            # Row divider
            draw.line([(0, fy + row_h), (W, fy + row_h)], fill=theme.border, width=1)

        # Score bars row
        score_y = feat_y + len(features) * row_h + 8
        for oi, opt in enumerate(options):
            ox = label_col_w + oi * opt_col_w
            score = scores[oi] if oi < len(scores) else 5
            pb_img = _progress_bar(score, 10, opt_col_w - 20, 16, theme.accent, theme.bg)
            img.paste(pb_img, (ox + 10, score_y))
            score_str = f"{score}/10"
            sw, _ = _text_size(draw, score_str, font_sm)
            draw.text((ox + (opt_col_w - sw) // 2, score_y + 20), score_str,
                      font=font_sm, fill=theme.text)

        return img
    except Exception:
        img = Image.new("RGB", (800, 600), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "comparison render error", fill=(100, 100, 100))
        return img


def _render_process_steps(data: dict, theme: _Theme, rng: Any) -> Image.Image:
    try:
        W, H = 1200, 580
        img = Image.new("RGB", (W, H), theme.bg)
        draw = ImageDraw.Draw(img)

        font_title = _load_font(18, bold=True)
        font_sub   = _load_font(11)
        font_icon  = _load_font(16, bold=True)
        font_num   = _load_font(10)
        font_step  = _load_font(14, bold=True)
        font_desc  = _load_font(11)
        font_dur   = _load_font(10)
        font_prog  = _load_font(11)

        # Header
        _gradient_rect(img, 0, 0, W, 64, theme.primary, _lighten(theme.primary, 0.2))
        draw.text((20, 12), data["title"], font=font_title, fill=(255, 255, 255))
        draw.text((20, 42), data["subtitle"], font=font_sub, fill=_lighten(theme.primary, 0.7))

        steps = data.get("steps", [])
        N = min(len(steps), 6)
        if N == 0:
            return img

        margin = 40
        gap = 12
        box_y = 82
        box_h = 320
        box_w = (W - 2 * margin - gap * (N - 1)) // N

        for si, step in enumerate(steps[:N]):
            bx = margin + si * (box_w + gap)
            _shadow_card(img, draw, bx, box_y, box_w, box_h, theme.surface, theme, 8)

            # Icon circle
            icon_cx = bx + box_w // 2
            icon_cy = box_y + 36
            icon_r = 24
            draw.ellipse([icon_cx - icon_r, icon_cy - icon_r,
                          icon_cx + icon_r, icon_cy + icon_r],
                         fill=theme.accent)
            _draw_text_center(draw, icon_cx, icon_cy, step.get("icon_char", "?"),
                              font_icon, (255, 255, 255))

            # Step number
            num_str = f"Step {step.get('number', si + 1)}"
            nw, _ = _text_size(draw, num_str, font_num)
            draw.text((bx + (box_w - nw) // 2, icon_cy + icon_r + 6),
                      num_str, font=font_num, fill=theme.subtext)

            # Title
            title_str = step.get("title", "")
            title_lines = _wrap_text(draw, title_str, font_step, box_w - 16)
            ty = icon_cy + icon_r + 22
            for li, line in enumerate(title_lines[:2]):
                tw, _ = _text_size(draw, line, font_step)
                draw.text((bx + (box_w - tw) // 2, ty + li * 20),
                          line, font=font_step, fill=theme.text)
            ty += len(title_lines[:2]) * 20 + 8

            # Description
            desc_str = step.get("desc", "")
            desc_lines = _wrap_text(draw, desc_str, font_desc, box_w - 20)
            for li, line in enumerate(desc_lines[:4]):
                dw, _ = _text_size(draw, line, font_desc)
                draw.text((bx + (box_w - dw) // 2, ty + li * 16),
                          line, font=font_desc, fill=theme.subtext)

            # Duration badge at bottom
            dur_str = step.get("duration", "")
            if dur_str:
                dw, dh = _text_size(draw, dur_str, font_dur)
                badge_x = bx + (box_w - dw - 16) // 2
                badge_y = box_y + box_h - 30
                _rounded_rect(draw, badge_x, badge_y, badge_x + dw + 16, badge_y + dh + 8,
                              4, fill=theme.border)
                draw.text((badge_x + 8, badge_y + 4), dur_str, font=font_dur, fill=theme.text)

            # Arrow between boxes
            if si < N - 1:
                ax = bx + box_w + gap // 2
                ay = box_y + box_h // 2
                draw.line([(bx + box_w, ay), (ax + 4, ay)], fill=theme.accent, width=2)
                # Arrowhead triangle
                draw.polygon([(ax + 4, ay - 5), (ax + 4, ay + 5), (ax + 10, ay)],
                             fill=theme.accent)

        # Progress bar
        prog_y = box_y + box_h + 22
        prog_val = data.get("progress", 0.0)
        prog_img = _progress_bar(prog_val, 1.0, W - 2 * margin, 22,
                                 theme.accent, theme.border)
        img.paste(prog_img, (margin, prog_y))

        # Progress label
        prog_label = data.get("progress_label", "")
        draw.text((margin, prog_y + 28), prog_label, font=font_prog, fill=theme.subtext)

        return img
    except Exception:
        img = Image.new("RGB", (800, 600), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "process_steps render error", fill=(100, 100, 100))
        return img


def _render_hub_spoke(data: dict, theme: _Theme, rng: Any) -> Image.Image:
    try:
        W, H = 1100, 960
        img = Image.new("RGB", (W, H), theme.bg)
        draw = ImageDraw.Draw(img)

        font_title   = _load_font(18, bold=True)
        font_sub     = _load_font(11)
        font_hub_v   = _load_font(28, bold=True)
        font_hub_t   = _load_font(18, bold=True)
        font_hub_l   = _load_font(11)
        font_spoke_t = _load_font(12, bold=True)
        font_spoke_i = _load_font(12, bold=True)
        font_stat_l  = _load_font(9)
        font_stat_v  = _load_font(11, bold=True)

        # Header strip
        _gradient_rect(img, 0, 0, W, 52, theme.primary, _lighten(theme.primary, 0.2))
        draw.text((20, 10), data["title"], font=font_title, fill=(255, 255, 255))
        draw.text((20, 36), data["subtitle"], font=font_sub, fill=_lighten(theme.primary, 0.7))

        center = data.get("center", {})
        spokes = data.get("spokes", [])

        hub_cx, hub_cy = W // 2, 510
        hub_r = 128
        sat_dist = 270
        sat_r = 105

        def _hex_pts(cx, cy, r):
            return [(int(cx + r * math.cos(math.radians(60 * i))),
                     int(cy + r * math.sin(math.radians(60 * i))))
                    for i in range(6)]

        # Draw connection lines FIRST
        for si, spoke in enumerate(spokes[:6]):
            angle_deg = 60 * si - 90
            sx = int(hub_cx + sat_dist * math.cos(math.radians(angle_deg)))
            sy = int(hub_cy + sat_dist * math.sin(math.radians(angle_deg)))
            draw.line([(hub_cx, hub_cy), (sx, sy)], fill=theme.border, width=2)

        # Draw hub hex
        hub_pts = _hex_pts(hub_cx, hub_cy, hub_r)
        draw.polygon(hub_pts, fill=theme.primary, outline=_lighten(theme.primary, 0.3))

        # Hub text
        center_title = center.get("title", "")
        center_lines = center_title.split("\n")
        line_h = 22
        total_h = len(center_lines) * line_h + 40
        start_y = hub_cy - total_h // 2 + 4
        for li, line in enumerate(center_lines):
            _draw_text_center(draw, hub_cx, start_y + li * line_h, line,
                              font_hub_t, (255, 255, 255))
        vy = start_y + len(center_lines) * line_h + 4
        _draw_text_center(draw, hub_cx, vy, center.get("value", ""),
                          font_hub_v, (255, 255, 255))
        _draw_text_center(draw, hub_cx, vy + 36, center.get("label", ""),
                          font_hub_l, _lighten(theme.primary, 0.6))

        # Draw satellite hexes
        for si, spoke in enumerate(spokes[:6]):
            angle_deg = 60 * si - 90
            sx = int(hub_cx + sat_dist * math.cos(math.radians(angle_deg)))
            sy = int(hub_cy + sat_dist * math.sin(math.radians(angle_deg)))

            sat_pts = _hex_pts(sx, sy, sat_r)
            draw.polygon(sat_pts, fill=theme.surface, outline=theme.accent)
            # Also draw outline with width
            for i in range(6):
                p1 = sat_pts[i]
                p2 = sat_pts[(i + 1) % 6]
                draw.line([p1, p2], fill=theme.accent, width=2)

            # Icon circle
            ic_r = 16
            draw.ellipse([sx - ic_r, sy - sat_r + 20 - ic_r,
                          sx + ic_r, sy - sat_r + 20 + ic_r],
                         fill=theme.accent)
            _draw_text_center(draw, sx, sy - sat_r + 20, spoke.get("icon_char", "?"),
                              font_spoke_i, (255, 255, 255))

            # Title
            _draw_text_center(draw, sx, sy - sat_r + 48,
                              spoke.get("title", ""), font_spoke_t, theme.text)

            # Stats
            stats = spoke.get("stats", [])
            for sti, stat in enumerate(stats[:2]):
                stat_y = sy - sat_r + 72 + sti * 26
                _draw_text_center(draw, sx, stat_y, stat.get("label", ""),
                                  font_stat_l, theme.subtext)
                _draw_text_center(draw, sx, stat_y + 13, stat.get("value", ""),
                                  font_stat_v, theme.text)

        return img
    except Exception:
        img = Image.new("RGB", (800, 600), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "hub_spoke render error", fill=(100, 100, 100))
        return img


def _render_magazine(data: dict, theme: _Theme, rng: Any) -> Image.Image:
    try:
        W, H = 1200, 680
        img = Image.new("RGB", (W, H), theme.bg)
        draw = ImageDraw.Draw(img)

        font_title  = _load_font(16, bold=True)
        font_hero   = _load_font(72, bold=True)
        font_label  = _load_font(16, bold=True)
        font_body   = _load_font(12)
        font_callout = _load_font(11)

        # Title strip
        draw.rectangle([0, 0, W, 52], fill=theme.primary)
        draw.text((18, 12), data["title"], font=font_title, fill=(255, 255, 255))

        # Vertical accent bar
        draw.rectangle([0, 52, 6, H], fill=theme.accent)

        # Left column
        lx = 18
        ly = 64

        # Hero number
        hero_str = data.get("hero_number", "")
        hw, hh = _text_size(draw, hero_str, font_hero)
        draw.text((lx, ly + 16), hero_str, font=font_hero, fill=theme.primary)

        # Hero label
        label_str = data.get("hero_label", "")
        draw.text((lx, ly + hh + 24), label_str, font=font_label, fill=theme.accent)

        # Horizontal rule
        rule_y = ly + hh + 52
        draw.line([(lx, rule_y), (lx + 600, rule_y)], fill=theme.border, width=2)

        # Context text
        context_str = data.get("hero_context", "")
        context_lines = _wrap_text(draw, context_str, font_body, 600)
        cy2 = rule_y + 14
        for line in context_lines:
            draw.text((lx, cy2), line, font=font_body, fill=theme.text)
            cy2 += 18

        # Right column
        rx = 660
        ry = 64

        # Lollipop chart
        stats = data.get("stats", [])
        labels = [s["label"] for s in stats]
        values = [s["value"] for s in stats]
        max_vals = [s["max"] for s in stats]
        suffixes = [s.get("suffix", "") for s in stats]
        mv = max(max_vals) if max_vals else 1

        lollipop_labels = labels
        lollipop_values = values
        # Normalize values relative to their own max for visual, but show raw
        lollipop_img = _lollipop_chart(
            lollipop_labels, lollipop_values,
            520, 300, theme, max_val=mv,
        )
        img.paste(lollipop_img, (rx, ry))
        # Add suffix labels
        font_val = _load_font(9)
        for si, (val, suf) in enumerate(zip(values, suffixes)):
            if suf:
                sv = f"{val}{suf}"
                _, label_h = _text_size(draw, sv, font_val)
                # These are drawn over the pasted chart; position roughly
                # (We rely on lollipop_chart's own value label instead)

        # Callout box
        callout_str = data.get("callout", "")
        callout_y = ry + 310
        callout_lines = _wrap_text(draw, callout_str, font_callout, 490)
        callout_h = len(callout_lines) * 18 + 20
        _rounded_rect(draw, rx, callout_y, rx + 510, callout_y + callout_h,
                      6, fill=theme.surface, outline=theme.border, outline_w=1)
        # Left accent border on callout
        draw.rectangle([rx, callout_y, rx + 3, callout_y + callout_h], fill=theme.accent)
        for li, line in enumerate(callout_lines):
            draw.text((rx + 14, callout_y + 10 + li * 18), line,
                      font=font_callout, fill=theme.text)

        return img
    except Exception:
        img = Image.new("RGB", (800, 600), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "magazine render error", fill=(100, 100, 100))
        return img


def _render_pictograph(data: dict, theme: _Theme, rng: Any) -> Image.Image:
    try:
        W, H = 900, 660
        img = Image.new("RGB", (W, H), theme.bg)
        draw = ImageDraw.Draw(img)

        font_title  = _load_font(18, bold=True)
        font_sub    = _load_font(11)
        font_label  = _load_font(11)
        font_pct    = _load_font(12, bold=True)
        font_legend = _load_font(10)

        # Header gradient
        _gradient_rect(img, 0, 0, W, 64, theme.primary, _lighten(theme.primary, 0.2))
        draw.text((20, 10), data["title"], font=font_title, fill=(255, 255, 255))
        draw.text((20, 38), data.get("subtitle", ""), font=font_sub,
                  fill=_lighten(theme.primary, 0.7))

        rows = data.get("rows", [])
        n_rows = len(rows)
        if n_rows == 0:
            return img

        icon_area_h = 460
        row_h = icon_area_h // n_rows
        row_start_y = 80

        for ri, row in enumerate(rows):
            ry = row_start_y + ri * row_h
            row_mid_y = ry + row_h // 2

            # Row label
            draw.text((10, row_mid_y - 6), row["label"], font=font_label, fill=theme.text)

            # Icons: 50 total, each = 2%
            n_icons = 50
            val_pct = float(row.get("value", 0))
            filled = max(0, min(n_icons, round(val_pct / 2)))
            icon_color = row.get("icon_color", theme.accent)

            icon_r = 7
            icon_gap = 4
            icon_x0 = 200
            icons_per_row = 25
            for ii in range(n_icons):
                col = ii % icons_per_row
                row_i = ii // icons_per_row
                ix = icon_x0 + col * (icon_r * 2 + icon_gap)
                iy = ry + 4 + row_i * (icon_r * 2 + icon_gap)
                if ii < filled:
                    draw.ellipse([ix, iy, ix + icon_r * 2, iy + icon_r * 2],
                                 fill=icon_color)
                else:
                    draw.ellipse([ix, iy, ix + icon_r * 2, iy + icon_r * 2],
                                 fill=theme.bg, outline=theme.border)

            # Percentage label at right
            pct_str = f"{val_pct:.0f}%"
            pw, ph = _text_size(draw, pct_str, font_pct)
            draw.text((W - pw - 10, row_mid_y - ph // 2),
                      pct_str, font=font_pct, fill=icon_color)

            # Divider between rows
            if ri < n_rows - 1:
                div_y = ry + row_h - 1
                draw.line([(0, div_y), (W, div_y)], fill=theme.border, width=1)

        # Legend row
        legend_y = 548
        unit_label = data.get("unit_label", "")
        uw, _ = _text_size(draw, unit_label, font_legend)
        draw.text(((W - uw) // 2, legend_y), unit_label, font=font_legend, fill=theme.subtext)

        # Summary gauge
        summary = data.get("summary", {})
        gauge_img = _gauge_semicircle(
            float(summary.get("value", 0)),
            float(summary.get("min", 0)),
            float(summary.get("max", 100)),
            300, 90, theme,
            label=summary.get("label", ""),
        )
        img.paste(gauge_img, ((W - 300) // 2, 575))

        return img
    except Exception:
        img = Image.new("RGB", (800, 600), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "pictograph render error", fill=(100, 100, 100))
        return img


# ---------------------------------------------------------------------------
# Floor plan helpers (kept from original implementation)
# ---------------------------------------------------------------------------

def _get_fp_font(size: int) -> Any:
    candidates = [
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


def _wrap_label(text: str, max_chars: int) -> list:
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [text]


def _draw_dimension(
    draw: Any, x0: int, y0: int, x1: int, y1: int,
    label: str, above: bool, font: Any, vertical: bool = False,
) -> None:
    TICK = 6
    color = (90, 90, 90)
    if not vertical:
        yl = y0 - 22 if above else y0 + 22
        draw.line([(x0, yl), (x1, yl)], fill=color, width=1)
        draw.line([(x0, yl - TICK), (x0, yl + TICK)], fill=color, width=1)
        draw.line([(x1, yl - TICK), (x1, yl + TICK)], fill=color, width=1)
        try:
            draw.text(((x0 + x1) // 2, yl - 8 if above else yl + 8),
                      label, fill=color, font=font, anchor="mm")
        except TypeError:
            w, h = _text_size(draw, label, font)
            draw.text(((x0 + x1) // 2 - w // 2, yl - 8 if above else yl + 8),
                      label, fill=color, font=font)
    else:
        xl = x1 + 22
        draw.line([(xl, y0), (xl, y1)], fill=color, width=1)
        draw.line([(xl - TICK, y0), (xl + TICK, y0)], fill=color, width=1)
        draw.line([(xl - TICK, y1), (xl + TICK, y1)], fill=color, width=1)
        try:
            draw.text((xl + 12, (y0 + y1) // 2), label, fill=color, font=font, anchor="lm")
        except TypeError:
            w, h = _text_size(draw, label, font)
            draw.text((xl + 12, (y0 + y1) // 2 - h // 2), label, fill=color, font=font)


def _draw_north_arrow(draw: Any, cx: int, cy: int, font: Any) -> None:
    r = 18
    draw.polygon([(cx, cy - r), (cx - 7, cy + r // 2), (cx + 7, cy + r // 2)],
                 fill=(60, 60, 60))
    _draw_text_center(draw, cx, cy + r + 10, "N", font, (60, 60, 60))


def _render_floor_plan(spec: dict, rng: Any) -> bytes:
    try:
        CANVAS_W, CANVAS_H = 1280, 820
        MARGIN_L, MARGIN_T = 60, 70
        MARGIN_R, MARGIN_B = 40, 120

        draw_w = CANVAS_W - MARGIN_L - MARGIN_R
        draw_h = CANVAS_H - MARGIN_T - MARGIN_B

        grid_w = spec["grid_w"]
        grid_h = spec["grid_h"]
        cell_px = min(draw_w // grid_w, draw_h // grid_h)

        actual_w = cell_px * grid_w
        actual_h = cell_px * grid_h

        img = Image.new("RGB", (CANVAS_W, CANVAS_H), (240, 240, 238))
        draw = ImageDraw.Draw(img)

        ox, oy = MARGIN_L, MARGIN_T

        # Background grid
        for gx in range(grid_w + 1):
            x = ox + gx * cell_px
            draw.line([(x, oy), (x, oy + actual_h)], fill=(210, 210, 210), width=1)
        for gy in range(grid_h + 1):
            y = oy + gy * cell_px
            draw.line([(ox, y), (ox + actual_w, y)], fill=(210, 210, 210), width=1)

        font_small = _get_fp_font(12)
        font_label = _get_fp_font(13)
        font_area  = _get_fp_font(10)
        WALL = 3

        for room in spec["rooms"]:
            name, rx, ry, rw, rh, rtype = room
            px0 = ox + rx * cell_px + WALL
            py0 = oy + ry * cell_px + WALL
            px1 = ox + (rx + rw) * cell_px - WALL
            py1 = oy + (ry + rh) * cell_px - WALL

            fill_color = _ROOM_COLORS.get(rtype, (235, 235, 235))
            draw.rectangle([px0, py0, px1, py1], fill=fill_color, outline=None)
            draw.rectangle([px0 - WALL, py0 - WALL, px1 + WALL, py1 + WALL],
                           outline=(60, 60, 60), width=WALL)

            cx = (px0 + px1) // 2
            cy = (py0 + py1) // 2
            room_area_m2 = rw * rh
            max_chars = max(6, (px1 - px0) // 9)
            wrapped = _wrap_label(name, max_chars)
            line_h = 15
            total_text_h = len(wrapped) * line_h + 14
            y_start = cy - total_text_h // 2
            for li, line in enumerate(wrapped):
                try:
                    draw.text((cx, y_start + li * line_h), line,
                              fill=(30, 30, 30), font=font_label, anchor="mm")
                except TypeError:
                    tw, th = _text_size(draw, line, font_label)
                    draw.text((cx - tw // 2, y_start + li * line_h - th // 2), line,
                              fill=(30, 30, 30), font=font_label)
            area_label = f"{room_area_m2 * 10} m2"
            try:
                draw.text((cx, y_start + len(wrapped) * line_h + 2),
                          area_label, fill=(90, 90, 90), font=font_area, anchor="mm")
            except TypeError:
                aw, ah = _text_size(draw, area_label, font_area)
                draw.text((cx - aw // 2, y_start + len(wrapped) * line_h + 2 - ah // 2),
                          area_label, fill=(90, 90, 90), font=font_area)

        # Dimension lines
        _draw_dimension(draw, ox, oy, ox + actual_w, oy,
                        f"{grid_w * 5} m", above=True, font=font_small)
        _draw_dimension(draw, ox + actual_w, oy, ox + actual_w, oy + actual_h,
                        f"{grid_h * 5} m", above=False, font=font_small, vertical=True)

        # Column / row grid references
        for gx in range(grid_w):
            lx = ox + gx * cell_px + cell_px // 2
            label = chr(ord("A") + gx)
            try:
                draw.text((lx, oy - 22), label, fill=(80, 80, 80),
                          font=font_small, anchor="mm")
            except TypeError:
                lw, lh = _text_size(draw, label, font_small)
                draw.text((lx - lw // 2, oy - 22 - lh // 2), label,
                          fill=(80, 80, 80), font=font_small)
        for gy in range(grid_h):
            ly = oy + gy * cell_px + cell_px // 2
            num_str = str(gy + 1)
            try:
                draw.text((ox - 22, ly), num_str, fill=(80, 80, 80),
                          font=font_small, anchor="mm")
            except TypeError:
                nw, nh = _text_size(draw, num_str, font_small)
                draw.text((ox - 22 - nw // 2, ly - nh // 2), num_str,
                          fill=(80, 80, 80), font=font_small)

        # Title block
        title_y = oy + actual_h + 14
        draw.rectangle([ox, title_y, ox + actual_w, CANVAS_H - 6],
                       fill=(255, 255, 255), outline=(80, 80, 80), width=1)
        font_title = _get_fp_font(14)
        font_sub   = _get_fp_font(11)
        draw.text((ox + 12, title_y + 14), spec["title"], fill=(20, 20, 20), font=font_title)
        draw.text((ox + 12, title_y + 35),
                  f"Scale: {spec['scale']}   |   {spec['description']}",
                  fill=(80, 80, 80), font=font_sub)

        # North arrow
        _draw_north_arrow(draw, CANVAS_W - 55, MARGIN_T + 30, font=font_small)

        # Legend
        used_types = {r[5] for r in spec["rooms"]}
        legend_x = ox + actual_w + 8
        legend_y = oy
        lf = _get_fp_font(10)
        draw.text((legend_x, legend_y - 2), "Legend", fill=(30, 30, 30),
                  font=_get_fp_font(11))
        for li, rtype in enumerate(sorted(used_types)):
            lry = legend_y + 14 + li * 16
            clr = _ROOM_COLORS.get(rtype, (200, 200, 200))
            draw.rectangle([legend_x, lry, legend_x + 12, lry + 12],
                           fill=clr, outline=(80, 80, 80), width=1)
            draw.text((legend_x + 16, lry + 1),
                      rtype.replace("_", " ").title(),
                      fill=(50, 50, 50), font=lf)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()
    except Exception:
        img = Image.new("RGB", (1280, 820), (240, 240, 238))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "floor_plan render error", fill=(100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()


# ===========================================================================
# SECTION 7: LLM helpers for data-driven infographics
# ===========================================================================

def _gen_element_qa(
    client: Any,
    topic: Any,
    element_type: str,
    ground_truth: dict,
    style_examples: list[str],
    count: int = 3,
) -> list[dict]:
    """Call ContentGenerator.generate_element_qa() if client is available; else return [].

    The style_examples are used as few-shot phrasing inspiration -- the LLM generates
    varied question formulations rather than copying the templates verbatim.
    Falls back gracefully so callers can use ``or fallback_list``.
    """
    if not (client and topic):
        return []
    try:
        from synthpdf.llm.content_generator import ContentGenerator
        return ContentGenerator(client, topic).generate_element_qa(
            element_type, ground_truth, style_examples, count
        )
    except Exception:
        return []


def _llm_generate_stat_dashboard(client: Any, topic: Any) -> "dict | None":
    """Ask the LLM to generate a fresh stat-dashboard data spec for *topic*."""
    import json as _json
    _DEFAULT_COLORS = [
        (33, 150, 243), (100, 181, 246), (187, 222, 251), (144, 202, 249),
        (77, 182, 172), (178, 223, 219),
    ]
    prompt = (
        f"Generate a realistic business performance dashboard dataset for the topic: {topic.description}.\n"
        "Return ONLY valid JSON — no extra text:\n"
        "{\n"
        '  "title": "short dashboard title (8 words max, e.g. Q2 2025 Sales Performance)",\n'
        '  "subtitle": "subtitle describing the metrics (e.g. Revenue & Pipeline Metrics)",\n'
        '  "period_badge": "short period label (e.g. Q2 2025 or Week 22)",\n'
        '  "kpis": [\n'
        '    {"label": "metric name", "value": "formatted value", "delta": "+/-X%", "trend": "up or down", "sparkline": [n1, n2, n3, n4, n5]},\n'
        '    {"label": "...", "value": "...", "delta": "...", "trend": "...", "sparkline": [...]},\n'
        '    {"label": "...", "value": "...", "delta": "...", "trend": "...", "sparkline": [...]},\n'
        '    {"label": "...", "value": "...", "delta": "...", "trend": "...", "sparkline": [...]}\n'
        '  ],\n'
        '  "chart": {\n'
        '    "type": "bar",\n'
        '    "title": "chart title (e.g. Revenue by Region)",\n'
        '    "labels": ["cat1", "cat2", "cat3", "cat4"],\n'
        '    "series": [\n'
        '      {"name": "Current Period", "values": [n, n, n, n]},\n'
        '      {"name": "Prior Period",   "values": [n, n, n, n]}\n'
        '    ]\n'
        '  },\n'
        '  "donut": {\n'
        '    "title": "mix chart title (e.g. Revenue Mix)",\n'
        '    "segments": [\n'
        '      {"label": "segment1", "value": 55},\n'
        '      {"label": "segment2", "value": 30},\n'
        '      {"label": "segment3", "value": 15}\n'
        '    ]\n'
        '  },\n'
        '  "callouts": ["highlight 1", "highlight 2", "highlight 3"]\n'
        "}"
    )
    try:
        raw = client.complete(prompt, json_mode=True, tier="medium", _label="llm_dashboard")
        data = _json.loads(raw)
        required = ("title", "subtitle", "kpis", "chart", "donut", "callouts")
        if not all(k in data for k in required):
            return None
        if not isinstance(data["kpis"], list) or len(data["kpis"]) < 2:
            return None
        # Inject color tuples into donut segments (JSON has no tuple type)
        for j, seg in enumerate(data.get("donut", {}).get("segments", [])):
            if "color" not in seg:
                seg["color"] = _DEFAULT_COLORS[j % len(_DEFAULT_COLORS)]
            elif isinstance(seg["color"], list):
                seg["color"] = tuple(seg["color"])
        return data
    except Exception:
        return None


def _llm_generate_floor_plan(client: Any, topic: Any) -> "dict | None":
    """Ask the LLM to generate a floor-plan spec relevant to *topic*."""
    import json as _json
    prompt = (
        f"Generate a realistic floor plan for a building or space related to: {topic.description}.\n"
        "Use a grid coordinate system where x/y are grid cell positions and w/h are cell spans.\n"
        "Grid must be 10-14 cells wide and 6-9 cells tall. All rooms must fit within the grid.\n"
        "Return ONLY valid JSON — no extra text:\n"
        "{\n"
        '  "title": "building and floor name (e.g. Nexus HQ — Floor 2)",\n'
        '  "scale": "scale string (e.g. 1:100)",\n'
        '  "grid_w": 12,\n'
        '  "grid_h": 8,\n'
        '  "rooms": [\n'
        '    {"name": "Reception",       "x": 0, "y": 0, "w": 3, "h": 2, "type": "reception"},\n'
        '    {"name": "Meeting Room A",  "x": 0, "y": 2, "w": 2, "h": 2, "type": "meeting"},\n'
        '    {"name": "Open Plan Office","x": 3, "y": 0, "w": 5, "h": 4, "type": "open_office"}\n'
        "  ]\n"
        "}\n"
        "Valid room types: reception, meeting, open_office, server, storage, bathroom, kitchen, lounge, lab, corridor"
    )
    try:
        raw = client.complete(prompt, json_mode=True, tier="medium", _label="llm_floor_plan")
        data = _json.loads(raw)
        required = ("title", "scale", "grid_w", "grid_h", "rooms")
        if not all(k in data for k in required):
            return None
        rooms = data.get("rooms", [])
        if not isinstance(rooms, list) or len(rooms) < 3:
            return None
        # Normalize rooms: LLM returns dicts; renderer expects tuples (name, x, y, w, h, type)
        data["rooms"] = [
            (r["name"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]),
             r.get("type", "general"))
            for r in rooms
            if all(k in r for k in ("name", "x", "y", "w", "h"))
        ]
        if len(data["rooms"]) < 3:
            return None
        return data
    except Exception:
        return None


# ===========================================================================
# SECTION 8: Public Variant Generators (static + LLM-aware)
# ===========================================================================

def generate_stat_dashboard_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Any = None,
) -> list:
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"dash_{i}")
        data = _STAT_DASHBOARD_DATA[int(child_rng.integers(0, len(_STAT_DASHBOARD_DATA)))]
        theme_key = _THEME_KEYS[int(child_rng.integers(0, len(_THEME_KEYS)))]
        theme = _THEMES[theme_key]
        img = _render_stat_dashboard(data, theme, child_rng)
        vid = f"dash_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img.save(str(img_path), "PNG")
        caption = f"Dashboard: {data['title']}"
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "stat_dashboard", "title": data["title"],
                      "theme": theme_key, "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "stat_dashboard",
                "title": data["title"],
                "subtitle": data.get("subtitle", ""),
                "period_badge": data.get("period_badge", ""),
                "theme": theme_key,
                "kpis": data.get("kpis", []),
                "chart": data.get("chart", {}),
                "donut": data.get("donut", {}),
                "callouts": data.get("callouts", []),
                "caption": caption,
                "qa_candidates": [
                    {
                        "question": f"In the '{data['title']}' dashboard, what value is shown for the {data['kpis'][0]['label']} KPI?",
                        "answer": str(data["kpis"][0]["value"]),
                        "assertions": [{"text": f"The response must state {data['kpis'][0]['value']}."}],
                    },
                    {
                        "question": f"What is the subtitle of the '{data['title']}' dashboard?",
                        "answer": data.get("subtitle", ""),
                        "assertions": [{"text": f"The response must state '{data.get('subtitle', '')}'."}],
                    },
                    {
                        "question": f"What period badge is displayed on the '{data['title']}' dashboard?",
                        "answer": data.get("period_badge", ""),
                        "assertions": [{"text": f"The response must state '{data.get('period_badge', '')}'."}],
                    },
                ],
            },
        ))
    return variants


def generate_llm_stat_dashboard_variants(
    client: Any,
    topic: Any,
    rng: "np.random.Generator",
    count: int,
    assets_dir: Any = None,
) -> list:
    """Generate stat dashboard variants using LLM-generated data.

    Each dashboard is freshly generated by the LLM for *topic*, so every
    variant has unique KPIs, chart labels, and callouts.  Falls back to the
    static corpus on LLM failure.  Full data (kpis, chart, donut, callouts)
    is stored in ``ground_truth`` and therefore flows into the manifest.
    """
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"llm_dash_{i}")
        data = _llm_generate_stat_dashboard(client, topic) if (client and topic) else None
        if data is None:
            data = _STAT_DASHBOARD_DATA[int(child_rng.integers(0, len(_STAT_DASHBOARD_DATA)))]
            # Ensure donut segments have color tuples (static corpus already has them)
        else:
            _DEFAULT_COLORS = [
                (33, 150, 243), (100, 181, 246), (187, 222, 251),
                (144, 202, 249), (77, 182, 172), (178, 223, 219),
            ]
            for j, seg in enumerate(data.get("donut", {}).get("segments", [])):
                if "color" not in seg:
                    seg["color"] = _DEFAULT_COLORS[j % len(_DEFAULT_COLORS)]

        theme_key = _THEME_KEYS[int(child_rng.integers(0, len(_THEME_KEYS)))]
        theme = _THEMES[theme_key]
        img = _render_stat_dashboard(data, theme, child_rng)
        vid = f"dash_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img.save(str(img_path), "PNG")
        caption = f"Dashboard: {data['title']}"
        title = data["title"]
        subtitle = data.get("subtitle", "")
        kpi0 = data["kpis"][0]
        period_badge = data.get("period_badge", "")
        gt_data = {
            "title": title,
            "subtitle": subtitle,
            "period_badge": period_badge,
            "kpis": [{"label": k.get("label", ""), "value": str(k.get("value", ""))} for k in data.get("kpis", [])],
            "callouts": data.get("callouts", []),
        }
        _fallback_qa = [
            {
                "question": f"In the '{title}' dashboard, what value is shown for the {kpi0['label']} KPI?",
                "answer": str(kpi0["value"]),
                "assertions": [{"text": f"The response must state {kpi0['value']}."}],
            },
            {
                "question": f"What is the subtitle of the '{title}' dashboard?",
                "answer": subtitle,
                "assertions": [{"text": f"The response must state '{subtitle}'."}],
            },
            {
                "question": f"What period badge is displayed on the '{title}' dashboard?",
                "answer": period_badge,
                "assertions": [{"text": f"The response must state '{period_badge}'."}],
            },
        ]
        qa = _gen_element_qa(client, topic, "KPI dashboard", gt_data,
                             [q["question"] for q in _fallback_qa]) or _fallback_qa
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "stat_dashboard", "title": title,
                      "theme": theme_key, "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "stat_dashboard",
                "title": title,
                "subtitle": subtitle,
                "period_badge": period_badge,
                "theme": theme_key,
                "kpis": data.get("kpis", []),
                "chart": data.get("chart", {}),
                "donut": {
                    "title": data.get("donut", {}).get("title", ""),
                    "segments": [
                        {"label": s.get("label", ""), "value": s.get("value", 0)}
                        for s in data.get("donut", {}).get("segments", [])
                    ],
                },
                "callouts": data.get("callouts", []),
                "caption": caption,
                "qa_candidates": qa,
            },
        ))
    return variants


def generate_timeline_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Any = None,
) -> list:
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"tl_{i}")
        data = _TIMELINE_DATA[int(child_rng.integers(0, len(_TIMELINE_DATA)))]
        theme_key = _THEME_KEYS[int(child_rng.integers(0, len(_THEME_KEYS)))]
        theme = _THEMES[theme_key]
        img = _render_timeline(data, theme, child_rng)
        vid = f"tl_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img.save(str(img_path), "PNG")
        caption = f"Timeline: {data['title']}"
        milestones = data.get("milestones", [])
        current_ms = next(
            (m["title"] for m in milestones if m.get("status") == "current"),
            milestones[-1]["title"] if milestones else "",
        )
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "timeline", "title": data["title"],
                      "theme": theme_key, "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "timeline",
                "title": data["title"],
                "theme": theme_key,
                "qa_candidates": [
                    {
                        "question": f"How many milestones are shown on the '{data['title']}' timeline?",
                        "answer": str(len(milestones)),
                        "assertions": [{"text": f"The response must state {len(milestones)}."}],
                    },
                    {
                        "question": f"What is the current milestone on the '{data['title']}' timeline?",
                        "answer": current_ms,
                        "assertions": [{"text": f"The response must state '{current_ms}'."}],
                    },
                ],
            },
        ))
    return variants


def generate_comparison_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Any = None,
) -> list:
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"cmp_{i}")
        data = _COMPARISON_DATA[int(child_rng.integers(0, len(_COMPARISON_DATA)))]
        theme_key = _THEME_KEYS[int(child_rng.integers(0, len(_THEME_KEYS)))]
        theme = _THEMES[theme_key]
        img = _render_comparison(data, theme, child_rng)
        vid = f"cmp_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img.save(str(img_path), "PNG")
        caption = f"Comparison: {data['title']}"
        options = data.get("options", [])
        rec_name = next(
            (o["name"] for o in options if o.get("recommended")),
            options[0]["name"] if options else "",
        )
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "comparison", "title": data["title"],
                      "theme": theme_key, "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "comparison",
                "title": data["title"],
                "theme": theme_key,
                "qa_candidates": [
                    {
                        "question": f"In the '{data['title']}' comparison diagram, which option is marked as recommended?",
                        "answer": rec_name,
                        "assertions": [{"text": f"The response must state '{rec_name}'."}],
                    },
                    {
                        "question": f"How many options are compared in the '{data['title']}' diagram?",
                        "answer": str(len(options)),
                        "assertions": [{"text": f"The response must state {len(options)}."}],
                    },
                ],
            },
        ))
    return variants


def generate_process_step_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Any = None,
) -> list:
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"proc_{i}")
        data = _PROCESS_STEPS_DATA[int(child_rng.integers(0, len(_PROCESS_STEPS_DATA)))]
        theme_key = _THEME_KEYS[int(child_rng.integers(0, len(_THEME_KEYS)))]
        theme = _THEMES[theme_key]
        img = _render_process_steps(data, theme, child_rng)
        vid = f"proc_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img.save(str(img_path), "PNG")
        caption = f"Process: {data['title']}"
        steps = data.get("steps", [])
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "process_steps", "title": data["title"],
                      "theme": theme_key, "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "process_steps",
                "title": data["title"],
                "theme": theme_key,
                "qa_candidates": [
                    {
                        "question": f"How many steps are shown in the '{data['title']}' process diagram?",
                        "answer": str(len(steps)),
                        "assertions": [{"text": f"The response must state {len(steps)}."}],
                    },
                    {
                        "question": f"In the '{data['title']}' process diagram, what is the title of the first step?",
                        "answer": steps[0]["title"] if steps else "",
                        "assertions": [{"text": f"The response must state '{steps[0]['title'] if steps else ''}'."}],
                    },
                ],
            },
        ))
    return variants


def generate_hub_spoke_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Any = None,
) -> list:
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"hub_{i}")
        data = _HUB_SPOKE_DATA[int(child_rng.integers(0, len(_HUB_SPOKE_DATA)))]
        theme_key = _THEME_KEYS[int(child_rng.integers(0, len(_THEME_KEYS)))]
        theme = _THEMES[theme_key]
        img = _render_hub_spoke(data, theme, child_rng)
        vid = f"hub_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img.save(str(img_path), "PNG")
        caption = f"Hub-Spoke: {data['title']}"
        spokes = data.get("spokes", [])
        center_title = data.get("center", {}).get("title", "").replace("\n", " ")
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "hub_spoke", "title": data["title"],
                      "theme": theme_key, "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "hub_spoke",
                "title": data["title"],
                "theme": theme_key,
                "qa_candidates": [
                    {
                        "question": f"In the '{data['title']}' hub-spoke diagram, what is the central hub topic?",
                        "answer": center_title,
                        "assertions": [{"text": f"The response must state '{center_title}'."}],
                    },
                    {
                        "question": f"In the '{data['title']}' hub-spoke diagram, how many spokes (branches) are shown?",
                        "answer": str(len(spokes)),
                        "assertions": [{"text": f"The response must state {len(spokes)}."}],
                    },
                ],
            },
        ))
    return variants


def generate_magazine_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Any = None,
) -> list:
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"mag_{i}")
        data = _MAGAZINE_DATA[int(child_rng.integers(0, len(_MAGAZINE_DATA)))]
        theme_key = _THEME_KEYS[int(child_rng.integers(0, len(_THEME_KEYS)))]
        theme = _THEMES[theme_key]
        img = _render_magazine(data, theme, child_rng)
        vid = f"mag_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img.save(str(img_path), "PNG")
        caption = f"Magazine: {data['title']}"
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "magazine", "title": data["title"],
                      "theme": theme_key, "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "magazine",
                "title": data["title"],
                "theme": theme_key,
                "qa_candidates": [
                    {
                        "question": f"In the '{data['title']}' magazine layout, what large headline number is displayed?",
                        "answer": str(data["hero_number"]),
                        "assertions": [{"text": f"The response must state '{data['hero_number']}'."}],
                    },
                    {
                        "question": f"In the '{data['title']}' magazine layout, what does the headline number represent?",
                        "answer": str(data["hero_label"]),
                        "assertions": [{"text": f"The response must state '{data['hero_label']}'."}],
                    },
                ],
            },
        ))
    return variants


def generate_pictograph_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Any = None,
) -> list:
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"pict_{i}")
        data = _PICTOGRAPH_DATA[int(child_rng.integers(0, len(_PICTOGRAPH_DATA)))]
        theme_key = _THEME_KEYS[int(child_rng.integers(0, len(_THEME_KEYS)))]
        theme = _THEMES[theme_key]
        img = _render_pictograph(data, theme, child_rng)
        vid = f"pict_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img.save(str(img_path), "PNG")
        caption = f"Pictograph: {data['title']}"
        rows = data.get("rows", [])
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "pictograph", "title": data["title"],
                      "theme": theme_key, "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "pictograph",
                "title": data["title"],
                "theme": theme_key,
                "qa_candidates": [
                    {
                        "question": f"In the '{data['title']}' pictograph, what does each icon represent?",
                        "answer": data.get("unit_label", ""),
                        "assertions": [{"text": f"The response must state '{data.get('unit_label', '')}'."}],
                    },
                    {
                        "question": f"How many data rows are shown in the '{data['title']}' pictograph?",
                        "answer": str(len(rows)),
                        "assertions": [{"text": f"The response must state {len(rows)}."}],
                    },
                ],
            },
        ))
    return variants


def generate_floor_plan_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Any = None,
) -> list:
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"fp_{i}")
        spec = _FLOOR_PLANS[int(child_rng.integers(0, len(_FLOOR_PLANS)))]
        png = _render_floor_plan(spec, child_rng)
        vid = f"fp_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img_path.write_bytes(png)
        caption = f"Floor plan: {spec['title']}"
        room_names = [r[0] for r in spec["rooms"]]
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "floor_plan", "title": spec["title"],
                      "scale": spec["scale"],
                      "room_count": len(spec["rooms"]),
                      "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "floor_plan",
                "title": spec["title"],
                "scale": spec["scale"],
                "rooms": room_names,
                "room_count": len(spec["rooms"]),
                "caption": caption,
                "qa_candidates": [
                    {
                        "question": f"How many rooms are labeled on the '{spec['title']}' floor plan?",
                        "answer": str(len(spec["rooms"])),
                        "assertions": [{"text": f"The response must state {len(spec['rooms'])}."}],
                    },
                    {
                        "question": f"What scale is indicated on the '{spec['title']}' floor plan?",
                        "answer": spec["scale"],
                        "assertions": [{"text": f"The response must state '{spec['scale']}'."}],
                    },
                    {
                        "question": f"What is the name of the first room shown on the '{spec['title']}' floor plan?",
                        "answer": room_names[0] if room_names else "",
                        "assertions": [{"text": f"The response must state '{room_names[0] if room_names else ''}'."}],
                    },
                ],
            },
        ))
    return variants


def generate_llm_floor_plan_variants(
    client: Any,
    topic: Any,
    rng: "np.random.Generator",
    count: int,
    assets_dir: Any = None,
) -> list:
    """Generate floor plan variants using LLM-generated room layouts.

    The LLM produces a building title, scale, and room list relevant to
    *topic*.  Falls back to the static ``_FLOOR_PLANS`` corpus on failure.
    Room names and count are stored in ``ground_truth`` for the manifest.
    """
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"llm_fp_{i}")
        spec = _llm_generate_floor_plan(client, topic) if (client and topic) else None
        if spec is None:
            spec = _FLOOR_PLANS[int(child_rng.integers(0, len(_FLOOR_PLANS)))]

        png = _render_floor_plan(spec, child_rng)
        vid = f"fp_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img_path.write_bytes(png)
        caption = f"Floor plan: {spec['title']}"
        room_names = [r[0] for r in spec["rooms"]]
        fp_title = spec["title"]
        fp_scale = spec["scale"]
        gt_data = {
            "title": fp_title,
            "scale": fp_scale,
            "room_count": len(spec["rooms"]),
            "rooms": room_names,
            "first_room": room_names[0] if room_names else "",
        }
        _fallback_qa = [
            {
                "question": f"How many rooms are labeled on the '{fp_title}' floor plan?",
                "answer": str(len(spec["rooms"])),
                "assertions": [{"text": f"The response must state {len(spec['rooms'])}."}],
            },
            {
                "question": f"What scale is indicated on the '{fp_title}' floor plan?",
                "answer": fp_scale,
                "assertions": [{"text": f"The response must state '{fp_scale}'."}],
            },
            {
                "question": f"What is the name of the first room shown on the '{fp_title}' floor plan?",
                "answer": room_names[0] if room_names else "",
                "assertions": [{"text": f"The response must state '{room_names[0] if room_names else ''}'."}],
            },
        ]
        qa = _gen_element_qa(client, topic, "floor plan", gt_data,
                             [q["question"] for q in _fallback_qa]) or _fallback_qa
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "floor_plan", "title": fp_title,
                      "scale": fp_scale,
                      "room_count": len(spec["rooms"]),
                      "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "floor_plan",
                "title": fp_title,
                "scale": fp_scale,
                "rooms": room_names,
                "room_count": len(spec["rooms"]),
                "caption": caption,
                "qa_candidates": qa,
            },
        ))
    return variants


def _llm_generate_timeline(client: Any, topic: Any) -> "dict | None":
    """Ask the LLM to generate a timeline milestone spec for *topic*."""
    import json as _json
    prompt = (
        f"Generate a realistic project or process timeline for the topic: {topic.description}.\n"
        "Return ONLY valid JSON — no extra text:\n"
        "{\n"
        '  "title": "timeline title (8 words max)",\n'
        '  "subtitle": "brief subtitle describing the timeline",\n'
        '  "milestones": [\n'
        '    {"date": "Q1 2024", "title": "Milestone Name", "desc": "Short description", "status": "done"},\n'
        '    {"date": "Q2 2024", "title": "Milestone Name", "desc": "Short description", "status": "done"},\n'
        '    {"date": "Q3 2024", "title": "Milestone Name", "desc": "Short description", "status": "current"},\n'
        '    {"date": "Q4 2024", "title": "Milestone Name", "desc": "Short description", "status": "future"},\n'
        '    {"date": "Q1 2025", "title": "Milestone Name", "desc": "Short description", "status": "future"}\n'
        '  ],\n'
        '  "kpi_callouts": [\n'
        '    {"label": "metric name", "value": "formatted value"},\n'
        '    {"label": "metric name", "value": "formatted value"},\n'
        '    {"label": "metric name", "value": "formatted value"}\n'
        '  ]\n'
        "}\n"
        'Valid status values: "done", "current", "future". Exactly one milestone must have "current".'
    )
    try:
        raw = client.complete(prompt, json_mode=True, tier="medium", _label="llm_timeline")
        data = _json.loads(raw)
        required = ("title", "milestones")
        if not all(k in data for k in required):
            return None
        milestones = data.get("milestones", [])
        if not isinstance(milestones, list) or len(milestones) < 3:
            return None
        # Ensure at least one current milestone
        if not any(m.get("status") == "current" for m in milestones):
            milestones[len(milestones) // 2]["status"] = "current"
        return data
    except Exception:
        return None


def _llm_generate_comparison(client: Any, topic: Any) -> "dict | None":
    """Ask the LLM to generate a feature comparison spec for *topic*."""
    import json as _json
    prompt = (
        f"Generate a realistic feature/pricing comparison diagram for the topic: {topic.description}.\n"
        "Compare exactly 3 options (e.g. product tiers, vendors, approaches).\n"
        "Return ONLY valid JSON — no extra text:\n"
        "{\n"
        '  "title": "comparison title (6 words max)",\n'
        '  "subtitle": "brief subtitle",\n'
        '  "options": [\n'
        '    {"name": "Option A", "subtitle": "price or tier label", "recommended": false},\n'
        '    {"name": "Option B", "subtitle": "price or tier label", "recommended": true},\n'
        '    {"name": "Option C", "subtitle": "price or tier label", "recommended": false}\n'
        '  ],\n'
        '  "features": [\n'
        '    {"name": "Feature 1", "values": ["val_A", "val_B", "val_C"]},\n'
        '    {"name": "Feature 2", "values": [true, true, false]},\n'
        '    {"name": "Feature 3", "values": ["basic", "advanced", "enterprise"]}\n'
        '  ],\n'
        '  "scores": [6, 9, 7]\n'
        "}\n"
        "Provide 5-8 features. Exactly one option must have recommended=true."
    )
    try:
        raw = client.complete(prompt, json_mode=True, tier="medium", _label="llm_comparison")
        data = _json.loads(raw)
        required = ("title", "options", "features")
        if not all(k in data for k in required):
            return None
        options = data.get("options", [])
        if not isinstance(options, list) or len(options) < 2:
            return None
        # Ensure at least one recommended option
        if not any(o.get("recommended") for o in options):
            options[1]["recommended"] = True
        if "scores" not in data:
            data["scores"] = [5] * len(options)
        return data
    except Exception:
        return None


def _llm_generate_process_steps(client: Any, topic: Any) -> "dict | None":
    """Ask the LLM to generate a process/workflow step spec for *topic*."""
    import json as _json
    prompt = (
        f"Generate a realistic step-by-step process or workflow for the topic: {topic.description}.\n"
        "Use 4-6 numbered steps.\n"
        "Return ONLY valid JSON — no extra text:\n"
        "{\n"
        '  "title": "process title (6 words max)",\n'
        '  "subtitle": "brief subtitle describing the workflow",\n'
        '  "steps": [\n'
        '    {"number": "01", "title": "Step Name", "desc": "What happens in this step", "duration": "1 day", "icon_char": "S"},\n'
        '    {"number": "02", "title": "Step Name", "desc": "What happens in this step", "duration": "2 days", "icon_char": "R"},\n'
        '    {"number": "03", "title": "Step Name", "desc": "What happens in this step", "duration": "1 wk",  "icon_char": "A"}\n'
        '  ],\n'
        '  "progress": 0.4,\n'
        '  "progress_label": "Steps 1-2 complete (40%)"\n'
        "}\n"
        "icon_char must be a single uppercase letter. progress is a float 0.0-1.0."
    )
    try:
        raw = client.complete(prompt, json_mode=True, tier="medium", _label="llm_process_steps")
        data = _json.loads(raw)
        required = ("title", "steps")
        if not all(k in data for k in required):
            return None
        steps = data.get("steps", [])
        if not isinstance(steps, list) or len(steps) < 2:
            return None
        # Ensure icon_char is a single character
        for step in steps:
            ic = str(step.get("icon_char", "X"))
            step["icon_char"] = ic[0].upper() if ic else "X"
        if "progress" not in data:
            data["progress"] = 0.5
        if "progress_label" not in data:
            data["progress_label"] = f"Step 1 complete"
        return data
    except Exception:
        return None


def _llm_generate_hub_spoke(client: Any, topic: Any) -> "dict | None":
    """Ask the LLM to generate a hub-and-spoke diagram spec for *topic*."""
    import json as _json
    prompt = (
        f"Generate a hub-and-spoke diagram for the topic: {topic.description}.\n"
        "The hub is the central theme; spokes are 4-6 key sub-topics with 2 metrics each.\n"
        "Return ONLY valid JSON — no extra text:\n"
        "{\n"
        '  "title": "diagram title (6 words max)",\n'
        '  "subtitle": "brief subtitle",\n'
        '  "center": {"title": "Hub\\nTheme", "value": "key metric value", "label": "metric label"},\n'
        '  "spokes": [\n'
        '    {"title": "Spoke 1", "icon_char": "S", "stats": [{"label": "Metric", "value": "val"}, {"label": "Metric", "value": "val"}]},\n'
        '    {"title": "Spoke 2", "icon_char": "R", "stats": [{"label": "Metric", "value": "val"}, {"label": "Metric", "value": "val"}]},\n'
        '    {"title": "Spoke 3", "icon_char": "A", "stats": [{"label": "Metric", "value": "val"}, {"label": "Metric", "value": "val"}]},\n'
        '    {"title": "Spoke 4", "icon_char": "T", "stats": [{"label": "Metric", "value": "val"}, {"label": "Metric", "value": "val"}]}\n'
        '  ]\n'
        "}\n"
        "icon_char must be a single uppercase letter. center.title can use \\n to split across 2 lines."
    )
    try:
        raw = client.complete(prompt, json_mode=True, tier="medium", _label="llm_hub_spoke")
        data = _json.loads(raw)
        required = ("title", "center", "spokes")
        if not all(k in data for k in required):
            return None
        spokes = data.get("spokes", [])
        if not isinstance(spokes, list) or len(spokes) < 3:
            return None
        # Ensure icon_char is a single character
        for spoke in spokes:
            ic = str(spoke.get("icon_char", "X"))
            spoke["icon_char"] = ic[0].upper() if ic else "X"
        return data
    except Exception:
        return None


def _llm_generate_magazine(client: Any, topic: Any) -> "dict | None":
    """Ask the LLM to generate a magazine-layout hero-number spec for *topic*."""
    import json as _json
    prompt = (
        f"Generate a magazine-style infographic with one large headline number for the topic: {topic.description}.\n"
        "Return ONLY valid JSON — no extra text:\n"
        "{\n"
        '  "title": "infographic title (5 words max)",\n'
        '  "hero_number": "formatted headline value (e.g. 94.7%, $12.4M, 3,280)",\n'
        '  "hero_label": "what the hero number represents (8 words max)",\n'
        '  "hero_context": "2-3 sentence description explaining the headline figure and its significance",\n'
        '  "stats": [\n'
        '    {"label": "metric name", "value": 73.5, "max": 100, "suffix": "%"},\n'
        '    {"label": "metric name", "value": 28,   "max": 50,  "suffix": ""},\n'
        '    {"label": "metric name", "value": 4.2,  "max": 10,  "suffix": "x"},\n'
        '    {"label": "metric name", "value": 91,   "max": 200, "suffix": ""}\n'
        '  ],\n'
        '  "callout": "one sentence management commentary or insight"\n'
        "}\n"
        "stats value and max must be numbers (int or float). Provide exactly 4 stats."
    )
    try:
        raw = client.complete(prompt, json_mode=True, tier="medium", _label="llm_magazine")
        data = _json.loads(raw)
        required = ("title", "hero_number", "hero_label", "stats")
        if not all(k in data for k in required):
            return None
        stats = data.get("stats", [])
        if not isinstance(stats, list) or len(stats) < 2:
            return None
        # Ensure value/max are numbers
        for s in stats:
            s["value"] = float(s.get("value", 0))
            s["max"]   = float(s.get("max", 100))
        if "hero_context" not in data:
            data["hero_context"] = f"Key metric for {topic.description}."
        if "callout" not in data:
            data["callout"] = ""
        return data
    except Exception:
        return None


def _llm_generate_pictograph(client: Any, topic: Any) -> "dict | None":
    """Ask the LLM to generate a pictograph (icon-array) spec for *topic*."""
    import json as _json
    _DEFAULT_ICON_COLORS = [
        [16, 185, 129], [251, 191, 36], [239, 68, 68],
        [33, 150, 243], [168, 85, 247], [251, 146, 60],
    ]
    prompt = (
        f"Generate a pictograph (icon-array chart) for the topic: {topic.description}.\n"
        "Each row represents a category with a percentage value (values must sum to 100).\n"
        "Return ONLY valid JSON — no extra text:\n"
        "{\n"
        '  "title": "pictograph title (6 words max)",\n'
        '  "subtitle": "subtitle with total count and key metric",\n'
        '  "unit_label": "Each icon = X% of total (e.g. Each icon = 5% of respondents)",\n'
        '  "rows": [\n'
        '    {"label": "Category A", "value": 55},\n'
        '    {"label": "Category B", "value": 30},\n'
        '    {"label": "Category C", "value": 15}\n'
        '  ],\n'
        '  "summary": {"label": "summary metric label", "value": 72, "min": 0, "max": 100, "suffix": "%"}\n'
        "}\n"
        "Row values must be integers summing to 100. Provide 3-5 rows. summary value/min/max are numbers."
    )
    try:
        raw = client.complete(prompt, json_mode=True, tier="medium", _label="llm_pictograph")
        data = _json.loads(raw)
        required = ("title", "rows")
        if not all(k in data for k in required):
            return None
        rows = data.get("rows", [])
        if not isinstance(rows, list) or len(rows) < 2:
            return None
        # Normalize: ensure integer values and add icon_color tuples
        for j, row in enumerate(rows):
            row["value"] = int(round(float(row.get("value", 10))))
            color = _DEFAULT_ICON_COLORS[j % len(_DEFAULT_ICON_COLORS)]
            row["icon_color"] = tuple(color)
        # Normalize summary
        if "summary" not in data:
            data["summary"] = {"label": "Total", "value": 100, "min": 0, "max": 100, "suffix": "%"}
        else:
            s = data["summary"]
            s["value"] = float(s.get("value", 0))
            s["min"]   = float(s.get("min", 0))
            s["max"]   = float(s.get("max", 100))
        return data
    except Exception:
        return None


def generate_llm_timeline_variants(
    client: Any,
    topic: Any,
    rng: "np.random.Generator",
    count: int,
    assets_dir: Any = None,
) -> list:
    """Generate timeline variants using LLM-generated milestone data for *topic*.

    Falls back to the static ``_TIMELINE_DATA`` corpus on LLM failure.
    """
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"llm_tl_{i}")
        data = _llm_generate_timeline(client, topic) if (client and topic) else None
        if data is None:
            data = _TIMELINE_DATA[int(child_rng.integers(0, len(_TIMELINE_DATA)))]
        theme_key = _THEME_KEYS[int(child_rng.integers(0, len(_THEME_KEYS)))]
        theme = _THEMES[theme_key]
        img = _render_timeline(data, theme, child_rng)
        vid = f"tl_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img.save(str(img_path), "PNG")
        caption = f"Timeline: {data['title']}"
        milestones = data.get("milestones", [])
        current_ms = next(
            (m["title"] for m in milestones if m.get("status") == "current"),
            milestones[-1]["title"] if milestones else "",
        )
        title = data["title"]
        ms_list = [{"date": m.get("date", ""), "title": m.get("title", ""), "status": m.get("status", "")} for m in milestones]
        gt_data = {
            "title": title,
            "subtitle": data.get("subtitle", ""),
            "milestone_count": len(milestones),
            "current_milestone": current_ms,
            "milestones": ms_list,
        }
        _fallback_qa = [
            {
                "question": f"How many milestones are shown on the '{title}' timeline?",
                "answer": str(len(milestones)),
                "assertions": [{"text": f"The response must state {len(milestones)}."}],
            },
            {
                "question": f"What is the current milestone on the '{title}' timeline?",
                "answer": current_ms,
                "assertions": [{"text": f"The response must state '{current_ms}'."}],
            },
        ]
        qa = _gen_element_qa(client, topic, "project timeline", gt_data,
                             [q["question"] for q in _fallback_qa]) or _fallback_qa
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "timeline", "title": title,
                      "theme": theme_key, "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "timeline",
                "title": title,
                "subtitle": data.get("subtitle", ""),
                "theme": theme_key,
                "milestones": ms_list,
                "caption": caption,
                "qa_candidates": qa,
            },
        ))
    return variants


def generate_llm_comparison_variants(
    client: Any,
    topic: Any,
    rng: "np.random.Generator",
    count: int,
    assets_dir: Any = None,
) -> list:
    """Generate comparison diagram variants using LLM-generated data for *topic*.

    Falls back to the static ``_COMPARISON_DATA`` corpus on LLM failure.
    """
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"llm_cmp_{i}")
        data = _llm_generate_comparison(client, topic) if (client and topic) else None
        if data is None:
            data = _COMPARISON_DATA[int(child_rng.integers(0, len(_COMPARISON_DATA)))]
        theme_key = _THEME_KEYS[int(child_rng.integers(0, len(_THEME_KEYS)))]
        theme = _THEMES[theme_key]
        img = _render_comparison(data, theme, child_rng)
        vid = f"cmp_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img.save(str(img_path), "PNG")
        caption = f"Comparison: {data['title']}"
        options = data.get("options", [])
        rec_name = next(
            (o["name"] for o in options if o.get("recommended")),
            options[0]["name"] if options else "",
        )
        title = data["title"]
        opt_list = [{"name": o["name"], "recommended": o.get("recommended", False)} for o in options]
        gt_data = {
            "title": title,
            "subtitle": data.get("subtitle", ""),
            "option_count": len(options),
            "recommended_option": rec_name,
            "options": opt_list,
        }
        _fallback_qa = [
            {
                "question": f"In the '{title}' comparison diagram, which option is marked as recommended?",
                "answer": rec_name,
                "assertions": [{"text": f"The response must state '{rec_name}'."}],
            },
            {
                "question": f"How many options are compared in the '{title}' diagram?",
                "answer": str(len(options)),
                "assertions": [{"text": f"The response must state {len(options)}."}],
            },
        ]
        qa = _gen_element_qa(client, topic, "feature comparison diagram", gt_data,
                             [q["question"] for q in _fallback_qa]) or _fallback_qa
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "comparison", "title": title,
                      "theme": theme_key, "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "comparison",
                "title": title,
                "subtitle": data.get("subtitle", ""),
                "theme": theme_key,
                "options": opt_list,
                "caption": caption,
                "qa_candidates": qa,
            },
        ))
    return variants


def generate_llm_process_step_variants(
    client: Any,
    topic: Any,
    rng: "np.random.Generator",
    count: int,
    assets_dir: Any = None,
) -> list:
    """Generate process-step diagram variants using LLM-generated data for *topic*.

    Falls back to the static ``_PROCESS_STEPS_DATA`` corpus on LLM failure.
    """
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"llm_proc_{i}")
        data = _llm_generate_process_steps(client, topic) if (client and topic) else None
        if data is None:
            data = _PROCESS_STEPS_DATA[int(child_rng.integers(0, len(_PROCESS_STEPS_DATA)))]
        theme_key = _THEME_KEYS[int(child_rng.integers(0, len(_THEME_KEYS)))]
        theme = _THEMES[theme_key]
        img = _render_process_steps(data, theme, child_rng)
        vid = f"proc_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img.save(str(img_path), "PNG")
        caption = f"Process: {data['title']}"
        steps = data.get("steps", [])
        title = data["title"]
        step_list = [{"number": s.get("number", ""), "title": s.get("title", ""), "duration": s.get("duration", "")} for s in steps]
        first_step_title = steps[0]["title"] if steps else ""
        gt_data = {
            "title": title,
            "subtitle": data.get("subtitle", ""),
            "step_count": len(steps),
            "first_step": first_step_title,
            "steps": step_list,
        }
        _fallback_qa = [
            {
                "question": f"How many steps are shown in the '{title}' process diagram?",
                "answer": str(len(steps)),
                "assertions": [{"text": f"The response must state {len(steps)}."}],
            },
            {
                "question": f"In the '{title}' process diagram, what is the title of the first step?",
                "answer": first_step_title,
                "assertions": [{"text": f"The response must state '{first_step_title}'."}],
            },
        ]
        qa = _gen_element_qa(client, topic, "step-by-step process diagram", gt_data,
                             [q["question"] for q in _fallback_qa]) or _fallback_qa
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "process_steps", "title": title,
                      "theme": theme_key, "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "process_steps",
                "title": title,
                "subtitle": data.get("subtitle", ""),
                "theme": theme_key,
                "steps": step_list,
                "caption": caption,
                "qa_candidates": qa,
            },
        ))
    return variants


def generate_llm_hub_spoke_variants(
    client: Any,
    topic: Any,
    rng: "np.random.Generator",
    count: int,
    assets_dir: Any = None,
) -> list:
    """Generate hub-spoke diagram variants using LLM-generated data for *topic*.

    Falls back to the static ``_HUB_SPOKE_DATA`` corpus on LLM failure.
    """
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"llm_hub_{i}")
        data = _llm_generate_hub_spoke(client, topic) if (client and topic) else None
        if data is None:
            data = _HUB_SPOKE_DATA[int(child_rng.integers(0, len(_HUB_SPOKE_DATA)))]
        theme_key = _THEME_KEYS[int(child_rng.integers(0, len(_THEME_KEYS)))]
        theme = _THEMES[theme_key]
        img = _render_hub_spoke(data, theme, child_rng)
        vid = f"hub_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img.save(str(img_path), "PNG")
        caption = f"Hub-Spoke: {data['title']}"
        spokes = data.get("spokes", [])
        center_title = data.get("center", {}).get("title", "").replace("\n", " ")
        title = data["title"]
        spoke_titles = [s.get("title", "") for s in spokes]
        gt_data = {
            "title": title,
            "subtitle": data.get("subtitle", ""),
            "center_topic": center_title,
            "spoke_count": len(spokes),
            "spokes": spoke_titles,
        }
        _fallback_qa = [
            {
                "question": f"In the '{title}' hub-spoke diagram, what is the central hub topic?",
                "answer": center_title,
                "assertions": [{"text": f"The response must state '{center_title}'."}],
            },
            {
                "question": f"In the '{title}' hub-spoke diagram, how many spokes (branches) are shown?",
                "answer": str(len(spokes)),
                "assertions": [{"text": f"The response must state {len(spokes)}."}],
            },
        ]
        qa = _gen_element_qa(client, topic, "hub-and-spoke diagram", gt_data,
                             [q["question"] for q in _fallback_qa]) or _fallback_qa
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "hub_spoke", "title": title,
                      "theme": theme_key, "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "hub_spoke",
                "title": title,
                "subtitle": data.get("subtitle", ""),
                "theme": theme_key,
                "center": center_title,
                "spokes": spoke_titles,
                "caption": caption,
                "qa_candidates": qa,
            },
        ))
    return variants


def generate_llm_magazine_variants(
    client: Any,
    topic: Any,
    rng: "np.random.Generator",
    count: int,
    assets_dir: Any = None,
) -> list:
    """Generate magazine-layout variants using LLM-generated data for *topic*.

    Falls back to the static ``_MAGAZINE_DATA`` corpus on LLM failure.
    """
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"llm_mag_{i}")
        data = _llm_generate_magazine(client, topic) if (client and topic) else None
        if data is None:
            data = _MAGAZINE_DATA[int(child_rng.integers(0, len(_MAGAZINE_DATA)))]
        theme_key = _THEME_KEYS[int(child_rng.integers(0, len(_THEME_KEYS)))]
        theme = _THEMES[theme_key]
        img = _render_magazine(data, theme, child_rng)
        vid = f"mag_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img.save(str(img_path), "PNG")
        caption = f"Magazine: {data['title']}"
        title = data["title"]
        hero_number = str(data.get("hero_number", ""))
        hero_label  = str(data.get("hero_label", ""))
        gt_data = {
            "title": title,
            "hero_number": hero_number,
            "hero_label": hero_label,
            "callout": str(data.get("callout", "")),
        }
        _fallback_qa = [
            {
                "question": f"In the '{title}' magazine layout, what large headline number is displayed?",
                "answer": hero_number,
                "assertions": [{"text": f"The response must state '{hero_number}'."}],
            },
            {
                "question": f"In the '{title}' magazine layout, what does the headline number represent?",
                "answer": hero_label,
                "assertions": [{"text": f"The response must state '{hero_label}'."}],
            },
        ]
        qa = _gen_element_qa(client, topic, "magazine-style infographic", gt_data,
                             [q["question"] for q in _fallback_qa]) or _fallback_qa
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "magazine", "title": title,
                      "theme": theme_key, "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "magazine",
                "title": title,
                "hero_number": hero_number,
                "hero_label": hero_label,
                "theme": theme_key,
                "caption": caption,
                "qa_candidates": qa,
            },
        ))
    return variants


def generate_llm_pictograph_variants(
    client: Any,
    topic: Any,
    rng: "np.random.Generator",
    count: int,
    assets_dir: Any = None,
) -> list:
    """Generate pictograph (icon-array) variants using LLM-generated data for *topic*.

    Falls back to the static ``_PICTOGRAPH_DATA`` corpus on LLM failure.
    """
    out_dir = (Path(assets_dir) if assets_dir else Path(".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        child_rng = fork_rng(rng, f"llm_pict_{i}")
        data = _llm_generate_pictograph(client, topic) if (client and topic) else None
        if data is None:
            data = _PICTOGRAPH_DATA[int(child_rng.integers(0, len(_PICTOGRAPH_DATA)))]
        theme_key = _THEME_KEYS[int(child_rng.integers(0, len(_THEME_KEYS)))]
        theme = _THEMES[theme_key]
        img = _render_pictograph(data, theme, child_rng)
        vid = f"pict_{i:04d}_{_uid()}"
        img_path = out_dir / f"{vid}.png"
        img.save(str(img_path), "PNG")
        caption = f"Pictograph: {data['title']}"
        rows = data.get("rows", [])
        title = data["title"]
        unit_label = data.get("unit_label", "")
        row_list = [{"label": r.get("label", ""), "value": r.get("value", 0)} for r in rows]
        gt_data = {
            "title": title,
            "unit_label": unit_label,
            "row_count": len(rows),
            "rows": row_list,
        }
        _fallback_qa = [
            {
                "question": f"In the '{title}' pictograph, what does each icon represent?",
                "answer": unit_label,
                "assertions": [{"text": f"The response must state '{unit_label}'."}],
            },
            {
                "question": f"How many data rows are shown in the '{title}' pictograph?",
                "answer": str(len(rows)),
                "assertions": [{"text": f"The response must state {len(rows)}."}],
            },
        ]
        qa = _gen_element_qa(client, topic, "pictograph / icon-array chart", gt_data,
                             [q["question"] for q in _fallback_qa]) or _fallback_qa
        variants.append(Variant(
            id=vid, kind="image",
            metadata={"subtype": "pictograph", "title": title,
                      "theme": theme_key, "caption": caption},
            render_payload={"image_path": str(img_path), "caption": caption},
            ground_truth={
                "template": "pictograph",
                "title": title,
                "unit_label": unit_label,
                "theme": theme_key,
                "rows": row_list,
                "caption": caption,
                "qa_candidates": qa,
            },
        ))
    return variants


def generate_infographic_pool(
    rng: np.random.Generator,
    count: int,
    assets_dir: Any = None,
) -> list:
    generators = [
        ("pool_dash",  generate_stat_dashboard_variants),
        ("pool_tl",    generate_timeline_variants),
        ("pool_cmp",   generate_comparison_variants),
        ("pool_proc",  generate_process_step_variants),
        ("pool_hub",   generate_hub_spoke_variants),
        ("pool_mag",   generate_magazine_variants),
        ("pool_pict",  generate_pictograph_variants),
        ("pool_fp",    generate_floor_plan_variants),
    ]
    n = len(generators)
    base = count // n
    extras = count % n

    all_variants = []
    for gi, (salt, gen_fn) in enumerate(generators):
        sub_count = base + (1 if gi < extras else 0)
        if sub_count == 0:
            continue
        child_rng = fork_rng(rng, salt)
        all_variants.extend(gen_fn(child_rng, sub_count, assets_dir))

    return all_variants


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

def generate_flow_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Any = None,
) -> list:
    """Alias for generate_process_step_variants (process_steps template)."""
    return generate_process_step_variants(rng, count, assets_dir)


def generate_org_chart_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Any = None,
) -> list:
    """Alias for generate_hub_spoke_variants (hub_spoke template)."""
    return generate_hub_spoke_variants(rng, count, assets_dir)


def generate_plotly_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Any = None,
) -> list:
    """Alias for generate_stat_dashboard_variants (stat_dashboard template)."""
    return generate_stat_dashboard_variants(rng, count, assets_dir)


def generate_dashboard_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Any = None,
) -> list:
    """Alias for generate_stat_dashboard_variants (stat_dashboard template)."""
    return generate_stat_dashboard_variants(rng, count, assets_dir)

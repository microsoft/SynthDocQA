# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Shared handwriting renderer, extracted without algorithm changes."""
from __future__ import annotations

import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_DEJAVU_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
_DEJAVU_SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
_DEJAVU_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Handwriting fonts bundled in the fonts/ subdirectory
_HW_FONTS_DIR = Path(__file__).parent.parent / "resources" / "fonts"

def _hw_font(name: str) -> str:
    p = _HW_FONTS_DIR / name
    return str(p) if p.exists() else _DEJAVU_SERIF

# Style → font mapping (real handwriting/script fonts)
_HW_FONT_MAP: dict[str, str] = {
    "normal":   _hw_font("Kalam-Regular.ttf"),      # natural, slightly informal
    "bold":     _hw_font("Kalam-Bold.ttf"),          # bold natural writing
    "neat":     _hw_font("PatrickHand-Regular.ttf"), # clean, legible
    "rushed":   _hw_font("IndieFlower-Regular.ttf"), # loose, casual
    "sans":     _hw_font("Caveat-Regular.ttf"),      # clean modern handwriting
    "cursive":  _hw_font("DancingScript-Regular.ttf"), # flowing script
    "formal":   _hw_font("GrandHotel.ttf"),          # elegant formal
    "gochi":    _hw_font("Gochi-Hand.ttf"),          # relaxed informal
    "shadows":  _hw_font("ShadowsIntoLight.ttf"),    # light, airy
    "caveat_bold": _hw_font("Caveat-Bold.ttf"),      # bold casual
}


def simulate_handwriting(
    text_lines: list[str],
    width: int = 600,
    height: int = 200,
    color: tuple = (0, 0, 100),
    background_color: tuple = (255, 255, 240),
    seed: int | None = None,
    style: str = "normal",  # see _HW_FONT_MAP for valid styles
) -> Image.Image:
    """Create a PIL Image with realistic handwriting using bundled handwriting fonts.

    Renders each line with a real handwriting/script font, then applies
    per-line rotation, baseline drift, ink colour variation and paper grain
    so the result looks like a genuine handwritten note.
    Returns an image sized to fit all content (height may exceed the hint).
    """
    import textwrap
    from PIL import ImageEnhance
    import numpy as np

    if seed is None:
        seed = hash(tuple(text_lines)) % (2 ** 31)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # --- Font selection (real handwriting fonts) ---
    font_path = _HW_FONT_MAP.get(style, _HW_FONT_MAP["normal"])

    # Font size: fit ~28 chars per line; cap at 22px so text stays legible
    size_by_width  = max(14, width  // 28)
    size_by_height = max(14, height // max(len(text_lines) + 2, 3))
    font_size = min(size_by_width, size_by_height, 22)

    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        font = ImageFont.truetype(font_path, font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(_DEJAVU_SERIF, font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
            font_size = 12

    # --- Pre-wrap long lines (avg handwriting char ≈ 0.58 × font_size px wide) ---
    usable_w = width - 30
    chars_per_line = max(15, int(usable_w / (font_size * 0.58)))
    wrapped: list[str] = []
    for line in text_lines:
        if not line:
            wrapped.append("")
        elif len(line) <= chars_per_line:
            wrapped.append(line)
        else:
            wrapped.extend(textwrap.wrap(line, width=chars_per_line) or [line])

    # --- Compute canvas height to fit all wrapped lines ---
    line_h = int(font_size * 1.65)
    actual_height = max(height, len(wrapped) * line_h + 30)

    # --- Paper background with subtle grain ---
    img = Image.new("RGB", (width, actual_height), background_color)
    grain = np_rng.normal(0, 2.5, (actual_height, width, 3)).astype(np.int16)
    arr = np.clip(np.array(img).astype(np.int16) + grain, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # --- Per-style variation parameters (line-level, not char-level) ---
    style_params: dict[str, dict] = {
        "normal":      {"x_jitter": 8,  "y_jitter": 3, "line_rot": 0.8,  "opacity_lo": 210, "opacity_hi": 245, "blur_lo": 0.3, "blur_hi": 0.7},
        "bold":        {"x_jitter": 5,  "y_jitter": 2, "line_rot": 0.5,  "opacity_lo": 225, "opacity_hi": 255, "blur_lo": 0.2, "blur_hi": 0.5},
        "neat":        {"x_jitter": 4,  "y_jitter": 1, "line_rot": 0.4,  "opacity_lo": 220, "opacity_hi": 250, "blur_lo": 0.2, "blur_hi": 0.45},
        "rushed":      {"x_jitter": 14, "y_jitter": 5, "line_rot": 2.5,  "opacity_lo": 180, "opacity_hi": 235, "blur_lo": 0.5, "blur_hi": 1.2},
        "sans":        {"x_jitter": 6,  "y_jitter": 2, "line_rot": 0.6,  "opacity_lo": 215, "opacity_hi": 245, "blur_lo": 0.25,"blur_hi": 0.6},
        "cursive":     {"x_jitter": 6,  "y_jitter": 2, "line_rot": 0.8,  "opacity_lo": 210, "opacity_hi": 245, "blur_lo": 0.3, "blur_hi": 0.6},
        "formal":      {"x_jitter": 4,  "y_jitter": 1, "line_rot": 0.4,  "opacity_lo": 215, "opacity_hi": 248, "blur_lo": 0.2, "blur_hi": 0.45},
        "gochi":       {"x_jitter": 10, "y_jitter": 4, "line_rot": 1.8,  "opacity_lo": 195, "opacity_hi": 240, "blur_lo": 0.4, "blur_hi": 0.9},
        "shadows":     {"x_jitter": 5,  "y_jitter": 2, "line_rot": 0.6,  "opacity_lo": 180, "opacity_hi": 225, "blur_lo": 0.3, "blur_hi": 0.7},
        "caveat_bold": {"x_jitter": 6,  "y_jitter": 2, "line_rot": 0.7,  "opacity_lo": 220, "opacity_hi": 255, "blur_lo": 0.2, "blur_hi": 0.55},
    }
    sp = style_params.get(style, style_params["normal"])

    for line_idx, line in enumerate(wrapped):
        if not line:
            continue

        # Slight ink colour variation per line (mimics pen pressure / angle changes)
        r0, g0, b0 = color
        dr, dg, db = rng.randint(-18, 18), rng.randint(-12, 12), rng.randint(-12, 12)
        line_color = (max(0, min(255, r0+dr)), max(0, min(255, g0+dg)), max(0, min(255, b0+db)))
        opacity = int(rng.uniform(sp["opacity_lo"], sp["opacity_hi"]))

        # Render the whole line onto a transparent tile
        try:
            bbox = font.getbbox(line)
            tw = bbox[2] - bbox[0] + 8
            th = bbox[3] - bbox[1] + 8
        except AttributeError:
            tw = len(line) * font_size
            th = font_size + 8

        tile = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((4, 4), line, font=font, fill=(*line_color, opacity))

        # Rotate the whole line slightly (natural tilt of the hand)
        rot = rng.uniform(-sp["line_rot"], sp["line_rot"])
        tile = tile.rotate(rot, expand=True, resample=Image.BICUBIC)

        # Position on canvas
        x = rng.randint(12, 12 + sp["x_jitter"])
        y = int(line_h * (line_idx + 0.75)) + rng.randint(-sp["y_jitter"], sp["y_jitter"])
        px = max(0, min(x, width - tile.width))
        py = max(0, min(y, actual_height - tile.height))
        img.paste(tile, (px, py), tile)

    # Gentle blur: soften digital sharpness (pen ink bleeds slightly into paper)
    blur_r = rng.uniform(sp["blur_lo"], sp["blur_hi"])
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_r))

    # Slight contrast boost so ink stands out from paper grain
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(1.05, 1.15))

    return img

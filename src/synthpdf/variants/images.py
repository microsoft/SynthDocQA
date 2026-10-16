# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Image variant generator.

Generates synthetic chart images (bar, line, scatter, pie) and
simple diagram images, with optional perturbations (blur, rotation,
watermark, low contrast, occlusion).

Each variant writes a PNG asset to *assets_dir* and returns a Variant
with a Markdown embedding payload and ground truth.

Also provides generate_llm_image_variants() which uses seaborn charts
with LLM-provided titles, labels, and value ranges for topic-specific visuals.
"""
from __future__ import annotations

import io
import threading
import time
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

import click
import numpy as np

# matplotlib/seaborn are not thread-safe; serialize all rendering operations.
_matplotlib_lock = threading.Lock()

from synthpdf.rng import fork_rng
from synthpdf.types import Variant

if TYPE_CHECKING:
    from synthpdf.llm.client import AzureOpenAIClient
    from synthpdf.llm.content_generator import TopicSpec


def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def _bar_chart(rng: np.random.Generator, fig_ax: Any) -> dict[str, Any]:
    import matplotlib.pyplot as plt  # noqa: F401

    fig, ax = fig_ax
    n_bars = int(rng.integers(4, 9))
    labels = [f"Cat{i+1}" for i in range(n_bars)]
    values = list(rng.uniform(10, 100, size=n_bars).astype(float))
    ax.bar(labels, values, color=[f"C{i}" for i in range(n_bars)])
    ax.set_title("Bar Chart")
    ax.set_xlabel("Category")
    ax.set_ylabel("Value")
    return {"chart_type": "bar", "labels": labels, "values": values}


def _line_chart(rng: np.random.Generator, fig_ax: Any) -> dict[str, Any]:
    fig, ax = fig_ax
    n_points = int(rng.integers(10, 30))
    x = list(range(n_points))
    y = list(np.cumsum(rng.normal(0, 1, size=n_points)).astype(float))
    ax.plot(x, y, marker="o", markersize=3)
    ax.set_title("Line Chart")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    return {"chart_type": "line", "x": x, "y": y}


def _scatter_chart(rng: np.random.Generator, fig_ax: Any) -> dict[str, Any]:
    fig, ax = fig_ax
    n_points = int(rng.integers(20, 60))
    x = list(rng.normal(0, 1, size=n_points).astype(float))
    y = list(rng.normal(0, 1, size=n_points).astype(float))
    ax.scatter(x, y, alpha=0.6)
    ax.set_title("Scatter Plot")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return {"chart_type": "scatter", "x": x, "y": y}


def _pie_chart(rng: np.random.Generator, fig_ax: Any) -> dict[str, Any]:
    fig, ax = fig_ax
    n_slices = int(rng.integers(3, 7))
    labels = [f"Slice{i+1}" for i in range(n_slices)]
    sizes = list(rng.uniform(5, 40, size=n_slices).astype(float))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%")
    ax.set_title("Pie Chart")
    return {"chart_type": "pie", "labels": labels, "sizes": sizes}


def _diagram(rng: np.random.Generator, fig_ax: Any) -> dict[str, Any]:
    """A simple box-and-arrow diagram using matplotlib patches."""
    import matplotlib.patches as mpatches

    fig, ax = fig_ax
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Process Diagram")

    n_boxes = int(rng.integers(3, 6))
    xs = list(rng.uniform(0.5, 8.0, size=n_boxes).astype(float))
    xs.sort()
    box_labels = [f"Step {i+1}" for i in range(n_boxes)]
    for i, (bx, label) in enumerate(zip(xs, box_labels)):
        rect = mpatches.FancyBboxPatch(
            (bx, 4.0), 1.2, 0.8,
            boxstyle="round,pad=0.1",
            edgecolor="navy",
            facecolor="lightblue",
        )
        ax.add_patch(rect)
        ax.text(bx + 0.6, 4.4, label, ha="center", va="center", fontsize=7)
        if i < n_boxes - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1], 4.4),
                xytext=(bx + 1.2, 4.4),
                arrowprops=dict(arrowstyle="->", color="black"),
            )
    return {"chart_type": "diagram", "steps": box_labels}


_CHART_GENERATORS = [
    ("bar_chart", _bar_chart),
    ("line_chart", _line_chart),
    ("scatter_chart", _scatter_chart),
    ("pie_chart", _pie_chart),
    ("diagram", _diagram),
]


# ---------------------------------------------------------------------------
# Perturbations (applied to PIL images)
# ---------------------------------------------------------------------------

def _apply_blur(img: Any, rng: np.random.Generator) -> Any:
    from PIL import ImageFilter
    radius = float(rng.uniform(1.0, 3.0))
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _apply_rotation(img: Any, rng: np.random.Generator) -> Any:
    angle = float(rng.uniform(-5.0, 5.0))
    return img.rotate(angle, expand=False, fillcolor=(255, 255, 255))


def _apply_low_contrast(img: Any, rng: np.random.Generator) -> Any:
    from PIL import ImageEnhance
    factor = float(rng.uniform(0.4, 0.7))
    return ImageEnhance.Contrast(img).enhance(factor)


def _apply_watermark(img: Any, rng: np.random.Generator) -> Any:
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    w, h = img.size
    texts = ["DRAFT", "CONFIDENTIAL", "SAMPLE", "REVIEW"]
    text = texts[int(rng.integers(0, len(texts)))]
    draw.text((w // 4, h // 2), text, fill=(200, 200, 200, 128), font=None)
    return img


def _apply_occlusion(img: Any, rng: np.random.Generator) -> Any:
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x0 = int(rng.integers(0, w // 2))
    y0 = int(rng.integers(0, h // 2))
    x1 = x0 + int(rng.integers(20, 80))
    y1 = y0 + int(rng.integers(20, 80))
    draw.rectangle([x0, y0, x1, y1], fill=(128, 128, 128))
    return img


_PERTURBATIONS = {
    "blur": _apply_blur,
    "rotation": _apply_rotation,
    "low_contrast": _apply_low_contrast,
    "watermark": _apply_watermark,
    "occlusion": _apply_occlusion,
}


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

def _static_image_qa(
    caption: str,
    chart_data: "dict[str, Any]",
    chart_name: str,
) -> "list[dict[str, Any]]":
    """Return simple, deterministic QA anchored by figure caption.

    Used by the random/static image path so every image variant carries
    testable QA candidates without requiring an LLM call.  Questions are
    self-sufficient: each embeds the figure caption.
    """
    qa: list[dict] = []
    chart_type = chart_data.get("chart_type", chart_name.replace("_", " "))
    qa.append({
        "question": f"What type of chart or diagram is shown in the figure titled '{caption}'?",
        "answer": chart_type,
        "assertions": [{"text": f"The response must state '{chart_type}'."}],
    })
    labels = chart_data.get("labels") or chart_data.get("categories") or chart_data.get("steps", [])
    if labels:
        qa.append({
            "question": f"In the figure titled '{caption}', what is the first category or label shown?",
            "answer": str(labels[0]),
            "assertions": [{"text": f"The response must state '{labels[0]}'."}],
        })
    vrange = chart_data.get("value_range", [])
    if len(vrange) >= 2:
        qa.append({
            "question": f"In the figure titled '{caption}', what is the approximate maximum value shown?",
            "answer": str(vrange[1]),
            "assertions": [{"text": f"The response must state approximately {vrange[1]}."}],
        })
    return qa


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_image_variants(
    rng: np.random.Generator,
    count: int,
    assets_dir: Path,
) -> list[Variant]:
    """Generate *count* image variants and write PNGs to *assets_dir*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    assets_dir.mkdir(parents=True, exist_ok=True)
    variants: list[Variant] = []
    perturb_keys = list(_PERTURBATIONS.keys())

    for i in range(count):
        child_rng = fork_rng(rng, f"image_{i}")

        chart_name, chart_fn = _CHART_GENERATORS[i % len(_CHART_GENERATORS)]
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            chart_data = chart_fn(child_rng, (fig, ax))
        finally:
            pass

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf).copy()

        # Apply 0-2 perturbations
        n_perturb = int(child_rng.integers(0, 3))
        chosen_perturbs = list(
            child_rng.choice(len(perturb_keys), size=min(n_perturb, len(perturb_keys)), replace=False)
        )
        applied = []
        for p_idx in chosen_perturbs:
            p_name = perturb_keys[p_idx]
            img = _PERTURBATIONS[p_name](img, child_rng)
            applied.append(p_name)

        img_id = f"img_{i:04d}_{_uid()[:8]}"
        img_filename = f"{img_id}.png"
        img_path = assets_dir / img_filename
        img.save(img_path, format="PNG")

        caption = f"Figure {i+1}: {chart_name.replace('_', ' ').title()}"

        variants.append(
            Variant(
                id=img_id,
                kind="image",
                metadata={
                    "chart_type": chart_data.get("chart_type", chart_name),
                    "perturbations": applied,
                    "caption": caption,
                    "filename": img_filename,
                },
                render_payload={
                    # Use an absolute POSIX path so Markdown links never contain
                    # Windows backslashes (which Markdown treats as escape chars).
                    "image_path": img_path.resolve().as_posix(),
                    "caption": caption,
                    "alt_text": caption,
                },
                ground_truth={
                    "chart_type": chart_data.get("chart_type", chart_name),
                    "chart_data": chart_data,
                    "caption_semantics": {
                        "caption": caption,
                        "chart_type": chart_data.get("chart_type", chart_name),
                    },
                    "perturbations": applied,
                    "caption": caption,
                    "qa_candidates": _static_image_qa(caption, chart_data, chart_name),
                },
            )
        )

    return variants


# ---------------------------------------------------------------------------
# Seaborn chart generators (LLM-driven)
# ---------------------------------------------------------------------------

def _seaborn_grouped_bar(
    rng: np.random.Generator,
    spec: dict[str, Any],
    fig: Any,
    ax: Any,
) -> dict[str, Any]:
    import pandas as pd
    import seaborn as sns

    cats = spec.get("categories", [f"Cat{i+1}" for i in range(4)])[:6]
    series = spec.get("series_names", ["Series A", "Series B"])[:3]
    vmin, vmax = spec.get("value_range", [0, 100])

    data = []
    for cat in cats:
        for ser in series:
            val = float(rng.uniform(vmin, vmax))
            data.append({"Category": cat, "Series": ser, "Value": val})

    import pandas as pd
    df = pd.DataFrame(data)
    sns.barplot(data=df, x="Category", y="Value", hue="Series", ax=ax)
    ax.set_title(spec.get("title", "Grouped Bar Chart"))
    ax.set_xlabel(spec.get("x_label", "Category"))
    ax.set_ylabel(spec.get("y_label", "Value"))
    ax.tick_params(axis="x", rotation=30)
    return {"chart_type": "grouped_bar", "categories": cats, "series": series,
            "value_range": [vmin, vmax]}


def _seaborn_violin(
    rng: np.random.Generator,
    spec: dict[str, Any],
    fig: Any,
    ax: Any,
) -> dict[str, Any]:
    import pandas as pd
    import seaborn as sns

    cats = spec.get("categories", [f"Group{i+1}" for i in range(4)])[:5]
    vmin, vmax = spec.get("value_range", [0, 100])

    data = []
    for cat in cats:
        n = int(rng.integers(30, 80))
        vals = rng.uniform(vmin, vmax, size=n).tolist()
        for v in vals:
            data.append({"Group": cat, "Value": float(v)})

    df = pd.DataFrame(data)
    sns.violinplot(data=df, x="Group", y="Value", ax=ax)
    ax.set_title(spec.get("title", "Violin Plot"))
    ax.set_xlabel(spec.get("x_label", "Group"))
    ax.set_ylabel(spec.get("y_label", "Value"))
    return {"chart_type": "violin", "categories": cats, "value_range": [vmin, vmax]}


def _seaborn_heatmap(
    rng: np.random.Generator,
    spec: dict[str, Any],
    fig: Any,
    ax: Any,
) -> dict[str, Any]:
    import seaborn as sns

    cats = spec.get("categories", [f"Item{i+1}" for i in range(5)])[:6]
    series = spec.get("series_names", [f"Metric{i+1}" for i in range(4)])[:5]
    vmin, vmax = spec.get("value_range", [0, 1])

    matrix = rng.uniform(vmin, vmax, size=(len(cats), len(series)))
    sns.heatmap(
        matrix,
        xticklabels=series,
        yticklabels=cats,
        annot=True,
        fmt=".2f",
        ax=ax,
        cmap="YlOrRd",
    )
    ax.set_title(spec.get("title", "Heatmap"))
    return {"chart_type": "heatmap", "categories": cats, "series": series,
            "value_range": [vmin, vmax]}


def _seaborn_stacked_area(
    rng: np.random.Generator,
    spec: dict[str, Any],
    fig: Any,
    ax: Any,
) -> dict[str, Any]:
    import seaborn as sns

    series = spec.get("series_names", ["Series A", "Series B", "Series C"])[:4]
    vmin, vmax = spec.get("value_range", [0, 100])
    n_points = int(rng.integers(10, 20))
    x = list(range(n_points))
    trend = spec.get("trend", "volatile")

    palette = sns.color_palette("deep", len(series))
    ys_cumulative = np.zeros(n_points)
    for i, ser in enumerate(series):
        if trend == "increasing":
            base = np.linspace(vmin, vmax * 0.5, n_points)
        elif trend == "decreasing":
            base = np.linspace(vmax * 0.5, vmin, n_points)
        else:
            base = rng.uniform(vmin * 0.1, vmax * 0.3, size=n_points)
        y = base + rng.uniform(0, (vmax - vmin) * 0.2, size=n_points)
        ax.fill_between(x, ys_cumulative, ys_cumulative + y,
                        alpha=0.7, label=ser, color=palette[i])
        ys_cumulative = ys_cumulative + y

    ax.set_title(spec.get("title", "Stacked Area Chart"))
    ax.set_xlabel(spec.get("x_label", "Period"))
    ax.set_ylabel(spec.get("y_label", "Value"))
    ax.legend(loc="upper left", fontsize=7)
    return {"chart_type": "stacked_area", "series": series, "value_range": [vmin, vmax]}


def _seaborn_bubble_scatter(
    rng: np.random.Generator,
    spec: dict[str, Any],
    fig: Any,
    ax: Any,
) -> dict[str, Any]:
    import pandas as pd
    import seaborn as sns

    series = spec.get("series_names", ["Group A", "Group B"])[:3]
    vmin, vmax = spec.get("value_range", [0, 100])

    data = []
    for ser in series:
        n = int(rng.integers(15, 30))
        for _ in range(n):
            data.append({
                "x": float(rng.uniform(vmin, vmax)),
                "y": float(rng.uniform(vmin, vmax)),
                "size": float(rng.uniform(20, 300)),
                "group": ser,
            })

    df = pd.DataFrame(data)
    sns.scatterplot(data=df, x="x", y="y", size="size", hue="group",
                    sizes=(20, 300), alpha=0.6, ax=ax)
    ax.set_title(spec.get("title", "Bubble Scatter Plot"))
    ax.set_xlabel(spec.get("x_label", "X"))
    ax.set_ylabel(spec.get("y_label", "Y"))
    return {"chart_type": "bubble_scatter", "series": series, "value_range": [vmin, vmax]}


def _seaborn_ridgeline(
    rng: np.random.Generator,
    spec: dict[str, Any],
    fig: Any,
    ax: Any,
) -> dict[str, Any]:
    import seaborn as sns

    cats = spec.get("categories", [f"Group{i+1}" for i in range(4)])[:5]
    vmin, vmax = spec.get("value_range", [0, 100])
    palette = sns.color_palette("coolwarm", len(cats))

    for i, cat in enumerate(cats):
        n = int(rng.integers(40, 100))
        vals = rng.uniform(vmin, vmax, size=n)
        offset = i * (vmax - vmin) * 0.15
        sns.kdeplot(vals, ax=ax, label=cat, color=palette[i],
                    fill=True, alpha=0.4)

    ax.set_title(spec.get("title", "Ridgeline Plot"))
    ax.set_xlabel(spec.get("x_label", "Value"))
    ax.legend(fontsize=7)
    return {"chart_type": "ridgeline", "categories": cats, "value_range": [vmin, vmax]}


_SEABORN_CHART_GENERATORS = [
    ("grouped_bar", _seaborn_grouped_bar),
    ("violin", _seaborn_violin),
    ("heatmap", _seaborn_heatmap),
    ("stacked_area", _seaborn_stacked_area),
    ("bubble_scatter", _seaborn_bubble_scatter),
    ("ridgeline", _seaborn_ridgeline),
]


def generate_llm_image_variants(
    client: "AzureOpenAIClient",
    topic: "TopicSpec",
    rng: np.random.Generator,
    count: int,
    assets_dir: Path,
    max_workers: int = 1,
    chart_briefs: "list[dict] | None" = None,
) -> list[Variant]:
    """Generate *count* seaborn chart variants using LLM-provided specifications.

    LLM calls (chart spec, QA) run concurrently when max_workers > 1.
    Matplotlib rendering is always serialized via _matplotlib_lock (not thread-safe).

    chart_briefs (optional): list of brief dicts from generate_document_brief().
      Each dict may contain 'title', 'chart_type', 'x_label', 'y_label', 'purpose'.
      When provided, the chart type and axis labels are constrained to the brief values,
      ensuring visual coherence with the document's prose sections and tables.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from PIL import Image
    from synthpdf.llm.content_generator import ContentGenerator

    assets_dir.mkdir(parents=True, exist_ok=True)
    generator = ContentGenerator(client, topic)
    perturb_keys = list(_PERTURBATIONS.keys())

    def _generate_one(i: int) -> Variant:
        child_rng = fork_rng(rng, f"llm_image_{i}")

        # Use brief slot if available
        brief_slot: dict | None = chart_briefs[i] if (chart_briefs and i < len(chart_briefs)) else None
        default_chart_name, chart_fn = _SEABORN_CHART_GENERATORS[i % len(_SEABORN_CHART_GENERATORS)]
        if brief_slot and brief_slot.get("chart_type"):
            brief_chart_type = brief_slot["chart_type"].lower().replace(" ", "_")
            # Find matching seaborn generator (fall back to default if no match)
            chart_name = brief_chart_type
            matched = [(n, fn) for n, fn in _SEABORN_CHART_GENERATORS if n == brief_chart_type]
            chart_fn = matched[0][1] if matched else chart_fn
        else:
            chart_name = default_chart_name

        # LLM call — thread-safe
        t0 = time.perf_counter()
        try:
            spec = generator.generate_chart_spec(chart_name, brief_slot=brief_slot)
        except Exception:
            spec = {}
        spec_time = time.perf_counter() - t0

        # Matplotlib rendering — serialized via lock
        with _matplotlib_lock:
            sns.set_theme(style="whitegrid", palette="deep")
            fig, ax = plt.subplots(figsize=(7, 4))
            try:
                chart_data = chart_fn(child_rng, spec, fig, ax)
            except Exception:
                chart_data = {"chart_type": chart_name}
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf).copy()

        # PIL perturbations — thread-safe (each thread has its own img object)
        n_perturb = int(child_rng.integers(0, 3))
        chosen_perturbs = list(
            child_rng.choice(len(perturb_keys), size=min(n_perturb, len(perturb_keys)), replace=False)
        )
        applied = []
        for p_idx in chosen_perturbs:
            p_name = perturb_keys[p_idx]
            img = _PERTURBATIONS[p_name](img, child_rng)
            applied.append(p_name)

        img_id = f"img_{i:04d}_{_uid()[:8]}"
        img_filename = f"{img_id}.png"
        img_path = assets_dir / img_filename
        img.save(img_path, format="PNG")

        chart_type = chart_data.get("chart_type", chart_name)
        title = spec.get("title", chart_name.replace("_", " ").title())
        caption = f"Figure {i+1}: {title}"

        chart_content = (
            f"Chart type: {chart_type}\nTitle: {title}\n"
            f"X-axis: {spec.get('x_label', '')}\nY-axis: {spec.get('y_label', '')}\n"
            f"Categories: {', '.join(str(c) for c in spec.get('categories', []))}\n"
            f"Series: {', '.join(str(s) for s in spec.get('series_names', []))}\n"
            f"Value range: {spec.get('value_range', [])}\nTrend: {spec.get('trend', '')}\n"
            f"Data summary: {chart_data}"
        )

        # QA call — thread-safe
        t0 = time.perf_counter()
        qa_candidates = generator.generate_qa_candidates(chart_content, content_type="image", n=3)
        qa_time = time.perf_counter() - t0

        click.echo(
            f"      image [{i+1}/{count}] {chart_name!r}: "
            f"spec->{spec_time:.0f}s | QA({len(qa_candidates)})->{qa_time:.0f}s"
        )
        try:
            from synthpdf import stats_sink as _ss
            _ss.log_artifact(
                type="image", index=i + 1, total=count,
                subtype=chart_name, caption=caption[:80],
                spec_time_s=round(spec_time, 1),
                qa_count=len(qa_candidates), qa_time_s=round(qa_time, 1),
            )
        except Exception:
            pass
        return Variant(
            id=img_id,
            kind="image",
            metadata={
                "chart_type": chart_type,
                "perturbations": applied,
                "caption": caption,
                "filename": img_filename,
                "topic_id": topic.id,
            },
            render_payload={
                "image_path": img_path.resolve().as_posix(),
                "caption": caption,
                "alt_text": caption,
            },
            ground_truth={
                "chart_type": chart_type,
                "title": title,
                "x_label": spec.get("x_label", ""),
                "y_label": spec.get("y_label", ""),
                "categories": spec.get("categories", []),
                "series_names": spec.get("series_names", []),
                "value_range": spec.get("value_range", []),
                "trend": spec.get("trend", ""),
                "caption": caption,
                "chart_data": chart_data,
                "perturbations": applied,
                "qa_candidates": qa_candidates,
            },
        )

    if count == 0:
        return []
    results: list[Variant | None] = [None] * count
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, count)) as executor:
            future_to_idx = {executor.submit(_generate_one, i): i for i in range(count)}
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()
    else:
        for i in range(count):
            results[i] = _generate_one(i)
    return results  # type: ignore[return-value]

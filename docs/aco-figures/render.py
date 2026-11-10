#!/usr/bin/env python3
"""Render the ten numbered decode and prefill figures from curated chart data."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "figures"
RESEARCH = ROOT / "data"
PROMPTS = (("P1", "Python merge"), ("P2", "mmap vs read"), ("P3", "Bash watcher"))
COLORS = {
    "GPU": "#0072B2", "ACO": "#D55E00", "CPU": "#009E73",
    "CPU + GPU MTP": "#009E73", "Master CPU + MTP": "#CC79A7",
    "MTP on": "#D55E00", "MTP off": "#0072B2",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 14,
    "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "legend.fontsize": 12, "figure.facecolor": "white", "axes.facecolor": "#FCFDFE",
    "axes.edgecolor": "#B8C2CC", "axes.labelcolor": "#273444",
    "xtick.color": "#465363", "ytick.color": "#465363", "grid.color": "#DCE3EA",
    "grid.linewidth": .75, "savefig.facecolor": "white",
})


def load_data():
    with (RESEARCH / "chart-data.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"kind", "figure", "mtp", "series", "prompt", "context_tokens", "context_k", "rate_tok_s", "statistic", "reasoning"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Missing chart fields: {required - set(rows[0] if rows else [])}")
    for row in rows:
        for key in ("context_tokens", "context_k", "rate_tok_s"):
            row[key] = float(row[key])
        if not all(math.isfinite(row[key]) for key in ("context_tokens", "context_k", "rate_tok_s")):
            raise ValueError(f"Nonfinite chart point: {row}")
        if row["kind"] in ("decode", "cpu-placement") and (row["statistic"] != "max" or row["reasoning"] != "off"):
            raise ValueError(f"Unexpected decode selection: {row}")
    return rows


def style(ax, ylabel="Decode speed (tokens/s)"):
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax.grid(True, color="#DCE3EA", alpha=.9, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#B8C2CC")
    ax.tick_params(axis="both", length=0, pad=6)


def xcontext(ax):
    ax.set_xlim(0, 272)
    ax.xaxis.set_major_locator(MultipleLocator(32))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_xlabel("Context (K tokens; 1K = 1,024 tokens)")


def artists(ax, points, order):
    result = []
    for series in order:
        data = sorted((row for row in points if row["series"] == series), key=lambda row: row["context_tokens"])
        if not data:
            continue
        xs = [row["context_k"] for row in data]
        ys = [row["rate_tok_s"] for row in data]
        line, = ax.plot(xs, ys, color=COLORS[series], linewidth=3.5, marker="o", markersize=8.2,
                        markerfacecolor=COLORS[series], markeredgecolor="white", markeredgewidth=1,
                        solid_capstyle="round", solid_joinstyle="round", label=series, zorder=3)
        result.append((line, xs, ys, series))
    return result


def labels(ax, sets, integer=False, fontsize=8.2):
    """Place value labels in clear nearby positions while keeping them in the axes."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer)
    used = []
    base_offsets = ((0, 12), (0, -19), (14, 11), (-14, 11), (14, -19), (-14, -19),
                    (0, 29), (0, -35), (26, 10), (-26, 10), (26, -19), (-26, -19),
                    (0, 46), (0, -52), (40, 0), (-40, 0))
    ordered = [(line, x, y, series, idx) for line, xs, ys, series in sets
               for idx, (x, y) in enumerate(zip(xs, ys))]
    # Place the most crowded high-value series first; all labels still retain their series colors.
    for line, x, y, series, idx in ordered:
        color = to_hex(tuple(channel * .72 for channel in to_rgb(line.get_color())))
        text = f"{int(round(y)):,}" if integer else f"{y:.1f}"
        ann = ax.annotate(text, (x, y), xytext=(0, 12), textcoords="offset points",
                          ha="center", va="bottom", fontsize=fontsize, color=color, zorder=5,
                          bbox={"boxstyle": "round,pad=.12", "facecolor": "white", "edgecolor": "none", "alpha": .94},
                          arrowprops={"arrowstyle": "-", "color": line.get_color(), "linewidth": .7,
                                      "alpha": .8, "shrinkA": 2, "shrinkB": 5})
        chosen = None
        # Rotate the preferred side by series and point to separate labels around nearby lines.
        start = (idx + (0 if series in ("GPU", "CPU", "CPU + GPU MTP", "MTP off") else 1)) % len(base_offsets)
        for step in range(len(base_offsets)):
            candidate = base_offsets[(start + step) % len(base_offsets)]
            ann.set_position(candidate)
            box = ann.get_window_extent(renderer).expanded(1.04, 1.08)
            inside = box.x0 >= bounds.x0 + 1 and box.x1 <= bounds.x1 - 1 and box.y0 >= bounds.y0 + 1 and box.y1 <= bounds.y1 - 1
            if inside and all(not box.overlaps(prev) for prev in used):
                used.append(box)
                chosen = candidate
                ann.arrow_patch.set_visible(math.hypot(*chosen) > 14)
                break
        if chosen is None:
            chosen = base_offsets[start]
            ann.set_position(chosen)
            used.append(ann.get_window_extent(renderer).expanded(1, 1))
            ann.arrow_patch.set_visible(math.hypot(*chosen) > 14)


def legend(fig, order):
    handles = [Line2D([0], [0], color=COLORS[name], lw=3.5, marker="o", markersize=8.2,
                      markerfacecolor=COLORS[name], markeredgecolor="white", label=name) for name in order]
    fig.legend(handles=handles, labels=list(order), loc="upper center", bbox_to_anchor=(.5, .952),
               ncol=len(order), frameon=False, handlelength=2.3, columnspacing=2.4, handletextpad=.6)


def save(fig, slug):
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / f"{slug}.png", dpi=220, bbox_inches="tight")
    svg = ASSETS / f"{slug}.svg"
    fig.savefig(svg, bbox_inches="tight", metadata={"Title": slug.replace("_", " ").title()})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def line_panel(ax, points, order, *, integer=False, title=None):
    sets = artists(ax, points, order)
    style(ax, "Prefill speed (tokens/s)" if integer else "Decode speed (tokens/s)")
    xcontext(ax)
    if not points:
        raise ValueError(f"No points for panel {title}")
    ax.set_ylim(0, max(row["rate_tok_s"] for row in points) * (1.16 if integer else 1.27))
    labels(ax, sets, integer=integer, fontsize=8 if integer else 8.3)


def make_triptych(points, slug, state, order=("GPU", "ACO", "CPU"), *, cpu_placement=False):
    fig, axes = plt.subplots(3, 1, figsize=(18.5, 14), sharex=True)
    heading = "Maximum decode · CPU MTP placement · MTP on · reasoning off" if cpu_placement else f"Maximum decode · MTP {state} · reasoning off"
    fig.suptitle(heading, fontsize=20, fontweight="semibold", color="#1E2B39", y=.99)
    for ax, (prompt, prompt_type) in zip(axes, PROMPTS):
        panel = [row for row in points if row["prompt"] == prompt]
        if not cpu_placement:
            panel = [row for row in panel if row["mtp"] == state]
        line_panel(ax, panel, order, title=prompt)
        ax.set_title(f"{prompt} · {prompt_type}", loc="left", fontweight="semibold", fontsize=14, color="#334155", pad=10)
    legend(fig, order)
    fig.subplots_adjust(left=.105, right=.985, top=.91, bottom=.075, hspace=.34)
    save(fig, slug)


def make_single(points, slug, title, order=("GPU", "ACO", "CPU"), *, integer=False):
    fig, ax = plt.subplots(figsize=(18.5, 9.5))
    fig.suptitle(title, fontsize=20, fontweight="semibold", color="#1E2B39", y=.99)
    line_panel(ax, points, order, integer=integer)
    legend(fig, order)
    fig.subplots_adjust(left=.105, right=.985, top=.87, bottom=.12)
    save(fig, slug)


def make_bar(points, slug):
    labels_text = [row["series"] for row in points]
    vals = [row["rate_tok_s"] for row in points]
    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    fig.suptitle("Maximum decode · MTP on · P1 Python merge · 64K context · reasoning off",
                 fontsize=19, fontweight="semibold", color="#1E2B39", y=.97)
    y = list(range(len(vals)))
    bars = ax.barh(y, vals, color=[COLORS[row["series"]] for row in points], height=.58,
                   edgecolor="white", linewidth=1, zorder=3)
    limit = max(vals) * 1.30
    for bar, value in zip(bars, vals):
        ax.text(value + limit * .018, bar.get_y() + bar.get_height() / 2, f"{value:.1f} tok/s",
                va="center", fontsize=12, color="#23313F", fontweight="semibold")
    ax.set_yticks(y, labels_text, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, limit)
    ax.set_xlabel("Decode speed (tokens/s)")
    ax.xaxis.grid(True, color="#DCE3EA", alpha=.9, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, pad=6)
    fig.subplots_adjust(left=.12, right=.96, top=.83, bottom=.13)
    save(fig, slug)


def make_long_context(decode):
    points = [row for row in decode if row["mtp"] == "on" and row["context_tokens"] == 256000 and row["series"] in ("ACO", "CPU")]
    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    fig.suptitle("Maximum decode · MTP on · 250K filled context · reasoning off",
                 fontsize=20, fontweight="semibold", color="#1E2B39", y=.97)
    prompts = ["P1", "P2", "P3"]
    width = .34
    max_rate = max(row["rate_tok_s"] for row in points)
    for index, series in enumerate(("ACO", "CPU")):
        vals = [next(row["rate_tok_s"] for row in points if row["prompt"] == prompt and row["series"] == series) for prompt in prompts]
        positions = [slot + (index - .5) * width for slot in range(3)]
        bars = ax.bar(positions, vals, width, color=COLORS[series], label=series,
                      edgecolor="white", linewidth=1, zorder=3)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max_rate * .028, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=10, color=to_hex(tuple(c * .72 for c in to_rgb(COLORS[series]))), fontweight="semibold")
    ax.set_ylim(0, max_rate * 1.18)
    ax.set_xticks(range(3), [f"{prompt}\n{title}" for prompt, title in PROMPTS])
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Decode speed (tokens/s)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax.grid(axis="y", color="#DCE3EA", alpha=.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    legend(fig, ("ACO", "CPU"))
    fig.subplots_adjust(left=.105, right=.97, top=.85, bottom=.13)
    save(fig, "05_mtp_long_context")


def make_on_off(decode):
    fig, axes = plt.subplots(3, 1, figsize=(18.5, 14), sharex=True)
    fig.suptitle("Maximum decode · ACO · MTP on versus off · reasoning off",
                 fontsize=20, fontweight="semibold", color="#1E2B39", y=.99)
    for ax, (prompt, prompt_type) in zip(axes, PROMPTS):
        points = [dict(row, series="MTP on" if row["mtp"] == "on" else "MTP off")
                  for row in decode if row["series"] == "ACO" and row["prompt"] == prompt]
        line_panel(ax, points, ("MTP on", "MTP off"))
        ax.set_title(f"{prompt} · {prompt_type}", loc="left", fontweight="semibold", fontsize=14, color="#334155", pad=10)
    legend(fig, ("MTP on", "MTP off"))
    fig.subplots_adjust(left=.105, right=.985, top=.91, bottom=.075, hspace=.34)
    save(fig, "08_aco_mtp_on_off")


def main():
    rows = load_data()
    decode = [row for row in rows if row["kind"] == "decode"]
    placement = [row for row in rows if row["kind"] == "cpu-placement"]
    prefill = [row for row in rows if row["kind"] == "prefill"]

    p1_bar = [next(row for row in decode if row["mtp"] == "on" and row["prompt"] == "P1" and row["series"] == series and row["context_tokens"] == context)
              for series, context in (("GPU", 61440), ("ACO", 62464), ("CPU", 61440))]
    make_bar(p1_bar, "01_mtp_64k_comparison")
    make_triptych(decode, "02_mtp_context_comparison", "on")
    make_single([row for row in decode if row["mtp"] == "on" and row["prompt"] == "P1"],
                "03_mtp_prompt_1", "Maximum decode · MTP on · P1 Python merge · reasoning off")
    make_single([row for row in decode if row["mtp"] == "on" and row["prompt"] == "P3"],
                "04_mtp_prompt_3", "Maximum decode · MTP on · P3 Bash watcher · reasoning off")
    make_long_context(decode)
    make_triptych(placement, "06_cpu_mtp_placement", "on", ("CPU + GPU MTP", "Master CPU + MTP"), cpu_placement=True)
    make_triptych(decode, "07_non_mtp_context_comparison", "off")
    make_on_off(decode)
    make_single([row for row in prefill if row["mtp"] == "on"], "09_prefill_mtp_on_context",
                "Backend prefill · MTP on",
                integer=True)
    make_single([row for row in decode if row["mtp"] == "on" and row["prompt"] == "P2"],
                "10_mtp_prompt_2", "Maximum decode · MTP on · P2 mmap vs read · reasoning off")

    captions = [
        {"file": "01_mtp_64k_comparison", "title": "Maximum decode with MTP on around 64K context", "caption": "Around 64K on P1, ACO reached 79.69 tokens/s, GPU reached 80.70, and CPU reached 17.68."},
        {"file": "02_mtp_context_comparison", "title": "Maximum decode with MTP on across prompt types", "caption": "At 250K filled context, ACO measured 61.13, 48.83, and 55.35 tokens/s on P1, P2, and P3, compared with CPU at 7.21, 5.49, and 6.50."},
        {"file": "03_mtp_prompt_1", "title": "Maximum decode with MTP on for P1 Python merge", "caption": "ACO measured 102.48 tokens/s at 6K context and 61.23 tokens/s near full 256K context on P1."},
        {"file": "04_mtp_prompt_3", "title": "Maximum decode with MTP on for P3 Bash watcher", "caption": "ACO measured 92.93 tokens/s at 6K context and 55.35 tokens/s at 250K filled context on P3."},
        {"file": "05_mtp_long_context", "title": "Maximum decode with MTP on at 250K filled context", "caption": "At 256,000 occupied tokens, ACO measured 61.13, 48.83, and 55.35 tokens/s across P1, P2, and P3."},
        {"file": "06_cpu_mtp_placement", "title": "Maximum decode by CPU MTP placement", "caption": "Around 64K on P1, CPU + GPU MTP measured 17.68 tokens/s and master CPU + MTP measured 15.12, a 16.9% gain."},
        {"file": "07_non_mtp_context_comparison", "title": "Maximum decode with MTP off across prompt types", "caption": "At 250K filled context, ACO measured 37.51, 37.57, and 37.45 tokens/s on P1, P2, and P3."},
        {"file": "08_aco_mtp_on_off", "title": "Maximum ACO decode with MTP on and off", "caption": "At 250K filled context on P1, ACO measured 61.13 tokens/s with MTP on and 37.51 tokens/s with MTP off."},
        {"file": "09_prefill_mtp_on_context", "title": "Backend prefill with MTP on across context lengths", "caption": "With MTP on, ACO ingested fresh repository chunks at 1,402 backend tokens/s on the first 8K append and 614 tokens/s near full 256K context."},
        {"file": "10_mtp_prompt_2", "title": "Maximum decode with MTP on for P2 mmap vs read", "caption": "ACO measured 82.01 tokens/s at 6K context and 48.83 tokens/s at 250K filled context on P2."},
    ]
    (RESEARCH / "chart_captions.json").write_text(json.dumps(captions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Rendered {len(captions)} figure pairs from {len(rows)} curated rows")


if __name__ == "__main__":
    main()

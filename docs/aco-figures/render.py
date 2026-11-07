#!/usr/bin/env python3
"""Render article figures from the three portable, curated CSV exports.

Inputs are read only from ./data beside this script. PNG and SVG files are
written to ./figures. No raw benchmark records or source chart packs are used.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
PROMPTS = (("P1", "Python merge"), ("P2", "mmap vs read"), ("P3", "Bash watcher"))
MTP_STATES = ("on", "off")
COLORS = {
    "GPU": "#0072B2", "ACO": "#D55E00", "CPU": "#009E73",
    "CPU + GPU MTP": "#009E73", "Master CPU + MTP": "#CC79A7",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 14,
    "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "legend.fontsize": 13, "figure.facecolor": "white", "axes.facecolor": "#FCFDFE",
    "axes.edgecolor": "#B8C2CC", "axes.labelcolor": "#273444",
    "xtick.color": "#465363", "ytick.color": "#465363",
    "grid.color": "#DCE3EA", "grid.linewidth": 0.75, "savefig.facecolor": "white",
})


def read_rows(name: str, required: set[str]) -> list[dict[str, str]]:
    path = DATA / name
    if not path.is_file():
        raise FileNotFoundError(f"Expected local input CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def as_float(row: dict[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {name} in exported CSV")
    return value


def read_decode() -> list[dict]:
    fields = {"figure", "mtp", "series", "prompt", "context_tokens", "context_k",
              "rate_tok_s", "statistic", "reasoning", "eligible_reps", "repetitions_tok_s"}
    output = []
    for row in read_rows("decode.csv", fields):
        output.append({**row, "context_tokens": as_float(row, "context_tokens"),
                       "context_k": as_float(row, "context_k"),
                       "rate_tok_s": as_float(row, "rate_tok_s")})
    return output


def read_prefill() -> list[dict]:
    fields = {"figure", "mtp", "series", "context_tokens", "context_k", "rate_tok_s", "net_new_tokens"}
    output = []
    for row in read_rows("prefill.csv", fields):
        output.append({**row, "context_tokens": as_float(row, "context_tokens"),
                       "context_k": as_float(row, "context_k"),
                       "rate_tok_s": as_float(row, "rate_tok_s")})
    return output


def read_cpu_placement() -> list[dict]:
    fields = {"figure", "series", "prompt", "context_tokens", "context_k", "rate_tok_s",
              "statistic", "reasoning", "eligible_reps"}
    output = []
    for row in read_rows("cpu-mtp-placement.csv", fields):
        output.append({**row, "context_tokens": as_float(row, "context_tokens"),
                       "context_k": as_float(row, "context_k"),
                       "rate_tok_s": as_float(row, "rate_tok_s")})
    return output


def style_axis(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax.grid(True, color="#DCE3EA", alpha=0.9, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8C2CC")
    ax.spines["bottom"].set_color("#B8C2CC")
    ax.tick_params(axis="both", length=0, pad=6)


def set_context_axis(ax, xmax: int = 272) -> None:
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(32 if xmax > 160 else 16))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_xlabel("Context (K tokens; 1K = 1,024 tokens)")


def draw_series(ax, points: list[dict], series_order: tuple[str, ...]):
    artists = []
    for series in series_order:
        data = sorted((r for r in points if r["series"] == series), key=lambda r: r["context_tokens"])
        if not data:
            continue
        xs = [r["context_k"] for r in data]
        ys = [r["rate_tok_s"] for r in data]
        line, = ax.plot(xs, ys, color=COLORS[series], linewidth=3.5,
                        marker="o", markersize=8.2, markerfacecolor=COLORS[series],
                        markeredgecolor="white", markeredgewidth=1.0,
                        solid_capstyle="round", solid_joinstyle="round",
                        label=series, zorder=3)
        artists.append((line, xs, ys, series))
    return artists


def place_labels(ax, artists, *, integer: bool = False, fontsize: float = 8.5) -> None:
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    placed = []
    offsets = {
        "GPU": [(0, 11), (0, -17), (12, 11), (-12, 11), (12, -17), (-12, -17), (0, 25), (0, -31), (22, 10), (-22, 10), (22, -17), (-22, -17)],
        "ACO": [(0, -17), (0, 11), (12, -17), (-12, -17), (12, 11), (-12, 11), (0, -31), (0, 25), (22, -17), (-22, -17), (22, 10), (-22, 10)],
        "CPU": [(0, 11), (0, -17), (12, 11), (-12, 11), (12, -17), (-12, -17), (0, 25), (0, -31), (22, 10), (-22, 10), (22, -17), (-22, -17)],
        "CPU + GPU MTP": [(0, 11), (0, -17), (12, 11), (-12, 11), (12, -17), (-12, -17), (0, 25), (0, -31), (22, 10), (-22, 10), (22, -17), (-22, -17)],
        "Master CPU + MTP": [(0, -17), (0, 11), (12, -17), (-12, -17), (12, 11), (-12, 11), (0, -31), (0, 25), (22, -17), (-22, -17), (22, 10), (-22, 10)],
    }
    for line, xs, ys, series in artists:
        label_color = to_hex(tuple(channel * 0.72 for channel in to_rgb(line.get_color())))
        candidates = offsets[series]
        for x, y in zip(xs, ys):
            label = f"{int(round(y)):,}" if integer else f"{y:.1f}"
            text = ax.annotate(label, (x, y), xytext=(0, 11), textcoords="offset points",
                               ha="center", va="bottom", fontsize=fontsize, color=label_color,
                               zorder=5, bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                                              "edgecolor": "none", "alpha": 0.94},
                               arrowprops={"arrowstyle": "-", "color": line.get_color(),
                                           "linewidth": 0.7, "alpha": 0.8,
                                           "shrinkA": 2, "shrinkB": 5})
            chosen = None
            for dx, dy in candidates:
                text.set_position((dx, dy))
                box = text.get_window_extent(renderer).expanded(1.05, 1.08)
                inside = (box.x0 >= axes_box.x0 + 1 and box.x1 <= axes_box.x1 - 1
                          and box.y0 >= axes_box.y0 + 1 and box.y1 <= axes_box.y1 - 1)
                if inside and all(not box.overlaps(prev) for prev in placed):
                    placed.append(box)
                    chosen = (dx, dy)
                    break
            if chosen is None:
                chosen = candidates[0]
                text.set_position(chosen)
                placed.append(text.get_window_extent(renderer).expanded(1.0, 1.0))
            if math.hypot(*chosen) > 14:
                text.arrow_patch.set_visible(True)
            else:
                text.arrow_patch.set_visible(False)


def add_legend(fig, labels: tuple[str, ...]) -> None:
    handles = [Line2D([0], [0], color=COLORS[name], lw=3.5, marker="o", markersize=8.2,
                      markerfacecolor=COLORS[name], markeredgecolor="white",
                      markeredgewidth=1.0, label=name) for name in labels]
    fig.legend(handles=handles, labels=list(labels), loc="upper center", bbox_to_anchor=(0.5, 0.952),
               ncol=len(labels), frameon=False, handlelength=2.3, columnspacing=2.4, handletextpad=0.6)


def save(fig, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    svg = FIGURES / f"{stem}.svg"
    fig.savefig(svg, bbox_inches="tight", metadata={"Title": stem.replace("-", " ").title()})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def make_decode(points: list[dict]) -> None:
    for state in MTP_STATES:
        fig, axes = plt.subplots(3, 1, figsize=(18.5, 14.0), sharex=True)
        fig.suptitle(f"Decode speed · MTP {state} · max · reasoning OFF", fontsize=20,
                     fontweight="semibold", color="#1E2B39", y=0.99)
        for ax, (prompt, title) in zip(axes, PROMPTS):
            panel = [r for r in points if r["mtp"] == state and r["prompt"] == prompt]
            artists = draw_series(ax, panel, ("GPU", "ACO", "CPU"))
            ax.set_title(f"{prompt} · {title}", loc="left", fontweight="semibold", fontsize=14,
                         color="#334155", pad=10)
            style_axis(ax, "Decode speed (tokens/s)")
            set_context_axis(ax)
            ax.set_ylim(0, max(r["rate_tok_s"] for r in panel) * 1.28)
            place_labels(ax, artists, fontsize=9.0)
        add_legend(fig, ("GPU", "ACO", "CPU"))
        fig.subplots_adjust(left=0.105, right=0.985, top=0.91, bottom=0.075, hspace=0.34)
        save(fig, f"decode-mtp-{state}")


def make_prefill(points: list[dict]) -> None:
    for state in MTP_STATES:
        fig, ax = plt.subplots(figsize=(19.5, 9.0))
        fig.suptitle(f"Prefill speed · MTP {state}", fontsize=20, fontweight="semibold",
                     color="#1E2B39", y=0.99)
        selected = [r for r in points if r["mtp"] == state]
        artists = draw_series(ax, selected, ("GPU", "ACO", "CPU"))
        style_axis(ax, "Prefill speed (tokens/s)")
        set_context_axis(ax)
        ax.set_ylim(0, max(r["rate_tok_s"] for r in selected) * 1.16)
        place_labels(ax, artists, integer=True, fontsize=8.0)
        add_legend(fig, ("GPU", "ACO", "CPU"))
        fig.subplots_adjust(left=0.105, right=0.985, top=0.87, bottom=0.12)
        save(fig, f"prefill-mtp-{state}")


def make_cpu_placement(points: list[dict]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(18.5, 14.0), sharex=True)
    fig.suptitle("Decode speed · CPU placement · max · reasoning OFF", fontsize=20,
                 fontweight="semibold", color="#1E2B39", y=0.99)
    order = ("CPU + GPU MTP", "Master CPU + MTP")
    for ax, (prompt, title) in zip(axes, PROMPTS):
        panel = [r for r in points if r["prompt"] == prompt]
        artists = draw_series(ax, panel, order)
        ax.set_title(f"{prompt} · {title}", loc="left", fontweight="semibold", fontsize=14,
                     color="#334155", pad=10)
        style_axis(ax, "Decode speed (tokens/s)")
        set_context_axis(ax)
        ax.set_ylim(0, max(r["rate_tok_s"] for r in panel) * 1.28)
        place_labels(ax, artists, fontsize=9.0)
    add_legend(fig, order)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.91, bottom=0.075, hspace=0.34)
    save(fig, "cpu-mtp-placement")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    decode = read_decode()
    prefill = read_prefill()
    cpu = read_cpu_placement()
    make_decode(decode)
    make_prefill(prefill)
    make_cpu_placement(cpu)
    print(f"Rendered 5 figures to {FIGURES}")


if __name__ == "__main__":
    main()

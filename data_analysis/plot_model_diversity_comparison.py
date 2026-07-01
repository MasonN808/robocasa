#!/usr/bin/env python3
"""Compare Gemini Flash and Pro sampling-method diversity metrics."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_analysis.plotting import (
    METHOD_ORDER,
    PngCanvas,
    TaskMetric,
    axis_ticks,
    blend_color,
    color_method,
    format_value,
    label_method,
    metrics_by_task,
    nice_max,
    quartiles,
    read_metrics,
    stable_jitter,
)


DEFAULT_FLASH_INPUT = Path(
    "data_generation/task_level/data_k3/raw/sampling_method_analysis_qwen3/task_metrics.csv"
)
DEFAULT_PRO_INPUT = Path(
    "data_analysis/raw_gemini_pro_traj/raw/sampling_method_analysis_qwen3/task_metrics.csv"
)
DEFAULT_OUTPUT_DIR = Path("data_analysis/plots/gemini_model_diversity_comparison")
MODEL_FLASH = "Gemini Flash Preview"
MODEL_PRO = "Gemini Pro"
MODEL_COLORS = {
    MODEL_FLASH: "#4c78a8",
    MODEL_PRO: "#e45756",
}


@dataclass(frozen=True)
class ModelMetric:
    model: str
    metric: TaskMetric


@dataclass(frozen=True)
class PairedDelta:
    task: str
    sampling_method: str
    flash_diversity: float
    pro_diversity: float
    delta: float


@dataclass(frozen=True)
class BaselineDelta:
    model: str
    task: str
    sampling_method: str
    delta_from_base: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create PNG comparison plots for Gemini Flash Preview and Gemini Pro "
            "sampling-method diversity. Diversity is 1 - trajectory_cosine_avg."
        )
    )
    parser.add_argument("--flash-input", type=Path, default=DEFAULT_FLASH_INPUT)
    parser.add_argument("--pro-input", type=Path, default=DEFAULT_PRO_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sorted_methods_from_metrics(metrics: list[TaskMetric]) -> list[str]:
    methods = sorted({metric.sampling_method for metric in metrics})
    known = [method for method in METHOD_ORDER if method in methods]
    extra = [method for method in methods if method not in METHOD_ORDER]
    return [*known, *extra]


def model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#777777")


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.mean(values)


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def draw_y_axis(
    canvas: PngCanvas,
    *,
    x0: float,
    y0: float,
    plot_width: float,
    plot_height: float,
    min_y: float,
    max_y: float,
    ylabel: str,
) -> Callable[[float], float]:
    span = max(max_y - min_y, 1e-9)

    def y_scale(value: float) -> float:
        return y0 + plot_height - (value - min_y) / span * plot_height

    canvas.line(x0, y0, x0, y0 + plot_height, stroke="#555")
    canvas.line(x0, y0 + plot_height, x0 + plot_width, y0 + plot_height, stroke="#555")
    for fraction in axis_ticks(1.0):
        tick = min_y + span * fraction
        y = y_scale(tick)
        canvas.line(x0, y, x0 + plot_width, y, stroke="#ddd")
        canvas.text(
            x0 - 8,
            y + 4,
            format_value(tick),
            anchor="end",
            size_class="small",
        )
    canvas.text(
        20,
        y0 + plot_height / 2,
        ylabel,
        anchor="middle",
        size_class="label",
        rotate=-90,
    )
    return y_scale


def grouped_values(
    model_metrics: list[ModelMetric],
) -> dict[tuple[str, str], list[float]]:
    groups: dict[tuple[str, str], list[float]] = {}
    for row in model_metrics:
        key = (row.model, row.metric.sampling_method)
        groups.setdefault(key, []).append(row.metric.diversity)
    return groups


def plot_diversity_by_model_and_method(
    model_metrics: list[ModelMetric],
    output_path: Path,
) -> None:
    methods = sorted_methods_from_metrics([row.metric for row in model_metrics])
    models = [MODEL_FLASH, MODEL_PRO]
    groups = grouped_values(model_metrics)
    values = [value for group in groups.values() for value in group]

    width, height = 980, 540
    left, right, top, bottom = 82, 52, 58, 86
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_y = nice_max(values)
    x_step = plot_width / len(methods)
    model_gap = min(52, x_step * 0.23)

    canvas = PngCanvas(width, height)
    canvas.text(
        left, 30, "Trajectory diversity by model and method", size_class="title"
    )
    y_scale = draw_y_axis(
        canvas,
        x0=left,
        y0=top,
        plot_width=plot_width,
        plot_height=plot_height,
        min_y=0.0,
        max_y=max_y,
        ylabel="1 - avg cosine",
    )

    for method_index, method in enumerate(methods):
        center = left + x_step * (method_index + 0.5)
        for model_index, model in enumerate(models):
            method_values = groups.get((model, method), [])
            if not method_values:
                continue
            x = center + (model_index - 0.5) * model_gap
            min_v, q1, median, q3, max_v = quartiles(method_values)
            box_width = min(32, x_step * 0.16)
            color = model_color(model)
            canvas.line(x, y_scale(min_v), x, y_scale(max_v), stroke=color, width=1.8)
            canvas.rect(
                x - box_width / 2,
                y_scale(q3),
                box_width,
                max(1, y_scale(q1) - y_scale(q3)),
                fill=color,
                stroke=color,
                opacity=0.24,
            )
            canvas.line(
                x - box_width / 2,
                y_scale(median),
                x + box_width / 2,
                y_scale(median),
                stroke=color,
                width=1.8,
            )
            for offset_index, value in enumerate(method_values):
                jitter = stable_jitter(
                    f"{model}-{method}-{offset_index}", box_width * 0.7
                )
                canvas.circle(x + jitter, y_scale(value), 2.3, fill=color, opacity=0.58)

        canvas.text(center, height - 48, label_method(method), anchor="middle")
        canvas.text(
            center,
            height - 30,
            f"n={sum(len(groups.get((model, method), [])) for model in models)}",
            anchor="middle",
            size_class="small",
        )

    legend_x = left + plot_width - 210
    legend_y = top + 10
    for index, model in enumerate(models):
        y = legend_y + index * 22
        canvas.rect(legend_x, y - 10, 12, 12, fill=model_color(model))
        canvas.text(legend_x + 20, y, model, size_class="small")
    canvas.save(output_path)


def compute_paired_deltas(
    flash_metrics: list[TaskMetric],
    pro_metrics: list[TaskMetric],
) -> list[PairedDelta]:
    flash_by_key = {
        (metric.task, metric.sampling_method): metric for metric in flash_metrics
    }
    pro_by_key = {
        (metric.task, metric.sampling_method): metric for metric in pro_metrics
    }
    deltas: list[PairedDelta] = []
    for key in sorted(flash_by_key.keys() & pro_by_key.keys()):
        flash = flash_by_key[key]
        pro = pro_by_key[key]
        deltas.append(
            PairedDelta(
                task=key[0],
                sampling_method=key[1],
                flash_diversity=flash.diversity,
                pro_diversity=pro.diversity,
                delta=pro.diversity - flash.diversity,
            )
        )
    return deltas


def plot_pro_minus_flash_by_method(
    deltas: list[PairedDelta],
    output_path: Path,
) -> None:
    methods = sorted_methods_from_metrics(
        [
            TaskMetric(
                task=row.task,
                sampling_method=row.sampling_method,
                diversity=row.delta,
                outlier_diversity=row.delta,
                conversation_words=0.0,
                conversation_messages=0.0,
                num_trajectories=0,
            )
            for row in deltas
        ]
    )
    values_by_method = {
        method: [row.delta for row in deltas if row.sampling_method == method]
        for method in methods
    }
    values = [
        value
        for values_for_method in values_by_method.values()
        for value in values_for_method
    ]
    max_abs = nice_max(abs(value) for value in values) if values else 1.0

    width, height = 920, 520
    left, right, top, bottom = 82, 46, 58, 86
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_step = plot_width / len(methods)

    canvas = PngCanvas(width, height)
    canvas.text(left, 30, "Gemini Pro minus Flash diversity", size_class="title")
    y_scale = draw_y_axis(
        canvas,
        x0=left,
        y0=top,
        plot_width=plot_width,
        plot_height=plot_height,
        min_y=-max_abs,
        max_y=max_abs,
        ylabel="Pro - Flash diversity",
    )
    canvas.line(left, y_scale(0.0), left + plot_width, y_scale(0.0), stroke="#555")

    for index, method in enumerate(methods):
        method_values = values_by_method[method]
        x = left + x_step * (index + 0.5)
        min_v, q1, median, q3, max_v = quartiles(method_values)
        box_width = min(64, x_step * 0.46)
        color = color_method(method)
        canvas.line(x, y_scale(min_v), x, y_scale(max_v), stroke=color, width=2)
        canvas.rect(
            x - box_width / 2,
            y_scale(q3),
            box_width,
            max(1, y_scale(q1) - y_scale(q3)),
            fill=color,
            stroke=color,
            opacity=0.24,
        )
        canvas.line(
            x - box_width / 2,
            y_scale(median),
            x + box_width / 2,
            y_scale(median),
            stroke=color,
            width=2,
        )
        for row in [item for item in deltas if item.sampling_method == method]:
            jitter = stable_jitter(row.task, box_width * 0.55)
            canvas.circle(x + jitter, y_scale(row.delta), 2.7, fill=color, opacity=0.72)
        canvas.text(x, height - 48, label_method(method), anchor="middle")
        canvas.text(
            x,
            height - 30,
            f"n={len(method_values)}",
            anchor="middle",
            size_class="small",
        )

    canvas.save(output_path)


def compute_baseline_deltas(
    model: str,
    metrics: list[TaskMetric],
) -> list[BaselineDelta]:
    by_task = metrics_by_task(metrics)
    deltas: list[BaselineDelta] = []
    for task, method_metrics in sorted(by_task.items()):
        base = method_metrics.get("base")
        if base is None:
            continue
        for method, metric in sorted(method_metrics.items()):
            if method == "base":
                continue
            deltas.append(
                BaselineDelta(
                    model=model,
                    task=task,
                    sampling_method=method,
                    delta_from_base=metric.diversity - base.diversity,
                )
            )
    return deltas


def plot_delta_from_base_by_model(
    baseline_deltas: list[BaselineDelta],
    output_path: Path,
) -> None:
    methods = [
        method
        for method in METHOD_ORDER
        if method != "base"
        and any(row.sampling_method == method for row in baseline_deltas)
    ]
    methods.extend(
        sorted(
            {
                row.sampling_method
                for row in baseline_deltas
                if row.sampling_method not in METHOD_ORDER
            }
        )
    )
    models = [MODEL_FLASH, MODEL_PRO]
    values_by_key = {
        (model, method): [
            row.delta_from_base
            for row in baseline_deltas
            if row.model == model and row.sampling_method == method
        ]
        for model in models
        for method in methods
    }
    values = [value for group in values_by_key.values() for value in group]
    max_abs = nice_max(abs(value) for value in values) if values else 1.0

    width, height = 940, 520
    left, right, top, bottom = 82, 56, 58, 86
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_step = plot_width / len(methods)
    model_gap = min(56, x_step * 0.3)

    canvas = PngCanvas(width, height)
    canvas.text(left, 30, "Diversity delta from base by model", size_class="title")
    y_scale = draw_y_axis(
        canvas,
        x0=left,
        y0=top,
        plot_width=plot_width,
        plot_height=plot_height,
        min_y=-max_abs,
        max_y=max_abs,
        ylabel="method - base diversity",
    )
    canvas.line(left, y_scale(0.0), left + plot_width, y_scale(0.0), stroke="#555")

    for method_index, method in enumerate(methods):
        center = left + x_step * (method_index + 0.5)
        for model_index, model in enumerate(models):
            values_for_key = values_by_key.get((model, method), [])
            if not values_for_key:
                continue
            x = center + (model_index - 0.5) * model_gap
            min_v, q1, median, q3, max_v = quartiles(values_for_key)
            box_width = min(34, x_step * 0.18)
            color = model_color(model)
            canvas.line(x, y_scale(min_v), x, y_scale(max_v), stroke=color, width=1.8)
            canvas.rect(
                x - box_width / 2,
                y_scale(q3),
                box_width,
                max(1, y_scale(q1) - y_scale(q3)),
                fill=color,
                stroke=color,
                opacity=0.25,
            )
            canvas.line(
                x - box_width / 2,
                y_scale(median),
                x + box_width / 2,
                y_scale(median),
                stroke=color,
                width=1.8,
            )
            for offset_index, value in enumerate(values_for_key):
                jitter = stable_jitter(
                    f"{model}-{method}-{offset_index}", box_width * 0.65
                )
                canvas.circle(x + jitter, y_scale(value), 2.2, fill=color, opacity=0.58)
        canvas.text(center, height - 48, label_method(method), anchor="middle")

    legend_x = left + plot_width - 210
    legend_y = top + 10
    for index, model in enumerate(models):
        y = legend_y + index * 22
        canvas.rect(legend_x, y - 10, 12, 12, fill=model_color(model))
        canvas.text(legend_x + 20, y, model, size_class="small")
    canvas.save(output_path)


def plot_paired_difference_heatmap(
    deltas: list[PairedDelta],
    output_path: Path,
) -> None:
    if not deltas:
        raise ValueError("No paired Pro/Flash deltas available for heatmap")
    methods = [
        method
        for method in METHOD_ORDER
        if any(row.sampling_method == method for row in deltas)
    ]
    tasks = sorted(
        {row.task for row in deltas},
        key=lambda task: max(abs(row.delta) for row in deltas if row.task == task),
        reverse=True,
    )
    delta_by_key = {(row.task, row.sampling_method): row.delta for row in deltas}
    max_abs = max(abs(row.delta) for row in deltas)
    max_abs = max(max_abs, 1e-9)

    row_height = 17
    cell_width = 118
    left = 210
    top = 58
    width = left + cell_width * len(methods) + 58
    height = top + row_height * len(tasks) + 76

    canvas = PngCanvas(width, height)
    canvas.text(left, 30, "Pro minus Flash diversity by task", size_class="title")
    for col, method in enumerate(methods):
        x = left + col * cell_width + cell_width / 2
        canvas.text(
            x, top - 18, label_method(method), anchor="middle", size_class="small"
        )

    for row, task in enumerate(tasks):
        y = top + row * row_height
        canvas.text(left - 8, y + 12, task, anchor="end", size_class="small")
        for col, method in enumerate(methods):
            x = left + col * cell_width
            delta = delta_by_key.get((task, method))
            if delta is None:
                canvas.rect(
                    x,
                    y,
                    cell_width - 2,
                    row_height - 2,
                    fill="#eeeeee",
                    stroke="white",
                )
                continue
            fraction = abs(delta) / max_abs
            fill = blend_color(
                "#f7f7f7", "#2166ac" if delta >= 0 else "#b2182b", fraction
            )
            canvas.rect(x, y, cell_width - 2, row_height - 2, fill=fill, stroke="white")

    legend_x = left
    legend_y = height - 38
    legend_width = 180
    half = legend_width // 2
    for index in range(legend_width):
        if index < half:
            fill = blend_color("#b2182b", "#f7f7f7", index / max(1, half - 1))
        else:
            fill = blend_color("#f7f7f7", "#2166ac", (index - half) / max(1, half - 1))
        canvas.rect(legend_x + index, legend_y, 1, 12, fill=fill)
    canvas.text(legend_x, legend_y + 28, f"-{max_abs:.3f}", size_class="small")
    canvas.text(
        legend_x + half, legend_y + 28, "0", anchor="middle", size_class="small"
    )
    canvas.text(
        legend_x + legend_width,
        legend_y + 28,
        f"+{max_abs:.3f}",
        anchor="end",
        size_class="small",
    )
    canvas.text(
        legend_x + legend_width + 14,
        legend_y + 10,
        "Pro - Flash",
        size_class="small",
    )
    canvas.save(output_path)


def summarize_model_metrics(
    model_metrics: list[ModelMetric],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in [MODEL_FLASH, MODEL_PRO]:
        model_rows = [row.metric for row in model_metrics if row.model == model]
        for method in sorted_methods_from_metrics(model_rows):
            values = [
                metric.diversity
                for metric in model_rows
                if metric.sampling_method == method
            ]
            rows.append(
                {
                    "model": model,
                    "sampling_method": method,
                    "num_task_method_rows": len(values),
                    "mean_diversity": mean_or_none(values),
                    "median_diversity": median_or_none(values),
                    "min_diversity": min(values) if values else None,
                    "max_diversity": max(values) if values else None,
                }
            )
    return rows


def summarize_paired_deltas(deltas: list[PairedDelta]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = [
        method
        for method in METHOD_ORDER
        if any(row.sampling_method == method for row in deltas)
    ]
    methods.extend(
        sorted(
            {
                row.sampling_method
                for row in deltas
                if row.sampling_method not in METHOD_ORDER
            }
        )
    )
    for method in methods:
        values = [row.delta for row in deltas if row.sampling_method == method]
        rows.append(
            {
                "sampling_method": method,
                "num_paired_tasks": len(values),
                "mean_pro_minus_flash_diversity": mean_or_none(values),
                "median_pro_minus_flash_diversity": median_or_none(values),
                "min_pro_minus_flash_diversity": min(values) if values else None,
                "max_pro_minus_flash_diversity": max(values) if values else None,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    flash_metrics: list[TaskMetric],
    pro_metrics: list[TaskMetric],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_metrics = [
        *[ModelMetric(MODEL_FLASH, metric) for metric in flash_metrics],
        *[ModelMetric(MODEL_PRO, metric) for metric in pro_metrics],
    ]
    paired_deltas = compute_paired_deltas(flash_metrics, pro_metrics)
    baseline_deltas = [
        *compute_baseline_deltas(MODEL_FLASH, flash_metrics),
        *compute_baseline_deltas(MODEL_PRO, pro_metrics),
    ]

    outputs = [
        output_dir / "diversity_by_model_and_method.png",
        output_dir / "pro_minus_flash_diversity_by_method.png",
        output_dir / "delta_from_base_by_model.png",
        output_dir / "pro_minus_flash_diversity_heatmap.png",
        output_dir / "model_diversity_summary.csv",
        output_dir / "paired_task_method_deltas.csv",
        output_dir / "paired_delta_summary.csv",
    ]
    plot_diversity_by_model_and_method(model_metrics, outputs[0])
    plot_pro_minus_flash_by_method(paired_deltas, outputs[1])
    plot_delta_from_base_by_model(baseline_deltas, outputs[2])
    plot_paired_difference_heatmap(paired_deltas, outputs[3])
    write_csv(outputs[4], summarize_model_metrics(model_metrics))
    write_csv(
        outputs[5],
        [
            {
                "task": row.task,
                "sampling_method": row.sampling_method,
                "flash_diversity": row.flash_diversity,
                "pro_diversity": row.pro_diversity,
                "pro_minus_flash_diversity": row.delta,
            }
            for row in paired_deltas
        ],
    )
    write_csv(outputs[6], summarize_paired_deltas(paired_deltas))
    return outputs


def main() -> int:
    args = parse_args()
    flash_metrics = read_metrics(args.flash_input)
    pro_metrics = read_metrics(args.pro_input)
    outputs = write_outputs(flash_metrics, pro_metrics, args.output_dir)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plot sampling-method trajectory diversity and tool-call distributions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_INPUT = Path(
    "data_generation/task_level/data_k3/raw/sampling_method_analysis_qwen3/task_metrics.csv"
)
DEFAULT_RAW_INPUT = Path("data_generation/task_level/data_k3/raw/sampling_methods")
DEFAULT_OUTPUT_DIR = Path("data_analysis/plots/sampling_method_diversity")
METHOD_ORDER = ["base", "high_temperature", "random", "verbalized"]
METHOD_LABELS = {
    "base": "base",
    "high_temperature": "high temp",
    "random": "random",
    "verbalized": "verbalized",
}
METHOD_COLORS = {
    "base": "#4c78a8",
    "high_temperature": "#f58518",
    "random": "#54a24b",
    "verbalized": "#b279a2",
}
TOOL_COLORS = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
]


@dataclass(frozen=True)
class TaskMetric:
    task: str
    sampling_method: str
    diversity: float
    outlier_diversity: float
    conversation_words: float
    conversation_messages: float
    num_trajectories: int


@dataclass(frozen=True)
class ToolTrajectory:
    task: str
    sampling_method: str
    trajectory_id: str
    tool_counts: Counter[str]

    @property
    def total_calls(self) -> int:
        return sum(self.tool_counts.values())


class PdfCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = []
        self.rect(0, 0, width, height, fill="white")

    def pdf_y(self, y: float) -> float:
        return self.height - y

    def pdf_rect_y(self, y: float, height: float) -> float:
        return self.height - y - height

    def color_components(
        self, color: str, opacity: float = 1.0
    ) -> tuple[float, float, float]:
        if color == "none":
            return 0.0, 0.0, 0.0
        named_colors = {
            "black": "#000000",
            "white": "#ffffff",
        }
        color = named_colors.get(color, color)
        color = color.lstrip("#")
        if len(color) == 3:
            color = "".join(channel * 2 for channel in color)
        red = int(color[0:2], 16) / 255.0
        green = int(color[2:4], 16) / 255.0
        blue = int(color[4:6], 16) / 255.0
        if opacity < 1.0:
            red = 1.0 - (1.0 - red) * opacity
            green = 1.0 - (1.0 - green) * opacity
            blue = 1.0 - (1.0 - blue) * opacity
        return red, green, blue

    def set_stroke(self, color: str, opacity: float = 1.0) -> str:
        red, green, blue = self.color_components(color, opacity)
        return f"{red:.4f} {green:.4f} {blue:.4f} RG"

    def set_fill(self, color: str, opacity: float = 1.0) -> str:
        red, green, blue = self.color_components(color, opacity)
        return f"{red:.4f} {green:.4f} {blue:.4f} rg"

    def text_size(self, size_class: str) -> int:
        if size_class == "title":
            return 18
        if size_class == "small":
            return 10
        return 12

    def estimate_text_width(self, value: str, size: int) -> float:
        return len(value) * size * 0.54

    def escape_text(self, value: str) -> str:
        return (
            value.encode("latin-1", errors="replace")
            .decode("latin-1")
            .replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )

    def append(self, command: str) -> None:
        self.parts.append(command)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = "#333",
        width: float = 1.0,
        opacity: float = 1.0,
        dash: str | None = None,
    ) -> None:
        dash_command = "[3 3] 0 d" if dash else "[] 0 d"
        self.append(
            "q\n"
            f"{self.set_stroke(stroke, opacity)}\n"
            f"{width:.2f} w\n"
            f"{dash_command}\n"
            f"{x1:.2f} {self.pdf_y(y1):.2f} m "
            f"{x2:.2f} {self.pdf_y(y2):.2f} l S\n"
            "Q"
        )

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str = "none",
        stroke: str = "none",
        opacity: float = 1.0,
    ) -> None:
        if fill == "none" and stroke == "none":
            return
        operator = (
            "B"
            if fill != "none" and stroke != "none"
            else "f"
            if fill != "none"
            else "S"
        )
        commands = ["q"]
        if fill != "none":
            commands.append(self.set_fill(fill, opacity))
        if stroke != "none":
            commands.append(self.set_stroke(stroke, opacity))
        commands.append(
            f"{x:.2f} {self.pdf_rect_y(y, height):.2f} {width:.2f} {height:.2f} re {operator}"
        )
        commands.append("Q")
        self.append("\n".join(commands))

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        *,
        fill: str,
        stroke: str = "white",
        opacity: float = 1.0,
    ) -> None:
        kappa = 0.5522847498
        x = cx
        y = self.pdf_y(cy)
        c = r * kappa
        operator = "B" if stroke != "none" else "f"
        self.append(
            "q\n"
            f"{self.set_fill(fill, opacity)}\n"
            f"{self.set_stroke(stroke, opacity)}\n"
            f"{x + r:.2f} {y:.2f} m\n"
            f"{x + r:.2f} {y + c:.2f} {x + c:.2f} {y + r:.2f} {x:.2f} {y + r:.2f} c\n"
            f"{x - c:.2f} {y + r:.2f} {x - r:.2f} {y + c:.2f} {x - r:.2f} {y:.2f} c\n"
            f"{x - r:.2f} {y - c:.2f} {x - c:.2f} {y - r:.2f} {x:.2f} {y - r:.2f} c\n"
            f"{x + c:.2f} {y - r:.2f} {x + r:.2f} {y - c:.2f} {x + r:.2f} {y:.2f} c\n"
            f"{operator}\n"
            "Q"
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        anchor: str = "start",
        size_class: str = "label",
        rotate: float | None = None,
    ) -> None:
        size = self.text_size(size_class)
        escaped = self.escape_text(value)
        width = self.estimate_text_width(value, size)
        anchor_dx = 0.0
        if anchor == "middle":
            anchor_dx = -width / 2
        elif anchor == "end":
            anchor_dx = -width

        pdf_y = self.pdf_y(y)
        if rotate is None:
            self.append(
                "BT\n"
                f"{self.set_fill('#222222')}\n"
                f"/F1 {size} Tf\n"
                f"{x + anchor_dx:.2f} {pdf_y:.2f} Td\n"
                f"({escaped}) Tj\n"
                "ET"
            )
            return

        radians = math.radians(rotate)
        cos_value = math.cos(radians)
        sin_value = -math.sin(radians)
        self.append(
            "BT\n"
            f"{self.set_fill('#222222')}\n"
            f"/F1 {size} Tf\n"
            f"{cos_value:.5f} {sin_value:.5f} {-sin_value:.5f} {cos_value:.5f} "
            f"{x:.2f} {pdf_y:.2f} Tm\n"
            f"{anchor_dx:.2f} 0 Td\n"
            f"({escaped}) Tj\n"
            "ET"
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(self.parts).encode("latin-1", errors="replace")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
                f"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
            ).encode("ascii"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length "
            + str(len(content)).encode("ascii")
            + b" >>\nstream\n"
            + content
            + b"\nendstream",
        ]
        pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(pdf))
            pdf.extend(f"{index} 0 obj\n".encode("ascii"))
            pdf.extend(obj)
            pdf.extend(b"\nendobj\n")
        xref_offset = len(pdf)
        pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        pdf.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        pdf.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        path.write_bytes(bytes(pdf))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create simple PDF plots for sampling-method trajectory diversity "
            "and tool-call distributions. Diversity is defined as "
            "1 - trajectory_cosine_avg."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_RAW_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-tools", type=int, default=8)
    return parser.parse_args()


def float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def read_metrics(path: Path) -> list[TaskMetric]:
    metrics: list[TaskMetric] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            avg_cosine = float_or_none(row.get("trajectory_cosine_avg"))
            min_cosine = float_or_none(row.get("trajectory_cosine_min"))
            if avg_cosine is None or min_cosine is None:
                continue
            metrics.append(
                TaskMetric(
                    task=row["task"],
                    sampling_method=row["sampling_method"],
                    diversity=1.0 - avg_cosine,
                    outlier_diversity=1.0 - min_cosine,
                    conversation_words=float(row["conversation_length_words_avg"]),
                    conversation_messages=float(row["conversation_messages_avg"]),
                    num_trajectories=int(float(row["num_trajectories"])),
                )
            )
    if not metrics:
        raise ValueError(f"No rows with trajectory cosine metrics found in {path}")
    return metrics


def extract_tool_name(step: dict) -> str:
    tool = step.get("tool")
    if isinstance(tool, str) and tool:
        return tool
    tool_call = step.get("tool_call")
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        if isinstance(name, str) and name:
            return name
        function = tool_call.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
    return "<missing>"


def read_tool_trajectories(raw_input: Path) -> list[ToolTrajectory]:
    trajectories: list[ToolTrajectory] = []
    for path in sorted(raw_input.glob("*/*/trajectories/traj_*.json")):
        sampling_method = path.parents[2].name
        task = path.parents[1].name
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        steps = data.get("steps")
        if not isinstance(steps, list):
            continue
        counts = Counter()
        for step in steps:
            if isinstance(step, dict):
                counts[extract_tool_name(step)] += 1
        if not counts:
            continue
        trajectories.append(
            ToolTrajectory(
                task=task,
                sampling_method=sampling_method,
                trajectory_id=str(data.get("trajectory_id", path.stem)),
                tool_counts=counts,
            )
        )
    if not trajectories:
        raise ValueError(
            f"No raw trajectory JSON files with tool calls found in {raw_input}"
        )
    return trajectories


def sorted_methods(metrics: Iterable[TaskMetric]) -> list[str]:
    methods = sorted({metric.sampling_method for metric in metrics})
    known = [method for method in METHOD_ORDER if method in methods]
    extra = [method for method in methods if method not in METHOD_ORDER]
    return [*known, *extra]


def label_method(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def color_method(method: str) -> str:
    return METHOD_COLORS.get(method, "#777777")


def grouped_by_method(
    metrics: Iterable[TaskMetric],
    value_fn: Callable[[TaskMetric], float],
) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {}
    for metric in metrics:
        groups.setdefault(metric.sampling_method, []).append(value_fn(metric))
    return groups


def top_tools(trajectories: Iterable[ToolTrajectory], count: int) -> list[str]:
    totals = Counter()
    for trajectory in trajectories:
        totals.update(trajectory.tool_counts)
    return [tool for tool, _ in totals.most_common(max(1, count))]


def tool_color(index: int) -> str:
    return TOOL_COLORS[index % len(TOOL_COLORS)]


def tool_fraction(trajectory: ToolTrajectory, tool: str) -> float:
    total = trajectory.total_calls
    if total == 0:
        return 0.0
    return trajectory.tool_counts.get(tool, 0) / total


def quartiles(values: list[float]) -> tuple[float, float, float, float, float]:
    values = sorted(values)
    if len(values) == 1:
        only = values[0]
        return only, only, only, only, only
    q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return values[0], q1, median, q3, values[-1]


def nice_max(values: Iterable[float]) -> float:
    maximum = max(values)
    if maximum <= 0:
        return 1.0
    step = 10 ** math.floor(math.log10(maximum))
    return math.ceil((maximum * 1.08) / step) * step


def axis_ticks(max_value: float, count: int = 5) -> list[float]:
    if count <= 1:
        return [max_value]
    return [max_value * index / (count - 1) for index in range(count)]


def stable_jitter(text: str, width: float) -> float:
    total = 0
    for index, char in enumerate(text):
        total += (index + 1) * ord(char)
    return ((total % 997) / 996.0 - 0.5) * width


def format_value(value: float) -> str:
    if abs(value) < 0.1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def draw_y_axis(
    svg: PdfCanvas,
    *,
    x0: float,
    y0: float,
    plot_width: float,
    plot_height: float,
    max_y: float,
    ylabel: str,
) -> Callable[[float], float]:
    def y_scale(value: float) -> float:
        return y0 + plot_height - value / max_y * plot_height

    svg.line(x0, y0, x0, y0 + plot_height, stroke="#555")
    svg.line(x0, y0 + plot_height, x0 + plot_width, y0 + plot_height, stroke="#555")
    for tick in axis_ticks(max_y):
        y = y_scale(tick)
        svg.line(x0, y, x0 + plot_width, y, stroke="#ddd")
        svg.text(x0 - 8, y + 4, format_value(tick), anchor="end", size_class="small")
    svg.text(
        20,
        y0 + plot_height / 2,
        ylabel,
        anchor="middle",
        size_class="label",
        rotate=-90,
    )
    return y_scale


def plot_method_boxplot(
    metrics: list[TaskMetric],
    *,
    value_fn: Callable[[TaskMetric], float],
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    methods = sorted_methods(metrics)
    groups = grouped_by_method(metrics, value_fn)
    values = [value for method in methods for value in groups[method]]

    width, height = 840, 520
    left, right, top, bottom = 80, 40, 56, 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_y = nice_max(values)
    x_step = plot_width / len(methods)

    svg = PdfCanvas(width, height)
    svg.text(left, 30, title, size_class="title")
    y_scale = draw_y_axis(
        svg,
        x0=left,
        y0=top,
        plot_width=plot_width,
        plot_height=plot_height,
        max_y=max_y,
        ylabel=ylabel,
    )

    for index, method in enumerate(methods):
        method_values = groups[method]
        min_v, q1, median, q3, max_v = quartiles(method_values)
        x = left + x_step * (index + 0.5)
        box_width = min(72, x_step * 0.5)
        color = color_method(method)

        svg.line(x, y_scale(min_v), x, y_scale(max_v), stroke=color, width=2)
        svg.line(
            x - box_width / 3,
            y_scale(min_v),
            x + box_width / 3,
            y_scale(min_v),
            stroke=color,
            width=2,
        )
        svg.line(
            x - box_width / 3,
            y_scale(max_v),
            x + box_width / 3,
            y_scale(max_v),
            stroke=color,
            width=2,
        )
        svg.rect(
            x - box_width / 2,
            y_scale(q3),
            box_width,
            max(1, y_scale(q1) - y_scale(q3)),
            fill=color,
            stroke=color,
            opacity=0.22,
        )
        svg.line(
            x - box_width / 2,
            y_scale(median),
            x + box_width / 2,
            y_scale(median),
            stroke=color,
            width=2,
        )

        for metric in [item for item in metrics if item.sampling_method == method]:
            jitter = stable_jitter(metric.task, box_width * 0.55)
            svg.circle(
                x + jitter, y_scale(value_fn(metric)), 3.0, fill=color, opacity=0.74
            )

        label = f"{label_method(method)}\n(n={len(method_values)})"
        svg.text(x, height - 52, label.split("\n")[0], anchor="middle")
        svg.text(
            x, height - 34, label.split("\n")[1], anchor="middle", size_class="small"
        )

    svg.save(output_path)


def metrics_by_task(metrics: Iterable[TaskMetric]) -> dict[str, dict[str, TaskMetric]]:
    by_task: dict[str, dict[str, TaskMetric]] = {}
    for metric in metrics:
        by_task.setdefault(metric.task, {})[metric.sampling_method] = metric
    return by_task


def plot_paired_slope(metrics: list[TaskMetric], output_path: Path) -> None:
    methods = sorted_methods(metrics)
    by_task = metrics_by_task(metrics)
    tasks = [
        task
        for task, values in by_task.items()
        if "base" in values and sum(method in values for method in methods) >= 2
    ]

    width, height = 860, 520
    left, right, top, bottom = 82, 42, 56, 86
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_y = nice_max(metric.diversity for metric in metrics)
    x_step = plot_width / max(1, len(methods) - 1)

    svg = PdfCanvas(width, height)
    svg.text(left, 30, "Paired task diversity by sampling method", size_class="title")
    y_scale = draw_y_axis(
        svg,
        x0=left,
        y0=top,
        plot_width=plot_width,
        plot_height=plot_height,
        max_y=max_y,
        ylabel="1 - avg cosine",
    )

    x_positions = {
        method: left + x_step * index for index, method in enumerate(methods)
    }
    for task in sorted(tasks):
        present = [method for method in methods if method in by_task[task]]
        points = [
            (x_positions[method], y_scale(by_task[task][method].diversity))
            for method in present
        ]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            svg.line(x1, y1, x2, y2, stroke="#999", opacity=0.24)
        for method in present:
            metric = by_task[task][method]
            svg.circle(
                x_positions[method],
                y_scale(metric.diversity),
                3.0,
                fill=color_method(method),
                opacity=0.82,
            )

    for method, x in x_positions.items():
        svg.text(x, height - 50, label_method(method), anchor="middle")
    svg.text(
        left,
        height - 24,
        f"{len(tasks)} tasks with base plus at least one comparison method",
        size_class="small",
    )
    svg.save(output_path)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def blend_color(low: str, high: str, fraction: float) -> str:
    fraction = max(0.0, min(1.0, fraction))
    lo = hex_to_rgb(low)
    hi = hex_to_rgb(high)
    rgb = tuple(round(a + (b - a) * fraction) for a, b in zip(lo, hi))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def plot_task_heatmap(metrics: list[TaskMetric], output_path: Path) -> None:
    methods = sorted_methods(metrics)
    by_task = metrics_by_task(metrics)
    tasks = sorted(
        by_task,
        key=lambda task: max(metric.diversity for metric in by_task[task].values()),
        reverse=True,
    )
    values = [metric.diversity for metric in metrics]
    min_value, max_value = min(values), max(values)
    span = max(max_value - min_value, 1e-9)

    row_height = 17
    cell_width = 118
    left = 210
    top = 58
    width = left + cell_width * len(methods) + 50
    height = top + row_height * len(tasks) + 72

    svg = PdfCanvas(width, height)
    svg.text(left, 30, "Task-method diversity heatmap", size_class="title")

    for col, method in enumerate(methods):
        x = left + col * cell_width + cell_width / 2
        svg.text(x, top - 18, label_method(method), anchor="middle", size_class="small")

    for row, task in enumerate(tasks):
        y = top + row * row_height
        svg.text(left - 8, y + 12, task, anchor="end", size_class="small")
        for col, method in enumerate(methods):
            x = left + col * cell_width
            metric = by_task[task].get(method)
            if metric is None:
                svg.rect(
                    x, y, cell_width - 2, row_height - 2, fill="#eeeeee", stroke="white"
                )
                continue
            fraction = (metric.diversity - min_value) / span
            fill = blend_color("#f7fbff", "#2166ac", fraction)
            svg.rect(x, y, cell_width - 2, row_height - 2, fill=fill, stroke="white")

    legend_x = left
    legend_y = height - 38
    legend_width = 180
    for index in range(legend_width):
        fill = blend_color("#f7fbff", "#2166ac", index / max(1, legend_width - 1))
        svg.rect(legend_x + index, legend_y, 1, 12, fill=fill)
    svg.text(legend_x, legend_y + 28, format_value(min_value), size_class="small")
    svg.text(
        legend_x + legend_width,
        legend_y + 28,
        format_value(max_value),
        anchor="end",
        size_class="small",
    )
    svg.text(
        legend_x + legend_width + 14,
        legend_y + 10,
        "1 - avg cosine",
        size_class="small",
    )
    svg.save(output_path)


def plot_delta_from_base(metrics: list[TaskMetric], output_path: Path) -> None:
    methods = [method for method in sorted_methods(metrics) if method != "base"]
    by_task = metrics_by_task(metrics)
    deltas: dict[str, list[tuple[str, float]]] = {}
    for method in methods:
        rows: list[tuple[str, float]] = []
        for task, values in by_task.items():
            if "base" in values and method in values:
                rows.append((task, values[method].diversity - values["base"].diversity))
        deltas[method] = sorted(rows, key=lambda item: item[1])

    all_deltas = [abs(delta) for rows in deltas.values() for _, delta in rows]
    max_abs = nice_max(all_deltas) if all_deltas else 1.0
    row_height = 13
    panel_gap = 42
    panel_left = 230
    panel_width = 620
    top = 58
    width = 920
    height = (
        top
        + sum(max(1, len(rows)) * row_height + panel_gap for rows in deltas.values())
        + 30
    )

    svg = PdfCanvas(width, height)
    svg.text(panel_left, 30, "Diversity delta from base", size_class="title")

    y_cursor = top
    for method in methods:
        rows = deltas[method]
        panel_height = max(1, len(rows)) * row_height
        x_zero = panel_left + panel_width / 2

        svg.text(20, y_cursor + 12, label_method(method), size_class="label")
        svg.line(x_zero, y_cursor - 8, x_zero, y_cursor + panel_height, stroke="#555")
        svg.line(
            panel_left,
            y_cursor - 8,
            panel_left + panel_width,
            y_cursor - 8,
            stroke="#ddd",
        )
        svg.text(
            panel_left,
            y_cursor - 14,
            f"-{format_value(max_abs)}",
            anchor="start",
            size_class="small",
        )
        svg.text(x_zero, y_cursor - 14, "0", anchor="middle", size_class="small")
        svg.text(
            panel_left + panel_width,
            y_cursor - 14,
            f"+{format_value(max_abs)}",
            anchor="end",
            size_class="small",
        )

        for row, (task, delta) in enumerate(rows):
            y = y_cursor + row * row_height
            bar_width = abs(delta) / max_abs * (panel_width / 2)
            if delta >= 0:
                x = x_zero
                fill = "#54a24b"
            else:
                x = x_zero - bar_width
                fill = "#e45756"
            svg.text(panel_left - 8, y + 9, task, anchor="end", size_class="small")
            svg.rect(
                x, y + 2, max(1.0, bar_width), row_height - 4, fill=fill, opacity=0.78
            )

        y_cursor += panel_height + panel_gap

    svg.save(output_path)


def plot_diversity_vs_length(metrics: list[TaskMetric], output_path: Path) -> None:
    methods = sorted_methods(metrics)
    xs = [metric.conversation_words for metric in metrics]
    ys = [metric.diversity for metric in metrics]
    min_x, max_x = min(xs), max(xs)
    max_y = nice_max(ys)
    x_span = max(max_x - min_x, 1e-9)

    width, height = 840, 520
    left, right, top, bottom = 82, 150, 56, 76
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_scale(value: float) -> float:
        return left + (value - min_x) / x_span * plot_width

    svg = PdfCanvas(width, height)
    svg.text(left, 30, "Diversity vs conversation length", size_class="title")
    y_scale = draw_y_axis(
        svg,
        x0=left,
        y0=top,
        plot_width=plot_width,
        plot_height=plot_height,
        max_y=max_y,
        ylabel="1 - avg cosine",
    )

    svg.text(
        left + plot_width / 2, height - 22, "avg conversation words", anchor="middle"
    )
    for index in range(5):
        value = min_x + x_span * index / 4
        x = x_scale(value)
        svg.line(x, top + plot_height, x, top + plot_height + 5, stroke="#555")
        svg.text(
            x,
            top + plot_height + 20,
            f"{value:.0f}",
            anchor="middle",
            size_class="small",
        )

    for metric in metrics:
        svg.circle(
            x_scale(metric.conversation_words),
            y_scale(metric.diversity),
            3.4,
            fill=color_method(metric.sampling_method),
            opacity=0.78,
        )

    legend_x = left + plot_width + 24
    legend_y = top + 10
    for index, method in enumerate(methods):
        y = legend_y + index * 22
        svg.circle(legend_x, y - 4, 4, fill=color_method(method))
        svg.text(legend_x + 12, y, label_method(method), size_class="small")

    svg.save(output_path)


def plot_method_numeric_boxplot(
    method_values: dict[str, list[float]],
    output_path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    methods = [method for method in METHOD_ORDER if method in method_values]
    methods.extend(
        sorted(method for method in method_values if method not in METHOD_ORDER)
    )
    values = [value for method in methods for value in method_values[method]]

    width, height = 840, 520
    left, right, top, bottom = 82, 40, 56, 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_y = nice_max(values)
    x_step = plot_width / len(methods)

    svg = PdfCanvas(width, height)
    svg.text(left, 30, title, size_class="title")
    y_scale = draw_y_axis(
        svg,
        x0=left,
        y0=top,
        plot_width=plot_width,
        plot_height=plot_height,
        max_y=max_y,
        ylabel=ylabel,
    )

    for index, method in enumerate(methods):
        values_for_method = method_values[method]
        min_v, q1, median, q3, max_v = quartiles(values_for_method)
        x = left + x_step * (index + 0.5)
        box_width = min(72, x_step * 0.5)
        color = color_method(method)

        svg.line(x, y_scale(min_v), x, y_scale(max_v), stroke=color, width=2)
        svg.line(
            x - box_width / 3,
            y_scale(min_v),
            x + box_width / 3,
            y_scale(min_v),
            stroke=color,
            width=2,
        )
        svg.line(
            x - box_width / 3,
            y_scale(max_v),
            x + box_width / 3,
            y_scale(max_v),
            stroke=color,
            width=2,
        )
        svg.rect(
            x - box_width / 2,
            y_scale(q3),
            box_width,
            max(1, y_scale(q1) - y_scale(q3)),
            fill=color,
            stroke=color,
            opacity=0.22,
        )
        svg.line(
            x - box_width / 2,
            y_scale(median),
            x + box_width / 2,
            y_scale(median),
            stroke=color,
            width=2,
        )
        svg.text(x, height - 52, label_method(method), anchor="middle")
        svg.text(
            x,
            height - 34,
            f"(n={len(values_for_method)})",
            anchor="middle",
            size_class="small",
        )

    svg.save(output_path)


def plot_tool_fraction_by_method(
    trajectories: list[ToolTrajectory],
    output_path: Path,
    *,
    top_tool_count: int,
) -> None:
    methods = sorted_methods(trajectories)
    tools = top_tools(trajectories, top_tool_count)
    all_tools = set()
    totals_by_method: dict[str, Counter[str]] = {}
    for method in methods:
        totals = Counter()
        for trajectory in trajectories:
            if trajectory.sampling_method == method:
                totals.update(trajectory.tool_counts)
        totals_by_method[method] = totals
        all_tools.update(totals)
    if all_tools - set(tools):
        tools = [*tools, "other"]

    width, height = 940, 520
    left, right, top, bottom = 82, 210, 56, 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_step = plot_width / len(methods)
    bar_width = min(92, x_step * 0.55)

    def y_scale(value: float) -> float:
        return top + plot_height - value * plot_height

    svg = PdfCanvas(width, height)
    svg.text(left, 30, "Tool-call fraction by sampling method", size_class="title")
    draw_y_axis(
        svg,
        x0=left,
        y0=top,
        plot_width=plot_width,
        plot_height=plot_height,
        max_y=1.0,
        ylabel="fraction of tool calls",
    )

    for method_index, method in enumerate(methods):
        totals = totals_by_method[method]
        total_calls = sum(totals.values())
        x = left + x_step * (method_index + 0.5) - bar_width / 2
        y_cursor = top + plot_height
        selected_total = sum(totals.get(tool, 0) for tool in tools if tool != "other")
        for tool_index, tool in enumerate(tools):
            count = (
                total_calls - selected_total if tool == "other" else totals.get(tool, 0)
            )
            fraction = count / total_calls if total_calls else 0.0
            height_part = fraction * plot_height
            y_cursor -= height_part
            svg.rect(
                x,
                y_cursor,
                bar_width,
                height_part,
                fill=tool_color(tool_index),
                stroke="white",
                opacity=0.88,
            )
        svg.text(x + bar_width / 2, height - 52, label_method(method), anchor="middle")
        svg.text(
            x + bar_width / 2,
            height - 34,
            f"{total_calls} calls",
            anchor="middle",
            size_class="small",
        )

    legend_x = left + plot_width + 28
    legend_y = top + 8
    for index, tool in enumerate(tools):
        y = legend_y + index * 22
        svg.rect(legend_x, y - 10, 11, 11, fill=tool_color(index))
        svg.text(legend_x + 18, y, tool, size_class="small")

    svg.save(output_path)


def plot_tool_calls_per_trajectory(
    trajectories: list[ToolTrajectory], output_path: Path
) -> None:
    values: dict[str, list[float]] = {}
    for trajectory in trajectories:
        values.setdefault(trajectory.sampling_method, []).append(
            float(trajectory.total_calls)
        )
    plot_method_numeric_boxplot(
        values,
        output_path,
        title="Tool calls per trajectory",
        ylabel="tool calls",
    )


def plot_top_tool_fraction_boxplots(
    trajectories: list[ToolTrajectory],
    output_path: Path,
    *,
    top_tool_count: int,
) -> None:
    methods = sorted_methods(trajectories)
    tools = top_tools(trajectories, top_tool_count)
    columns = 2
    panel_width = 430
    panel_height = 220
    rows = math.ceil(len(tools) / columns)
    width = columns * panel_width + 40
    height = 58 + rows * panel_height + 26

    svg = PdfCanvas(width, height)
    svg.text(48, 30, "Per-trajectory fraction of top tools", size_class="title")

    for tool_index, tool in enumerate(tools):
        col = tool_index % columns
        row = tool_index // columns
        panel_x = 54 + col * panel_width
        panel_y = 64 + row * panel_height
        plot_width = panel_width - 92
        plot_height = panel_height - 70
        values_by_method: dict[str, list[float]] = {
            method: [
                tool_fraction(trajectory, tool)
                for trajectory in trajectories
                if trajectory.sampling_method == method
            ]
            for method in methods
        }
        max_y = min(
            1.0,
            nice_max(value for values in values_by_method.values() for value in values),
        )
        y_scale = draw_y_axis(
            svg,
            x0=panel_x,
            y0=panel_y,
            plot_width=plot_width,
            plot_height=plot_height,
            max_y=max_y,
            ylabel="fraction",
        )
        svg.text(panel_x, panel_y - 18, tool, size_class="label")

        x_step = plot_width / len(methods)
        for method_index, method in enumerate(methods):
            values = values_by_method[method]
            min_v, q1, median, q3, max_v = quartiles(values)
            x = panel_x + x_step * (method_index + 0.5)
            box_width = min(42, x_step * 0.45)
            color = color_method(method)
            svg.line(x, y_scale(min_v), x, y_scale(max_v), stroke=color, width=1.6)
            svg.rect(
                x - box_width / 2,
                y_scale(q3),
                box_width,
                max(1, y_scale(q1) - y_scale(q3)),
                fill=color,
                stroke=color,
                opacity=0.22,
            )
            svg.line(
                x - box_width / 2,
                y_scale(median),
                x + box_width / 2,
                y_scale(median),
                stroke=color,
                width=1.6,
            )
            svg.text(
                x,
                panel_y + plot_height + 18,
                label_method(method),
                anchor="middle",
                size_class="small",
            )

    svg.save(output_path)


def mean_tool_fractions(
    trajectories: list[ToolTrajectory],
    methods: list[str],
    tools: list[str],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for method in methods:
        method_trajectories = [
            trajectory
            for trajectory in trajectories
            if trajectory.sampling_method == method
        ]
        result[method] = {}
        for tool in tools:
            values = [
                tool_fraction(trajectory, tool) for trajectory in method_trajectories
            ]
            result[method][tool] = statistics.mean(values) if values else 0.0
    return result


def plot_tool_delta_from_base(
    trajectories: list[ToolTrajectory],
    output_path: Path,
    *,
    top_tool_count: int,
) -> None:
    methods = [method for method in sorted_methods(trajectories) if method != "base"]
    if not methods:
        return
    tools = top_tools(trajectories, top_tool_count)
    means = mean_tool_fractions(trajectories, ["base", *methods], tools)
    deltas = {
        method: {tool: means[method][tool] - means["base"][tool] for tool in tools}
        for method in methods
    }
    max_abs = max(
        abs(value)
        for method_values in deltas.values()
        for value in method_values.values()
    )
    max_abs = max(max_abs, 1e-9)

    cell_width = 142
    row_height = 34
    left = 190
    top = 66
    width = left + cell_width * len(methods) + 64
    height = top + row_height * len(tools) + 78

    svg = PdfCanvas(width, height)
    svg.text(left, 30, "Mean tool-fraction delta from base", size_class="title")
    for col, method in enumerate(methods):
        svg.text(
            left + col * cell_width + cell_width / 2,
            top - 18,
            label_method(method),
            anchor="middle",
        )

    for row, tool in enumerate(tools):
        y = top + row * row_height
        svg.text(left - 8, y + 21, tool, anchor="end", size_class="small")
        for col, method in enumerate(methods):
            delta = deltas[method][tool]
            fraction = abs(delta) / max_abs
            fill = blend_color(
                "#f7f7f7", "#2166ac" if delta >= 0 else "#b2182b", fraction
            )
            x = left + col * cell_width
            svg.rect(x, y, cell_width - 2, row_height - 2, fill=fill, stroke="white")
            svg.text(
                x + cell_width / 2,
                y + 21,
                f"{delta:+.3f}",
                anchor="middle",
                size_class="small",
            )

    legend_x = left
    legend_y = height - 38
    legend_width = 180
    half = legend_width // 2
    for index in range(legend_width):
        if index < half:
            fill = blend_color("#b2182b", "#f7f7f7", index / max(1, half - 1))
        else:
            fill = blend_color("#f7f7f7", "#2166ac", (index - half) / max(1, half - 1))
        svg.rect(legend_x + index, legend_y, 1, 12, fill=fill)
    svg.text(legend_x, legend_y + 28, f"-{max_abs:.3f}", size_class="small")
    svg.text(legend_x + half, legend_y + 28, "0", anchor="middle", size_class="small")
    svg.text(
        legend_x + legend_width,
        legend_y + 28,
        f"+{max_abs:.3f}",
        anchor="end",
        size_class="small",
    )
    svg.text(
        legend_x + legend_width + 14,
        legend_y + 10,
        "mean fraction delta",
        size_class="small",
    )
    svg.save(output_path)


def write_diversity_plots(metrics: list[TaskMetric], output_dir: Path) -> list[Path]:
    outputs = [
        output_dir / "diversity_by_method.pdf",
        output_dir / "paired_task_diversity.pdf",
        output_dir / "task_method_heatmap.pdf",
        output_dir / "delta_from_base.pdf",
        output_dir / "outlier_diversity_by_method.pdf",
        output_dir / "diversity_vs_conversation_length.pdf",
    ]
    plot_method_boxplot(
        metrics,
        value_fn=lambda metric: metric.diversity,
        output_path=outputs[0],
        title="Trajectory diversity by sampling method",
        ylabel="1 - avg cosine",
    )
    plot_paired_slope(metrics, outputs[1])
    plot_task_heatmap(metrics, outputs[2])
    plot_delta_from_base(metrics, outputs[3])
    plot_method_boxplot(
        metrics,
        value_fn=lambda metric: metric.outlier_diversity,
        output_path=outputs[4],
        title="Outlier diversity by sampling method",
        ylabel="1 - min cosine",
    )
    plot_diversity_vs_length(metrics, outputs[5])
    return outputs


def write_tool_plots(
    trajectories: list[ToolTrajectory],
    output_dir: Path,
    *,
    top_tool_count: int,
) -> list[Path]:
    outputs = [
        output_dir / "tool_fraction_by_method.pdf",
        output_dir / "tool_calls_per_trajectory.pdf",
        output_dir / "top_tool_fraction_boxplots.pdf",
        output_dir / "tool_distribution_delta_from_base.pdf",
    ]
    plot_tool_fraction_by_method(
        trajectories,
        outputs[0],
        top_tool_count=top_tool_count,
    )
    plot_tool_calls_per_trajectory(trajectories, outputs[1])
    plot_top_tool_fraction_boxplots(
        trajectories,
        outputs[2],
        top_tool_count=min(top_tool_count, 8),
    )
    plot_tool_delta_from_base(
        trajectories,
        outputs[3],
        top_tool_count=top_tool_count,
    )
    return outputs


def main() -> int:
    args = parse_args()
    metrics = read_metrics(args.input)
    outputs = write_diversity_plots(metrics, args.output_dir)
    if args.raw_input.exists():
        trajectories = read_tool_trajectories(args.raw_input)
        outputs.extend(
            write_tool_plots(
                trajectories,
                args.output_dir,
                top_tool_count=args.top_tools,
            )
        )
    else:
        print(
            f"Skipping tool-call plots because raw input does not exist: {args.raw_input}"
        )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

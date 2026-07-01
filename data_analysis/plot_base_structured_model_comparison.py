#!/usr/bin/env python3
"""Plot base vs structured-random diversity across generation models as PNGs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

try:
    from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by environment.
    raise SystemExit(
        "Pillow is required for readable PNG plots. Run this script with "
        "/work/hdd/bgjs/mnakamura/robocasa/.venv_d/bin/python or install Pillow "
        "in the selected Python environment."
    ) from exc

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_analysis.plotting import TaskMetric, label_method, read_metrics


DEFAULT_OUTPUT_DIR = Path("data_analysis/plots/model_sampling_comparison")
DEFAULT_MODEL_INPUTS = (
    "Gemini-3-Flash=data_generation/task_level/data_k3/raw/"
    "sampling_method_analysis_qwen3/task_metrics.csv",
    "Gemini-3.1-Pro=data_analysis/model_sampling_comparison/analysis/"
    "gemini_3_1_pro/task_metrics.csv",
    "GPT-5.4-Mini=data_analysis/model_sampling_comparison/analysis/"
    "gpt_5_4_mini_azure/task_metrics.csv",
)
DEFAULT_METHODS = ("base", "structured_random")
BASELINE_METHOD = "base"
COMPARISON_METHOD = "structured_random"

BACKGROUND_COLOR = "#ffffff"
PANEL_FILL = "#ffffff"
GRID_COLOR = "#d0d8e6"
AXIS_COLOR = "#566172"
TEXT_COLOR = "#1f2937"
MUTED_TEXT_COLOR = "#64748b"
FONT_SCALE = 1.30
LINE_WIDTH_SCALE = 1.30
PALETTE_DARKEN_FACTOR = 0.74
POINT_RADIUS_SCALE = 1.50
LEGEND_ICON_SCALE = 1.50
SEABORN_SPECTRAL_CONTROL_COLORS = (
    "#9e0142",
    "#d53e4f",
    "#f46d43",
    "#fdae61",
    "#fee08b",
    "#ffffbf",
    "#e6f598",
    "#abdda4",
    "#66c2a5",
    "#3288bd",
    "#5e4fa2",
)


def scaled_font(size: int) -> int:
    return max(1, int(round(size * FONT_SCALE)))


REGULAR_FONT_CANDIDATES = (
    ("/System/Library/Fonts/SFNS.ttf", "SF Pro"),
    ("/System/Library/Fonts/SFNSDisplay.ttf", "SF Pro Display"),
    ("/Library/Fonts/SF-Pro-Text-Regular.otf", "SF Pro Text"),
    ("/Library/Fonts/SF-Pro-Display-Regular.otf", "SF Pro Display"),
    ("/usr/share/fonts/urw-base35/NimbusSans-Regular.otf", "Nimbus Sans"),
    ("/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf", "Nimbus Sans"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVu Sans"),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "DejaVu Sans"),
    ("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", "DejaVu Sans"),
    ("/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf", "Liberation Sans"),
)
BOLD_FONT_CANDIDATES = (
    ("/System/Library/Fonts/SFNS.ttf", "SF Pro"),
    ("/System/Library/Fonts/SFNSDisplay.ttf", "SF Pro Display"),
    ("/Library/Fonts/SF-Pro-Text-Bold.otf", "SF Pro Text"),
    ("/Library/Fonts/SF-Pro-Display-Bold.otf", "SF Pro Display"),
    ("/usr/share/fonts/urw-base35/NimbusSans-Bold.otf", "Nimbus Sans"),
    ("/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf", "Nimbus Sans"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVu Sans"),
    ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", "DejaVu Sans"),
    ("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", "DejaVu Sans"),
    ("/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf", "Liberation Sans"),
)


@dataclass(frozen=True)
class ModelInput:
    label: str
    path: Path


@dataclass(frozen=True)
class ModelMetric:
    model: str
    metric: TaskMetric


@dataclass(frozen=True)
class BaselineDelta:
    model: str
    task: str
    sampling_method: str
    base_diversity: float
    method_diversity: float
    delta_from_base: float


@dataclass(frozen=True)
class FontSet:
    regular_path: Path | None
    bold_path: Path | None
    family: str


class PngCanvas:
    def __init__(
        self, width: int, height: int, fonts: FontSet, *, scale: int = 2
    ) -> None:
        self.width = width
        self.height = height
        self.scale = scale
        self.fonts = fonts
        self.image = Image.new(
            "RGB", (width * scale, height * scale), hex_to_rgb(BACKGROUND_COLOR)
        )
        self.draw = ImageDraw.Draw(self.image)
        self._font_cache: dict[tuple[int, bool], ImageFont.ImageFont] = {}

    def _s(self, value: float) -> int:
        return int(round(value * self.scale))

    def font(self, size: int, *, bold: bool = False) -> ImageFont.ImageFont:
        key = (size, bold)
        if key in self._font_cache:
            return self._font_cache[key]
        font_path = (
            self.fonts.bold_path
            if bold and self.fonts.bold_path
            else self.fonts.regular_path
        )
        if font_path is not None:
            font = ImageFont.truetype(str(font_path), self._s(size))
        else:
            font = ImageFont.load_default()
        self._font_cache[key] = font
        return font

    def measure_text(
        self, value: str, size: int, *, bold: bool = False
    ) -> tuple[float, float]:
        bbox = self.draw.textbbox((0, 0), value, font=self.font(size, bold=bold))
        return (
            (bbox[2] - bbox[0]) / self.scale,
            (bbox[3] - bbox[1]) / self.scale,
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: int = scaled_font(14),
        bold: bool = False,
        fill: str = TEXT_COLOR,
        anchor: str = "la",
    ) -> None:
        self.draw.text(
            (self._s(x), self._s(y)),
            value,
            font=self.font(size, bold=bold),
            fill=hex_to_rgb(fill),
            anchor=anchor,
        )

    def rotated_text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: int = scaled_font(14),
        bold: bool = False,
        fill: str = TEXT_COLOR,
        angle: float = -90.0,
    ) -> None:
        font = self.font(size, bold=bold)
        bbox = self.draw.textbbox((0, 0), value, font=font)
        pad = self._s(8)
        temp_width = bbox[2] - bbox[0] + pad * 2
        temp_height = bbox[3] - bbox[1] + pad * 2
        temp = Image.new("RGBA", (temp_width, temp_height), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp)
        temp_draw.text(
            (pad - bbox[0], pad - bbox[1]),
            value,
            font=font,
            fill=(*hex_to_rgb(fill), 255),
        )
        resample = getattr(Image, "Resampling", Image).BICUBIC
        rotated = temp.rotate(angle, expand=True, resample=resample)
        left = self._s(x) - rotated.width // 2
        top = self._s(y) - rotated.height // 2
        self.image.paste(rotated.convert("RGB"), (left, top), rotated.getchannel("A"))

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        fill: str = AXIS_COLOR,
        width: float = 1.0,
    ) -> None:
        self.draw.line(
            (self._s(x1), self._s(y1), self._s(x2), self._s(y2)),
            fill=hex_to_rgb(fill),
            width=max(1, self._s(width * LINE_WIDTH_SCALE)),
        )

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str | tuple[int, int, int] = BACKGROUND_COLOR,
        outline: str | None = None,
        stroke_width: float = 1.0,
        radius: float = 0.0,
    ) -> None:
        xy = (self._s(x), self._s(y), self._s(x + width), self._s(y + height))
        fill_value = fill if isinstance(fill, tuple) else hex_to_rgb(fill)
        outline_value = None if outline is None else hex_to_rgb(outline)
        if radius > 0:
            self.draw.rounded_rectangle(
                xy,
                radius=self._s(radius),
                fill=fill_value,
                outline=outline_value,
                width=max(1, self._s(stroke_width * LINE_WIDTH_SCALE)),
            )
        else:
            self.draw.rectangle(
                xy,
                fill=fill_value,
                outline=outline_value,
                width=max(1, self._s(stroke_width * LINE_WIDTH_SCALE)),
            )

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        fill: str | tuple[int, int, int],
        outline: str | None = None,
        stroke_width: float = 1.0,
    ) -> None:
        fill_value = fill if isinstance(fill, tuple) else hex_to_rgb(fill)
        outline_value = None if outline is None else hex_to_rgb(outline)
        xy = (
            self._s(x - radius),
            self._s(y - radius),
            self._s(x + radius),
            self._s(y + radius),
        )
        self.draw.ellipse(
            xy,
            fill=fill_value,
            outline=outline_value,
            width=max(1, self._s(stroke_width * LINE_WIDTH_SCALE)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        resample = getattr(Image, "Resampling", Image).LANCZOS
        output = self.image.resize((self.width, self.height), resample=resample)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("FontFamily", self.fonts.family)
        metadata.add_text("GeneratedBy", Path(__file__).name)
        output.save(path, pnginfo=metadata)


def parse_model_input(value: str) -> ModelInput:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Model inputs must use LABEL=PATH, for example "
            "'Gemini-3-Flash=data_analysis/.../task_metrics.csv'."
        )
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("Model input label and path are required.")
    return ModelInput(label=label, path=Path(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create PNG plots comparing base and structured-random trajectory "
            "diversity across model analysis outputs."
        )
    )
    parser.add_argument(
        "--model-input",
        action="append",
        type=parse_model_input,
        default=None,
        help="Model analysis input as LABEL=task_metrics.csv. Repeat per model.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Sampling methods to include. Defaults to base structured_random.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def discover_fonts() -> FontSet:
    regular_path, family = first_existing_font(REGULAR_FONT_CANDIDATES)
    bold_path, bold_family = first_existing_font(BOLD_FONT_CANDIDATES)
    return FontSet(
        regular_path=regular_path,
        bold_path=bold_path,
        family=family or bold_family or "Pillow default",
    )


def first_existing_font(
    candidates: Iterable[tuple[str, str]]
) -> tuple[Path | None, str | None]:
    for raw_path, family in candidates:
        path = Path(raw_path)
        if path.exists():
            return path, family
    return None, None


def load_model_metrics(
    model_inputs: list[ModelInput],
    *,
    methods: list[str],
) -> list[ModelMetric]:
    method_set = set(methods)
    rows: list[ModelMetric] = []
    for model_input in model_inputs:
        metrics = [
            metric
            for metric in read_metrics(model_input.path)
            if metric.sampling_method in method_set
        ]
        if not metrics:
            raise ValueError(
                f"No requested methods found in {model_input.path}: "
                f"{', '.join(methods)}"
            )
        rows.extend(
            ModelMetric(model=model_input.label, metric=metric) for metric in metrics
        )
    return rows


def metrics_by_model_method(
    model_metrics: list[ModelMetric],
) -> dict[tuple[str, str], list[TaskMetric]]:
    groups: dict[tuple[str, str], list[TaskMetric]] = {}
    for row in model_metrics:
        key = (row.model, row.metric.sampling_method)
        groups.setdefault(key, []).append(row.metric)
    return groups


def baseline_deltas(model_metrics: list[ModelMetric]) -> list[BaselineDelta]:
    by_model_task_method: dict[tuple[str, str, str], TaskMetric] = {}
    for row in model_metrics:
        by_model_task_method[
            (row.model, row.metric.task, row.metric.sampling_method)
        ] = row.metric

    deltas: list[BaselineDelta] = []
    for (model, task, method), metric in sorted(by_model_task_method.items()):
        if method != COMPARISON_METHOD:
            continue
        base = by_model_task_method.get((model, task, BASELINE_METHOD))
        if base is None:
            continue
        deltas.append(
            BaselineDelta(
                model=model,
                task=task,
                sampling_method=method,
                base_diversity=base.diversity,
                method_diversity=metric.diversity,
                delta_from_base=metric.diversity - base.diversity,
            )
        )
    if not deltas:
        raise ValueError(
            f"No paired {COMPARISON_METHOD} vs {BASELINE_METHOD} rows were found."
        )
    return deltas


def write_summary_csv(
    path: Path,
    groups: dict[tuple[str, str], list[TaskMetric]],
    *,
    model_order: list[str],
    method_order: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "sampling_method",
                "n_tasks",
                "mean_diversity",
                "median_diversity",
                "min_diversity",
                "max_diversity",
                "mean_num_trajectories",
            ],
        )
        writer.writeheader()
        for model in model_order:
            for method in method_order:
                metrics = groups.get((model, method), [])
                if not metrics:
                    continue
                values = [metric.diversity for metric in metrics]
                writer.writerow(
                    {
                        "model": model,
                        "sampling_method": method,
                        "n_tasks": len(values),
                        "mean_diversity": f"{statistics.mean(values):.6f}",
                        "median_diversity": f"{statistics.median(values):.6f}",
                        "min_diversity": f"{min(values):.6f}",
                        "max_diversity": f"{max(values):.6f}",
                        "mean_num_trajectories": (
                            f"{statistics.mean(metric.num_trajectories for metric in metrics):.2f}"
                        ),
                    }
                )


def write_delta_csv(path: Path, deltas: list[BaselineDelta]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "task",
                "sampling_method",
                "base_diversity",
                "method_diversity",
                "delta_from_base",
            ],
        )
        writer.writeheader()
        for row in deltas:
            writer.writerow(
                {
                    "model": row.model,
                    "task": row.task,
                    "sampling_method": row.sampling_method,
                    "base_diversity": f"{row.base_diversity:.6f}",
                    "method_diversity": f"{row.method_diversity:.6f}",
                    "delta_from_base": f"{row.delta_from_base:.6f}",
                }
            )


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {color!r}")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def rgb_to_hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def fallback_spectral_palette(n_colors: int) -> tuple[str, ...]:
    if n_colors <= 0:
        return ()
    stops = [hex_to_rgb(color) for color in SEABORN_SPECTRAL_CONTROL_COLORS]
    positions = [(index + 1) / (n_colors + 1) for index in range(n_colors)]
    colors: list[str] = []
    for position in positions:
        scaled = position * (len(stops) - 1)
        lower = int(math.floor(scaled))
        upper = min(lower + 1, len(stops) - 1)
        fraction = scaled - lower
        rgb = tuple(
            int(
                round(
                    stops[lower][channel] * (1.0 - fraction)
                    + stops[upper][channel] * fraction
                )
            )
            for channel in range(3)
        )
        colors.append(rgb_to_hex(rgb))
    return tuple(colors)


def spectral_palette(n_colors: int) -> tuple[str, ...]:
    if n_colors <= 0:
        return ()
    try:
        import seaborn as sns
    except ModuleNotFoundError:
        return fallback_spectral_palette(n_colors)
    return tuple(sns.color_palette("Spectral", n_colors=n_colors).as_hex())


def darken_color(color: str, *, factor: float = PALETTE_DARKEN_FACTOR) -> str:
    rgb = hex_to_rgb(color)
    factor = max(0.0, min(1.0, factor))
    return rgb_to_hex(tuple(int(round(channel * factor)) for channel in rgb))


def palette_mapping(labels: list[str]) -> dict[str, str]:
    colors = tuple(darken_color(color) for color in spectral_palette(len(labels)))
    return {label: colors[index] for index, label in enumerate(labels)}


def blend(
    color: str, *, opacity: float, background: str = BACKGROUND_COLOR
) -> tuple[int, int, int]:
    foreground = hex_to_rgb(color)
    bg = hex_to_rgb(background)
    opacity = max(0.0, min(1.0, opacity))
    return tuple(
        int(round(foreground[index] * opacity + bg[index] * (1.0 - opacity)))
        for index in range(3)
    )


def stable_jitter(text: str, width: float) -> float:
    total = 0
    for index, char in enumerate(text):
        total += (index + 1) * ord(char)
    return ((total % 1000) / 999.0 - 0.5) * width


def quartiles(values: list[float]) -> tuple[float, float, float, float, float]:
    values = sorted(values)
    if len(values) == 1:
        only = values[0]
        return only, only, only, only, only
    q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return values[0], q1, median, q3, values[-1]


def padded_limits(
    values: Iterable[float],
    *,
    include_zero: bool = False,
    symmetric: bool = False,
) -> tuple[float, float]:
    values = list(values)
    if not values:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        maximum = max(abs(value) for value in values)
        maximum = max(maximum * 1.18, 0.001)
        return -maximum, maximum

    low = min(values)
    high = max(values)
    if include_zero:
        low = min(0.0, low)
        high = max(0.0, high)
    if math.isclose(low, high):
        padding = max(abs(low) * 0.12, 0.001)
    else:
        padding = (high - low) * 0.12
    padded_low = low - padding
    if include_zero:
        padded_low = max(0.0, padded_low)
    return padded_low, high + padding


def tick_values(low: float, high: float, count: int = 6) -> list[float]:
    if count <= 1:
        return [low]
    step = (high - low) / (count - 1)
    return [low + index * step for index in range(count)]


def fixed_step_tick_values(low: float, high: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("Tick step must be positive.")
    start = math.floor(low / step) * step
    stop = math.ceil(high / step) * step
    count = int(round((stop - start) / step))
    return [round(start + index * step, 10) for index in range(count + 1)]


def format_tick(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return "0"
    if abs(value) < 1:
        text = f"{value:.2f}"
        if text.startswith("-0"):
            return f"-{text[2:]}"
        if text.startswith("0"):
            return text[1:]
        return text
    return f"{value:.2f}".rstrip("0").rstrip(".")


def value_to_y(
    value: float, low: float, high: float, top: float, bottom: float
) -> float:
    if math.isclose(low, high):
        return (top + bottom) / 2
    return bottom - (value - low) / (high - low) * (bottom - top)


def draw_y_axis(
    canvas: PngCanvas,
    *,
    left: float,
    top: float,
    right: float,
    bottom: float,
    low: float,
    high: float,
    label: str,
    ticks: list[float] | None = None,
    tick_size: int = scaled_font(15),
    label_size: int = scaled_font(18),
    label_angle: float = -90.0,
) -> None:
    canvas.rect(left, top, right - left, bottom - top, fill=PANEL_FILL)
    for tick in ticks if ticks is not None else tick_values(low, high):
        y = value_to_y(tick, low, high, top, bottom)
        canvas.line(left, y, right, y, fill=GRID_COLOR, width=1.35)
        canvas.text(
            left - 14,
            y,
            format_tick(tick),
            size=tick_size,
            fill=MUTED_TEXT_COLOR,
            anchor="rm",
        )
    label_x = max(24, left - 124)
    canvas.line(left, top, left, bottom, fill=AXIS_COLOR, width=1.9)
    canvas.line(left, bottom, right, bottom, fill=AXIS_COLOR, width=1.9)
    canvas.rotated_text(
        label_x,
        (top + bottom) / 2,
        label,
        size=label_size,
        fill=TEXT_COLOR,
        angle=label_angle,
    )


def draw_boxplot(
    canvas: PngCanvas,
    *,
    center_x: float,
    values: list[float],
    box_width: float,
    color: str,
    y_scale: Callable[[float], float],
) -> None:
    min_v, q1, median, q3, max_v = quartiles(values)
    y_min = y_scale(min_v)
    y_q1 = y_scale(q1)
    y_med = y_scale(median)
    y_q3 = y_scale(q3)
    y_max = y_scale(max_v)
    left = center_x - box_width / 2
    right = center_x + box_width / 2
    whisker_half = box_width * 0.22

    box_stroke_width = 4.2
    median_stroke_width = 5.0

    canvas.line(center_x, y_min, center_x, y_max, fill=color, width=box_stroke_width)
    canvas.line(
        center_x - whisker_half,
        y_min,
        center_x + whisker_half,
        y_min,
        fill=color,
        width=box_stroke_width,
    )
    canvas.line(
        center_x - whisker_half,
        y_max,
        center_x + whisker_half,
        y_max,
        fill=color,
        width=box_stroke_width,
    )

    box_top = min(y_q1, y_q3)
    box_height = max(abs(y_q3 - y_q1), 2.5)
    canvas.rect(
        left,
        box_top,
        right - left,
        box_height,
        fill=blend(color, opacity=0.20),
        outline=color,
        stroke_width=box_stroke_width,
        radius=3,
    )
    canvas.line(left, y_med, right, y_med, fill=color, width=median_stroke_width)


def wrap_text(
    canvas: PngCanvas, value: str, *, max_width: float, size: int
) -> list[str]:
    words = value.split()
    if not words:
        return [value]
    if len(words) == 1:
        return split_hyphenated_label(canvas, words[0], max_width=max_width, size=size)
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if canvas.measure_text(candidate, size)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def split_hyphenated_label(
    canvas: PngCanvas,
    value: str,
    *,
    max_width: float,
    size: int,
) -> list[str]:
    if canvas.measure_text(value, size)[0] <= max_width or "-" not in value:
        return [value]
    pieces = value.split("-")
    chunks = [
        f"{piece}-" if index < len(pieces) - 1 else piece
        for index, piece in enumerate(pieces)
    ]
    lines: list[str] = []
    current = chunks[0]
    for chunk in chunks[1:]:
        candidate = f"{current}{chunk}"
        if canvas.measure_text(candidate, size)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = chunk
    lines.append(current)
    return lines


def draw_wrapped_centered(
    canvas: PngCanvas,
    x: float,
    y: float,
    value: str,
    *,
    max_width: float,
    size: int = scaled_font(13),
    fill: str = TEXT_COLOR,
) -> None:
    for index, line in enumerate(
        wrap_text(canvas, value, max_width=max_width, size=size)
    ):
        canvas.text(
            x,
            y + index * (size + 5),
            line,
            size=size,
            fill=fill,
            anchor="ma",
        )


def draw_method_legend(
    canvas: PngCanvas,
    *,
    methods: list[str],
    method_colors: dict[str, str],
    x: float,
    y: float,
    size: int = scaled_font(15),
) -> None:
    cursor = x
    for method in methods:
        color = method_colors[method]
        label = label_method(method)
        marker_width = size * 0.78 * LEGEND_ICON_SCALE
        marker_height = size * 0.54 * LEGEND_ICON_SCALE
        canvas.rect(
            cursor,
            y - marker_height * 0.68,
            marker_width,
            marker_height,
            fill=blend(color, opacity=0.26),
            outline=color,
            stroke_width=2.1,
            radius=3,
        )
        marker_line_y = y - marker_height * 0.18
        canvas.line(
            cursor,
            marker_line_y,
            cursor + marker_width,
            marker_line_y,
            fill=color,
            width=3.6,
        )
        canvas.text(
            cursor + marker_width + size * 0.29,
            marker_line_y,
            label,
            size=size,
            fill=MUTED_TEXT_COLOR,
            anchor="lm",
        )
        label_width = canvas.measure_text(label, size)[0]
        cursor += marker_width + size * 1.12 + label_width


def plot_diversity_by_model(
    groups: dict[tuple[str, str], list[TaskMetric]],
    *,
    model_order: list[str],
    method_order: list[str],
    output_path: Path,
    fonts: FontSet,
) -> None:
    width = 1600
    height = 1600
    left = 160
    right = width - 28
    top = 160
    bottom = height - 250
    title_size = scaled_font(54)
    y_tick_size = scaled_font(36)
    axis_label_size = scaled_font(44)
    legend_size = scaled_font(41)
    model_label_size = scaled_font(40)
    x_axis_label_size = axis_label_size
    values = [
        metric.diversity
        for model in model_order
        for method in method_order
        for metric in groups.get((model, method), [])
    ]
    y_low, y_high = padded_limits(values, include_zero=True)
    tick_step = 0.05
    y_low = max(0.0, math.floor(y_low / tick_step) * tick_step)
    y_high = math.ceil(y_high / tick_step) * tick_step
    if math.isclose(y_low, y_high):
        y_high = y_low + tick_step
    y_ticks = fixed_step_tick_values(y_low, y_high, tick_step)

    canvas = PngCanvas(width, height, fonts)
    method_colors = palette_mapping(method_order)
    canvas.text(
        width / 2,
        44,
        "Trajectory Conversation Diversity Over Models",
        size=title_size,
        bold=True,
        anchor="ma",
    )
    draw_y_axis(
        canvas,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        low=y_low,
        high=y_high,
        label="Trajectory Conversation Diversity ↑",
        ticks=y_ticks,
        tick_size=y_tick_size,
        label_size=axis_label_size,
        label_angle=90.0,
    )
    draw_method_legend(
        canvas,
        methods=method_order,
        method_colors=method_colors,
        x=right - 720,
        y=top + 64,
        size=legend_size,
    )

    group_width = (right - left) / max(len(model_order), 1)
    method_count = max(len(method_order), 1)
    method_step = min(205, group_width / max(method_count, 1))
    box_width = min(150, method_step * 0.72)
    y_scale = lambda value: value_to_y(value, y_low, y_high, top, bottom)

    for model_index, model in enumerate(model_order):
        group_left = left + model_index * group_width
        group_center = group_left + group_width / 2
        canvas.line(
            group_center,
            bottom,
            group_center,
            bottom + 7,
            fill=AXIS_COLOR,
            width=1.8,
        )
        for method_index, method in enumerate(method_order):
            metrics = groups.get((model, method), [])
            if not metrics:
                continue
            offset = (method_index - (method_count - 1) / 2) * method_step
            center_x = group_center + offset
            color = method_colors[method]
            values_for_method = [metric.diversity for metric in metrics]
            draw_boxplot(
                canvas,
                center_x=center_x,
                values=values_for_method,
                box_width=box_width,
                color=color,
                y_scale=y_scale,
            )
            for metric in metrics:
                jitter = stable_jitter(
                    f"{model}|{method}|{metric.task}",
                    box_width * 0.68,
                )
                point_fill = blend(color, opacity=0.70)
                canvas.circle(
                    center_x + jitter,
                    y_scale(metric.diversity),
                    3.8 * POINT_RADIUS_SCALE,
                    fill=point_fill,
                    outline=BACKGROUND_COLOR,
                    stroke_width=0.9,
                )
        draw_wrapped_centered(
            canvas,
            group_center,
            bottom + 34,
            model,
            max_width=group_width - 34,
            size=model_label_size,
            fill=TEXT_COLOR,
        )

    canvas.text(
        (left + right) / 2,
        bottom + 138,
        "Model",
        size=x_axis_label_size,
        fill=TEXT_COLOR,
        anchor="ma",
    )
    canvas.save(output_path)


def plot_delta_by_model(
    deltas: list[BaselineDelta],
    *,
    model_order: list[str],
    output_path: Path,
    fonts: FontSet,
) -> None:
    width = 1300
    height = 800
    left = 105
    right = width - 62
    top = 125
    bottom = height - 155
    values = [row.delta_from_base for row in deltas]
    y_low, y_high = padded_limits(values, symmetric=True)
    by_model: dict[str, list[BaselineDelta]] = {}
    for row in deltas:
        by_model.setdefault(row.model, []).append(row)

    canvas = PngCanvas(width, height, fonts)
    model_colors = palette_mapping(model_order)
    canvas.text(
        64,
        42,
        "Structured Random Change from Base by Model",
        size=scaled_font(28),
        bold=True,
    )
    canvas.text(
        64,
        79,
        "Positive values mean structured random produced higher task-level diversity than base.",
        size=scaled_font(15),
        fill=MUTED_TEXT_COLOR,
    )
    draw_y_axis(
        canvas,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        low=y_low,
        high=y_high,
        label="Diversity delta from base",
    )
    zero_y = value_to_y(0, y_low, y_high, top, bottom)
    canvas.line(left, zero_y, right, zero_y, fill="#2f3645", width=2.1)

    group_width = (right - left) / max(len(model_order), 1)
    box_width = min(94, group_width * 0.28)
    y_scale = lambda value: value_to_y(value, y_low, y_high, top, bottom)

    for model_index, model in enumerate(model_order):
        rows = by_model.get(model, [])
        if not rows:
            continue
        center_x = left + model_index * group_width + group_width / 2
        color = model_colors[model]
        values_for_model = [row.delta_from_base for row in rows]
        draw_boxplot(
            canvas,
            center_x=center_x,
            values=values_for_model,
            box_width=box_width,
            color=color,
            y_scale=y_scale,
        )
        for row in rows:
            jitter = stable_jitter(f"{model}|{row.task}", box_width * 0.72)
            canvas.circle(
                center_x + jitter,
                y_scale(row.delta_from_base),
                3.2 * POINT_RADIUS_SCALE,
                fill=blend(color, opacity=0.70),
                outline=BACKGROUND_COLOR,
                stroke_width=0.7,
            )
        draw_wrapped_centered(
            canvas,
            center_x,
            bottom + 34,
            model,
            max_width=group_width - 32,
            size=scaled_font(15),
            fill=TEXT_COLOR,
        )
        mean_delta = statistics.mean(values_for_model)
        canvas.text(
            center_x,
            top - 25,
            f"mean {mean_delta:+.3f}",
            size=scaled_font(13),
            fill=MUTED_TEXT_COLOR,
            anchor="ma",
        )

    canvas.text(
        (left + right) / 2,
        height - 55,
        "Model",
        size=scaled_font(15),
        fill=TEXT_COLOR,
        anchor="ma",
    )
    canvas.save(output_path)


def write_outputs(
    model_metrics: list[ModelMetric],
    *,
    model_order: list[str],
    method_order: list[str],
    output_dir: Path,
) -> list[Path]:
    groups = metrics_by_model_method(model_metrics)
    deltas = baseline_deltas(model_metrics)
    fonts = discover_fonts()

    summary_csv = output_dir / "model_method_diversity_summary.csv"
    delta_csv = output_dir / "structured_random_delta_from_base.csv"
    diversity_png = output_dir / "base_structured_diversity_by_model.png"
    delta_png = output_dir / "structured_random_delta_from_base_by_model.png"

    write_summary_csv(
        summary_csv,
        groups,
        model_order=model_order,
        method_order=method_order,
    )
    write_delta_csv(delta_csv, deltas)
    plot_diversity_by_model(
        groups,
        model_order=model_order,
        method_order=method_order,
        output_path=diversity_png,
        fonts=fonts,
    )
    plot_delta_by_model(
        deltas,
        model_order=model_order,
        output_path=delta_png,
        fonts=fonts,
    )
    print(f"Using font family: {fonts.family}")
    return [summary_csv, delta_csv, diversity_png, delta_png]


def main() -> None:
    args = parse_args()
    model_inputs = (
        args.model_input
        if args.model_input is not None
        else [parse_model_input(value) for value in DEFAULT_MODEL_INPUTS]
    )
    methods = list(dict.fromkeys(args.methods))
    model_metrics = load_model_metrics(model_inputs, methods=methods)
    outputs = write_outputs(
        model_metrics,
        model_order=[model_input.label for model_input in model_inputs],
        method_order=methods,
        output_dir=args.output_dir,
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

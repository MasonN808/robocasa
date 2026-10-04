#!/usr/bin/env python3
"""Plot phase distributions for the canonical 43/10 task split."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = ROOT / "configs/splits/43_train_10_heldout.json"
DEFAULT_ATTRIBUTES = ROOT / "docs/composite_tasks/task_attributes.json"
DEFAULT_SPECS = ROOT / "robotalk/tasks/specs/verified"

PHASES = ("Phase 1", "Phase 2", "Phase 3", "Phase 4")
PHASE_SUBTITLES = ("1 stage", "2–3 stages", "4–5 stages", "6+ stages")
BLUE = "#3266A8"
ORANGE = "#D9813B"
INK = "#24313D"
GRID = "#DCE2E8"


class CairoCanvas:
    """Tiny ctypes wrapper around the system Cairo PNG renderer."""

    def __init__(self, width: int, height: int):
        lib_path = ctypes.util.find_library("cairo")
        if not lib_path:
            raise RuntimeError("System Cairo library not found")
        self.c = ctypes.CDLL(lib_path)
        self.c.cairo_image_surface_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.c.cairo_image_surface_create.restype = ctypes.c_void_p
        self.c.cairo_create.argtypes = [ctypes.c_void_p]
        self.c.cairo_create.restype = ctypes.c_void_p
        self.c.cairo_text_extents.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        self.surface = ctypes.c_void_p(self.c.cairo_image_surface_create(0, width, height))
        self.ctx = ctypes.c_void_p(self.c.cairo_create(self.surface))
        self.rect(0, 0, width, height, "#FFFFFF")

    @staticmethod
    def rgb(hex_color: str):
        return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))

    def color(self, value: str):
        self.c.cairo_set_source_rgb(self.ctx, *map(ctypes.c_double, self.rgb(value)))

    def rect(self, x, y, width, height, color):
        self.color(color)
        self.c.cairo_rectangle(self.ctx, *map(ctypes.c_double, (x, y, width, height)))
        self.c.cairo_fill(self.ctx)

    def line(self, x1, y1, x2, y2, color, width=1):
        self.color(color)
        self.c.cairo_set_line_width(self.ctx, ctypes.c_double(width))
        self.c.cairo_move_to(self.ctx, ctypes.c_double(x1), ctypes.c_double(y1))
        self.c.cairo_line_to(self.ctx, ctypes.c_double(x2), ctypes.c_double(y2))
        self.c.cairo_stroke(self.ctx)

    def text(self, x, y, value, size, anchor="start", color=INK, bold=False):
        self.color(color)
        self.c.cairo_select_font_face(self.ctx, b"Nimbus Sans", 0, 1 if bold else 0)
        self.c.cairo_set_font_size(self.ctx, ctypes.c_double(size))
        encoded = value.encode("utf-8")
        if anchor != "start":
            extents = (ctypes.c_double * 6)()
            self.c.cairo_text_extents(self.ctx, encoded, extents)
            x -= extents[2] / (2 if anchor == "middle" else 1)
        self.c.cairo_move_to(self.ctx, ctypes.c_double(x), ctypes.c_double(y))
        self.c.cairo_show_text(self.ctx, encoded)

    def vertical_text(self, x, y, value, size, color=INK):
        self.c.cairo_save(self.ctx)
        self.c.cairo_translate(self.ctx, ctypes.c_double(x), ctypes.c_double(y))
        self.c.cairo_rotate(self.ctx, ctypes.c_double(-3.141592653589793 / 2))
        self.text(0, 0, value, size, "middle", color)
        self.c.cairo_restore(self.ctx)

    def save(self, path: Path):
        status = self.c.cairo_surface_write_to_png(self.surface, str(path).encode())
        self.c.cairo_destroy(self.ctx)
        self.c.cairo_surface_destroy(self.surface)
        if status:
            raise RuntimeError(f"Cairo PNG export failed with status {status}")


def snake_case(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def phase_for(num_stages: int) -> str:
    if num_stages == 1:
        return "Phase 1"
    if num_stages <= 3:
        return "Phase 2"
    if num_stages <= 5:
        return "Phase 3"
    return "Phase 4"


def load_counts(split_path: Path, attributes_path: Path, specs_dir: Path):
    split = json.loads(split_path.read_text(encoding="utf-8"))
    attributes = json.loads(attributes_path.read_text(encoding="utf-8"))["tasks"]
    stages = {snake_case(row["name"]): int(row["num_subtasks"]) for row in attributes}
    verified = {
        snake_case(json.loads(path.read_text(encoding="utf-8"))["composite_task"])
        for path in specs_dir.glob("*.json")
    }

    trained = set(split["train_tasks"])
    held_out = set(split["held_out_tasks"])
    if trained & held_out:
        raise ValueError(f"Tasks occur in both splits: {sorted(trained & held_out)}")
    if trained | held_out != verified:
        raise ValueError(
            "Split and verified inventory differ: "
            f"missing={sorted(verified - trained - held_out)}, "
            f"extra={sorted((trained | held_out) - verified)}"
        )
    missing_stages = sorted(verified - stages.keys())
    if missing_stages:
        raise ValueError(f"Tasks missing num_subtasks metadata: {missing_stages}")

    split_counts = {
        "In training": Counter(phase_for(stages[task]) for task in trained),
        "Held out": Counter(phase_for(stages[task]) for task in held_out),
    }
    overall = split_counts["In training"] + split_counts["Held out"]
    return overall, split_counts


def save_overall(overall: Counter, output: Path):
    values = [overall[phase] for phase in PHASES]
    width, height = 1100, 650
    left, top, plot_width, plot_height = 105, 155, 930, 380
    ymax = 30
    canvas = CairoCanvas(width, height)
    draw_titles(canvas, "Verified task specs by RoboCasa phase", "All 53 tasks in the seeded 43-trained / 10-held-out split")
    draw_axes(canvas, left, top, plot_width, plot_height, ymax)
    slot = plot_width / len(PHASES)
    bar_width = 125
    for index, (phase, subtitle, value) in enumerate(zip(PHASES, PHASE_SUBTITLES, values)):
        center = left + slot * (index + 0.5)
        bar_height = plot_height * value / ymax
        y = top + plot_height - bar_height
        canvas.rect(center - bar_width / 2, y, bar_width, bar_height, BLUE)
        canvas.text(center, y - 12, str(value), 19, "middle", bold=True)
        canvas.text(center, top + plot_height + 34, phase, 17, "middle", bold=True)
        canvas.text(center, top + plot_height + 57, subtitle, 15, "middle", "#586674")
    canvas.save(output)


def save_split(split_counts: dict[str, Counter], output: Path):
    trained = [split_counts["In training"][phase] for phase in PHASES]
    held_out = [split_counts["Held out"][phase] for phase in PHASES]
    width, height = 1100, 650
    left, top, plot_width, plot_height = 105, 180, 930, 355
    ymax = 24
    canvas = CairoCanvas(width, height)
    draw_titles(canvas, "RoboCasa phase distribution by task split", "Counts of verified task specs; seeded 43/10 assignment")
    canvas.rect(105, 130, 16, 16, BLUE)
    canvas.text(130, 143, "In training (n=43)", 15)
    canvas.rect(305, 130, 16, 16, ORANGE)
    canvas.text(330, 143, "Held out (n=10)", 15)
    draw_axes(canvas, left, top, plot_width, plot_height, ymax, tick_step=4)
    slot = plot_width / len(PHASES)
    bar_width = 70
    for index, (phase, subtitle, train_value, held_value) in enumerate(zip(PHASES, PHASE_SUBTITLES, trained, held_out)):
        center = left + slot * (index + 0.5)
        for x, value, color in ((center - bar_width - 5, train_value, BLUE), (center + 5, held_value, ORANGE)):
            bar_height = plot_height * value / ymax
            y = top + plot_height - bar_height
            if value:
                canvas.rect(x, y, bar_width, bar_height, color)
            canvas.text(x + bar_width / 2, y - 10, str(value), 17, "middle", bold=True)
        canvas.text(center, top + plot_height + 34, phase, 17, "middle", bold=True)
        canvas.text(center, top + plot_height + 57, subtitle, 15, "middle", "#586674")
    canvas.save(output)


def draw_titles(canvas, title, subtitle):
    canvas.text(70, 65, title, 26, bold=True)
    canvas.text(70, 96, subtitle, 16, color="#586674")


def draw_axes(canvas, left, top, width, height, ymax, tick_step=5):
    baseline = top + height
    for value in range(0, ymax + 1, tick_step):
        y = baseline - height * value / ymax
        canvas.line(left, y, left + width, y, GRID)
        canvas.text(left - 16, y + 5, str(value), 14, "end", "#586674")
    canvas.line(left, baseline, left + width, baseline, "#AEB8C2", 1.2)
    canvas.vertical_text(28, top + height / 2, "Number of tasks", 15, "#586674")


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: Inter, Arial, sans-serif; }</style>',
    ]


def svg_text(x, y, text, size, anchor="start", *, weight=400, fill=INK) -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{text}</text>'


def svg_titles(title: str, subtitle: str) -> list[str]:
    return [
        svg_text(70, 65, title, 26, weight=700),
        svg_text(70, 96, subtitle, 16, fill="#586674"),
    ]


def svg_axes(left, top, width, height, ymax, tick_step=5) -> list[str]:
    label_y = top + height / 2
    parts = [
        f'<text x="30" y="{label_y}" font-size="15" text-anchor="middle" '
        f'fill="#586674" transform="rotate(-90 30 {label_y})">Number of task specs</text>'
    ]
    baseline = top + height
    for value in range(0, ymax + 1, tick_step):
        y = baseline - height * value / ymax
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+width}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(svg_text(left - 16, y + 5, str(value), 14, "end", fill="#586674"))
    parts.append(f'<line x1="{left}" y1="{baseline}" x2="{left+width}" y2="{baseline}" stroke="#AEB8C2" stroke-width="1.2"/>')
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--attributes", type=Path, default=DEFAULT_ATTRIBUTES)
    parser.add_argument("--specs-dir", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data_analysis/plots/task_phase_distribution")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall, split_counts = load_counts(args.split, args.attributes, args.specs_dir)
    combined_path = args.output_dir / "phase_distribution_combined.png"
    split_path = args.output_dir / "phase_distribution_by_train_heldout.png"
    save_overall(overall, combined_path)
    save_split(split_counts, split_path)
    print(combined_path)
    print(split_path)


if __name__ == "__main__":
    main()

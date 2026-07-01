#!/usr/bin/env python3
"""Plot sampling-method trajectory diversity and tool-call distributions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFont, PngImagePlugin
except ModuleNotFoundError:  # Keep the original dependency-free renderer available.
    Image = None
    ImageChops = None
    ImageDraw = None
    ImageFont = None
    PngImagePlugin = None


DEFAULT_INPUT = Path(
    "data_generation/task_level/data_k3/raw/sampling_method_analysis_qwen3/task_metrics.csv"
)
DEFAULT_RAW_INPUT = Path("data_generation/task_level/data_k3/raw/sampling_methods")
DEFAULT_OUTPUT_DIR = Path("data_analysis/plots/sampling_method_diversity")
METHOD_ORDER = [
    "base",
    "high_temperature",
    "random",
    "structured_random",
    "verbalized",
]
METHOD_LABELS = {
    "base": "base",
    "high_temperature": "high temp",
    "random": "random",
    "structured_random": "structured random",
    "verbalized": "verbalized",
}
METHOD_COLORS = {
    "base": "#4c78a8",
    "high_temperature": "#f58518",
    "random": "#54a24b",
    "structured_random": "#e45756",
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
PIL_CANVAS_SCALE = 2
PIL_TEXT_SIZES = {
    "title": 36,
    "label": 14,
    "method_label": 18,
    "task_label": 13,
    "interval_label": 15,
    "small": 10,
}
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


@dataclass(frozen=True)
class FontSet:
    regular_path: Path | None
    bold_path: Path | None
    family: str


def first_existing_font(
    candidates: Iterable[tuple[str, str]],
) -> tuple[Path | None, str | None]:
    for raw_path, family in candidates:
        path = Path(raw_path)
        if path.exists():
            return path, family
    return None, None


def discover_fonts() -> FontSet:
    regular_path, regular_family = first_existing_font(REGULAR_FONT_CANDIDATES)
    bold_path, bold_family = first_existing_font(BOLD_FONT_CANDIDATES)
    return FontSet(
        regular_path=regular_path,
        bold_path=bold_path,
        family=regular_family or bold_family or "Pillow default",
    )


class PillowPngCanvas:
    def __init__(
        self, width: int, height: int, *, scale: int = PIL_CANVAS_SCALE
    ) -> None:
        if Image is None or ImageDraw is None or ImageFont is None:
            raise RuntimeError("Pillow is not available")
        self.width = width
        self.height = height
        self.scale = scale
        self.fonts = discover_fonts()
        self.image = Image.new("RGB", (width * scale, height * scale), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.text_values: list[str] = []
        self._font_cache: dict[tuple[int, bool], ImageFont.ImageFont] = {}

    def _s(self, value: float) -> int:
        return int(round(value * self.scale))

    def color_components(
        self, color: str, opacity: float = 1.0
    ) -> tuple[int, int, int]:
        named_colors = {
            "black": "#000000",
            "white": "#ffffff",
        }
        color = named_colors.get(color, color).lstrip("#")
        if len(color) == 3:
            color = "".join(channel * 2 for channel in color)
        red = int(color[0:2], 16)
        green = int(color[2:4], 16)
        blue = int(color[4:6], 16)
        if opacity < 1.0:
            red = round(255 - (255 - red) * opacity)
            green = round(255 - (255 - green) * opacity)
            blue = round(255 - (255 - blue) * opacity)
        return red, green, blue

    def text_size(self, size_class: str) -> int:
        return PIL_TEXT_SIZES.get(size_class, PIL_TEXT_SIZES["label"])

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
        fill = self.color_components(stroke, opacity)
        scaled_width = max(1, self._s(width))
        if dash:
            steps = max(1, int(max(abs(x2 - x1), abs(y2 - y1)) / 8))
            for index in range(steps):
                if index % 2:
                    continue
                start = index / steps
                end = min(1.0, (index + 1) / steps)
                self.draw.line(
                    (
                        self._s(x1 + (x2 - x1) * start),
                        self._s(y1 + (y2 - y1) * start),
                        self._s(x1 + (x2 - x1) * end),
                        self._s(y1 + (y2 - y1) * end),
                    ),
                    fill=fill,
                    width=scaled_width,
                )
        else:
            self.draw.line(
                (self._s(x1), self._s(y1), self._s(x2), self._s(y2)),
                fill=fill,
                width=scaled_width,
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
        xy = (
            self._s(x),
            self._s(y),
            self._s(x + width),
            self._s(y + height),
        )
        self.draw.rectangle(
            xy,
            fill=None if fill == "none" else self.color_components(fill, opacity),
            outline=None
            if stroke == "none"
            else self.color_components(stroke, opacity),
            width=max(1, self._s(1.0)),
        )

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
        xy = (
            self._s(cx - r),
            self._s(cy - r),
            self._s(cx + r),
            self._s(cy + r),
        )
        self.draw.ellipse(
            xy,
            fill=None if fill == "none" else self.color_components(fill, opacity),
            outline=None
            if stroke == "none"
            else self.color_components(stroke, opacity),
            width=max(1, self._s(1.0)),
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
        bold = size_class == "title"
        font = self.font(size, bold=bold)
        self.text_values.append(value)
        fill = self.color_components("#222222")
        bbox = self.draw.textbbox((0, 0), value, font=font)
        width = bbox[2] - bbox[0]
        anchor_dx = 0
        if anchor == "middle":
            anchor_dx = -width // 2
        elif anchor == "end":
            anchor_dx = -width

        if rotate is None:
            draw_x = self._s(x) + anchor_dx - bbox[0]
            draw_y = self._s(y) - bbox[3]
            self.draw.text((draw_x, draw_y), value, font=font, fill=fill)
            return

        pad = self._s(6)
        temp = Image.new(
            "RGBA",
            (max(1, bbox[2] - bbox[0] + pad * 2), max(1, bbox[3] - bbox[1] + pad * 2)),
            (0, 0, 0, 0),
        )
        temp_draw = ImageDraw.Draw(temp)
        temp_draw.text((pad - bbox[0], pad - bbox[1]), value, font=font, fill=fill)
        resample = getattr(Image, "Resampling", Image).BICUBIC
        rotated = temp.rotate(rotate, expand=True, resample=resample)
        left = self._s(x) - rotated.width // 2
        top = self._s(y) - rotated.height // 2
        self.image.paste(rotated.convert("RGB"), (left, top), rotated.getchannel("A"))
        self.draw = ImageDraw.Draw(self.image)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        resample = getattr(Image, "Resampling", Image).LANCZOS
        output = self.image.resize((self.width, self.height), resample=resample)
        if ImageChops is not None:
            background = Image.new(output.mode, output.size, (255, 255, 255))
            bbox = ImageChops.difference(output, background).getbbox()
            if bbox is not None:
                crop_padding = 12
                left_crop_padding = 28
                output = output.crop(
                    (
                        max(0, bbox[0] - left_crop_padding),
                        max(0, bbox[1] - crop_padding),
                        min(output.width, bbox[2] + crop_padding),
                        min(output.height, bbox[3] + crop_padding),
                    )
                )
        metadata = PngImagePlugin.PngInfo() if PngImagePlugin is not None else None
        if metadata is not None:
            metadata.add_text("FontFamily", self.fonts.family)
            metadata.add_text("Labels", "; ".join(dict.fromkeys(self.text_values)))
        output.save(path, pnginfo=metadata)


BITMAP_FONT = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "11100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "01100", "01100", "01000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "<": ["00010", "00100", "01000", "10000", "01000", "00100", "00010"],
    ">": ["01000", "00100", "00010", "00001", "00010", "00100", "01000"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    "%": ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    "?": ["01110", "10001", "00001", "00010", "00100", "00000", "00100"],
}


class PngCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.pixels = [bytearray([255, 255, 255] * width) for _ in range(height)]
        self.text_values: list[str] = []
        self.rect(0, 0, width, height, fill="white")

    def color_components(
        self, color: str, opacity: float = 1.0
    ) -> tuple[int, int, int]:
        if color == "none":
            return 0, 0, 0
        named_colors = {
            "black": "#000000",
            "white": "#ffffff",
        }
        color = named_colors.get(color, color)
        color = color.lstrip("#")
        if len(color) == 3:
            color = "".join(channel * 2 for channel in color)
        red = int(color[0:2], 16)
        green = int(color[2:4], 16)
        blue = int(color[4:6], 16)
        if opacity < 1.0:
            red = round(255 - (255 - red) * opacity)
            green = round(255 - (255 - green) * opacity)
            blue = round(255 - (255 - blue) * opacity)
        return red, green, blue

    def set_pixel(
        self,
        x: int,
        y: int,
        color: tuple[int, int, int],
        opacity: float = 1.0,
    ) -> None:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        offset = x * 3
        row = self.pixels[y]
        if opacity >= 1.0:
            row[offset : offset + 3] = bytes(color)
            return
        row[offset] = round(row[offset] * (1.0 - opacity) + color[0] * opacity)
        row[offset + 1] = round(row[offset + 1] * (1.0 - opacity) + color[1] * opacity)
        row[offset + 2] = round(row[offset + 2] * (1.0 - opacity) + color[2] * opacity)

    def text_size(self, size_class: str) -> int:
        if size_class in {"title", "method_label", "interval_label"}:
            return 2
        return 1

    def text_height(self, scale: int) -> int:
        return 7 * scale

    def estimate_text_width(self, value: str, size: int) -> float:
        return max(0, len(value)) * 6 * size

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
        color = self.color_components(stroke)
        dx = x2 - x1
        dy = y2 - y1
        steps = max(1, int(max(abs(dx), abs(dy))))
        radius = max(0.5, width / 2.0)
        for index in range(steps + 1):
            if dash and (index // 4) % 2:
                continue
            x = x1 + dx * index / steps
            y = y1 + dy * index / steps
            self.draw_disc(x, y, radius, color, opacity)

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
        if fill != "none":
            color = self.color_components(fill)
            left = max(0, math.floor(x))
            top = max(0, math.floor(y))
            right = min(self.width, math.ceil(x + width))
            bottom = min(self.height, math.ceil(y + height))
            for py in range(top, bottom):
                for px in range(left, right):
                    self.set_pixel(px, py, color, opacity)
        if stroke != "none":
            self.line(x, y, x + width, y, stroke=stroke, opacity=opacity)
            self.line(
                x + width, y, x + width, y + height, stroke=stroke, opacity=opacity
            )
            self.line(
                x + width, y + height, x, y + height, stroke=stroke, opacity=opacity
            )
            self.line(x, y + height, x, y, stroke=stroke, opacity=opacity)

    def draw_disc(
        self,
        cx: float,
        cy: float,
        radius: float,
        color: tuple[int, int, int],
        opacity: float = 1.0,
    ) -> None:
        left = math.floor(cx - radius)
        right = math.ceil(cx + radius)
        top = math.floor(cy - radius)
        bottom = math.ceil(cy + radius)
        radius_sq = radius * radius
        for py in range(top, bottom + 1):
            for px in range(left, right + 1):
                if (px - cx) * (px - cx) + (py - cy) * (py - cy) <= radius_sq:
                    self.set_pixel(px, py, color, opacity)

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
        if fill != "none":
            self.draw_disc(cx, cy, r, self.color_components(fill), opacity)
        if stroke != "none":
            stroke_color = self.color_components(stroke)
            for angle in range(0, 360, 2):
                radians = math.radians(angle)
                self.draw_disc(
                    cx + math.cos(radians) * r,
                    cy + math.sin(radians) * r,
                    0.8,
                    stroke_color,
                    opacity,
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
        self.text_values.append(value)
        width = self.estimate_text_width(value, size)
        height = self.text_height(size)
        anchor_dx = 0.0
        if anchor == "middle":
            anchor_dx = -width / 2
        elif anchor == "end":
            anchor_dx = -width

        radians = math.radians(rotate or 0.0)
        cos_value = math.cos(radians)
        sin_value = math.sin(radians)
        color = self.color_components("#222222")
        cursor = 0
        for raw_char in value:
            char = raw_char.upper()
            glyph = BITMAP_FONT.get(char, BITMAP_FONT["?"])
            for row_index, row in enumerate(glyph):
                for column_index, pixel in enumerate(row):
                    if pixel != "1":
                        continue
                    for sy in range(size):
                        for sx in range(size):
                            local_x = anchor_dx + cursor + column_index * size + sx
                            local_y = row_index * size + sy - height
                            if rotate is None:
                                draw_x = x + local_x
                                draw_y = y + local_y
                            else:
                                draw_x = x + local_x * cos_value - local_y * sin_value
                                draw_y = y + local_x * sin_value + local_y * cos_value
                            self.set_pixel(round(draw_x), round(draw_y), color)
            cursor += 6 * size

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_rows = b"".join(b"\x00" + bytes(row) for row in self.pixels)

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        labels = "; ".join(dict.fromkeys(self.text_values)).encode(
            "latin-1", errors="replace"
        )
        png = bytearray(b"\x89PNG\r\n\x1a\n")
        png.extend(
            chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0),
            )
        )
        png.extend(chunk(b"tEXt", b"Labels\x00" + labels))
        png.extend(chunk(b"IDAT", zlib.compress(raw_rows, level=9)))
        png.extend(chunk(b"IEND", b""))
        path.write_bytes(bytes(png))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create simple PNG plots for sampling-method trajectory diversity "
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
    svg: PngCanvas,
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

    svg = PngCanvas(width, height)
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

    svg = PngCanvas(width, height)
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

    svg = PngCanvas(width, height)
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
    row_height = 14
    panel_gap = 14
    label_width = 165
    bar_width = 215
    panel_width = label_width + bar_width
    left = 28
    right = 6
    top = 70
    max_rows = max((max(1, len(rows)) for rows in deltas.values()), default=1)
    width = (
        left + len(methods) * panel_width + max(0, len(methods) - 1) * panel_gap + right
    )
    method_label_y = top + max_rows * row_height + 30
    height = top + max_rows * row_height + 46

    svg = (
        PillowPngCanvas(width, height)
        if Image is not None
        else PngCanvas(width, height)
    )
    svg.text(
        width / 2,
        38,
        "Diverstiy Delta from Base Sampling",
        anchor="middle",
        size_class="title",
    )

    for index, method in enumerate(methods):
        rows = deltas[method]
        panel_height = max(1, len(rows)) * row_height
        panel_left = left + index * (panel_width + panel_gap)
        bar_left = panel_left + label_width
        x_zero = bar_left + bar_width / 2

        svg.line(x_zero, top - 5, x_zero, top + panel_height, stroke="#555")
        svg.line(
            bar_left,
            top - 5,
            bar_left + bar_width,
            top - 5,
            stroke="#ddd",
            width=2.0,
        )
        svg.text(
            bar_left,
            top - 7,
            f"-{format_value(max_abs)}",
            anchor="start",
            size_class="interval_label",
        )
        svg.text(x_zero, top - 7, "0", anchor="middle", size_class="interval_label")
        svg.text(
            bar_left + bar_width,
            top - 7,
            f"+{format_value(max_abs)}",
            anchor="end",
            size_class="interval_label",
        )

        for row, (task, delta) in enumerate(rows):
            y = top + row * row_height
            delta_width = abs(delta) / max_abs * (bar_width / 2)
            if delta >= 0:
                x = x_zero
                fill = "#54a24b"
            else:
                x = x_zero - delta_width
                fill = "#e45756"
            svg.text(
                bar_left - 6,
                y + 10,
                task,
                anchor="end",
                size_class="task_label",
            )
            svg.rect(
                x,
                y + 2,
                max(1.0, delta_width),
                row_height - 4,
                fill=fill,
                opacity=0.78,
            )

        svg.text(
            bar_left + bar_width / 2,
            method_label_y,
            label_method(method).title(),
            anchor="middle",
            size_class="method_label",
        )

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

    svg = PngCanvas(width, height)
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

    svg = PngCanvas(width, height)
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

    svg = PngCanvas(width, height)
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

    svg = PngCanvas(width, height)
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

    svg = PngCanvas(width, height)
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
        output_dir / "diversity_by_method.png",
        output_dir / "paired_task_diversity.png",
        output_dir / "task_method_heatmap.png",
        output_dir / "delta_from_base.png",
        output_dir / "outlier_diversity_by_method.png",
        output_dir / "diversity_vs_conversation_length.png",
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
        output_dir / "tool_fraction_by_method.png",
        output_dir / "tool_calls_per_trajectory.png",
        output_dir / "top_tool_fraction_boxplots.png",
        output_dir / "tool_distribution_delta_from_base.png",
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

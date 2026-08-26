"""Build a self-contained HTML report from fixed-cohort summary JSON files."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _vertical_ci_svg(*, low: float, value: float, high: float, height: int) -> str:
    y_low, y_value, y_high = (height * (1 - point) for point in (low, value, high))
    return (
        f'<line x1="0" x2="0" y1="{y_high:.1f}" y2="{y_low:.1f}" class="ci"/>'
        f'<line x1="-4" x2="4" y1="{y_high:.1f}" y2="{y_high:.1f}" class="ci"/>'
        f'<line x1="-4" x2="4" y1="{y_low:.1f}" y2="{y_low:.1f}" class="ci"/>'
        f'<circle cx="0" cy="{y_value:.1f}" r="3" class="point"/>'
    )


_SERIES = (
    ("fsm_success", "Final FSM success", "final"),
    ("fsm_error_free_success", "Error-free FSM success", "error-free"),
    ("fsm_single_agent_success", "Success, one agent did 100%", "single-agent"),
    ("fsm_dominant_agent_success_ge_0_8", "Success, one agent did ≥80%", "dominant-agent"),
)

_ERROR_SERIES = (
    ("call_construction_grounding", "Call construction & grounding", "call-grounding"),
    ("state_action_execution", "State & action execution", "state-action"),
    ("coordination_progress", "Coordination & progress", "coordination-progress"),
)

_SPLIT_ORDER = {"train_task_types": 0, "heldout_task_types": 1}


def _legacy_metric(row: dict) -> dict:
    return {
        "count": row["successes"],
        "rate": row["success_rate"],
        "wilson_95_ci": row["ci"],
    }


def _split_spans(rows: list[dict]) -> list[tuple[str, int, int]]:
    spans = []
    for split in sorted({row["split"] for row in rows}, key=lambda x: _SPLIT_ORDER.get(x, 99)):
        indices = [index for index, row in enumerate(rows) if row["split"] == split]
        if indices:
            spans.append((split, min(indices), max(indices)))
    return spans


def _task_success_chart(rows: list[dict]) -> str:
    if not rows:
        return '<section class="chart"><h3>Success by task</h3><p>No results.</p></section>'
    left, top, plot_height, group_width = 70, 72, 350, 78
    width = max(1120, left + group_width * len(rows) + 45)
    height = 625
    baseline = top + plot_height
    marks: list[str] = []
    spans = _split_spans(rows)
    for split, first, last in spans:
        x0 = left + first * group_width
        x1 = left + (last + 1) * group_width
        if split == "heldout_task_types":
            marks.append(f'<rect x="{x0}" y="{top-26}" width="{x1-x0}" height="{plot_height+34}" class="heldout-bg"/>')
        marks.append(f'<text x="{(x0+x1)/2:.1f}" y="{top-34}" text-anchor="middle" class="split-label">{html.escape(split)}</text>')
        for key, _series_label, css_class in _SERIES:
            values = [rows[i]["metrics"][key]["rate"] for i in range(first, last + 1)]
            mean = sum(values) / len(values)
            y = top + plot_height * (1 - mean)
            marks.append(f'<line x1="{x0+2}" x2="{x1-2}" y1="{y:.1f}" y2="{y:.1f}" class="mean-line {css_class}"/>')
    if len(spans) > 1:
        divider = left + spans[0][2] * group_width + group_width
        marks.append(f'<line x1="{divider}" x2="{divider}" y1="{top-27}" y2="{baseline+8}" class="split-divider"/>')
    bar_width = 12
    for index, row in enumerate(rows):
        group_x = left + index * group_width
        metrics = row.get("metrics") or {"fsm_success": _legacy_metric(row)}
        for series_index, (key, series_label, css_class) in enumerate(_SERIES):
            metric = metrics.get(key)
            if metric is None:
                continue
            rate = metric["rate"]
            low, high = metric["wilson_95_ci"]
            x = group_x + 8 + series_index * 15
            y = top + plot_height * (1 - rate)
            tooltip = html.escape(
                f'{row["task"]} — {series_label}: {_pct(rate)} '
                f'({metric["count"]}/{row["num_episodes"]}); 95% CI {_pct(low)}–{_pct(high)}'
            )
            marks.append(
                f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{baseline-y:.1f}" class="bar-mark {css_class}"><title>{tooltip}</title></rect>'
            )
            marks.append(
                f'<g transform="translate({x+bar_width/2:.1f},{top})">'
                f'{_vertical_ci_svg(low=low, value=rate, high=high, height=plot_height)}</g>'
            )
        marks.append(f'<text transform="translate({group_x+34},{baseline+16}) rotate(58)" text-anchor="start" class="task-label">{html.escape(row["task"])}</text>')
    ticks: list[str] = []
    for fraction in (0, .25, .5, .75, 1):
        y = top + plot_height * (1 - fraction)
        ticks.append(f'<line x1="{left}" x2="{width-25}" y1="{y}" y2="{y}" class="gridline"/>')
        ticks.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" class="axis">{fraction:.0%}</text>')
    legend = "".join(
        f'<span><i class="legend-swatch {css_class}"></i>{html.escape(label)}</span>'
        for _key, label, css_class in _SERIES
    )
    return (
        '<section class="chart"><h3>Success by task</h3>'
        '<p class="muted">All four rates use all evaluated episodes as the denominator; whiskers are 95% Wilson binomial confidence intervals. Final minus error-free success is success that required recovery.</p>'
        f'<div class="legend">{legend}</div>'
        f'<div class="scroll"><svg viewBox="0 0 {width} {height}" style="min-width:{width}px" role="img" aria-label="Success by task">'
        f'{"".join(ticks)}{"".join(marks)}</svg></div></section>'
    )


def _task_error_chart(rows: list[dict]) -> str:
    if not rows:
        return '<section class="chart"><h3>Errors by task</h3><p>No results.</p></section>'
    left, top, plot_height, group_width = 70, 72, 350, 78
    width = max(1120, left + group_width * len(rows) + 45)
    height = 625
    baseline = top + plot_height
    maximum = max(
        1,
        max(
            row["errors"]["categories"][key]["total_events"]
            for row in rows for key, _, _ in _ERROR_SERIES
        ),
    )
    tick_max = max(5, ((maximum + 4) // 5) * 5)
    marks: list[str] = []
    spans = _split_spans(rows)
    for split, first, last in spans:
        x0 = left + first * group_width
        x1 = left + (last + 1) * group_width
        if split == "heldout_task_types":
            marks.append(f'<rect x="{x0}" y="{top-26}" width="{x1-x0}" height="{plot_height+34}" class="heldout-bg"/>')
        marks.append(f'<text x="{(x0+x1)/2:.1f}" y="{top-34}" text-anchor="middle" class="split-label">{html.escape(split)}</text>')
        for key, _label, css_class in _ERROR_SERIES:
            values = [rows[i]["errors"]["categories"][key]["total_events"] for i in range(first, last + 1)]
            mean = sum(values) / len(values)
            y = top + plot_height * (1 - mean / tick_max)
            marks.append(f'<line x1="{x0+2}" x2="{x1-2}" y1="{y:.1f}" y2="{y:.1f}" class="mean-line {css_class}"/>')
    if len(spans) > 1:
        divider = left + spans[0][2] * group_width + group_width
        marks.append(f'<line x1="{divider}" x2="{divider}" y1="{top-27}" y2="{baseline+8}" class="split-divider"/>')
    bar_width = 16
    for index, row in enumerate(rows):
        group_x = left + index * group_width
        for series_index, (key, label, css_class) in enumerate(_ERROR_SERIES):
            metric = row["errors"]["categories"][key]
            failed = metric["failed_trajectory_events"]
            recovered = metric["recovered_success_events"]
            total = failed + recovered
            x = group_x + 8 + series_index * 20
            failed_h = plot_height * failed / tick_max
            recovered_h = plot_height * recovered / tick_max
            subtype_text = ", ".join(f"{name}: {count}" for name, count in metric["subtypes"].items()) or "none"
            tooltip = html.escape(
                f'{row["task"]} — {label}: {total} errors; {metric["events_per_episode"]:.2f}/episode; '
                f'{metric["affected_episodes"]}/{row["num_episodes"]} episodes affected; '
                f'{recovered} in eventual successes; {failed} in failed trajectories; subtypes: {subtype_text}'
            )
            if failed:
                marks.append(f'<rect x="{x}" y="{baseline-failed_h:.1f}" width="{bar_width}" height="{failed_h:.1f}" class="error-mark {css_class} failed"><title>{tooltip}</title></rect>')
            if recovered:
                marks.append(f'<rect x="{x}" y="{baseline-failed_h-recovered_h:.1f}" width="{bar_width}" height="{recovered_h:.1f}" class="error-mark {css_class} recovered"><title>{tooltip}</title></rect>')
            if total == 0:
                marks.append(f'<rect x="{x}" y="{baseline-1}" width="{bar_width}" height="1" class="error-mark {css_class} zero"><title>{tooltip}</title></rect>')
        marks.append(f'<text transform="translate({group_x+34},{baseline+16}) rotate(58)" text-anchor="start" class="task-label">{html.escape(row["task"])}</text>')
    ticks: list[str] = []
    for fraction in (0, .25, .5, .75, 1):
        value = tick_max * fraction
        y = top + plot_height * (1 - fraction)
        ticks.append(f'<line x1="{left}" x2="{width-25}" y1="{y}" y2="{y}" class="gridline"/>')
        ticks.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" class="axis">{value:g}</text>')
    category_legend = "".join(
        f'<span><i class="legend-swatch {css_class}"></i>{html.escape(label)}</span>'
        for _key, label, css_class in _ERROR_SERIES
    )
    outcome_legend = '<span><i class="legend-swatch outcome-failed"></i>Errors in failed trajectories</span><span><i class="legend-swatch outcome-recovered"></i>Errors in eventual successes</span>'
    return (
        '<section class="chart"><h3>Errors by task</h3>'
        '<p class="muted">Three error-category bars per task. Each bar is stacked by final episode outcome; dashed lines show the per-task category mean within each split. Hover a bar for raw subtypes and affected-episode counts.</p>'
        f'<div class="legend">{category_legend}{outcome_legend}</div>'
        f'<div class="scroll"><svg viewBox="0 0 {width} {height}" style="min-width:{width}px" role="img" aria-label="Errors by task">'
        f'{"".join(ticks)}{"".join(marks)}</svg></div></section>'
    )


def _overall_success_chart(rows: list[dict]) -> str:
    """Compare pooled outcomes across every model × communication arm."""
    if not rows:
        return '<section class="chart"><h2>Overall success by model and communication mode</h2><p>No results.</p></section>'
    left, top, plot_height, group_width = 76, 62, 360, 126
    width = max(1120, left + group_width * len(rows) + 38)
    height = 565
    baseline = top + plot_height
    bar_width = 34
    series = (
        ("fsm_rate", "dominant_rate", "FSM success", "final"),
        (
            "error_free_rate",
            "error_free_dominant_rate",
            "Error-free FSM success",
            "error-free",
        ),
    )
    marks: list[str] = []
    previous_model = None
    for index, row in enumerate(rows):
        group_x = left + index * group_width
        if previous_model is not None and row["model"] != previous_model:
            divider = group_x - 10
            marks.append(f'<line x1="{divider}" x2="{divider}" y1="{top-24}" y2="{baseline+8}" class="split-divider"/>')
        previous_model = row["model"]
        for series_index, (total_key, subset_key, label, css_class) in enumerate(series):
            rate = row[total_key]
            subset_rate = row[subset_key]
            low, high = row[f"{total_key}_ci"]
            numerator, denominator = row[f"{total_key}_counts"]
            subset_numerator, subset_denominator = row[f"{subset_key}_counts"]
            x = group_x + 22 + series_index * 49
            y = top + plot_height * (1 - rate)
            subset_y = top + plot_height * (1 - subset_rate)
            tooltip = html.escape(
                f'{row["model"]} — {row["mode"]} — {label}: {_pct(rate)} '
                f'({numerator}/{denominator}); success + one agent ≥80%: '
                f'{_pct(subset_rate)} ({subset_numerator}/{subset_denominator}); '
                f'total-success 95% Wilson CI {_pct(low)}–{_pct(high)}'
            )
            marks.append(
                f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{subset_y-y:.1f}" class="bar-mark {css_class} remainder"><title>{tooltip}</title></rect>'
            )
            marks.append(
                f'<rect x="{x}" y="{subset_y:.1f}" width="{bar_width}" height="{baseline-subset_y:.1f}" class="bar-mark {css_class} dominant-subset"><title>{tooltip}</title></rect>'
            )
            marks.append(
                f'<g transform="translate({x+bar_width/2:.1f},{top})">'
                f'{_vertical_ci_svg(low=low, value=rate, high=high, height=plot_height)}</g>'
            )
        marks.append(f'<text x="{group_x+63}" y="{baseline+19}" text-anchor="middle" class="axis">{html.escape(row["mode"])}</text>')

    for model in dict.fromkeys(row["model"] for row in rows):
        indices = [index for index, row in enumerate(rows) if row["model"] == model]
        midpoint = left + ((min(indices) + max(indices) + 1) * group_width) / 2
        marks.append(f'<text x="{midpoint:.1f}" y="{baseline+48}" text-anchor="middle" class="split-label">{html.escape(model)}</text>')

    ticks: list[str] = []
    for fraction in (0, .25, .5, .75, 1):
        y = top + plot_height * (1 - fraction)
        ticks.append(f'<line x1="{left}" x2="{width-25}" y1="{y}" y2="{y}" class="gridline"/>')
        ticks.append(f'<text x="{left-9}" y="{y+4}" text-anchor="end" class="axis">{fraction:.0%}</text>')
    legend = "".join(
        f'<span><i class="legend-swatch {css_class} remainder"></i>{html.escape(label)}: other successes</span>'
        f'<span><i class="legend-swatch {css_class} dominant-subset"></i>{html.escape(label)} + one agent did ≥80%</span>'
        for _total_key, _subset_key, label, css_class in series
    )
    return (
        '<section class="chart overall-chart"><h2>Overall success by model and communication mode</h2>'
        '<p class="muted">Pooled across all 530 fixed-cohort episodes (47 train-task types and 6 held-out task types). Every segment uses all episodes as the denominator. Each dark segment is the subset of that bar where one agent performed ≥80% of task-changing actions; dark plus light equals the complete success rate. Whiskers show the total bar’s 95% Wilson interval.</p>'
        f'<div class="legend">{legend}</div>'
        f'<div class="scroll"><svg viewBox="0 0 {width} {height}" style="min-width:{width}px" role="img" aria-label="Overall success by model and communication mode">'
        f'{"".join(ticks)}{"".join(marks)}</svg></div></section>'
    )


def _epoch_split_chart(summaries: list[dict]) -> str:
    """Show scale-30 checkpoint success by epoch and task split."""
    if not summaries:
        return ""
    left, top, plot_height, group_width = 82, 64, 360, 230
    width = max(1060, left + group_width * len(summaries) + 42)
    height = 590
    baseline = top + plot_height
    bar_width = 34
    bars = (
        ("train_task_types", "fsm_success", "fsm_dominant_agent_success_ge_0_8", "Train · FSM", "final"),
        ("train_task_types", "fsm_error_free_success", "fsm_error_free_dominant_agent_success_ge_0_8", "Train · error-free", "error-free"),
        ("heldout_task_types", "fsm_success", "fsm_dominant_agent_success_ge_0_8", "Held-out · FSM", "final"),
        ("heldout_task_types", "fsm_error_free_success", "fsm_error_free_dominant_agent_success_ge_0_8", "Held-out · error-free", "error-free"),
    )
    marks: list[str] = []
    for index, summary in enumerate(summaries):
        group_x = left + index * group_width
        epoch = str(summary.get("epoch", summary["label"]))
        for bar_index, (split, total_key, subset_key, label, css_class) in enumerate(bars):
            split_values = summary["splits"][split]
            total = split_values["metrics"][total_key]
            subset = split_values["metrics"][subset_key]
            rate, subset_rate = total["rate"], subset["rate"]
            low, high = total["wilson_95_ci"]
            x = group_x + 14 + bar_index * 49
            y = top + plot_height * (1 - rate)
            subset_y = top + plot_height * (1 - subset_rate)
            split_css = "heldout" if split == "heldout_task_types" else "train"
            tooltip = html.escape(
                f'Epoch {epoch} — {label}: {_pct(rate)} ({total["count"]}/{split_values["num_episodes"]}); '
                f'success + one agent ≥80%: {_pct(subset_rate)} '
                f'({subset["count"]}/{split_values["num_episodes"]}); '
                f'95% Wilson CI {_pct(low)}–{_pct(high)}'
            )
            marks.append(
                f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{subset_y-y:.1f}" class="bar-mark {css_class} remainder {split_css}"><title>{tooltip}</title></rect>'
            )
            marks.append(
                f'<rect x="{x}" y="{subset_y:.1f}" width="{bar_width}" height="{baseline-subset_y:.1f}" class="bar-mark {css_class} dominant-subset {split_css}"><title>{tooltip}</title></rect>'
            )
            marks.append(
                f'<g transform="translate({x+bar_width/2:.1f},{top})">'
                f'{_vertical_ci_svg(low=low, value=rate, high=high, height=plot_height)}</g>'
            )
            short_label = ("T" if split_css == "train" else "H") + ("·FSM" if total_key == "fsm_success" else "·EF")
            marks.append(f'<text x="{x+bar_width/2:.1f}" y="{baseline+18}" text-anchor="middle" class="epoch-bar-label">{short_label}</text>')
        marks.append(f'<text x="{group_x+88}" y="{baseline+51}" text-anchor="middle" class="split-label">Epoch {html.escape(epoch)}</text>')

    ticks: list[str] = []
    for fraction in (0, .25, .5, .75, 1):
        y = top + plot_height * (1 - fraction)
        ticks.append(f'<line x1="{left}" x2="{width-25}" y1="{y}" y2="{y}" class="gridline"/>')
        ticks.append(f'<text x="{left-9}" y="{y+4}" text-anchor="end" class="axis">{fraction:.0%}</text>')
    legend = (
        '<span><i class="legend-swatch final remainder"></i>FSM: other successes</span>'
        '<span><i class="legend-swatch final dominant-subset"></i>FSM success + one agent did ≥80%</span>'
        '<span><i class="legend-swatch error-free remainder"></i>Error-free: other successes</span>'
        '<span><i class="legend-swatch error-free dominant-subset"></i>Error-free success + one agent did ≥80%</span>'
        '<span><i class="legend-swatch heldout-outline"></i>Held-out-task split outline</span>'
    )
    return (
        '<section class="chart epoch-chart"><h2>Success rate by epoch and split — 30 trajectories per task</h2>'
        '<p class="muted">Four bars per checkpoint: train-task FSM (T·FSM), train-task error-free (T·EF), held-out-task FSM (H·FSM), and held-out-task error-free (H·EF). Each bar uses all episodes in its split as the denominator and is divided into its ≥80%-single-agent subset and remaining successes. Outlined bars are held-out tasks; whiskers are total-bar 95% Wilson intervals.</p>'
        f'<div class="legend">{legend}</div>'
        f'<div class="scroll"><svg viewBox="0 0 {width} {height}" style="min-width:{width}px" role="img" aria-label="Success rate by epoch and split">'
        f'{"".join(ticks)}{"".join(marks)}</svg></div></section>'
    )


def build_html(
    summaries: list[dict], cohort: dict, epoch_summaries: list[dict] | None = None
) -> str:
    cards = []
    rows = []
    run_charts = []
    overall_rows = []
    for summary in summaries:
        label = html.escape(summary["label"])
        task_chart_rows = []
        overall_n = 0
        overall_successes = 0
        overall_error_free = 0
        overall_dominant = 0
        overall_error_free_dominant = 0
        for split, values in summary["splits"].items():
            micro = values["micro_success_rate"]
            macro = values["macro_success_rate"]
            ci = values["hierarchical_bootstrap_95_ci"]["micro"]
            metrics = values["metrics"]
            error_free = metrics["fsm_error_free_success"]
            dominant = metrics["fsm_dominant_agent_success_ge_0_8"]
            error_free_dominant = metrics[
                "fsm_error_free_dominant_agent_success_ge_0_8"
            ]
            successes = metrics["fsm_success"]["count"]
            overall_n += values["num_episodes"]
            overall_successes += successes
            overall_error_free += error_free["count"]
            overall_dominant += dominant["count"]
            overall_error_free_dominant += error_free_dominant["count"]
            cards.append(
                f'<section class="card"><h3>{label}</h3><p class="split">{html.escape(split)}</p>'
                f'<div class="bar"><span style="width:{100*micro:.3f}%"></span></div>'
                f'<p class="big">{_pct(micro)} <small>FSM success</small></p><p>Micro 95% CI: {_pct(ci[0])}–{_pct(ci[1])}</p>'
                f'<p class="card-metric"><strong>{_pct(error_free["rate"])}</strong> error-free FSM success</p>'
                f'<p class="card-metric"><strong>{_pct(dominant["rate"])}</strong> FSM success + one agent performed ≥80% of task-changing actions</p>'
                f'<p class="card-metric"><strong>{_pct(error_free_dominant["rate"])}</strong> error-free FSM success + one agent performed ≥80% of task-changing actions</p>'
                f'<p>Macro: {_pct(macro)} · n={values["num_episodes"]}</p></section>'
            )
            rows.append(
                f"<tr><td>{label}</td><td>{html.escape(split)}</td>"
                f"<td>{values['num_episodes']}</td><td>{_pct(micro)}</td>"
                f"<td>{_pct(macro)}</td><td>{_pct(ci[0])}–{_pct(ci[1])}</td></tr>"
            )
            per_task = values.get("per_task", {})
            for task, task_values in per_task.items():
                task_chart_rows.append({
                    "task": task,
                    "split": split,
                    "successes": task_values["successes"],
                    "num_episodes": task_values["num_episodes"],
                    "success_rate": task_values["success_rate"],
                    "ci": task_values["wilson_95_ci"],
                    "metrics": task_values.get("metrics"),
                    "errors": task_values.get("errors", {
                        "categories": {
                            key: {
                                "total_events": 0,
                                "recovered_success_events": 0,
                                "failed_trajectory_events": 0,
                                "affected_episodes": 0,
                                "events_per_episode": 0.0,
                                "subtypes": {},
                            }
                            for key, _, _ in _ERROR_SERIES
                        }
                    }),
                })
        task_chart_rows.sort(
            key=lambda row: (_SPLIT_ORDER.get(row["split"], 99), row["task"])
        )
        run_charts.append(
            f'<section class="run"><h2>{label}</h2>'
            + _task_success_chart(task_chart_rows)
            + _task_error_chart(task_chart_rows)
            + '</section>'
        )
        fsm_rate = overall_successes / overall_n if overall_n else 0.0
        error_free_rate = overall_error_free / overall_n if overall_n else 0.0
        dominant_rate = overall_dominant / overall_n if overall_n else 0.0
        error_free_dominant_rate = (
            overall_error_free_dominant / overall_n if overall_n else 0.0
        )
        raw_label = summary["label"]
        model, _, communication = raw_label.partition(" · ")
        mode = communication.removesuffix(" communication") or "full"
        overall_rows.append({
            "model": model,
            "mode": mode,
            "fsm_rate": fsm_rate,
            "fsm_rate_ci": _wilson_interval(overall_successes, overall_n),
            "fsm_rate_counts": (overall_successes, overall_n),
            "error_free_rate": error_free_rate,
            "error_free_rate_ci": _wilson_interval(overall_error_free, overall_n),
            "error_free_rate_counts": (overall_error_free, overall_n),
            "dominant_rate": dominant_rate,
            "dominant_rate_counts": (overall_dominant, overall_n),
            "error_free_dominant_rate": error_free_dominant_rate,
            "error_free_dominant_rate_counts": (overall_error_free_dominant, overall_n),
        })
    contract = cohort.get("contract", {})
    contract_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in contract.items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fixed live-sim model comparison</title><style>
body{{font:16px system-ui,sans-serif;max-width:1180px;margin:40px auto;padding:0 20px;color:#17202a;background:#f6f8fb}}
h1{{margin-bottom:4px}} .muted,.split{{color:#687386}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}}
.card{{background:white;border:1px solid #dfe5ec;border-radius:12px;padding:18px;box-shadow:0 2px 8px #17202a0d}}
.card h3{{margin:0}} .big{{font-size:30px;font-weight:700;margin:8px 0}} .big small{{font-size:13px;font-weight:600;color:#687386}} .card-metric{{margin:8px 0;line-height:1.35}} .bar{{height:12px;background:#e8edf3;border-radius:8px;overflow:hidden}}
.bar span{{display:block;height:100%;background:#276ef1}} table{{border-collapse:collapse;width:100%;background:white;margin:20px 0}}
th,td{{text-align:left;padding:10px;border-bottom:1px solid #e2e7ed}} code{{background:#edf1f5;padding:2px 5px;border-radius:4px}}
.run{{margin:32px 0;padding:24px;background:white;border:1px solid #dfe5ec;border-radius:12px}}
.chart{{margin:28px 0}} .scroll{{overflow-x:auto}} svg{{min-width:760px;width:100%;height:auto}}
svg text{{font:12px system-ui,sans-serif;fill:#263241}} .axis{{fill:#687386}} .value{{font-weight:600}}
.gridline{{stroke:#e6ebf1;stroke-width:1}} .bar-mark.final,.legend-swatch.final{{fill:#3976d2;background:#3976d2}}
.bar-mark.error-free,.legend-swatch.error-free{{fill:#48a9a6;background:#48a9a6}}
.bar-mark.single-agent,.legend-swatch.single-agent{{fill:#e07a5f;background:#e07a5f}}
.bar-mark.dominant-agent,.legend-swatch.dominant-agent{{fill:#f2cc8f;background:#f2cc8f}}
.bar-mark.remainder,.legend-swatch.remainder{{opacity:.42}} .bar-mark.dominant-subset{{opacity:1}}
.legend-swatch.final.dominant-subset{{background:#3976d2}}
.legend-swatch.error-free.dominant-subset{{background:#48a9a6}}
.bar-mark.heldout{{stroke:#17202a;stroke-width:2;stroke-dasharray:4 2}}
.legend-swatch.heldout-outline{{background:white;border:2px dashed #17202a;box-sizing:border-box}}
.epoch-bar-label{{font-size:10px;fill:#687386}}
.error-mark.call-grounding,.legend-swatch.call-grounding{{fill:#3976d2;background:#3976d2}}
.error-mark.state-action,.legend-swatch.state-action{{fill:#e07a5f;background:#e07a5f}}
.error-mark.coordination-progress,.legend-swatch.coordination-progress{{fill:#7c6db0;background:#7c6db0}}
.error-mark.recovered{{opacity:.42}} .error-mark.failed{{opacity:1}} .error-mark.zero{{opacity:.25}}
.legend-swatch.outcome-failed{{background:#27313d}} .legend-swatch.outcome-recovered{{background:#aeb7c2}}
.ci{{stroke:#17202a;stroke-width:1.2}} .point{{fill:#17202a}}
.mean-line{{stroke-width:2;stroke-dasharray:5 4;opacity:.85}}
.mean-line.final{{stroke:#185bb5}} .mean-line.error-free{{stroke:#208b87}} .mean-line.single-agent{{stroke:#bd553d}} .mean-line.dominant-agent{{stroke:#b98529}}
.mean-line.call-grounding{{stroke:#185bb5}} .mean-line.state-action{{stroke:#bd553d}} .mean-line.coordination-progress{{stroke:#665398}}
.split-divider{{stroke:#17202a;stroke-width:2}} .heldout-bg{{fill:#f0f3f8}} .split-label{{font-weight:700}} .task-label{{font-size:11px}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;margin:10px 0 2px}} .legend span{{display:flex;align-items:center;gap:6px}}
.legend-swatch{{display:inline-block;width:13px;height:13px;border-radius:2px}}
</style></head><body>
<h1>Fixed live-sim model comparison</h1>
<p class="muted">One immutable cohort, identical episode configurations per model. Success means concurrent-FSM goal satisfaction.</p>
<div class="grid">{''.join(cards) or '<section class="card">No completed runs yet.</section>'}</div>
{_overall_success_chart(overall_rows)}
{_epoch_split_chart(epoch_summaries or [])}
{''.join(run_charts)}
<h2>Comparable results</h2><table><thead><tr><th>Model</th><th>Split</th><th>Episodes</th><th>Micro</th><th>Macro</th><th>Micro 95% CI</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Evaluation contract</h2><table>{contract_rows}</table>
<p>Cohort hash: <code>{html.escape(str(cohort.get('content_hash')))}</code> · Frozen: <code>{cohort.get('frozen')}</code>. Native RoboCasa success is retained as a diagnostic but is not the primary learning metric.</p>
</body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--summary", type=Path, action="append", default=[])
    parser.add_argument("--epoch-summary", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summary]
    epoch_summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.epoch_summary
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_html(summaries, cohort, epoch_summaries), encoding="utf-8"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

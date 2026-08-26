"""Build the portable visual report for nested place_next_to semantics."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "training/bc_task_vlm/eval_runs/nested_place_next_to_probe"
OUT = ROOT / "training/bc_task_vlm/eval_runs/nested_place_next_to_artifact"


def _data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def _gallery(record: dict) -> str:
    task = record["task"]
    slug = task.lower()
    panels = []
    for phase, phase_label in (("initial", "Initial state"), ("final", "After place_next_to")):
        for view, view_label in (
            ("top", "Top view"),
            ("agent_0_agentview_center", "Agent 0 view"),
            ("agent_1_agentview_center", "Agent 1 view"),
        ):
            path = PROBE / "assets" / f"{slug}_{phase}_{view}.jpg"
            panels.append(
                '<figure style="margin:0;border:1px solid #d8dee8;border-radius:10px;'
                'padding:10px;background:#fff;color:#17202a">'
                f'<img src="{_data_url(path)}" alt="{html.escape(task)} — '
                f'{html.escape(phase_label)} — {html.escape(view_label)}" '
                'style="display:block;width:100%;height:auto;border-radius:6px">'
                f'<figcaption style="margin-top:8px"><strong>{html.escape(phase_label)}</strong>'
                f'<br><span style="color:#596575">{html.escape(view_label)}</span></figcaption>'
                '</figure>'
            )
    return (
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));'
        'gap:12px;padding:4px">' + "".join(panels) + "</div>"
    )


def _fallback_html(records: list[dict], rows: list[dict], generated: str) -> str:
    """Render the same report when the canonical Node packager is unavailable."""

    table_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row[field]))}</td>"
            for field in (
                "task", "placed_object", "reference_object", "symbolic_support",
                "simulator_support", "xy_distance_cm", "physical_containment",
                "native_goal_compatible",
            )
        )
        + "</tr>"
        for row in rows
    )
    descriptions = {
        "GarnishCake": "The strawberry is beside the cake while remaining on the cake plate; this matches native success.",
        "SeasoningSteak": "The shaker is moved onto the steak plate, but native success requires it on the dining counter.",
        "TongBuffetSetup": "The tongs are moved onto the food tray, but native success requires them on the dining counter.",
        "PrepareCheeseStation": "The cheese is moved into/on the salad bowl, but native success requires it on the counter beside the bowl.",
    }
    case_sections = "".join(
        f"<section><h2>{html.escape(record['task'])}</h2>"
        f"<p>{html.escape(descriptions[record['task']])} The physical containment check passed; "
        f"the final center-to-center distance was {100 * record['xy_distance_to_reference_m']:.2f} cm. "
        f"Native-goal compatible: <strong>{'yes' if record['task'] == 'GarnishCake' else 'no'}</strong>.</p>"
        f"{_gallery(record)}</section>"
        for record in records
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark"><title>Nested place_next_to Simulator Validation</title>
<style>
:root{{--bg:#f5f7fa;--card:#fff;--ink:#17202a;--muted:#596575;--line:#d8dee8;--accent:#2457a6}}
@media(prefers-color-scheme:dark){{:root{{--bg:#111827;--card:#1f2937;--ink:#f3f4f6;--muted:#c7d0dc;--line:#475569;--accent:#8bb8ff}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}}
main{{max-width:1220px;margin:auto;padding:32px 22px 72px}} section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;margin:18px 0}}
h1{{font-size:2rem;margin:.2rem 0 1rem}} h2{{margin:.1rem 0 .7rem;color:var(--accent)}} p{{max-width:90ch}}
.result{{font-size:1.14rem}} table{{width:100%;border-collapse:collapse;display:block;overflow:auto}} th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}} th{{color:var(--muted)}}
figure{{background:var(--card)!important;border-color:var(--line)!important;color:var(--ink)!important}} figcaption span{{color:var(--muted)!important}}
code{{font-family:ui-monospace,SFMono-Regular,monospace}} .meta{{color:var(--muted);font-size:.9rem}}
</style></head><body><main>
<h1>Nested <code>place_next_to</code> Simulator Validation</h1>
<section><h2>Technical summary</h2><p class="result"><strong>The proposed global semantics are rejected: only 1/4 probes matched the native task goal.</strong> All four calls physically placed the incoming object on the reference object's immediate support, but three tasks require the moved object on the surrounding counter. The production change has therefore been reverted.</p></section>
<section><h2>Physical execution succeeded, but task semantics did not</h2><p>The containment test shows what the experimental implementation did; native-goal compatibility determines whether that behavior was correct.</p>
<table><thead><tr><th>Task</th><th>Placed object</th><th>Reference</th><th>Expected support</th><th>Simulator support</th><th>Distance (cm)</th><th>Contained</th><th>Native-goal compatible</th></tr></thead><tbody>{table_rows}</tbody></table></section>
{case_sections}
<section><h2>Validation checks the physical result</h2><p>Each probe initialized the real two-agent simulator, picked up the incoming object, and called <code>place_next_to</code> using the nested reference. A case passed only when executor location, direct support-parent bookkeeping, and RoboCasa's physical receptacle/contact check agreed. Both agents' views and a top view were captured before and after.</p></section>
<section><h2>Conclusion</h2><p>Keep the original fixture-surface semantics for <code>place_next_to</code>. GarnishCake should use <code>place_on_object(strawberry, cake_plate)</code>; its wording or shared tool guidance should be clarified rather than globally changing placement behavior.</p></section>
<p class="meta">Generated {html.escape(generated)}. Evidence: nested_place_next_to_probe/probe_results.json.</p>
</main></body></html>"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.loads((PROBE / "probe_results.json").read_text())
    records = payload["records"]
    generated = datetime.now(timezone.utc).isoformat()
    source = {
        "id": "nested_place_next_to_live_probe",
        "label": "Nested movable-support live-simulator probe",
        "path": "training/bc_task_vlm/eval_runs/nested_place_next_to_probe/probe_results.json",
        "query": {
            "description": "Execute object-relative place_next_to in four RoboCasa tasks and verify immediate-support containment.",
            "language": "python",
            "tables_used": ["probe_results.records"],
            "filters": ["layout 11", "style 14", "seed 42", "one episode per task"],
            "metric_definitions": [
                "Probe success requires executor location, support-parent bookkeeping, and physical containment to agree on the resolved immediate movable support.",
                "XY distance is Euclidean distance between the placed object and reference-object centers after placement.",
            ],
            "executed_at": generated,
        },
    }
    rows = [
        {
            "task": r["task"],
            "placed_object": r["object"],
            "reference_object": r["reference"],
            "symbolic_support": r["expected_support"],
            "simulator_support": r["resolved_support_object_id"],
            "xy_distance_cm": round(100 * r["xy_distance_to_reference_m"], 2),
            "physical_containment": r["physical_on_expected_support"],
            "native_goal_compatible": r["task"] == "GarnishCake",
        }
        for r in records
    ]
    blocks = [
        {"id": "title", "type": "markdown", "layout": "full", "body": "# Nested `place_next_to` Simulator Validation"},
        {
            "id": "summary", "type": "markdown", "layout": "full", "sourceId": source["id"],
            "body": (
                "## Technical summary\n\n"
                "**The proposed global semantics are rejected: only 1/4 probes matched native task success.** "
                "Although all four placements physically landed on the immediate movable support, three tasks require the moved item on the surrounding counter. The production change has been reverted."
            ),
        },
        {
            "id": "results_intro", "type": "markdown", "layout": "full", "sourceId": source["id"],
            "body": (
                "## Physical execution succeeded, but task semantics did not\n\n"
                "The four cases show that the experimental implementation reliably moved objects onto immediate supports. Native success, however, requires the surrounding counter in three tasks."
            ),
        },
        {"id": "results_table", "type": "table", "layout": "full", "tableId": "probe_results"},
    ]
    descriptions = {
        "GarnishCake": "The strawberry remains on the cake plate, matching native success.",
        "SeasoningSteak": "The shaker moves onto the steak plate, but native success requires counter contact.",
        "TongBuffetSetup": "The tongs move onto the food tray, but native success requires counter contact.",
        "PrepareCheeseStation": "The cheese moves into/on the salad bowl, but native success requires counter contact beside the bowl.",
    }
    for index, record in enumerate(records):
        blocks.extend(
            [
                {
                    "id": f"case_{index}_intro", "type": "markdown", "layout": "full",
                    "sourceId": source["id"],
                    "body": f"## {record['task']}\n\n{descriptions[record['task']]} Physical placement completed, with a final reference distance of **{100 * record['xy_distance_to_reference_m']:.2f} cm**. Native-goal compatible: **{'yes' if record['task'] == 'GarnishCake' else 'no'}**.",
                },
                {"id": f"case_{index}_gallery", "type": "html", "layout": "full", "body": _gallery(record)},
            ]
        )
    blocks.extend(
        [
            {
                "id": "method", "type": "markdown", "layout": "full", "sourceId": source["id"],
                "body": (
                    "## Validation checks the physical result, not only FSM bookkeeping\n\n"
                    "Each probe initialized the real two-agent simulator, picked up the incoming object, and called `place_next_to` using the nested reference object. Executor location, direct support-parent relation, and RoboCasa physical containment establish what the experimental implementation did. Native `_check_success()` establishes whether that result was actually correct for the task."
                ),
            },
            {
                "id": "limitations", "type": "markdown", "layout": "full",
                "body": (
                    "## What this establishes—and what remains uncertain\n\n"
                    "These probes disprove the proposed global rule: physical nesting is possible, but it conflicts with three native goals. The production change has been reverted. GarnishCake remains a prompt/tool-choice issue rather than justification for changing shared semantics."
                ),
            },
            {
                "id": "next_steps", "type": "markdown", "layout": "full",
                "body": (
                    "## Recommended next steps\n\n"
                    "- Keep `place_next_to` on the resolved fixture surface.\n"
                    "- Clarify that placing an object on a plate uses `place_on_object`, even when it must also be next to something already on that plate.\n"
                    "- Rerun the GarnishCake smoke after that prompt clarification."
                ),
            },
            {
                "id": "questions", "type": "markdown", "layout": "full",
                "body": (
                    "## Further questions\n\n"
                    "The remaining design question is whether tool guidance alone is sufficient or whether GarnishCake’s model-visible instruction should be rewritten as two short, unambiguous placement sentences."
                ),
            },
        ]
    )
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Nested place_next_to Simulator Validation",
            "description": "Live-simulator validation of object-relative placement on immediate movable supports.",
            "generatedAt": generated,
            "sources": [source],
            "tables": [
                {
                    "id": "probe_results",
                    "title": "Live-simulator probe results",
                    "subtitle": "Four representative nested movable-support placements; one scene per task",
                    "dataset": "probe_results",
                    "defaultSort": {"field": "task", "direction": "asc"},
                    "density": "spacious",
                    "sourceId": source["id"],
                    "layout": "full",
                    "columns": [
                        {"field": "task", "label": "Task", "type": "text"},
                        {"field": "placed_object", "label": "Placed object", "type": "text"},
                        {"field": "reference_object", "label": "Reference", "type": "text"},
                        {"field": "symbolic_support", "label": "Expected support", "type": "text"},
                        {"field": "simulator_support", "label": "Resolved simulator support", "type": "text"},
                        {"field": "xy_distance_cm", "label": "Distance (cm)", "format": "number"},
                        {"field": "physical_containment", "label": "Physically contained", "type": "text"},
                        {"field": "native_goal_compatible", "label": "Native-goal compatible", "type": "text"},
                    ],
                }
            ],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {"probe_results": rows},
        },
        "sources": [source],
    }
    (OUT / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (OUT / "report.html").write_text(_fallback_html(records, rows, generated))
    print(OUT / "artifact.json")


if __name__ == "__main__":
    main()

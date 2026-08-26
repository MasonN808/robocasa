"""Build a self-contained technical report for exclusive child-workspace placement."""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import glob
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/exclusive_workspace_map_artifact"
AUDIT = ROOT / "training/bc_task_vlm/eval_runs/exclusive_workspace_focused_ab_v1"
ASSETS = OUT / "assets"


def data_url(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def aggregate() -> tuple[list[dict], int]:
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    files = sorted(glob.glob(str(AUDIT / "*" / "audit.json")))
    for filename in files:
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        arm = Path(filename).parent.name.split("_layout_")[0]
        for row in payload["records"]:
            key = (arm, row["task_name"])
            counts[key]["probes"] += 1
            counts[key]["conflicts"] += bool(row["evidence"].get("detected"))
            counts[key]["errors"] += bool(row.get("error"))
    rows = []
    for (arm, task), values in sorted(counts.items()):
        rows.append({
            "policy": "Original placement" if arm == "baseline" else "Reserved child workspaces",
            "task": task,
            "completed_probes": values["probes"],
            "access_conflicts": values["conflicts"],
            "explicit_placement_failures": values["errors"],
        })
    return rows, len(files)


def map_panel(path: Path, title: str, note: str) -> str:
    return (
        '<figure style="margin:0;border:1px solid #d7dde5;border-radius:10px;padding:12px;'
        'background:#fff;color:#17202a">'
        f'<img src="{data_url(path)}" alt="{html.escape(title)}" '
        'style="display:block;width:100%;height:auto;border-radius:6px">'
        f'<figcaption style="margin-top:10px"><strong>{html.escape(title)}</strong><br>'
        f'<span style="color:#56616f">{html.escape(note)}</span></figcaption></figure>'
    )


def main() -> None:
    rows, completed_files = aggregate()
    examples = [
        (
            ASSETS / "before_top_view.jpg",
            "Before: original placement accepts both robots",
            "Both agents are assigned to the counter. Agent_1 occupies the new toaster-width, aisle-facing corridor shown in red.",
        ),
        (
            ASSETS / "after_top_view.jpg",
            "After: directional policy accepts both robots",
            "Both agents receive valid counter poses outside the narrow red corridor; no robot is omitted and initialization succeeds.",
        ),
        (
            ASSETS / "after_agent_1_navigates_top_view.jpg",
            "Then: agent_1 navigates to the toaster oven",
            "The previously rejected robot now moves into the red interaction corridor successfully while agent_0 remains at its accepted counter pose.",
        ),
        (
            ASSETS / "shared_corridor_agent_1_navigates_top_view.jpg",
            "New: navigation samples the full reserved corridor",
            "With reservation and navigation using the same fixed 46.5 cm lateral span, agent_1 again reaches the toaster successfully. The earlier three panels are retained for direct comparison.",
        ),
    ]
    missing = [str(path) for path, _title, _note in examples if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing report maps: " + ", ".join(missing))
    map_html = (
        '<div style="padding:16px;display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));'
        'gap:16px;background:#f4f6f8">'
        + "".join(map_panel(*item) for item in examples)
        + "</div>"
    )
    generated = datetime.now(timezone.utc).isoformat()
    source_id = "focused_audit"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Exclusive Child-Workspace Placement Audit",
            "description": "Before-and-after top views of the exclusive child-workspace placement rule, with its exact current exclusion zone.",
            "generatedAt": generated,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Exclusive Child-Workspace Placement Audit", "layout": "full"},
                {"id": "summary", "type": "markdown", "sourceId": source_id, "body": (
                    "## Technical summary\n\n"
                    "These views use the same `ArrangeBreadBowl` state and scene: layout 11, style 14, simulator seed 42. "
                    "Under the original policy, agent_1 occupies the toaster's aisle-facing working corridor. "
                    "Under the directional policy, both robots remain placeable: agent_0 is at **(3.074, -1.000)** and agent_1 moves to **(2.612, -1.000)**. "
                    "The red overlay is the exact revised rule: the toaster's full AABB width laterally, extending 0.75 m outward only toward the counter's accessible aisle face. There is no side or rear radius. "
                    f"The focused A/B is still in progress; this snapshot includes **{completed_files} of 150 scene-policy shards**."
                ), "layout": "full"},
                {"id": "maps_intro", "type": "markdown", "body": (
                    "## Placement before and after the change\n\n"
                    "The images are simulator top-camera renders, not occupancy maps. The first two use the same initial state; the third records the earlier navigation behavior. The fourth is newly rendered after making explicit appliance navigation sample the same 46.5 cm lateral span that parent placement reserves."
                ), "layout": "full"},
                {"id": "maps", "type": "html", "body": map_html, "layout": "full"},
                {"id": "ab_intro", "type": "markdown", "sourceId": source_id, "body": (
                    "## Prior radial-policy audit results are retained for context\n\n"
                    "The table and chart below describe the earlier rounded-radius policy, not the newly rendered directional corridor. They explain why that policy was replaced: it removed observed conflicts but caused explicit `ArrangeBreadBowl` placement failures. A fresh focused sweep is required before assigning aggregate rates to the directional policy."
                ), "layout": "full"},
                {"id": "ab_chart", "type": "chart", "chartId": "ab_outcomes", "layout": "full"},
                {"id": "ab_table", "type": "table", "tableId": "ab_results", "layout": "full"},
                {"id": "definitions", "type": "markdown", "body": (
                    "## What the audit measures\n\n"
                    "A probe is an initial symbolic configuration evaluated in a concrete layout, style, and simulator seed. An **access conflict** means the partner cannot navigate to the exclusive child with the suspected blocker present, can navigate after that blocker gives space, and still cannot navigate when only the navigator moves. An **explicit placement failure** means the new hard constraint found no legal parent pose; no resampling occurs."
                ), "layout": "full"},
                {"id": "method", "type": "markdown", "body": (
                    "## Revised placement rule\n\n"
                    "For a countertop appliance, infer the aisle-facing direction from the supporting counter's standable approach faces. Reserve a rectangle whose lateral bounds equal the appliance AABB and whose depth extends 0.75 m outward along that direction. Side and rear counter poses remain legal."
                ), "layout": "full"},
                {"id": "limits", "type": "markdown", "sourceId": source_id, "body": (
                    "## Limitations and robustness\n\n"
                    "The A/B counts are a live partial snapshot and should not be interpreted as final rates. The maps show baseline physical placement because the reserved-policy failure occurs before a complete two-agent state can be rendered. The focused sweep covers `ArrangeBreadBowl` and `SweetenCoffee`; regression checks for the other structurally eligible tasks follow after the geometry is corrected."
                ), "layout": "full"},
                {"id": "next", "type": "markdown", "body": (
                    "## Recommended next steps\n\n"
                    "1. Rerun the focused configuration-level A/B with the directional corridor.\n"
                    "2. Confirm that explicit navigation to each exclusive fixture remains legal.\n"
                    "3. Run deduplicated configuration regression on the other eligible tasks and negative controls.\n"
                    "4. Revisit the 0.75 m corridor depth only if simulator probes show that it is still unnecessarily long."
                ), "layout": "full"},
                {"id": "questions", "type": "markdown", "body": (
                    "## Remaining questions\n\n"
                    "- Should neighboring contiguous counter fixture IDs be interchangeable for robot placement while retaining the same symbolic location?\n"
                    "- How much clearance is actually required to preserve child navigation for each appliance class?\n"
                    "- Do any supported scenes genuinely lack two simultaneous safe poses at the parent surface?"
                ), "layout": "full"},
            ],
            "cards": [],
            "charts": [{
                "id": "ab_outcomes",
                "title": "Observed access conflicts and explicit placement failures",
                "subtitle": f"Partial snapshot from {completed_files} completed scene-policy shards; raw counts are shown because the sweep is incomplete.",
                "showDescription": True,
                "intent": "comparison",
                "question": "Does reserving exclusive child workspaces trade silent access conflicts for explicit initialization failures?",
                "rationale": "A grouped count comparison makes that tradeoff visible; the exact denominators remain in the table below.",
                "type": "bar",
                "dataset": "ab_results",
                "sourceId": source_id,
                "encodings": {
                    "x": {"field": "policy", "type": "nominal", "label": "Placement policy"},
                    "y": {
                        "fields": ["access_conflicts", "explicit_placement_failures"],
                        "type": "quantitative",
                        "format": "number",
                        "label": "Observed probes",
                    },
                },
                "layout": "full",
            }],
            "tables": [{
                "id": "ab_results",
                "title": "Focused A/B progress by task and placement policy",
                "subtitle": f"Partial snapshot from {completed_files} completed scene-policy shards; exact counts, not final rates.",
                "showDescription": True,
                "dataset": "ab_results",
                "sourceId": source_id,
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "policy", "label": "Placement policy", "type": "text"},
                    {"field": "task", "label": "Task", "type": "text"},
                    {"field": "completed_probes", "label": "Completed probes", "format": "number"},
                    {"field": "access_conflicts", "label": "Access conflicts", "format": "number"},
                    {"field": "explicit_placement_failures", "label": "Explicit placement failures", "format": "number"},
                ],
            }],
            "sources": [{
                "id": source_id,
                "label": "Focused exclusive-workspace A/B audit and geometry maps",
                "path": "training/bc_task_vlm/eval_runs/exclusive_workspace_focused_ab_v1",
                "query": {
                    "sql": "SELECT policy, task, COUNT(*) AS completed_probes, SUM(access_conflict) AS access_conflicts, SUM(placement_error) AS explicit_placement_failures FROM focused_audit_records GROUP BY policy, task ORDER BY policy, task;",
                    "description": "Aggregate each completed focused-audit probe by placement policy and task.",
                    "tables_used": ["focused_audit_records"],
                    "filters": ["Only completed audit.json shards present when the artifact was generated"],
                    "metric_definitions": {
                        "completed_probes": "Count of parsed probe records.",
                        "access_conflicts": "Count where evidence.detected is true.",
                        "explicit_placement_failures": "Count where the probe records a placement error.",
                    },
                },
            }],
        },
        "snapshot": {"version": 1, "generatedAt": generated, "status": "ready", "datasets": {"ab_results": rows}},
        "sources": [{
            "id": source_id,
            "label": "Focused exclusive-workspace A/B audit and geometry maps",
            "path": "training/bc_task_vlm/eval_runs/exclusive_workspace_focused_ab_v1",
            "query": {
                "sql": "SELECT policy, task, COUNT(*) AS completed_probes, SUM(access_conflict) AS access_conflicts, SUM(placement_error) AS explicit_placement_failures FROM focused_audit_records GROUP BY policy, task ORDER BY policy, task;",
                "description": "Aggregate each completed focused-audit probe by placement policy and task.",
                "tables_used": ["focused_audit_records"],
                "filters": ["Only completed audit.json shards present when the artifact was generated"],
                "metric_definitions": {
                    "completed_probes": "Count of parsed probe records.",
                    "access_conflicts": "Count where evidence.detected is true.",
                    "explicit_placement_failures": "Count where the probe records a placement error.",
                },
            },
        }],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    (OUT / "source_notes.md").write_text(
        "# Source and chart notes\n\n"
        "Audience: technical. Required report sections are all present; implications are integrated into findings and next steps.\n\n"
        "No quantitative chart is used because exact partial audit counts are better represented as a table, while the user's primary request is spatial map inspection. The embedded maps are the primary visual evidence.\n",
        encoding="utf-8",
    )
    print(OUT / "artifact.json")


if __name__ == "__main__":
    main()

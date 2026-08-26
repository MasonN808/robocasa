"""Build a self-contained report for one SweetenCoffee corridor failure."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/sweeten_coffee_workspace_artifact"
ASSETS = OUT / "assets"
SOURCE_ID = "directional_audit"
WIDE_SOURCE_ID = "shared_corridor_wide_audit"


def data_url(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def image_panel(path: Path, title: str, note: str) -> str:
    return (
        '<figure style="margin:0;border:1px solid #d7dde5;border-radius:10px;padding:12px;'
        'background:#fff;color:#17202a">'
        f'<img src="{data_url(path)}" alt="{html.escape(title)}" '
        'style="display:block;width:100%;height:auto;border-radius:6px">'
        f'<figcaption style="margin-top:10px"><strong>{html.escape(title)}</strong><br>'
        f'<span style="color:#56616f">{html.escape(note)}</span></figcaption></figure>'
    )


def main() -> None:
    metadata = json.loads((ASSETS / "example_metadata.json").read_text())
    layout50_metadata = json.loads((ASSETS / "layout50_issue_metadata.json").read_text())
    before = ASSETS / "before_top_view.jpg"
    after = ASSETS / "after_top_view.jpg"
    shared = ASSETS / "shared_corridor_navigator_navigates_top_view.jpg"
    layout50_blocked = ASSETS / "layout50_accepted_blocker_top_view.jpg"
    layout50_released = ASSETS / "layout50_after_give_space_navigation_top_view.jpg"
    for path in (before, after, shared, layout50_blocked, layout50_released):
        if not path.exists():
            raise FileNotFoundError(path)

    image_html = (
        '<div style="padding:16px;display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));'
        'gap:16px;background:#f4f6f8">'
        + image_panel(
            before,
            "Before: original placement",
            "Agent_0 stands just left of the red coffee-machine corridor and blocks agent_1's direct navigation.",
        )
        + image_panel(
            after,
            "After: fixed 46.5 cm corridor",
            "The same two poses are still accepted because agent_0's center remains about 4.5 cm beyond the fixed corridor's lateral edge.",
        )
        + image_panel(
            shared,
            "New: agent_1 navigates using the full counter corridor",
            "The old placement remains visible in the first two panels. With navigation candidates expanded to the same 46.5 cm lateral span, agent_1 reaches a safe offset beside agent_0.",
        )
        + "</div>"
    )
    layout50_html = (
        '<div style="padding:16px;display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));'
        'gap:16px;background:#f4f6f8">'
        + image_panel(
            layout50_blocked,
            "Layout 50: parent placement incorrectly accepts the blocker",
            "Agent_0 is symbolically at the counter but physically stands inside the red coffee-machine corridor. Agent_1's direct navigation fails.",
        )
        + image_panel(
            layout50_released,
            "Layout 50: giving space restores access",
            "After agent_0 gives space, agent_1 reaches the coffee machine. This counterfactual confirms that agent_0's accepted pose—not corridor width—is the obstruction.",
        )
        + "</div>"
    )

    rows = [
        {
            "policy": "Original placement",
            "completed_probes": 14,
            "access_conflicts": 5,
            "conflict_rate": 5 / 14,
            "layout11_conflict_rate": 2 / 10,
            "layout40_conflict_rate": 3 / 4,
            "placement_failures": 0,
        },
        {
            "policy": "Fixed 46.5 cm corridor",
            "completed_probes": 14,
            "access_conflicts": 3,
            "conflict_rate": 3 / 14,
            "layout11_conflict_rate": 0.0,
            "layout40_conflict_rate": 3 / 4,
            "placement_failures": 0,
        },
    ]
    position_rows = []
    for stage_key, stage_label in (("before_positions", "Before"), ("after_positions", "After")):
        for index, position in enumerate(metadata[stage_key]):
            position_rows.append({
                "stage": stage_label,
                "agent": f"agent_{index}",
                "x_m": position[0],
                "y_m": position[1],
                "role": (
                    "blocker" if f"agent_{index}" == metadata["blocker_agent"]
                    else "navigator"
                ),
            })
    wide_rows = [
        {"task": "ArrangeBreadBowl", "policy": "Reservation off", "probes": 180, "conflicts": 8, "conflict_rate": 8 / 180, "placement_failures": 1},
        {"task": "ArrangeBreadBowl", "policy": "Shared corridor", "probes": 180, "conflicts": 0, "conflict_rate": 0.0, "placement_failures": 0},
        {"task": "SweetenCoffee", "policy": "Reservation off", "probes": 300, "conflicts": 41, "conflict_rate": 41 / 300, "placement_failures": 0},
        {"task": "SweetenCoffee", "policy": "Shared corridor", "probes": 300, "conflicts": 21, "conflict_rate": 21 / 300, "placement_failures": 0},
    ]

    generated = datetime.now(timezone.utc).isoformat()
    source = {
        "id": SOURCE_ID,
        "label": "Directional exclusive-workspace audit, SweetenCoffee scene and probe outputs",
        "path": "training/bc_task_vlm/eval_runs/exclusive_workspace_fixed465_smoke_v1",
        "query": {
            "sql": "SELECT policy, scene_id, COUNT(*) AS completed_probes, SUM(access_conflict) AS access_conflicts, AVG(CAST(access_conflict AS DOUBLE)) AS conflict_rate, SUM(placement_failure) AS placement_failures FROM fixed465_smoke_records WHERE task_name IN ('arrange_bread_bowl', 'sweeten_coffee') GROUP BY policy, scene_id ORDER BY policy, scene_id;",
            "description": "Aggregate the targeted fixed-46.5-cm smoke probes by placement policy and scene.",
            "tables_used": ["fixed465_smoke_records"],
            "filters": ["Layout 11/style 14/seed 42 and layout 40/style 34/seed 99", "Baseline and fixed 46.5 cm policy"],
            "metric_definitions": {
                "access_conflicts": "Probe satisfies the navigation/give-space counterfactual identifying a blocking parent placement.",
                "conflict_rate": "Access conflicts divided by completed probes.",
                "placement_failures": "Initial-state construction rejects a robot because no legal pose exists.",
            },
        },
    }
    wide_source = {
        "id": WIDE_SOURCE_ID,
        "label": "Two-task shared-corridor scene sweep",
        "path": "training/bc_task_vlm/eval_runs/exclusive_workspace_shared_corridor_ab_v1",
        "query": {
            "sql": "SELECT task_name, policy, COUNT(*) AS probes, SUM(access_conflict) AS conflicts, AVG(CAST(access_conflict AS DOUBLE)) AS conflict_rate, SUM(placement_failure) AS placement_failures FROM shared_corridor_audit_records GROUP BY task_name, policy ORDER BY task_name, policy;",
            "description": "Aggregate all completed two-task shared-corridor probes by task and policy.",
            "tables_used": ["shared_corridor_audit_records"],
            "filters": ["25 layout/style pairs", "Three simulator seeds", "Task-supported scenes only"],
            "metric_definitions": {
                "access_conflict": "Partner navigation fails with the blocker present, succeeds after the blocker gives space, and still fails when only the navigator first moves away.",
                "placement_failure": "Initial-state construction cannot assign a requested robot pose.",
            },
        },
    }
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "SweetenCoffee Corridor: Layout-50 Parent-Matching Gap",
            "description": "Simulator-backed comparisons showing where the shared 46.5 cm corridor works and why layout 50 still accepts blocking parent-counter poses.",
            "generatedAt": generated,
            "sources": [source, wide_source],
            "cards": [],
            "charts": [{
                "id": "conflict_comparison",
                "title": "SweetenCoffee access conflicts by placement policy",
                "subtitle": "Targeted smoke: 10 layout-11 probes and four harder layout-40 probes per policy.",
                "showDescription": True,
                "intent": "comparison",
                "question": "Does the fixed 46.5 cm directional corridor prevent access conflicts in both inspected scenes?",
                "rationale": "A direct rate comparison shows whether the policy materially changes conflict frequency.",
                "type": "bar",
                "dataset": "policy_results",
                "sourceId": SOURCE_ID,
                "encodings": {
                    "x": {"field": "policy", "type": "nominal", "label": "Placement policy"},
                    "y": {"fields": ["layout11_conflict_rate", "layout40_conflict_rate"], "type": "quantitative", "format": "percent", "label": "Conflict rate"},
                },
                "layout": "full",
            }],
            "tables": [{
                "id": "positions",
                "title": "Robot center positions in the displayed example",
                "subtitle": "Layout 40, style 34, seed 99; positions are in world meters.",
                "showDescription": True,
                "dataset": "positions",
                "sourceId": SOURCE_ID,
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "stage", "label": "Stage", "type": "text"},
                    {"field": "agent", "label": "Agent", "type": "text"},
                    {"field": "role", "label": "Role", "type": "text"},
                    {"field": "x_m", "label": "X (m)", "format": "number"},
                    {"field": "y_m", "label": "Y (m)", "format": "number"},
                ],
            }, {
                "id": "wide_results",
                "title": "Wider two-task audit results",
                "subtitle": "Task-supported scenes across 25 layout/style pairs and three simulator seeds.",
                "showDescription": True,
                "dataset": "wide_results",
                "sourceId": WIDE_SOURCE_ID,
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "task", "label": "Task", "type": "text"},
                    {"field": "policy", "label": "Policy", "type": "text"},
                    {"field": "probes", "label": "Probes", "format": "number"},
                    {"field": "conflicts", "label": "Conflicts", "format": "number"},
                    {"field": "conflict_rate", "label": "Conflict rate", "format": "percent"},
                    {"field": "placement_failures", "label": "Placement failures", "format": "number"},
                ],
            }],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# SweetenCoffee Corridor: Layout-50 Parent-Matching Gap", "layout": "full"},
                {"id": "summary", "type": "markdown", "sourceId": WIDE_SOURCE_ID, "body": (
                    "## Technical summary\n\n"
                    "Making parent placement and appliance navigation share the **46.5 cm** corridor completely resolves `ArrangeBreadBowl` (**8/180 → 0/180 conflicts**) and removes 20 of 41 `SweetenCoffee` conflicts. The remaining **21/300** coffee conflicts occur only in layout 50. There, a parent-counter robot is accepted visibly inside the intended corridor, indicating a layout-specific parent/child fixture-matching gap rather than insufficient width."
                ), "layout": "full"},
                {"id": "visual_intro", "type": "markdown", "body": (
                    "## Earlier layout-40 evidence motivated the shared navigation span\n\n"
                    "These retained panels show the earlier reservation-only result. The red overlay is the exact tested corridor: **46.5 cm total width**, centered on the coffee machine and extending 0.75 m outward. The first two views use the same state and camera; the third adds the subsequently implemented shared navigation span."
                ), "layout": "full"},
                {"id": "images", "type": "html", "body": image_html, "layout": "full"},
                {"id": "shared_result", "type": "markdown", "sourceId": SOURCE_ID, "body": (
                    "## The shared-corridor change resolves the known hard scene\n\n"
                    "The earlier **reservation-only** experiment remains above for comparison: layout 40 stayed at 3/4 conflicts. After explicit appliance navigation was allowed to sample across the same 46.5 cm corridor, the targeted shared-corridor smoke found **0/4 conflicts in layout 40** and **0/10 in layout 11**, with no placement errors. The displayed new panel is one of those successful navigation states."
                ), "layout": "full"},
                {"id": "layout50_intro", "type": "markdown", "sourceId": WIDE_SOURCE_ID, "body": (
                    "## Layout 50 bypasses the intended reservation\n\n"
                    "This concrete case uses layout 50, style 34, seed 7. Agent_0 starts at **(5.287, -3.318)** and is accepted as a broad `counter` placement even though that point lies inside the red corridor. Agent_1's direct coffee-machine navigation fails. After agent_0 gives space, agent_1 reaches the machine at **(5.356, -3.324)**."
                ), "layout": "full"},
                {"id": "layout50_images", "type": "html", "body": layout50_html, "layout": "full"},
                {"id": "wide_results_intro", "type": "markdown", "sourceId": WIDE_SOURCE_ID, "body": (
                    "## The failure is isolated to SweetenCoffee layout 50\n\n"
                    "The broader audit found no reserved-policy conflict in any `ArrangeBreadBowl` scene and no remaining `SweetenCoffee` conflict in layouts 11, 15, 18, or 40. All 21 residual conflicts came from seven layout-50 style/seed scenes."
                ), "layout": "full"},
                {"id": "wide_results_table", "type": "table", "tableId": "wide_results", "layout": "full"},
                {"id": "example_result", "type": "markdown", "sourceId": SOURCE_ID, "body": (
                    "## The retained layout-40 example was a verified access conflict\n\n"
                    "In the earlier reservation-only implementation, agent_1's direct navigation failed with agent_0 present and succeeded after agent_0 called `give_space`. Expanding appliance navigation across the reserved span subsequently resolved this scene; the old evidence remains here to show why that change was made."
                ), "layout": "full"},
                {"id": "positions_block", "type": "table", "tableId": "positions", "layout": "full"},
                {"id": "aggregate_intro", "type": "markdown", "sourceId": SOURCE_ID, "body": (
                    "## Historical reservation-only smoke\n\n"
                    "Before navigation shared the corridor span, conflicts fell from 5/14 to 3/14, with layout 40 unchanged at 3/4. This historical comparison is intentionally retained; the newer shared-corridor smoke reduced both inspected layouts to zero conflicts."
                ), "layout": "full"},
                {"id": "conflict_chart", "type": "chart", "chartId": "conflict_comparison", "layout": "full"},
                {"id": "definitions", "type": "markdown", "body": (
                    "## Scope and definitions\n\n"
                    "A probe combines a task initial state with a concrete layout, style, and simulator seed. An access conflict requires the navigation/give-space counterfactual described above; it is not inferred from image overlap. The retained historical case uses layout 40/style 34/seed 99. The new residual case uses layout 50/style 34/seed 7; both use case `80aae4ea7d4246bf`."
                ), "layout": "full"},
                {"id": "method", "type": "markdown", "body": (
                    "## The remaining problem is reservation attachment, not width\n\n"
                    "The layout-50 blocker lies within both the corridor's lateral bounds and its 0.75 m forward depth, yet parent placement accepts it. Widening the corridor cannot fix a constraint that is not being attached to that concrete parent counter. The next diagnostic should inspect how symbolic `counter`, the resolved scene counter, and the coffee machine's supporting fixture are matched in layout 50."
                ), "layout": "full"},
                {"id": "limitations", "type": "markdown", "sourceId": WIDE_SOURCE_ID, "body": (
                    "## Limitations and robustness\n\n"
                    "The wider sweep covers the two tasks that originally surfaced the issue, not every task with an exclusive child fixture. It establishes that the current implementation handles the tested toaster cases and that coffee residuals cluster in layout 50; it does not yet establish the exact code-level cause or global safety across unrelated tasks."
                ), "layout": "full"},
                {"id": "next", "type": "markdown", "body": (
                    "## Recommended next steps\n\n"
                    "1. Trace the symbolic-to-concrete parent fixture mapping for the seven failing layout-50 scenes.\n"
                    "2. Fix that attachment without changing the 46.5 cm geometry.\n"
                    "3. Rerun the 21 residual cases and a sample of the 459 already-passing reserved-policy probes.\n"
                    "4. If those pass, extend the audit to other structurally eligible tasks and negative controls."
                ), "layout": "full"},
                {"id": "questions", "type": "markdown", "body": (
                    "## Further question\n\n"
                    "Does layout 50 resolve the symbolic `counter` to a different fixture object or fixture ID than the coffee machine declares as its supporting parent? Answering that should identify the narrowest safe implementation fix."
                ), "layout": "full"},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {"policy_results": rows, "positions": position_rows, "wide_results": wide_rows},
        },
        "sources": [source, wide_source],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (OUT / "source_notes.md").write_text(
        "# Source notes\n\nTechnical report structure is complete. The before/after top views are the primary spatial evidence; the native bar chart supplies aggregate context. The stopped partial cohort is explicitly labeled.\n"
    )
    print(OUT / "artifact.json")


if __name__ == "__main__":
    main()

"""Build a portable report for implicit versus explicit parent access."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "training/bc_task_vlm/eval_runs/arrange_bread_parent_access_probe"
EXPANDED_PROBE = ROOT / "training/bc_task_vlm/eval_runs/parent_access_retain_exclusive_pose_probe"
COUNTER_PROBE = ROOT / "training/bc_task_vlm/eval_runs/both_counter_to_cabinet_probe"
OUT = ROOT / "training/bc_task_vlm/eval_runs/arrange_bread_parent_access_artifact"


def _data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def _panel(name: str, title: str, note: str) -> str:
    path = PROBE / "assets" / name
    return (
        '<figure style="margin:0;border:1px solid #d7dde5;border-radius:10px;'
        'padding:12px;background:#fff;color:#17202a">'
        f'<img src="{_data_url(path)}" alt="{html.escape(title)}" '
        'style="display:block;width:100%;height:auto;border-radius:6px">'
        f'<figcaption style="margin-top:9px"><strong>{html.escape(title)}</strong><br>'
        f'<span style="color:#56616f">{html.escape(note)}</span></figcaption></figure>'
    )


def _path_panel(path: Path, title: str, note: str) -> str:
    return (
        '<figure style="margin:0;border:1px solid #d7dde5;border-radius:10px;padding:12px;background:#fff;color:#17202a">'
        f'<img src="{_data_url(path)}" alt="{html.escape(title)}" style="display:block;width:100%;height:auto;border-radius:6px">'
        f'<figcaption style="margin-top:9px"><strong>{html.escape(title)}</strong><br>'
        f'<span style="color:#56616f">{html.escape(note)}</span></figcaption></figure>'
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = json.loads((PROBE / "probe_results.json").read_text())
    branches = {row["branch"]: row for row in result["branches"]}
    implicit = branches["implicit_parent_access"]
    explicit = branches["explicit_navigation"]
    expanded = json.loads((EXPANDED_PROBE / "probe_results.json").read_text())
    expanded_rows = []
    expanded_panels = []
    manipulation_by_scenario = {
        "arrange_bread_place_in_receptacle": "place_in_receptacle",
        "arrange_tea_place_in_receptacle": "place_in_receptacle",
        "gather_marinade_place_next_to": "place_next_to",
        "spicy_marinade_place_on_surface": "place_on_surface",
    }
    for record in expanded["records"]:
        actor = "agent_0" if record["scenario"].startswith(("arrange_bread", "arrange_tea")) else "agent_1"
        partner = "agent_1" if actor == "agent_0" else "agent_0"
        expanded_rows.append({
            "task": record["task"],
            "manipulation": manipulation_by_scenario[record["scenario"]],
            "branch": record["branch"],
            "actor_final": str(record.get("final_positions", {}).get(actor)),
            "partner_final": str(record.get("final_positions", {}).get(partner)),
            "partner_setup_displacement_m": round(math.dist(
                record["initial_positions"][partner][:2],
                record["positions_after_setup"][partner][:2],
            ), 3),
            "clean_parent_occupancy_test": math.dist(
                record["initial_positions"][partner][:2],
                record["positions_after_setup"][partner][:2],
            ) < 0.1,
            "sim_success": record["success"],
        })
    records_by_scenario = {}
    for record in expanded["records"]:
        records_by_scenario.setdefault(record["scenario"], {})[record["branch"]] = record
    for scenario, branch_records in records_by_scenario.items():
        task = branch_records["implicit_parent_access"]["task"]
        frames = [
            (
                f"{scenario}_implicit_parent_access_initial.jpg",
                "1. Initial state",
                "Active agent starts at the exclusive child fixture; partner starts at its parent counter.",
            ),
            (
                f"{scenario}_explicit_parent_navigation_after_navigation.jpg",
                "2. After explicit navigate(parent)",
                "No placement yet: this isolates the result of explicit parent navigation.",
            ),
            (
                f"{scenario}_implicit_parent_access_final.jpg",
                "3. After direct manipulation",
                "No parent navigation call: the child-side agent directly performs the parent-targeted manipulation.",
            ),
        ]
        panels = []
        for filename, title, note in frames:
            path = EXPANDED_PROBE / "assets" / filename
            panels.append(_path_panel(path, title, note))
        implicit_record = branch_records["implicit_parent_access"]
        explicit_record = branch_records["explicit_parent_navigation"]
        camera_groups = [
            ("Initial active-agent view", implicit_record.get("active_agent_views_initial")),
        ]
        for opened in implicit_record.get("active_agent_views_after_open") or []:
            camera_groups.append(("After opening from retained pose", opened))
        camera_groups.extend([
            ("After explicit navigate(parent)", explicit_record.get("active_agent_views_after_parent_navigation")),
            ("After direct manipulation", implicit_record.get("active_agent_views_final")),
        ])
        for label, view_paths in camera_groups:
            for view_name in ("agentview_center", "wrist"):
                if not view_paths or not view_paths.get(view_name):
                    continue
                panels.append(_path_panel(
                    Path(view_paths[view_name]),
                    f"{label} — {view_name}",
                    "Active agent camera; inspect cabinet/appliance and object visibility.",
                ))
        expanded_panels.append(
            f'<section style="grid-column:1/-1;margin:0;background:#eef2f7;border-radius:10px;padding:12px"><h3 style="margin:0">{html.escape(task)}</h3></section>'
            + "".join(panels)
        )
    expanded_gallery = '<div style="padding:12px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;background:#f4f6f8">' + "".join(expanded_panels) + "</div>"
    counter = json.loads((COUNTER_PROBE / "probe_results.json").read_text())
    counter_rows = []
    counter_panels = []
    for record in counter["records"]:
        actor = record["actor"]
        partner = "agent_1" if actor == "agent_0" else "agent_0"
        displacement = math.dist(
            record["initial_positions"][partner][:2],
            record["final_positions"][partner][:2],
        )
        counter_rows.append({
            "task": record["task"], "mode": record["mode"],
            "navigation_success": record["navigation_success"],
            "partner_displacement_m": round(displacement, 3),
        })
        slug = "gather_marinade_place_next_to" if record["task"] == "GatherMarinadeIngredients" else "spicy_marinade_place_on_surface"
        for phase, label in (("initial", "Both agents at counter"), ("after_navigation", "After navigate(cabinet)")):
            counter_panels.append(_path_panel(
                COUNTER_PROBE / "assets" / f'{slug}_{record["mode"]}_{phase}.jpg',
                f'{record["task"]} — {record["mode"]} — {label}',
                "Compare whether the stationary partner remains at the counter.",
            ))
    counter_gallery = '<div style="padding:12px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;background:#f4f6f8">' + "".join(counter_panels) + "</div>"
    rows = [
        {
            "branch": "No explicit counter navigation",
            "tool_sequence": "open toaster → pick bread → place bread in bowl",
            "after_pickup": "(3.417, -0.834)",
            "after_navigation": "Not invoked",
            "after_placement": "(2.843, -1.000)",
            "sim_success": implicit["success"],
        },
        {
            "branch": "Explicit counter navigation",
            "tool_sequence": "open toaster → pick bread → navigate counter → place bread in bowl",
            "after_pickup": "(3.417, -0.834)",
            "after_navigation": "(3.069, -1.000)",
            "after_placement": "(2.843, -1.000)",
            "sim_success": explicit["success"],
        },
    ]
    gallery = '<div style="padding:12px;display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px;background:#f4f6f8">' + "".join([
        _panel("implicit_parent_access_after_pickup.jpg", "Shared starting point: after toaster pickup", "Agent 0 is at the toaster working pose and holds the bread."),
        _panel("implicit_parent_access_after_placement.jpg", "No navigation: after placement", "place_in_receptacle moves agent 0 to the bowl-side counter pose."),
        _panel("explicit_navigation_after_navigation.jpg", "Explicit branch: after navigate(counter)", "The extra navigation first moves agent 0 to the broad counter pose."),
        _panel("explicit_navigation_after_placement.jpg", "Explicit branch: after placement", "The final pose exactly matches the no-navigation branch."),
    ]) + "</div>"
    generated = datetime.now(timezone.utc).isoformat()
    source = {
        "id": "arrange_bread_live_probe",
        "label": "ArrangeBreadBowl parent-access live-simulator probe",
        "path": "training/bc_task_vlm/eval_runs/arrange_bread_parent_access_probe/probe_results.json",
        "query": {
            "description": "Compare identical simulator branches with and without navigate_to_fixture(counter) before placing toaster bread into the bowl.",
            "language": "python",
            "tables_used": ["branches"],
            "filters": ["ArrangeBreadBowl run 6", "layout 11", "style 14", "seed 42"],
            "metric_definitions": {"sim_success": "All executor calls returned successfully and the bread was placed in the bowl."},
        },
    }
    expanded_source = {
        "id": "occupied_parent_live_probe",
        "label": "Partner-occupied parent cross-tool simulator probe",
        "path": "training/bc_task_vlm/eval_runs/parent_access_retain_exclusive_pose_probe/probe_results.json",
        "query": {
            "description": "Compare implicit and explicit parent navigation while the other robot occupies the parent counter.",
            "language": "python", "tables_used": ["records"],
            "filters": ["layout 11", "style 14", "seed 42", "partner initialized at parent fixture"],
            "metric_definitions": {"sim_success": "All setup and manipulation executor calls completed successfully."},
        },
    }
    counter_source = {
        "id": "both_counter_to_cabinet_probe",
        "label": "Both-at-counter navigation-to-cabinet simulator probe",
        "path": "training/bc_task_vlm/eval_runs/both_counter_to_cabinet_probe/probe_results.json",
        "query": {
            "description": "Compare cabinet navigation with silent teammate clearing enabled versus disabled when both agents start at the parent counter.",
            "language": "python", "tables_used": ["records"],
            "filters": ["GatherMarinadeIngredients", "SpicyMarinade", "layout 11", "style 14", "seed 42"],
            "metric_definitions": {"partner_displacement_m": "Euclidean XY displacement of the non-navigating robot."},
        },
    }
    blocks = [
        {"id": "title", "type": "markdown", "body": "# ArrangeBreadBowl: Parent-Access Simulator Probe", "layout": "full"},
        {"id": "summary", "type": "markdown", "sourceId": source["id"], "body": (
            "## Technical summary\n\n"
            "**Both branches succeeded.** Omitting `navigate_to_fixture(counter)` before `place_in_receptacle(bread, bowl)` is supported by the current simulator executor. The placement moved agent 0 from the toaster pose `(3.417, -0.834)` to `(2.843, -1.000)`, exactly the same final pose as the explicit-navigation branch. Therefore this specific placement should update the symbolic location from `toaster_oven` to its parent `counter`."
        ), "layout": "full"},
        {"id": "expanded_intro", "type": "markdown", "sourceId": expanded_source["id"], "body": (
            "## Retaining an existing exclusive-fixture pose works in all four tasks\n\n"
            "With silent blocker clearing disabled and the active agent's existing exclusive-fixture pose retained, all **8/8** implicit/explicit branches succeeded. The partner remained at the parent counter. The top views show geometry; the active-agent center and wrist views show whether the cabinet/appliance and relevant object are visible."
        ), "layout": "full"},
        {"id": "expanded_gallery", "type": "html", "body": expanded_gallery, "layout": "full"},
        {"id": "expanded_results", "type": "table", "tableId": "expanded_results", "layout": "full"},
        {"id": "counter_intro", "type": "markdown", "sourceId": counter_source["id"], "body": (
            "## Entering the cabinet from a shared counter still requires a real handover\n\n"
            "When both agents start at the counter, current silent clearing makes cabinet navigation succeed by physically moving the partner about 0.72 m without a partner tool call. With silent clearing disabled, cabinet navigation fails and neither robot moves. Retaining an existing cabinet pose therefore solves only the already-at-cabinet case; it does not make a blocked counter-to-cabinet transition legal."
        ), "layout": "full"},
        {"id": "counter_gallery", "type": "html", "body": counter_gallery, "layout": "full"},
        {"id": "counter_results", "type": "table", "tableId": "counter_results", "layout": "full"},
        {"id": "interpretation", "type": "markdown", "sourceId": source["id"], "body": (
            "## Recommended semantic rule\n\n"
            "Use a two-part rule: retain a verified usable pose when an agent is already at an exclusive child fixture, but require an explicit `give_space` handover when an agent at the parent counter cannot enter the child fixture without displacing its partner. Do not silently move the partner, and do not declare every parent counter globally exclusive."
        ), "layout": "full"},
        {"id": "risks", "type": "markdown", "body": (
            "## Side effects to guard against\n\n"
            "1. **Do not make parent and child locations interchangeable.** Child → parent access may be safe for a tool that moves to a parent object; parent → exclusive child would bypass explicit appliance access and handovers.\n"
            "2. **Commit the location transition.** After this placement, later toaster work must require navigating back to `toaster_oven`.\n"
            "3. **Claim the destination during the atomic tick.** The action should contend for the bowl/object and parent workspace exactly as an explicit parent-side placement would.\n"
            "4. **Limit the first implementation to executor-backed transitions.** Other manipulation tools may reposition differently; do not globally permit every child-fixture action against every parent object without probes.\n"
            "5. **Revalidate old trajectories.** The rule broadens the current call, but the new post-call symbolic location can change whether later calls are legal."
        ), "layout": "full"},
        {"id": "method", "type": "markdown", "sourceId": source["id"], "body": (
            "## Method and scope\n\n"
            "Both branches used a fresh simulator initialized with ArrangeBreadBowl run 6, layout 11, style 14, and seed 42. Each opened the toaster and picked up the same bread. One branch immediately placed it in the bowl; the other first navigated to the parent counter. Top views and robot coordinates were recorded after each decisive call."
        ), "layout": "full"},
        {"id": "limits", "type": "markdown", "body": (
            "## Limitations\n\n"
            "This proves the behavior for `place_in_receptacle` in this concrete scene. It does not yet prove that every appliance-parent pair or every manipulation tool makes the same safe physical transition. A small tool-family regression probe is needed before making the rule fully generic."
        ), "layout": "full"},
        {"id": "next", "type": "markdown", "body": (
            "## Next steps\n\n"
            "1. Implement a shared concurrent-FSM transition for executor-backed child → immediate-parent manipulation.\n"
            "2. Add tests for location update, destination contention, and the asymmetric parent → exclusive-child rejection.\n"
            "3. Probe representative placement/pickup tool families before widening the rule.\n"
            "4. Then replace the current ArrangeBreadBowl-specific navigation reminder with the general rule in generation and policy prompts."
        ), "layout": "full"},
        {"id": "questions", "type": "markdown", "body": "## Remaining question\n\nShould implicit parent access initially cover only movable receptacles such as `bowl`, or every manipulation whose executor resolves and moves to an immediate parent fixture? The conservative first option is easier to certify.", "layout": "full"},
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": "ArrangeBreadBowl Parent-Access Probe",
            "description": "Simulator-backed comparison of implicit and explicit parent-counter access.",
            "generatedAt": generated, "blocks": blocks, "cards": [], "charts": [],
            "tables": [{
                "id": "branch_results", "title": "Simulator branch comparison",
                "subtitle": "Identical scene and task state; only explicit counter navigation differs.",
                "dataset": "branch_results", "sourceId": source["id"], "density": "spacious", "layout": "full",
                "columns": [
                    {"field": "branch", "label": "Branch", "type": "text"},
                    {"field": "tool_sequence", "label": "Tool sequence", "type": "text"},
                    {"field": "after_pickup", "label": "Pose after pickup", "type": "text"},
                    {"field": "after_navigation", "label": "Pose after navigation", "type": "text"},
                    {"field": "after_placement", "label": "Pose after placement", "type": "text"},
                    {"field": "sim_success", "label": "Simulator success", "type": "boolean"},
                ],
            }, {
                "id": "expanded_results", "title": "Parent occupied: cross-task and cross-tool results",
                "subtitle": "Four scenarios × implicit/explicit navigation; partner occupies the parent fixture.",
                "dataset": "expanded_results", "sourceId": expanded_source["id"], "density": "spacious", "layout": "full",
                "columns": [
                    {"field": "task", "label": "Task", "type": "text"},
                    {"field": "manipulation", "label": "Manipulation", "type": "text"},
                    {"field": "branch", "label": "Branch", "type": "text"},
                    {"field": "actor_final", "label": "Actor final pose", "type": "text"},
                    {"field": "partner_final", "label": "Partner final pose", "type": "text"},
                    {"field": "partner_setup_displacement_m", "label": "Partner moved during setup (m)", "format": "number"},
                    {"field": "clean_parent_occupancy_test", "label": "Clean occupancy test", "type": "boolean"},
                    {"field": "sim_success", "label": "Simulator success", "type": "boolean"},
                ],
            }, {
                "id": "counter_results", "title": "Both agents at counter, then one navigates to cabinet",
                "subtitle": "Silent clearing enabled versus disabled in two cabinet tasks.",
                "dataset": "counter_results", "sourceId": counter_source["id"], "density": "spacious", "layout": "full",
                "columns": [
                    {"field": "task", "label": "Task", "type": "text"},
                    {"field": "mode", "label": "Executor mode", "type": "text"},
                    {"field": "navigation_success", "label": "Navigation success", "type": "boolean"},
                    {"field": "partner_displacement_m", "label": "Partner displacement (m)", "format": "number"},
                ],
            }],
            "sources": [source, expanded_source, counter_source],
        },
        "snapshot": {"version": 1, "generatedAt": generated, "status": "ready", "datasets": {"branch_results": rows, "expanded_results": expanded_rows, "counter_results": counter_rows}},
        "sources": [source, expanded_source, counter_source],
    }
    (OUT / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (OUT / "source_notes.md").write_text(
        "# Source notes\n\nAudience: technical. The requested simulator images are the primary visual evidence; with only two exact branches, a chart would add no information beyond the comparison table.\n"
    )
    table_rows = "".join(
        "<tr>" + "".join(
            f"<td>{html.escape(str(row[key]))}</td>"
            for key in ("branch", "tool_sequence", "after_pickup", "after_navigation", "after_placement", "sim_success")
        ) + "</tr>"
        for row in rows
    )
    expanded_table_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in ("task", "manipulation", "branch", "actor_final", "partner_final", "partner_setup_displacement_m", "clean_parent_occupancy_test", "sim_success")) + "</tr>"
        for row in expanded_rows
    )
    counter_table_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in ("task", "mode", "navigation_success", "partner_displacement_m")) + "</tr>"
        for row in counter_rows
    )
    report_html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ArrangeBreadBowl Parent-Access Probe</title><style>
body{{margin:0;background:#f4f6f8;color:#17202a;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:32px 20px 70px}}section{{background:#fff;border:1px solid #d7dde5;border-radius:12px;padding:22px;margin:16px 0}}h1{{font-size:2rem}}h2{{margin-top:0}}code{{font-family:ui-monospace,monospace}}table{{width:100%;border-collapse:collapse;font-size:.9rem}}th,td{{padding:9px;border-bottom:1px solid #d7dde5;text-align:left;vertical-align:top}}.summary{{border-left:6px solid #2457c5}}.scroll{{overflow:auto}}
</style></head><body><main><h1>ArrangeBreadBowl: Parent-Access Simulator Probe</h1>
<section class="summary"><h2>Technical summary</h2><p><strong>Both branches succeeded.</strong> Without explicit counter navigation, placement moved agent 0 from the toaster pose <code>(3.417, -0.834)</code> to <code>(2.843, -1.000)</code>, exactly matching the explicit-navigation branch's final pose. This supports updating the symbolic location to <code>counter</code> after this placement.</p></section>
<section><h2>Retaining the existing exclusive-fixture pose</h2><p>With silent clearing disabled and the active agent's existing exclusive pose retained, all <strong>8/8</strong> branches succeeded and the partner remained at the counter. Each task includes top, center-agent, and wrist views.</p>{expanded_gallery}<div class="scroll"><table><thead><tr><th>Task</th><th>Manipulation</th><th>Branch</th><th>Actor final</th><th>Partner final</th><th>Setup displacement</th><th>Clean test</th><th>Success</th></tr></thead><tbody>{expanded_table_rows}</tbody></table></div></section>
<section><h2>Both agents start at counter, then one enters cabinet</h2><p>With silent clearing enabled, navigation succeeds by moving the partner roughly 0.72 m without a tool call. With clearing disabled, navigation fails and neither robot moves.</p>{counter_gallery}<div class="scroll"><table><thead><tr><th>Task</th><th>Mode</th><th>Navigation success</th><th>Partner displacement (m)</th></tr></thead><tbody>{counter_table_rows}</tbody></table></div></section>
<section><h2>Current interpretation</h2><p>Retain a verified usable pose for an agent already at the exclusive fixture. If an agent must enter that fixture from the counter and the partner blocks access, require an explicit handover. Never silently move the partner.</p></section>
<section><h2>Side effects to guard against</h2><ol><li>Keep the relation directional.</li><li>After the implicit move, require navigation back before later toaster work.</li><li>Claim the destination object and parent workspace during the atomic tick.</li><li>Start only with executor-backed tool transitions.</li><li>Revalidate trajectories because the post-call location changes later legality.</li></ol></section>
<section><h2>Method and limitations</h2><p>Both branches used ArrangeBreadBowl run 6, layout 11, style 14, seed 42 from fresh state. This proves <code>place_in_receptacle</code> in this scene, not every appliance-parent or manipulation-tool combination.</p></section>
<section><h2>Next steps</h2><ol><li>Add the transition to the shared concurrent FSM.</li><li>Test location update, contention, and asymmetric rejection.</li><li>Probe representative tool families before widening it.</li><li>Then replace the task-specific navigation reminder with general prompt guidance.</li></ol></section>
</main></body></html>"""
    (OUT / "report.html").write_text(report_html)
    print(OUT / "artifact.json")


if __name__ == "__main__":
    main()

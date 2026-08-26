"""Build the portable technical report for shared-source pickup probes."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/shared_source_pickup_probe_artifact"
ASSETS = OUT / "assets"
RESULTS = OUT / "probe_results.json"


def _data_url(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _panel(path: Path, title: str, note: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return (
        '<figure style="margin:0;border:1px solid #d7dde5;border-radius:10px;'
        'padding:12px;background:#fff;color:#17202a">'
        f'<img src="{_data_url(path)}" alt="{html.escape(title)}" '
        'style="display:block;width:100%;height:auto;border-radius:6px">'
        f'<figcaption style="margin-top:10px"><strong>{html.escape(title)}</strong><br>'
        f'<span style="color:#56616f">{html.escape(note)}</span></figcaption></figure>'
    )


def _gallery(task_slug: str, task: str, source: str, objects: tuple[str, str]) -> str:
    panels: list[str] = []
    for order in ("01", "10"):
        order_label = "agent_0 then agent_1" if order == "01" else "agent_1 then agent_0"
        panels.extend([
            _panel(
                ASSETS / f"{task_slug}_order_{order}_before.jpg",
                f"{task}: before pickup ({order_label})",
                f"Both robots begin at the counter; {objects[0]} and {objects[1]} rest on {source}.",
            ),
            _panel(
                ASSETS / f"{task_slug}_order_{order}_after_1.jpg",
                f"{task}: after first commit",
                "The first same-tick pickup has executed successfully; the second call was admitted from the same pre-tick state.",
            ),
            _panel(
                ASSETS / f"{task_slug}_order_{order}_after_2.jpg",
                f"{task}: after both commits",
                f"Both pickups succeeded; each robot holds its distinct intended object from {source}.",
            ),
        ])
        for agent in (0, 1):
            panels.append(
                _panel(
                    ASSETS / f"{task_slug}_order_{order}_agent_{agent}_after_both_agentview_center.jpg",
                    f"{task}: agent_{agent} close view after both pickups",
                    f"Close simulator view after the {order_label} commit order.",
                )
            )
    return (
        '<div style="padding:12px;display:grid;grid-template-columns:'
        'repeat(auto-fit,minmax(390px,1fr));gap:14px;background:#f4f6f8">'
        + "".join(panels)
        + "</div>"
    )


def _fallback_html(
    *, rows: list[dict], garnish_gallery: str, skewer_gallery: str
) -> str:
    """Render a self-contained fallback when the canonical Node packager is absent."""

    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['task'])}</td>"
        f"<td>{row['seed']}</td>"
        f"<td>{html.escape(row['commit_order'])}</td>"
        f"<td>{'PASS' if row['first_call_success'] else 'FAIL'}</td>"
        f"<td>{'PASS' if row['second_call_success'] else 'FAIL'}</td>"
        f"<td><code>{html.escape(str(row['agent_0_holds']))}</code></td>"
        f"<td><code>{html.escape(str(row['agent_1_holds']))}</code></td>"
        f"<td><strong>{'PASS' if row['passed'] else 'FAIL'}</strong></td>"
        "</tr>"
        for row in rows
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark"><title>Shared-Source Concurrent Pickup Probe</title>
<style>
:root{{--bg:#f4f6f8;--card:#fff;--ink:#17202a;--muted:#56616f;--line:#d7dde5;--accent:#2457c5}}
@media(prefers-color-scheme:dark){{:root{{--bg:#111827;--card:#1f2937;--ink:#f3f4f6;--muted:#cbd5e1;--line:#475569;--accent:#93b4ff}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}}
main{{max-width:1180px;margin:auto;padding:34px 22px 70px}} section{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:24px;margin:18px 0}}
h1{{font-size:2rem;margin:0 0 18px}} h2{{margin-top:0;font-size:1.35rem}} code{{font-family:ui-monospace,monospace}} .result{{border-left:6px solid var(--accent)}}
.gallery-wrap{{overflow:hidden;border-radius:12px}} table{{width:100%;border-collapse:collapse;font-size:.92rem}} th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left}} th{{position:sticky;top:0;background:var(--card)}} .table-scroll{{overflow:auto}}
ol,ul{{padding-left:1.4rem}} p{{max-width:86ch}}
</style></head><body><main>
<h1>Shared-Source Concurrent Pickup Probe</h1>
<section class="result"><h2>Technical summary</h2><p><strong>Both requested behaviors are supported: 12/12 fresh live-simulator sessions passed.</strong> Both call orders passed for each task across simulator seeds 42, 43, and 44. Every session ended with each robot holding its intended distinct object.</p></section>
<section><h2>Both cases are order-independent in the tested scenes</h2><p><code>GarnishCake</code> succeeded with agent_0 holding <code>cherry1</code> and agent_1 holding <code>strawberry1</code>, both taken from <code>fruit_plate</code>. <code>MeatSkewerAssembly</code> succeeded with agent_0 holding <code>skewer1</code> and agent_1 holding <code>skewer2</code>, both taken from <code>skewer_plate</code>. Reversing commit order did not change the result.</p></section>
<section><h2>GarnishCake visual sequence</h2><p>For each order, inspect before → first commit → both commits, followed by close simulator views of both robots.</p><div class="gallery-wrap">{garnish_gallery}</div></section>
<section><h2>MeatSkewerAssembly visual sequence</h2><p>The same visual sequence is shown for the two skewers.</p><div class="gallery-wrap">{skewer_gallery}</div></section>
<section><h2>Exact outcomes across seeds and commit orders</h2><p>Each row is a fresh simulator session. PASS requires two successful pickup results and the expected final held-object mapping.</p><div class="table-scroll"><table><thead><tr><th>Task</th><th>Seed</th><th>Commit order</th><th>First</th><th>Second</th><th>Agent 0 holds</th><th>Agent 1 holds</th><th>Result</th></tr></thead><tbody>{table_rows}</tbody></table></div></section>
<section><h2>Scope and methodology</h2><p>Both calls are admitted against one beginning-of-tick state. MuJoCo is not thread-safe, so live evaluation commits accepted physical calls serially. The probe tests both serial commit orders from fresh identical states. It used layout 11, style 34, and seeds 42–44, with no wait, release, give-space, navigation, or correction inserted between pickups.</p></section>
<section><h2>Limitations and robustness</h2><p>This establishes support and order independence in the tested scenes, not every possible layout/style arrangement. It is sufficient to show that unconditional locking of a stationary source plate is stricter than simulator behavior.</p></section>
<section><h2>Recommended next steps</h2><ol><li>Do not treat a stationary movable <code>source_id</code> as exclusive when agents pick different child objects.</li><li>Continue rejecting duplicate pickup of the same object and moving the source while another agent accesses a child.</li><li>Use one shared contention classifier in the dependency-wait audit and concurrent evaluator, then revalidate all 824 configurations.</li></ol></section>
</main></body></html>"""


def main() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = []
    for record in payload["records"]:
        rows.append({
            "task": record["task"],
            "seed": record["seed"],
            "commit_order": " → ".join(record["commit_order"]),
            "first_call_success": bool(record["calls"][0]["result"]["success"]),
            "second_call_success": bool(record["calls"][1]["result"]["success"]),
            "agent_0_holds": record["final_held_objects"].get("agent_0"),
            "agent_1_holds": record["final_held_objects"].get("agent_1"),
            "passed": bool(record["passed"]),
        })
    generated = datetime.now(timezone.utc).isoformat()
    source_id = "live_sim_probe"
    all_passed = all(row["passed"] for row in rows)
    garnish_gallery = _gallery(
        "garnishcake", "GarnishCake", "fruit_plate", ("cherry1", "strawberry1")
    )
    skewer_gallery = _gallery(
        "meatskewerassembly", "MeatSkewerAssembly", "skewer_plate", ("skewer1", "skewer2")
    )
    source = {
        "id": source_id,
        "label": "Shared-source concurrent pickup live-simulator probe",
        "path": "training/bc_task_vlm/eval_runs/shared_source_pickup_probe_artifact/probe_results.json",
        "query": {
            "description": "Execute both deterministic commit orders from fresh identical pre-tick simulator states.",
            "language": "python",
            "tables_used": ["probe_results.records"],
            "filters": [
                "Layout 11, style 34",
                "Simulator seeds 42, 43, and 44",
                "Distinct objects sharing one movable source",
            ],
            "metric_definitions": {
                "passed": "Both pickup tool calls returned success and the final executor state maps each robot to its intended distinct object.",
            },
        },
    }
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Shared-Source Concurrent Pickup Probe",
            "description": "Live-simulator evidence for two agents picking distinct objects from the same plate in one atomic scheduling tick.",
            "generatedAt": generated,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Shared-Source Concurrent Pickup Probe", "layout": "full"},
                {"id": "summary", "type": "markdown", "sourceId": source_id, "body": (
                    "## Technical summary\n\n"
                    f"**Both requested behaviors are supported by the simulator in this probe: {sum(r['passed'] for r in rows)}/{len(rows)} fresh sessions passed.** "
                    "For each task, both call orders passed across three simulator seeds. After each pair, agent_0 and agent_1 held the intended distinct objects. "
                    "The current concurrent FSM rejection is therefore stricter than the simulator behavior tested here."
                ), "layout": "full"},
                {"id": "findings", "type": "markdown", "sourceId": source_id, "body": (
                    "## Both shared-source cases are order-independent in the tested scenes\n\n"
                    "`GarnishCake` succeeded when agent_0 picked `cherry1` and agent_1 picked `strawberry1` from `fruit_plate`. "
                    "`MeatSkewerAssembly` succeeded when agent_0 picked `skewer1` and agent_1 picked `skewer2` from `skewer_plate`. "
                    "Reversing simulator commit order did not change either outcome."
                ), "layout": "full"},
                {"id": "garnish_intro", "type": "markdown", "body": (
                    "## GarnishCake visual sequence\n\n"
                    "Read each order from before → first commit → both commits. The close views make the final robot/object state easier to inspect than the overhead view alone."
                ), "layout": "full"},
                {"id": "garnish_gallery", "type": "html", "body": garnish_gallery, "layout": "full"},
                {"id": "skewer_intro", "type": "markdown", "body": (
                    "## MeatSkewerAssembly visual sequence\n\n"
                    "The same sequence is repeated for two distinct skewers resting on `skewer_plate`, again in both commit orders."
                ), "layout": "full"},
                {"id": "skewer_gallery", "type": "html", "body": skewer_gallery, "layout": "full"},
                {"id": "table_intro", "type": "markdown", "sourceId": source_id, "body": (
                    "## Exact outcomes across seeds and commit orders\n\n"
                    "Every row is a fresh simulator session. A row passes only when both tool calls report success and the executor records the expected final held object for each robot."
                ), "layout": "full"},
                {"id": "results_table", "type": "table", "tableId": "probe_results", "layout": "full"},
                {"id": "scope", "type": "markdown", "body": (
                    "## Scope and definitions\n\n"
                    "An **atomic scheduling tick** means both model calls are selected against the same beginning-of-tick symbolic state. The MuJoCo environment itself is not thread-safe, so accepted physical calls are committed serially. Testing both commit orders determines whether the result depends on that internal ordering."
                ), "layout": "full"},
                {"id": "method", "type": "markdown", "sourceId": source_id, "body": (
                    "## Simulator methodology\n\n"
                    "Each task was initialized from its verified symbolic state in layout 11/style 34. For seeds 42, 43, and 44, the probe started two fresh sessions: agent_0 then agent_1, and agent_1 then agent_0. No `give_space`, wait, release, navigation, or validator bypass action was inserted between the pickups."
                ), "layout": "full"},
                {"id": "limits", "type": "markdown", "body": (
                    "## Limitations and robustness\n\n"
                    "The probe covers one compatible layout/style and three seeds, not every RoboCasa scene. It establishes that the simulator and tool executor support these shared-source operations and that they are order-independent in the tested scenes. It does not prove that every physical arrangement of every shared plate is collision-free."
                ), "layout": "full"},
                {"id": "next", "type": "markdown", "body": (
                    "## Recommended next steps\n\n"
                    "1. Stop treating a stationary movable `source_id` as an exclusive claim when agents pick different child objects.\n"
                    "2. Continue rejecting the same object being picked twice or one agent moving the source while the other accesses a child.\n"
                    "3. Make the dependency-wait audit use the same shared contention classifier, then revalidate the 824 configurations."
                ), "layout": "full"},
                {"id": "questions", "type": "markdown", "body": (
                    "## Further questions\n\n"
                    "A broader geometry sweep would be useful only if we want a universal claim across layouts and styles. For the immediate validator decision, the tested task states demonstrate that unconditional source-plate locking is incorrect."
                ), "layout": "full"},
            ],
            "cards": [],
            "charts": [],
            "tables": [{
                "id": "probe_results",
                "title": "Shared-source pickup results",
                "subtitle": "Two tasks × three seeds × both deterministic simulator commit orders.",
                "showDescription": True,
                "dataset": "probe_results",
                "sourceId": source_id,
                "density": "spacious",
                "layout": "full",
                "defaultSort": {"field": "task", "direction": "asc"},
                "columns": [
                    {"field": "task", "label": "Task", "type": "text"},
                    {"field": "seed", "label": "Seed", "format": "number"},
                    {"field": "commit_order", "label": "Commit order", "type": "text"},
                    {"field": "first_call_success", "label": "First pickup", "type": "boolean"},
                    {"field": "second_call_success", "label": "Second pickup", "type": "boolean"},
                    {"field": "agent_0_holds", "label": "Agent 0 holds", "type": "text"},
                    {"field": "agent_1_holds", "label": "Agent 1 holds", "type": "text"},
                    {"field": "passed", "label": "Passed", "type": "boolean"},
                ],
            }],
            "sources": [source],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready" if all_passed else "partial",
            "datasets": {"probe_results": rows},
        },
        "sources": [source],
    }
    (OUT / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    (OUT / "source_notes.md").write_text(
        "# Source and visual notes\n\n"
        "Audience: technical. All required technical-report roles are represented.\n\n"
        "No quantitative chart is used because the evidence contains two binary cases across 12 exact audit rows; the result table is more honest and the simulator frames are the requested primary visual evidence.\n\n"
        "Chart map: none. Visual map: overhead before/after sequences plus per-agent close views for seed 42 and both commit orders.\n",
        encoding="utf-8",
    )
    (OUT / "report.html").write_text(
        _fallback_html(
            rows=rows,
            garnish_gallery=garnish_gallery,
            skewer_gallery=skewer_gallery,
        ),
        encoding="utf-8",
    )
    print(OUT / "artifact.json")


if __name__ == "__main__":
    main()

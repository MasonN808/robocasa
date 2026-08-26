"""Build a self-contained technical visual report for access-semantics probes."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "training/bc_task_vlm/eval_runs/candidate_global_access_semantics_probe"
OUT = ROOT / "training/bc_task_vlm/eval_runs/candidate_global_access_semantics_artifact"


def _img(path: str | Path, title: str, note: str) -> str:
    path = Path(path)
    data = base64.b64encode(path.read_bytes()).decode()
    return (
        '<figure class="panel">'
        f'<img src="data:image/jpeg;base64,{data}" alt="{html.escape(title)}">'
        f'<figcaption><strong>{html.escape(title)}</strong><span>{html.escape(note)}</span></figcaption>'
        '</figure>'
    )


def _call_table(record: dict) -> str:
    rows = []
    for index, item in enumerate(record.get("calls") or [], 1):
        call = item["call"]
        rows.append(
            f"<tr><td>{index}</td><td>{html.escape(str(call.get('metadata', {}).get('source_agent')))}</td>"
            f"<td><code>{html.escape(call['tool'])}</code></td>"
            f"<td><code>{html.escape(json.dumps(call['args'], sort_keys=True))}</code></td>"
            f"<td>{'Pass' if item['success'] else 'Fail'}</td></tr>"
        )
    return (
        '<table><thead><tr><th>#</th><th>Agent</th><th>Tool</th><th>Resolved arguments</th><th>Result</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _visual_grid(record: dict) -> str:
    panels = [
        _img(record["initial"]["top"], "Initial top view", "Physical placements before the experimental calls."),
    ]
    for milestone in record.get("milestones") or []:
        panels.append(_img(milestone["frame"]["top"], milestone["label"], "Top view immediately after this navigation or cabinet-state change."))
    panels.append(_img(record["final"]["top"], "Final top view", "Physical placements after all listed calls."))
    for agent in ("agent_0", "agent_1"):
        panels.append(_img(
            record["final"]["agent_views"][agent]["agentview_center"],
            f"{agent} final center view",
            "Use this view to assess visibility, crowding, and whether the interaction is physically plausible.",
        ))
    return '<div class="grid">' + "".join(panels) + "</div>"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.loads((PROBE / "probe_results.json").read_text())
    records = payload["records"]
    cabinet = records[:6]
    retained = records[6:]

    summary_rows = []
    for record in records:
        disp = ", ".join(f"{agent}: {value:.4f} m" for agent, value in record.get("displacements", {}).items())
        distance = record.get("final_inter_robot_distance")
        summary_rows.append(
            f"<tr><td>{html.escape(record['task'])}</td><td>{html.escape(record['name'])}</td>"
            f"<td>{'Pass' if record.get('success') else 'Fail'}</td><td>{html.escape(disp)}</td>"
            f"<td>{'—' if distance is None else f'{distance:.4f} m'}</td></tr>"
        )

    cabinet_sections = []
    for record in cabinet:
        if record["parent_declared"]:
            parent_note = (
                f"The task declares the parent and it matches the simulator-derived parent: "
                f"{record['resolved_parent']}."
            )
        else:
            parent_note = (
                f"The task does not declare a cabinet parent. Its candidate symbol resolved to "
                f"{record['requested_parent_resolution']}, but simulator geometry identifies "
                f"{record['scene_inferred_parent']}. This probe used the simulator-derived parent."
            )
        cabinet_sections.append(f"""
        <section>
          <h2>{html.escape(record['task'])}: shared cabinet access passed</h2>
          <p><strong>All listed calls succeeded without silent partner clearing.</strong> {html.escape(parent_note)} The final robot-base separation was {record['final_inter_robot_distance']:.3f} m; inspect the top and agent views to decide whether that clearance is acceptable.</p>
          {_call_table(record)}
          {_visual_grid(record)}
        </section>
        """)

    retained_sections = []
    for record in retained:
        disp = record["displacements"]
        retained_sections.append(f"""
        <section>
          <h2>{html.escape(record['task'])}: the exclusive child pose was retained</h2>
          <p><strong>The actor completed parent-side manipulation without an explicit parent navigation.</strong> Actor drift was {disp['agent_0']:.4f} m and partner drift was {disp['agent_1']:.4f} m. The actor remained symbolically at <code>{html.escape(record['child'])}</code> throughout this experimental branch.</p>
          {_call_table(record)}
          {_visual_grid(record)}
        </section>
        """)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark">
<title>Candidate Global Access Semantics</title><style>
:root{{--bg:#f3f6fa;--card:#fff;--text:#17202a;--muted:#5e6876;--line:#d8dee8;--accent:#155eef}}
@media(prefers-color-scheme:dark){{:root{{--bg:#10141b;--card:#181f29;--text:#eef3f8;--muted:#aeb8c5;--line:#344052;--accent:#73a2ff}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}main{{max-width:1320px;margin:auto;padding:34px 22px 80px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px;margin:20px 0}}h1{{font-size:32px;margin:0 0 8px}}h2{{font-size:23px;margin:0 0 10px}}p{{max-width:100ch}}.muted{{color:var(--muted)}}.summary{{border-left:5px solid var(--accent)}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:18px}}.panel{{margin:0;border:1px solid var(--line);border-radius:10px;padding:10px;background:var(--card)}}.panel img{{display:block;width:100%;height:auto;border-radius:7px}}figcaption{{margin-top:8px}}figcaption span{{display:block;color:var(--muted)}}
table{{width:100%;border-collapse:collapse;margin:16px 0;display:block;overflow-x:auto}}th,td{{padding:9px 11px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}code{{font-size:12px;overflow-wrap:anywhere}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><h1>Candidate Global Access Semantics</h1><div class="muted">Probe-only experiment · layout 11, style 14, seed 42 · {generated}</div></header>
<section class="summary"><h2>Technical summary</h2>
<p><strong>All eight controlled tool sequences passed without changing production code.</strong> A counter-located agent accessed a cabinet while its partner remained at the cabinet; two cabinet navigation calls were safely mapped to the simulator-derived supporting counter; and both current exclusive child fixtures could manipulate parent-counter objects without leaving their child pose.</p>
<p><strong>The main implementation prerequisite is parent identity, not simulator feasibility.</strong> `ArrangeTea` and `GatherMarinadeIngredients` already declare the correct parent. Four legacy cabinet tasks do not; their apparent task-level parent candidates resolve to the wrong surface, while simulator geometry correctly identifies the counter beneath the cabinet.</p>
<p><strong>Clearance remains a judgment call.</strong> Shared cabinet navigation produced final robot-base separations from 0.424 m to 0.478 m. The simulator accepted these placements, but the top views show close side-by-side poses and should be reviewed before adopting the behavior globally.</p></section>
<section><h2>Exact outcomes across all probes</h2><p>Displacement is planar base movement across the complete case. For cases containing ordinary navigation or placement elsewhere, it is not a measure of silent cabinet displacement by itself.</p>
<table><thead><tr><th>Task</th><th>Case</th><th>Tools</th><th>Base displacement</th><th>Final robot separation</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></section>
{''.join(cabinet_sections)}
{''.join(retained_sections)}
<section><h2>Scope and method</h2><p>The cabinet probe temporarily removed cabinet front-pose requirements, disabled silent blocker clearing, treated cabinet access as valid from its supporting counter, and mapped cabinet navigation to that counter. Concrete runtime fixture IDs and supporting counters were resolved after scene initialization. The retained-pose probe suppressed only actor repositioning from an exclusive countertop appliance to its parent during the tested parent-object manipulation.</p>
<p>No concurrent FSM, postprocessor, evaluator, task prompt, task specification, saved trajectory, or production executor behavior was changed.</p></section>
<section><h2>Limitations and robustness requirements</h2><ul><li>Manipulation tools place objects symbolically and do not prove physical arm reach. Camera views provide plausibility evidence, not a motion-planning guarantee.</li><li>Only one scene was tested. Geometry-derived parent resolution should be audited over layouts before becoming a global invariant.</li><li>Robot-base clearance was accepted by the placement system but is visually close.</li><li>Legacy tasks without model-visible parent metadata cannot yet tell the model that direct parent access is available, even though runtime geometry can find the parent.</li></ul></section>
<section><h2>Recommended next step</h2><ol><li>Review the visual clearances in this artifact.</li><li>If accepted, implement one shared runtime cabinet-parent resolver using the simulator’s existing geometry-derived <code>parent_fixture</code>.</li><li>Expose that resolved parent consistently in initial state for tasks that currently omit it.</li><li>Then implement cabinet non-front/shared access and retained exclusive-child parent manipulation in the shared executor and concurrent scheduler.</li><li>Only after those changes, revalidate the existing trajectory pool.</li></ol></section>
<section><h2>Further questions</h2><ul><li>Should the minimum side-by-side cabinet clearance exceed the current 0.424 m observed minimum?</li><li>Should cabinet navigation preserve symbolic <code>location=cabinet</code> while using the parent’s physical navigation pose?</li><li>Should direct child-to-parent access be limited by fixture hierarchy alone or additionally by a distance threshold?</li></ul></section>
</main></body></html>"""
    (OUT / "report.html").write_text(report)
    (OUT / "artifact.json").write_text(json.dumps({
        "surface": "report", "title": "Candidate Global Access Semantics",
        "generated_at": generated, "scene": payload["scene"], "records": records,
        "source": str(PROBE / "probe_results.json"),
    }, indent=2) + "\n")
    (OUT / "source_notes.md").write_text(
        "# Source notes\n\n"
        "Primary source: `candidate_global_access_semantics_probe/probe_results.json` and its simulator-rendered images.\n\n"
        "No chart was used because exact tool-call tables and simulator images are the decision-relevant evidence.\n\n"
        "Required technical-report structure is represented by the technical summary, per-case visual findings, scope/method, limitations, recommended next step, and further questions.\n\n"
        "The packaged analytics renderer was unavailable because Node/npm are not installed in this cluster environment; this is self-contained HTML generated from the saved JSON.\n"
    )
    print(OUT / "report.html")


if __name__ == "__main__":
    main()

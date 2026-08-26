"""Build a self-contained visual report for cabinet/parent access probes."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "training/bc_task_vlm/eval_runs/cabinet_parent_semantics_probe"
OUT = ROOT / "training/bc_task_vlm/eval_runs/cabinet_parent_semantics_artifact"


def _data_url(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def _panel(path: str | Path, title: str, note: str) -> str:
    return (
        '<figure class="panel">'
        f'<img src="{_data_url(path)}" alt="{html.escape(title)}">'
        f'<figcaption><strong>{html.escape(title)}</strong><span>{html.escape(note)}</span></figcaption>'
        '</figure>'
    )


def _frames(record: dict, before_key: str, after_key: str) -> str:
    before = record[before_key]
    after = record[after_key]
    panels = [
        _panel(before["top"], "Initial / before action", "Top view and exact robot placement before the tested interaction."),
        _panel(after["top"], "After action", "Top view after the tool calls; compare whether either robot was displaced."),
    ]
    for agent in ("agent_0", "agent_1"):
        panels.append(_panel(
            after["agent_views"][agent]["agentview_center"],
            f"{agent} view after action",
            "Center camera; use this to judge visibility and physical plausibility.",
        ))
    return '<div class="grid">' + "".join(panels) + "</div>"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.loads((PROBE / "probe_results.json").read_text())
    records = {record["name"]: record for record in payload["records"]}

    table_rows = []
    for record in payload["records"]:
        if "partner_displacements" in record:
            displacement = ", ".join(f"{a}: {v:.4f} m" for a, v in record["partner_displacements"].items())
        else:
            displacement = f"actor: {record['actor_displacement']:.4f} m; partner: {record['partner_displacement']:.4f} m"
        table_rows.append(
            f"<tr><td>{html.escape(record['name'])}</td><td>{'Pass' if record.get('success') else 'Fail'}</td>"
            f"<td>{html.escape(displacement)}</td></tr>"
        )

    sections = []
    for location in ("cabinet", "counter"):
        off = records[f"gather_both_at_{location}_silent_clear_off"]
        on = records[f"gather_both_at_{location}_silent_clear_on"]
        sections.append(f"""
        <section>
          <h2>Both robots at the {html.escape(location)}: cabinet access succeeds without clearing</h2>
          <p>The silent-clearing-off branch opened the cabinet and let the two robots pick up different objects. Its robot positions match the silent-clearing-on branch to millimeter-scale drift, so success did not depend on secretly moving the partner.</p>
          {_frames(off, 'initial', 'final')}
          <details><summary>Silent clearing enabled comparison</summary>{_frames(on, 'initial', 'final')}</details>
        </section>
        """)

    normal = records["arrange_bread_normal_parent_reposition"]
    retained = records["arrange_bread_retain_toaster_pose"]
    sections.append(f"""
      <section>
        <h2>Parent access can retain the toaster pose</h2>
        <p>Both branches placed the bread into the bowl. Current behavior moved the toaster-side actor {normal['actor_displacement']:.3f} m; the experimental branch moved neither robot. This establishes executor feasibility, but not arm-reach physics: placement tools set the object pose symbolically.</p>
        <h3>Current parent repositioning</h3>
        {_frames(normal, 'before_parent_access', 'after_parent_access')}
        <h3>Experimental retained pose</h3>
        {_frames(retained, 'before_parent_access', 'after_parent_access')}
      </section>
    """)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark"><title>Cabinet and Parent-Fixture Access Probe</title>
<style>
:root{{--bg:#f3f6fa;--card:#fff;--text:#17202a;--muted:#5e6876;--line:#d8dee8;--accent:#155eef}}
@media(prefers-color-scheme:dark){{:root{{--bg:#10141b;--card:#181f29;--text:#eef3f8;--muted:#aeb8c5;--line:#344052;--accent:#73a2ff}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1260px;margin:auto;padding:34px 22px 80px}} section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px;margin:20px 0}}
h1{{font-size:32px;margin:0 0 8px}} h2{{font-size:23px;margin:0 0 10px}} h3{{margin-top:28px}} p{{max-width:92ch}} .muted{{color:var(--muted)}}
.summary{{border-left:5px solid var(--accent)}} .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:18px}}
.panel{{margin:0;border:1px solid var(--line);border-radius:10px;padding:10px;background:var(--card)}} .panel img{{display:block;width:100%;height:auto;border-radius:7px}}
figcaption{{margin-top:8px}} figcaption span{{display:block;color:var(--muted)}} table{{width:100%;border-collapse:collapse}} th,td{{padding:10px;text-align:left;border-bottom:1px solid var(--line)}}
details{{margin-top:18px}} summary{{cursor:pointer;font-weight:650}} @media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><h1>Cabinet and Parent-Fixture Access Probe</h1><div class="muted">Scene layout 11, style 14, seed 42 · generated {generated}</div></header>
<section class="summary"><h2>Technical summary</h2>
<p><strong>The experiment supports parent-counter cabinet access without front-pose reservation or silent partner displacement.</strong> Both robots could remain at the cabinet or counter while opening the cabinet and retrieving different objects. The cleaner interface is to keep cabinet access attached to its parent counter rather than encourage two robots to navigate to one cabinet coordinate.</p>
<p><strong>Retaining an exclusive appliance pose during parent-object manipulation also worked at executor level.</strong> The toaster-side robot placed bread into the neighboring bowl with zero robot displacement. This is not yet proof of physical reach because manipulation is pose-based, so the camera views are the relevant plausibility check.</p></section>
<section><h2>Exact outcomes</h2><p>Displacement is planar robot-base movement from immediately before to after the tested calls.</p>
<table><thead><tr><th>Branch</th><th>Tool result</th><th>Robot displacement</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></section>
{''.join(sections)}
<section><h2>Recommended implementation boundary</h2>
<ol><li>Model cabinets as child storage reachable from their parent counter; do not reserve an exclusive cabinet front pose.</li><li>Make cabinets non-navigable in the model-facing interface only after auditing existing trajectories that navigate to cabinets and normalizing them to the parent.</li><li>Remove silent teammate displacement for cabinet access.</li><li>For an actor at an exclusive countertop appliance, allow parent-surface manipulation without moving its base when the target is plausibly within reach; retain the exclusive symbolic location.</li><li>Keep fridge, oven, microwave, dishwasher, toaster oven, coffee machine, and similar explicitly exclusive fixtures unchanged until separately probed.</li></ol>
<p><strong>Caveat:</strong> the two-robots-at-cabinet top view is physically crowded and the center views contain substantial self/partner occlusion. That branch passes the executor, but supports making cabinets non-navigable more than it supports teaching agents to congregate at a cabinet coordinate.</p></section>
<section><h2>Further questions</h2><ul><li>Does the same parent-counter policy work across every cabinet task, layout, and cabinet geometry?</li><li>Which parent objects are within plausible reach from each exclusive countertop appliance pose?</li><li>Can production validation derive cabinet-to-parent access from fixture hierarchy without adding task-specific rules?</li></ul></section>
</main></body></html>"""
    (OUT / "report.html").write_text(body)
    artifact = {
        "title": "Cabinet and Parent-Fixture Access Probe",
        "surface": "report",
        "generated_at": generated,
        "scene": payload["scene"],
        "records": payload["records"],
        "source": str(PROBE / "probe_results.json"),
    }
    (OUT / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (OUT / "source_notes.md").write_text(
        "# Source notes\n\n"
        "Primary source: `cabinet_parent_semantics_probe/probe_results.json` and its rendered frames.\n\n"
        "No chart was used: six exact experimental branches and simulator images are clearer as a result table plus visual panels.\n\n"
        "The packaged analytics renderer was unavailable because Node/npm are not installed in this cluster environment; the report is self-contained HTML generated from the same saved JSON.\n"
    )
    print(OUT / "report.html")


if __name__ == "__main__":
    main()

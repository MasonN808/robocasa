"""Build a self-contained visual report for reference-vs-support navigation."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/reference_fixture_navigation_probe"
ASSETS = OUT / "assets"


def _image(path: Path, title: str, note: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        '<figure><img src="data:image/jpeg;base64,' + data + '" alt="' + html.escape(title) + '">'
        '<figcaption><strong>' + html.escape(title) + '</strong><br>' + html.escape(note) + '</figcaption></figure>'
    )


def main() -> None:
    payload = json.loads((OUT / "probe_results.json").read_text())
    records = {(row["task"], row["start"]): row for row in payload["records"]}
    sections = []
    for task in ("SetUpSpiceStation", "PrepareSandwichStation", "SetupBowls"):
        ref = records[(task, "reference")]
        support = records[(task, "support")]
        slug = task.lower()
        panels = [
            _image(ASSETS / f"{slug}_reference_before.jpg", "Reference pose before placement", f"Robot navigated to {ref['reference']} while holding {ref['object']}."),
            _image(ASSETS / f"{slug}_reference_after.jpg", "Reference pose after placement", f"place_next_to success={ref['placement']['success']}; target support is {ref['support']}."),
            _image(ASSETS / f"{slug}_support_before.jpg", "Support pose before placement", f"Robot navigated to {support['support']} while holding {support['object']}."),
            _image(ASSETS / f"{slug}_support_after.jpg", "Support pose after placement", f"place_next_to success={support['placement']['success']}; reference is {support['reference']}."),
        ]
        sections.append(
            f"<section><h2>{html.escape(task)}</h2><p><code>{html.escape(ref['reference'])}</code> is the relation anchor; "
            f"<code>{html.escape(ref['support'])}</code> is the surface receiving the object. Compare the two before images first, then the resulting placements.</p>"
            '<div class="grid">' + ''.join(panels) + '</div></section>'
        )
    rows = ''.join(
        '<tr>'
        f"<td>{html.escape(row['task'])}</td><td>{html.escape(row['start'])}</td>"
        f"<td><code>{html.escape(row['reference'])}</code></td><td><code>{html.escape(row['support'])}</code></td>"
        f"<td>{'PASS' if row.get('navigation', {}).get('success') else 'FAIL'}</td>"
        f"<td>{'PASS' if row.get('placement', {}).get('success') else 'FAIL'}</td>"
        f"<td>{html.escape(row.get('error', ''))}</td></tr>"
        for row in payload['records']
    )
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark"><title>Reference Fixture Navigation Visual Audit</title><style>
:root{{--bg:#f3f5f8;--card:#fff;--ink:#17202a;--muted:#56616f;--line:#d7dde5;--accent:#315ea8}}
@media(prefers-color-scheme:dark){{:root{{--bg:#111827;--card:#1f2937;--ink:#f3f4f6;--muted:#cbd5e1;--line:#475569;--accent:#93b4ff}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,sans-serif}}main{{max-width:1400px;margin:auto;padding:32px 22px 70px}}section{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:22px;margin:18px 0}}h1,h2{{margin-top:0}}.summary{{border-left:6px solid var(--accent)}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}figure{{margin:0;border:1px solid var(--line);border-radius:10px;padding:10px}}img{{display:block;width:100%;border-radius:7px}}figcaption{{padding:9px 3px 2px;color:var(--muted)}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:left}}.scroll{{overflow:auto}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Reference Fixture Navigation Visual Audit</h1>
<section class="summary"><h2>Question</h2><p>Does it physically make sense to allow <code>place_next_to(..., reference_fixture_id=X)</code> when the robot navigated to X, in addition to the currently required adjacent support fixture?</p><p>This report is observational only. It does not change FSM semantics.</p></section>
{''.join(sections)}
<section><h2>Exact simulator outcomes</h2><div class="scroll"><table><thead><tr><th>Task</th><th>Starting pose</th><th>Reference</th><th>Support</th><th>Navigation</th><th>Placement</th><th>Error</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section><h2>How to judge the images</h2><ul><li>Is the reference-fixture pose physically close enough to manipulate the adjacent support?</li><li>Does placement silently move the robot to nearly the same pose as support navigation?</li><li>Would treating both symbolic locations as valid create an obvious collision or bypass an exclusive workspace?</li></ul></section>
<section><h2>Scope</h2><p>Layout {payload['scene']['layout']}, style {payload['scene']['style']}, seed {payload['scene']['seed']}. One representative fixture-anchor task from each current pattern was tested. Broader scene sampling should follow only if these images support changing the rule.</p></section>
</main></body></html>'''
    (OUT / "report.html").write_text(document)
    print(OUT / "report.html")


if __name__ == "__main__":
    main()

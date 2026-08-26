"""Build a self-contained report for the occupied-support visual probe."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/reference_fixture_occupied_support_probe"


def _image(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def main() -> None:
    payload = json.loads((OUT / "probe_results.json").read_text())
    cards = []
    rows = []
    for record in payload["records"]:
        slug = record["task"].lower()
        support = record.get("support_navigation") or {}
        reference = record.get("reference_navigation") or {}
        placement = record.get("placement") or {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['task'])}</td>"
            f"<td>{support.get('success')}</td>"
            f"<td>{reference.get('success')}</td>"
            f"<td>{placement.get('success')}</td>"
            f"<td><code>{html.escape(str(placement.get('details', '')))}</code></td>"
            "</tr>"
        )
        panels = []
        for stage, label in (("before", "Both navigation calls completed"), ("after", "After place_next_to")):
            src = _image(OUT / "assets" / f"{slug}_{stage}.jpg")
            panels.append(f"<figure><img src='{src}'><figcaption>{label}</figcaption></figure>")
        cards.append(
            f"<section><h2>{html.escape(record['task'])}</h2>"
            f"<p>Agent 0 holds <code>{record['object']}</code> at <code>{record['reference']}</code>; "
            f"agent 1 occupies <code>{record['support']}</code>.</p>"
            f"<div class='grid'>{''.join(panels)}</div></section>"
        )
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Reference Fixture with Occupied Support Probe</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 22px;background:#0b1020;color:#e8edf6}}h1,h2{{color:#fff}}section{{background:#131b2d;border:1px solid #29344d;border-radius:14px;padding:18px;margin:22px 0}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}figure{{margin:0}}img{{width:100%;border-radius:10px}}figcaption{{margin-top:7px;color:#b8c3d8}}table{{width:100%;border-collapse:collapse;background:#131b2d}}th,td{{border:1px solid #34415d;padding:9px;text-align:left}}code{{white-space:pre-wrap}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}</style>
</head><body><h1>Reference fixture placement while adjacent support is occupied</h1>
<p>Live simulator probe, layout 11 / style 34 / seed 42. This bypasses the symbolic FSM restriction only to test physical behavior.</p>
<table><thead><tr><th>Task</th><th>Support navigation</th><th>Reference navigation</th><th>Placement</th><th>Result details</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{''.join(cards)}</body></html>"""
    path = OUT / "report.html"
    path.write_text(document)
    print(path.resolve())


if __name__ == "__main__":
    main()

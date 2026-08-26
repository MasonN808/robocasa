"""Render one comparable occupancy map for every audited layout plus layout 18."""

from __future__ import annotations

import html
import json
from pathlib import Path

from robocasa.utils.trajectory_adapter import TrajectoryAdapter
from training.bc_task_vlm.diagnose_layout18_workspace_failures import _trajectory
from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/scene_layout_map_artifact"
ASSETS = OUT / "assets"
LAYOUTS = (11, 15, 18, 40, 50)
STYLE = 34
SEED = 42


def render_layout(layout: int) -> dict:
    trajectory = _trajectory("ArrangeTea", 0)
    session = SimSession(
        composite_task="ArrangeTea",
        sample_trajectory=trajectory,
        layout=layout,
        style=STYLE,
        seed=SEED,
        gl_backend="egl",
        render_size=768,
        map_dpi=130,
        map_renderer="raster",
    )
    error = None
    try:
        adapted = TrajectoryAdapter(
            executor=session.executor, allow_approximate_ids=True
        ).adapt(trajectory)
        try:
            session.executor.load_initial_state(adapted["initial_state"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        image_name = f"layout_{layout}_style_{STYLE}_seed_{SEED}_map.png"
        image_path = ASSETS / image_name
        session.executor.get_image(
            views=["map"], image_paths=[str(image_path)], agent_id="agent_0"
        )
        scene = session.executor.get_scene_description()
        return {
            "layout": layout,
            "style": STYLE,
            "seed": SEED,
            "included_in_certified_candidate_pool": layout != 18,
            "initial_state_loaded": error is None,
            "initialization_error": error,
            "map": f"assets/{image_name}",
            "fixture_count": len(scene.get("fixtures") or {}),
        }
    finally:
        session.close()


def build_report(records: list[dict]) -> None:
    cards = []
    for row in records:
        status = (
            "Candidate pool" if row["included_in_certified_candidate_pool"]
            else "Comparison only — excluded from candidate pool"
        )
        load = (
            "The representative two-robot initial state loaded successfully."
            if row["initial_state_loaded"]
            else "The map rendered, but the representative two-robot initial state failed: "
            + html.escape(row["initialization_error"] or "unknown error")
        )
        cards.append(
            f'''<section id="layout-{row['layout']}">
              <div class="heading"><div><h2>Layout {row['layout']}</h2>
              <p class="status">{status}</p></div>
              <span>{row['fixture_count']} fixtures</span></div>
              <p>{load}</p>
              <figure><a href="{row['map']}"><img src="{row['map']}"
                alt="Occupancy and fixture map for RoboCasa layout {row['layout']}"></a>
                <figcaption>Style {STYLE}, simulator seed {SEED}. Click the map for the full-resolution image.</figcaption>
              </figure>
            </section>'''
        )
    body = "\n".join(cards)
    report = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <meta name="color-scheme" content="light dark">
    <title>RoboCasa Scene Layout Maps</title>
    <style>
    :root{{--bg:#eef2f7;--card:#fff;--ink:#172033;--muted:#526079;--line:#d9e0ea;--blue:#225fc5}}
    @media(prefers-color-scheme:dark){{:root{{--bg:#10151e;--card:#19212d;--ink:#edf2f8;--muted:#abb8ca;--line:#344155;--blue:#87b4ff}}}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}}
    main{{max-width:1280px;margin:auto;padding:32px}} header,section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px;margin-bottom:20px}}
    h1,h2{{margin:0 0 8px}} header p,section p,figcaption{{color:var(--muted)}} nav{{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}}
    nav a{{color:var(--blue);border:1px solid var(--line);border-radius:999px;padding:6px 12px;text-decoration:none}}
    .heading{{display:flex;align-items:start;justify-content:space-between;gap:16px}} .heading span{{font-variant-numeric:tabular-nums;color:var(--muted)}}
    .status{{margin:0}} figure{{margin:18px 0 0}} img{{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fff}}
    figcaption{{margin-top:8px}} @media(max-width:700px){{main{{padding:14px}} header,section{{padding:16px}}}}
    </style></head><body><main><header><h1>RoboCasa Scene Layout Maps</h1>
    <p>One directly comparable occupancy/fixture map for every layout in the certified-scene audit, plus excluded layout 18. All maps use ArrangeTea, style {STYLE}, and simulator seed {SEED}; therefore differences primarily reflect layout geometry rather than styling.</p>
    <nav>{''.join(f'<a href="#layout-{x}">Layout {x}</a>' for x in LAYOUTS)}</nav></header>
    <section><h2>How to read these maps</h2><p>Colored fixture footprints show occupied geometry; the pale region is navigable workspace. Fixture and object labels identify the simulator-resolved scene. Layout 18 is shown for comparison because its narrow counter workspace caused valid two-agent configurations to fail, so it is not in the first certified candidate pool.</p></section>
    {body}</main></body></html>'''
    (OUT / "report.html").write_text(report)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    records = [render_layout(layout) for layout in LAYOUTS]
    (OUT / "metadata.json").write_text(json.dumps(records, indent=2) + "\n")
    build_report(records)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

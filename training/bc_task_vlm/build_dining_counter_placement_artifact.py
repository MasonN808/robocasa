"""Visualize failing and working layout-15 dining-counter scene variants."""

from __future__ import annotations

import html
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from robocasa.utils.placement import get_fixture_aabb
from robocasa.utils.trajectory_adapter import TrajectoryAdapter
from training.bc_task_vlm.audit_certified_scene_compatibility import _trajectory
from training.bc_task_vlm.diagnose_layout18_workspace_failures import _world_to_pixels
from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/dining_counter_placement_failure_artifact"
ASSETS = OUT / "assets"
PROBE_OUT = ROOT / "training/bc_task_vlm/eval_runs/dining_counter_placement_relaxation_probe"

PLACEMENT_CASES = (
    {
        "task": "GarnishCake",
        "region": ((0.30554635198950764, 0.932453741591331), (0.365, 0.715)),
        "placed_name": "cake_plate",
        "placed_center": (0.615813795, 0.524641377),
        "placed_size": (0.293691992, 0.293692843),
        "incoming_name": "fruit_plate",
        "incoming_size": (0.234147500, 0.234500001),
    },
    {
        "task": "MeatSkewerAssembly",
        "region": ((0.30554635198950764, 0.932453741591331), (0.365, 0.715)),
        "placed_name": "skewer_plate",
        "placed_center": (0.709107936, 0.538189408),
        "placed_size": (0.305929158, 0.305930045),
        "incoming_name": "oven_tray",
        "incoming_size": (0.358205862, 0.249745692),
    },
)

# Exact stool-referenced placement frame recorded from the failing
# layout-15/style-14 task initializations.
PLACEMENT_REFERENCE_XY = np.asarray([2.522999955434437, -3.084999975595049])
PLACEMENT_REFERENCE_ROTATION = -np.pi / 2


def _placement_schematic(case: dict, path: Path) -> None:
    """Draw the measured sampler strip and object footprints to scale."""

    width, height = 1200, 680
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    x_range, y_range = case["region"]
    margin = 120
    scale = min(
        (width - 2 * margin) / (x_range[1] - x_range[0]),
        (height - 2 * margin) / (y_range[1] - y_range[0]),
    )

    def point(x: float, y: float) -> tuple[float, float]:
        return (
            margin + (x - x_range[0]) * scale,
            height - margin - (y - y_range[0]) * scale,
        )

    def box(center, size):
        cx, cy = center
        sx, sy = size
        left, top = point(cx - sx / 2, cy + sy / 2)
        right, bottom = point(cx + sx / 2, cy - sy / 2)
        return (left, top, right, bottom)

    region_box = (*point(x_range[0], y_range[1]), *point(x_range[1], y_range[0]))
    draw.rounded_rectangle(region_box, radius=8, fill="#dbeafe", outline="#2563eb", width=5)

    incoming_buffer = min(case["incoming_size"]) / 2
    legal_x = (x_range[0] + incoming_buffer, x_range[1] - incoming_buffer)
    legal_y = (y_range[0] + incoming_buffer, y_range[1] - incoming_buffer)
    center_box = (*point(legal_x[0], legal_y[1]), *point(legal_x[1], legal_y[0]))
    draw.rounded_rectangle(center_box, radius=5, outline="#0891b2", width=4)

    placed_box = box(case["placed_center"], case["placed_size"])
    draw.rounded_rectangle(placed_box, radius=8, fill="#f59e0b", outline="#92400e", width=4)

    # Use the far-left / low-edge legal center: even this extreme candidate
    # overlaps the already placed support in the measured failing scene.
    incoming_center = (legal_x[0], legal_y[0])
    incoming_box = box(incoming_center, case["incoming_size"])
    draw.rounded_rectangle(incoming_box, radius=8, fill="#fca5a5", outline="#b91c1c", width=4)

    draw.text((margin, 35), f"{case['task']} — measured local placement geometry", fill="#172033")
    draw.text((margin, 68), f"Blue: allowed strip {x_range[1]-x_range[0]:.3f} × {y_range[1]-y_range[0]:.3f} m", fill="#2563eb")
    draw.text((margin, 96), f"Orange: placed {case['placed_name']} {case['placed_size'][0]:.3f} × {case['placed_size'][1]:.3f} m", fill="#92400e")
    draw.text((margin, 124), f"Red: {case['incoming_name']} {case['incoming_size'][0]:.3f} × {case['incoming_size'][1]:.3f} m at an extreme legal center", fill="#b91c1c")
    draw.text((margin, height - 78), "Cyan outline: where the incoming object's CENTER may be sampled after boundary padding", fill="#0e7490")
    draw.text((margin, height - 48), "The red and orange footprints still overlap; the full counter outside this blue strip is not considered.", fill="#475569")
    image.save(path)


def _annotated_top(session: SimSession, fixture_id: str, path: Path) -> None:
    image = session.executor.runner._render_top_view()
    result = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fixture = session.executor.runner._fixtures[fixture_id]
    fmin, fmax = get_fixture_aabb(fixture)
    corners = np.asarray(
        [
            [fmin[0], fmin[1]], [fmax[0], fmin[1]],
            [fmax[0], fmax[1]], [fmin[0], fmax[1]],
        ]
    )
    pixels = _world_to_pixels(corners, session, image.shape)
    draw.polygon(
        [tuple(row) for row in pixels],
        fill=(37, 99, 235, 55), outline=(37, 99, 235, 255), width=5,
    )
    result = Image.alpha_composite(result, overlay)
    result.convert("RGB").save(path, quality=94)


def _restriction_top_view(session: SimSession, case: dict, path: Path) -> None:
    """Overlay the measured task sampler geometry on the native top view."""

    image = session.executor.runner._render_top_view()
    result = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    c, s = np.cos(PLACEMENT_REFERENCE_ROTATION), np.sin(PLACEMENT_REFERENCE_ROTATION)
    rotation = np.asarray([[c, -s], [s, c]])

    def world(points):
        return np.asarray(points) @ rotation.T + PLACEMENT_REFERENCE_XY

    def rectangle(center, size):
        cx, cy = center
        sx, sy = size
        return world([
            [cx - sx / 2, cy - sy / 2], [cx + sx / 2, cy - sy / 2],
            [cx + sx / 2, cy + sy / 2], [cx - sx / 2, cy + sy / 2],
        ])

    x_range, y_range = case["region"]
    strip = world([
        [x_range[0], y_range[0]], [x_range[1], y_range[0]],
        [x_range[1], y_range[1]], [x_range[0], y_range[1]],
    ])
    incoming_buffer = min(case["incoming_size"]) / 2
    legal_center = (
        x_range[0] + incoming_buffer,
        y_range[0] + incoming_buffer,
    )
    placed = rectangle(case["placed_center"], case["placed_size"])
    incoming = rectangle(legal_center, case["incoming_size"])

    def polygon(points, fill, outline, width=5):
        pixels = _world_to_pixels(points, session, image.shape)
        draw.polygon([tuple(row) for row in pixels], fill=fill, outline=outline)
        # Pillow's polygon width support varies; reinforce the border.
        draw.line([tuple(row) for row in np.vstack([pixels, pixels[:1]])], fill=outline, width=width)

    polygon(strip, (37, 99, 235, 60), (37, 99, 235, 255), 6)
    polygon(placed, (245, 158, 11, 125), (146, 64, 14, 255), 5)
    polygon(incoming, (239, 68, 68, 115), (185, 28, 28, 255), 5)
    draw.rounded_rectangle((18, 18, 590, 112), radius=8, fill=(255, 255, 255, 225))
    draw.text((32, 30), f"{case['task']} exact sampler overlay", fill=(23, 32, 51, 255))
    draw.text((32, 57), "BLUE restricted strip  ORANGE first support", fill=(37, 99, 235, 255))
    draw.text((32, 82), "RED incoming support at an extreme legal center", fill=(185, 28, 28, 255))
    Image.alpha_composite(result, overlay).convert("RGB").save(path, quality=95)


def _render(style: int, restriction_cases=()) -> dict:
    # GarnishCupcake uses the same stool-referenced dining-counter selection,
    # but has a light enough object set to initialize in both styles. It is a
    # geometry proxy; the two failing target tasks cannot render at style 14.
    trajectory = _trajectory("GarnishCupcake", 0)
    session = SimSession(
        composite_task="GarnishCupcake", sample_trajectory=trajectory,
        layout=15, style=style, seed=42, gl_backend="egl",
        render_size=900, map_dpi=130, map_renderer="raster",
    )
    try:
        adapter = TrajectoryAdapter(
            executor=session.executor, allow_approximate_ids=True
        )
        adapted = adapter.adapt(trajectory)
        session.executor.load_initial_state(adapted["initial_state"])
        fixture_id = adapter._fixture_aliases.get("dining_counter")
        if fixture_id not in session.executor.runner._fixtures:
            raise ValueError(
                f"dining_counter resolved to unavailable fixture {fixture_id!r}"
            )
        top_name = f"layout_15_style_{style}_dining_counter_top.jpg"
        map_name = f"layout_15_style_{style}_map.png"
        _annotated_top(session, fixture_id, ASSETS / top_name)
        restriction_views = {}
        for case in restriction_cases:
            restriction_name = f"{case['task'].lower()}_top_view_restriction.jpg"
            _restriction_top_view(session, case, ASSETS / restriction_name)
            restriction_views[case["task"]] = f"assets/{restriction_name}"
        session.executor.get_image(
            views=["map"], image_paths=[str(ASSETS / map_name)], agent_id="agent_0"
        )
        fixture = session.executor.runner._fixtures[fixture_id]
        fmin, fmax = get_fixture_aabb(fixture)
        return {
            "layout": 15, "style": style, "seed": 42,
            "proxy_task": "GarnishCupcake",
            "resolved_dining_counter": fixture_id,
            "counter_aabb_xy": [fmin[:2].tolist(), fmax[:2].tolist()],
            "counter_size_xy_m": (np.asarray(fmax[:2]) - np.asarray(fmin[:2])).tolist(),
            "top_view": f"assets/{top_name}", "map": f"assets/{map_name}",
            "restriction_views": restriction_views,
        }
    finally:
        session.close()


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = [_render(14, PLACEMENT_CASES), _render(34)]
    for case in PLACEMENT_CASES:
        name = f"{case['task'].lower()}_placement_restriction.png"
        _placement_schematic(case, ASSETS / name)
        case["schematic"] = f"assets/{name}"
        case["top_view"] = rows[0]["restriction_views"][case["task"]]
    probe_results = []
    for case in PLACEMENT_CASES:
        for mode in ("widen_strip", "widen_along_edge", "center_in_strip"):
            result_path = PROBE_OUT / f"{case['task'].lower()}_{mode}.json"
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text())
            if result.get("top_view"):
                source = PROBE_OUT / Path(result["top_view"]).name
                destination = ASSETS / source.name
                shutil.copy2(source, destination)
                result["top_view"] = f"assets/{destination.name}"
            probe_results.append(result)
    (OUT / "metadata.json").write_text(json.dumps(rows, indent=2) + "\n")
    sections = []
    for row in rows:
        label = "failing target-task style" if row["style"] == 14 else "working comparison style"
        sections.append(f'''<section><h2>Layout 15 / style {row['style']}</h2>
        <p><strong>{label}.</strong> The blue overlay marks the dining-counter footprint selected relative to the stool.</p>
        <div class="grid"><figure><a href="{row['top_view']}"><img src="{row['top_view']}"></a><figcaption>Top view</figcaption></figure>
        <figure><a href="{row['map']}"><img src="{row['map']}"></a><figcaption>Occupancy and fixture map</figcaption></figure></div>
        <p>Resolved counter: <code>{html.escape(row['resolved_dining_counter'])}</code>; bounding-box size: {row['counter_size_xy_m'][0]:.2f} × {row['counter_size_xy_m'][1]:.2f} m.</p></section>''')
    schematic_sections = "".join(
        f"""<section><h2>{case['task']} measured placement restriction</h2>
        <p>The diagram is drawn to scale in the sampler's local coordinates. The blue region is only 0.627 × 0.350 m even though the physical counter is much larger.</p>
        <div class="grid"><figure><a href="{case['top_view']}"><img src="{case['top_view']}"></a>
        <figcaption>Exact restricted strip and approximate object footprints on the simulator top view.</figcaption></figure>
        <figure><a href="{case['schematic']}"><img src="{case['schematic']}"></a>
        <figcaption>Same measurements enlarged and drawn to scale.</figcaption></figure></div></section>"""
        for case in PLACEMENT_CASES
    )
    probe_sections = "".join(
        f"""<article><h3>{result['task']} — {result['mode'].replace('_', ' ')}</h3>
        {f'<figure><a href="{result["top_view"]}"><img src="{result["top_view"]}"></a><figcaption>Actual initialized simulator scene, not a schematic.</figcaption></figure>' if result.get('top_view') else ''}
        <p>Initialized: <strong>{result['initialized']}</strong>; native counter contact: <strong>{result.get('native_counter_contact')}</strong>;
        entire measured footprint inside the physical counter AABB: <strong>{result.get('whole_footprint_inside_counter_aabb')}</strong>;
        approximate footprint supported: <strong>{result.get('incoming_bbox_aabb_support_fraction', 'n/a')}</strong>;
        contact after 500 physics steps: <strong>{result.get('native_counter_contact_after_500_steps', 'n/a')}</strong>.</p>
        {f'<p>Error: <code>{html.escape(result["error"])}</code></p>' if result.get('error') else ''}</article>"""
        for result in probe_results
    )
    report = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark"><title>Layout 15 Dining-Counter Placement Failure</title>
    <style>:root{{--bg:#eef2f7;--card:#fff;--ink:#172033;--muted:#526079;--line:#d9e0ea}}@media(prefers-color-scheme:dark){{:root{{--bg:#10151e;--card:#19212d;--ink:#edf2f8;--muted:#abb8ca;--line:#344155}}}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui}}main{{max-width:1400px;margin:auto;padding:28px}}header,section{{background:var(--card);padding:22px;border:1px solid var(--line);border-radius:14px;margin-bottom:18px}}h1,h2{{margin-top:0}}p,figcaption{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}figure{{margin:0}}img{{width:100%;border:1px solid var(--line);border-radius:9px;background:#fff}}code{{font-size:.9em}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><main>
    <header><h1>Why two layout-15/style-14 tasks cannot initialize</h1><p>GarnishCake and MeatSkewerAssembly fail before robot placement: RoboCasa cannot sample their complete object sets onto the stool-referenced dining counter after 50 full retries. Because a failed task has no renderable final scene, these views use GarnishCupcake as a geometry proxy; it selects the same kind of dining counter but has a lighter object set.</p></header>
    <section><h2>The second large support cannot be placed</h2><p>Per-object tracing identifies the exact failure. In <strong>GarnishCake</strong>, RoboCasa places <code>cake_plate</code> first and then cannot place <code>fruit_plate</code>. In <strong>MeatSkewerAssembly</strong>, it places <code>skewer_plate</code> first and then cannot place <code>oven_tray</code>. Both tasks constrain these two large movable supports to overlapping, stool-referenced strips on the same dining counter. Every one of the 50 complete retries fails at that second support.</p><p>The gross counter bounding box is identical in these two rendered styles, which is itself informative: the failure is not visible merely as a shorter counter. It arises from the narrow task placement restriction and the sampled support footprints. These views show the involved workspace honestly; a completed style-14 target-task scene cannot be shown because RoboCasa never constructs one.</p></section>
    {schematic_sections}
    <section><h2>Actual relaxation probes</h2><p>These are real probes of the two previously failing tasks on layout 15/style 14/seed 42. “Widen strip” expands across counter depth; “widen along edge” expands the orthogonal requested dimension; “center in strip” permits the object footprint to cross the preferred-strip boundary.</p><div class="grid">{probe_sections}</div><p><strong>Result:</strong> widening across depth initialized both tasks but biased placement inward. Widening along the counter edge still failed both tasks because the stool-relative region clips that dimension. Center-only placement initialized both tasks and remained in contact after 500 physics steps. GarnishCake retained approximately 78.7% of its bounding-box footprint over the counter; MeatSkewerAssembly retained 100%.</p></section>
    {''.join(sections)}
    <section><h2>Why RoboCasa did not exclude it</h2><p>RoboCasa exclusions are manually declared by each task through <code>EXCLUDE_LAYOUTS</code> and <code>EXCLUDE_STYLES</code>. Both tasks inherit the dining-counter layout exclusions but declare no style exclusions. The runtime therefore considers layout 15/style 14 eligible, retries stochastic object placement 50 times, and only then fails. This audit discovered a missing task-specific style exclusion rather than violating an existing native restriction.</p></section>
    </main></body></html>'''
    (OUT / "report.html").write_text(report)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

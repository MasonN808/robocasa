"""Capture publication-oriented initial overviews for PlateStoreDinner traj 88."""

from __future__ import annotations

import json
from pathlib import Path

from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY = Path(
    "/work/umass/shlomo_umass/dbenhamougol_umass/"
    "tick53x150_state_grounded_cascade_v1_rendered/plate_store_dinner/"
    "traj_000088/original_trajectory.json"
)
OUT = ROOT / "training/bc_task_vlm/reports/plate_store_dinner_traj088_overview"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trajectory = json.loads(TRAJECTORY.read_text())
    session = SimSession(
        composite_task="PlateStoreDinner",
        sample_trajectory=trajectory,
        layout=50,
        style=34,
        seed=7,
        gl_backend="egl",
        render_size=1200,
        map_dpi=120,
        map_renderer="raster",
    )
    try:
        session.start_trajectory(trajectory)
        runner = session.executor.runner
        base = dict(runner._room_cam_config)
        variants = {
            "current_highres": base,
            "tight_80pct": {**base, "distance": base["distance"] * 0.80},
            "tight_70pct": {**base, "distance": base["distance"] * 0.70},
            "tight_65pct": {**base, "distance": base["distance"] * 0.65},
            "tight_70pct_rotated": {
                **base,
                "distance": base["distance"] * 0.70,
                "azimuth": base["azimuth"] + 12.0,
            },
            # Moving closer without retargeting shifts this asymmetric room to
            # the right of the frame. Shift the look-at point along the
            # camera's horizontal axis so the visible kitchen, rather than its
            # world-space bounding-box center, is centered in the image.
            "centered_shift_075": {
                **base,
                "lookat": [base["lookat"][0] - 0.75, base["lookat"][1], base["lookat"][2]],
                "distance": base["distance"] * 0.84,
            },
            "centered_shift_125": {
                **base,
                "lookat": [base["lookat"][0] - 1.25, base["lookat"][1], base["lookat"][2]],
                "distance": base["distance"] * 0.84,
            },
            "centered_shift_175": {
                **base,
                "lookat": [base["lookat"][0] - 1.75, base["lookat"][1], base["lookat"][2]],
                "distance": base["distance"] * 0.84,
            },
        }
        for name, config in variants.items():
            image = runner._render_free_camera(config)
            from PIL import Image

            Image.fromarray(image).save(OUT / f"{name}.png")

        # Paper-oriented landscape alternatives. In this camera orientation,
        # increasing world X moves the visible room left within the frame.
        runner.render_width = 1400
        runner.render_height = 900
        landscape_variants = {
            "landscape_centered_shift_075": {
                **base,
                "lookat": [base["lookat"][0] + 0.75, base["lookat"][1], base["lookat"][2]],
                "distance": base["distance"] * 0.76,
            },
            "landscape_centered_shift_125": {
                **base,
                "lookat": [base["lookat"][0] + 1.25, base["lookat"][1], base["lookat"][2]],
                "distance": base["distance"] * 0.76,
            },
            "landscape_centered_shift_175": {
                **base,
                "lookat": [base["lookat"][0] + 1.75, base["lookat"][1], base["lookat"][2]],
                "distance": base["distance"] * 0.76,
            },
            "landscape_room_fill_050": {
                **base,
                "lookat": [base["lookat"][0] + 1.25, base["lookat"][1], base["lookat"][2] + 0.35],
                "distance": base["distance"] * 0.50,
            },
            "landscape_room_fill_055": {
                **base,
                "lookat": [base["lookat"][0] + 1.25, base["lookat"][1], base["lookat"][2] + 0.35],
                "distance": base["distance"] * 0.55,
            },
            "landscape_room_fill_060": {
                **base,
                "lookat": [base["lookat"][0] + 1.25, base["lookat"][1], base["lookat"][2] + 0.35],
                "distance": base["distance"] * 0.60,
            },
            "landscape_close_centered_042": {
                **base,
                "lookat": [base["lookat"][0] + 1.65, base["lookat"][1], base["lookat"][2] + 0.05],
                "distance": base["distance"] * 0.42,
            },
            "landscape_close_centered_045": {
                **base,
                "lookat": [base["lookat"][0] + 1.65, base["lookat"][1], base["lookat"][2] + 0.05],
                "distance": base["distance"] * 0.45,
            },
            "landscape_close_centered_047": {
                **base,
                "lookat": [base["lookat"][0] + 1.65, base["lookat"][1], base["lookat"][2] + 0.05],
                "distance": base["distance"] * 0.47,
            },
        }
        for name, config in landscape_variants.items():
            Image.fromarray(runner._render_free_camera(config)).save(OUT / f"{name}.png")
        variants.update(landscape_variants)

        # Put the camera itself inside the kitchen and compensate for the much
        # shorter distance with a wider lens. These are interior scene views,
        # not exterior cutaway zooms.
        original_fovy = float(runner.env.sim.model.vis.global_.fovy)
        interior_variants = {
            "interior_wide_016": {
                "config": {
                    **base,
                    "lookat": [base["lookat"][0] + 1.65, base["lookat"][1], base["lookat"][2] + 0.10],
                    "distance": base["distance"] * 0.16,
                },
                "fovy": 82.0,
            },
            "interior_wide_019": {
                "config": {
                    **base,
                    "lookat": [base["lookat"][0] + 1.65, base["lookat"][1], base["lookat"][2] + 0.10],
                    "distance": base["distance"] * 0.19,
                },
                "fovy": 76.0,
            },
            "interior_wide_022": {
                "config": {
                    **base,
                    "lookat": [base["lookat"][0] + 1.65, base["lookat"][1], base["lookat"][2] + 0.10],
                    "distance": base["distance"] * 0.22,
                },
                "fovy": 70.0,
            },
        }
        for name, variant in interior_variants.items():
            runner.env.sim.model.vis.global_.fovy = variant["fovy"]
            Image.fromarray(runner._render_free_camera(variant["config"])).save(OUT / f"{name}.png")

        centered_interior_variants = {
            "interior_centered_down_24": {
                "config": {
                    **base,
                    "lookat": [base["lookat"][0] + 2.05, base["lookat"][1], base["lookat"][2] - 0.10],
                    "distance": base["distance"] * 0.19,
                    "elevation": -24.0,
                },
                "fovy": 76.0,
            },
            "interior_centered_down_28": {
                "config": {
                    **base,
                    "lookat": [base["lookat"][0] + 2.05, base["lookat"][1], base["lookat"][2] - 0.10],
                    "distance": base["distance"] * 0.19,
                    "elevation": -28.0,
                },
                "fovy": 76.0,
            },
            "interior_centered_down_32": {
                "config": {
                    **base,
                    "lookat": [base["lookat"][0] + 2.05, base["lookat"][1], base["lookat"][2] - 0.10],
                    "distance": base["distance"] * 0.19,
                    "elevation": -32.0,
                },
                "fovy": 76.0,
            },
            "interior_centered_down_36": {
                "config": {
                    **base,
                    "lookat": [base["lookat"][0] + 2.05, base["lookat"][1], base["lookat"][2] - 0.10],
                    "distance": base["distance"] * 0.19,
                    "elevation": -36.0,
                },
                "fovy": 76.0,
            },
            "interior_centered_down_40": {
                "config": {
                    **base,
                    "lookat": [base["lookat"][0] + 2.05, base["lookat"][1], base["lookat"][2] - 0.10],
                    "distance": base["distance"] * 0.19,
                    "elevation": -40.0,
                },
                "fovy": 76.0,
            },
            "interior_lower_tilt_up_30": {
                "config": {
                    **base,
                    "lookat": [base["lookat"][0] + 2.05, base["lookat"][1], 0.85],
                    "distance": base["distance"] * 0.19,
                    "elevation": -30.0,
                },
                "fovy": 76.0,
            },
            "interior_lower_tilt_up_32": {
                "config": {
                    **base,
                    "lookat": [base["lookat"][0] + 2.05, base["lookat"][1], 0.75],
                    "distance": base["distance"] * 0.19,
                    "elevation": -32.0,
                },
                "fovy": 76.0,
            },
            "interior_lower_tilt_up_34": {
                "config": {
                    **base,
                    "lookat": [base["lookat"][0] + 2.05, base["lookat"][1], 0.65],
                    "distance": base["distance"] * 0.19,
                    "elevation": -34.0,
                },
                "fovy": 76.0,
            },
        }
        for name, variant in centered_interior_variants.items():
            runner.env.sim.model.vis.global_.fovy = variant["fovy"]
            Image.fromarray(runner._render_free_camera(variant["config"])).save(OUT / f"{name}.png")
        runner.env.sim.model.vis.global_.fovy = original_fovy
        variants.update({name: value["config"] | {"fovy": value["fovy"]} for name, value in interior_variants.items()})
        variants.update({name: value["config"] | {"fovy": value["fovy"]} for name, value in centered_interior_variants.items()})
        (OUT / "camera_configs.json").write_text(
            json.dumps({"scene": {"layout": 50, "style": 34, "seed": 7}, "variants": variants}, indent=2)
            + "\n"
        )
        print(json.dumps({"output": str(OUT), "base": base}, indent=2))
    finally:
        session.close()


if __name__ == "__main__":
    main()

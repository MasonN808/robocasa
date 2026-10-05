"""Plotting one figure must not require the cells of the other figures."""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((REPO_ROOT / "configs/experiments.yaml").read_text())


def _fake_cell(root: Path, model_id: str) -> None:
    for split, episodes in (("train_task_types", 43), ("heldout_task_types", 10)):
        out = root / "eval" / model_id / split / "aggregate"
        out.mkdir(parents=True)
        rows = [
            {"trajectory_id": f"t{i}", "fsm_goal_satisfied": i % 2 == 0, "rejected_total": 0, "steps": []}
            for i in range(episodes)
        ]
        (out / "live_sim_trajectories.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")


def test_fig7_plots_without_the_fig6_cells(tmp_path):
    for model_id, model in CONFIG["models"].items():
        if model["figures"] == ["fig7"] or model_id in ("instruct_s30", "thinking_rationale_s30"):
            _fake_cell(tmp_path, model_id)
    env = {**os.environ, "ROBOTALK_OUTPUT_ROOT": str(tmp_path), "MPLBACKEND": "Agg"}
    for module in ("robotalk.analysis.paper_results", "robotalk.analysis.export_sft_scale_pngs"):
        result = subprocess.run(
            [sys.executable, "-m", module], cwd=REPO_ROOT, env=env,
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, result.stderr[-2000:]
    figures = tmp_path / "figures" / "43_10"
    assert (figures / "thinking_rationale_30task_ablations.png").exists()
    assert not (figures / "sft_scaling_combined.png").exists()
    assert "Fig. 6: skipped" in result.stdout

"""8B interface 2x2: forced JSON vs native tool calling, base and SFT."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = Path("training/bc_task_vlm/eval_runs")
rows = list(csv.DictReader(open(BASE / "results_table.csv")))


def get(model, prompt, split, key):
    for r in rows:
        if r["model"] == model and r["prompt"] == prompt and r["split"] == split:
            v = r.get(key)
            return float(v) if v not in (None, "", "-") else None
    return None


# model group -> (JSON-forced prompt label, native prompt label)
GROUPS = [
    ("Qwen3-VL-8B\nbase", "Qwen3-VL-8B base"),
    ("Qwen3-VL-8B\nSFT", "Qwen3-VL-8B SFT"),
]
JSON_PROMPT = "baseline (forced-JSON control)"
NATIVE_PROMPT = "baseline *"
ROWS = [("judged_overall_acc", "Judged overall accuracy"),
        ("judged_traj_all", "Full trajectories completed")]
SPLITS = [("heldout_trajectories", "Held-out trajectories (trained tasks)"),
          ("heldout_tasks", "Held-out tasks (never trained on)")]
JSON_C, NATIVE_C = "#2a78d6", "#eda100"
INK, MUTED, GRID = "#37352f", "#787066", "#e6e4dd"

fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True)
bar_w = 0.34

for ri, (mkey, mlabel) in enumerate(ROWS):
    for ci, (skey, stitle) in enumerate(SPLITS):
        ax = axes[ri][ci]
        for gi, (glabel, model) in enumerate(GROUPS):
            jv = get(model, JSON_PROMPT, skey, mkey)
            nv = get(model, NATIVE_PROMPT, skey, mkey)
            for off, val, color in ((-bar_w / 2, jv, JSON_C), (bar_w / 2, nv, NATIVE_C)):
                if val is None:
                    continue
                ax.bar([gi + off], [val], width=bar_w - 0.03, color=color, zorder=3)
                ax.annotate(f"{val:.3f}".lstrip("0") if val < 0.1 else f"{val:.2f}".lstrip("0"),
                            (gi + off, val), textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=9, color=INK,
                            fontweight="bold" if (mkey == "judged_traj_all" and val > 0) else "normal")
            if jv is not None and nv is not None:
                delta = nv - jv
                dcolor = MUTED if abs(delta) < 1e-9 else ("#1baf7a" if delta > 0 else "#c2410c")
                ax.annotate(f"{delta:+.3f}".rstrip("0").rstrip(".") if delta else "0",
                            (gi, max(jv, nv) + (0.055 if ri == 0 else 0.018)), ha="center",
                            fontsize=9, color=dcolor, fontweight="bold")
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([g[0] for g in GROUPS], fontsize=10.5)
        ax.set_ylim(0, 1.0 if ri == 0 else 0.30)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0, colors=MUTED)
        if ri == 0:
            ax.set_title(stitle, fontsize=11.5, color=INK)
        if ci == 0:
            ax.set_ylabel(mlabel, fontsize=10.5, color=INK)

fig.legend(handles=[Patch(facecolor=JSON_C, label="Forced JSON output"),
                    Patch(facecolor=NATIVE_C, label="Native tool calling (trained format)")],
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.945), fontsize=10.5)
fig.suptitle("Native tool calling beats forced JSON at both scales — for the base model per step, "
             "for the SFT model per trajectory",
             fontsize=13.5, fontweight="bold", y=0.995)
fig.text(0.5, 0.052,
         "Qwen3-VL-8B, identical model / prompt / images throughout; only the answer interface differs. "
         "Deltas above bars are native − forced JSON.",
         ha="center", fontsize=9, color=MUTED)
fig.text(0.5, 0.032,
         "Base gains most per step (action-exact +.19 / +.15) yet completes ZERO full trajectories either way "
         "— the interface cannot substitute for fine-tuning.",
         ha="center", fontsize=9, color=MUTED)
fig.text(0.5, 0.012,
         "For the SFT model native's payoff concentrates where it matters: full held-out-task trajectories "
         "roughly triple, .080 → .253.",
         ha="center", fontsize=9, color=MUTED)
fig.tight_layout(rect=[0, 0.075, 1, 0.9])
fig.savefig(BASE / "interface_comparison.png", dpi=200, facecolor="white")
print("wrote interface_comparison.png")

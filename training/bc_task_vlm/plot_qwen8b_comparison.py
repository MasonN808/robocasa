"""Qwen3-VL-8B only: does fine-tuning subsume prompt engineering?

Interface is held constant (forced JSON) across the prompt axis so the only
variable is the prompt aids; the separate interface figure varies the interface.
"""
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


# (csv prompt label, display label). Baseline is the forced-JSON control so all
# four prompt configs share one interface.
PROMPTS = [("baseline (forced-JSON control)", "baseline"), ("+task spec", "+task spec"),
           ("+few-shot", "+few-shot"), ("+spec+few-shot", "+spec\n+few-shot")]
SERIES = [("Qwen3-VL-8B base", "base (no fine-tuning)", "#2a78d6"),
          ("Qwen3-VL-8B SFT", "SFT (fine-tuned)", "#1baf7a")]
ROWS = [("judged_overall_acc", "Judged overall accuracy"),
        ("judged_traj_all", "Full trajectories completed")]
SPLITS = [("heldout_trajectories", "Held-out trajectories (trained tasks)"),
          ("heldout_tasks", "Held-out tasks (never trained on)")]
INK, MUTED, GRID = "#37352f", "#787066", "#e6e4dd"

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
bar_w = 0.34

for ri, (mkey, mlabel) in enumerate(ROWS):
    for ci, (skey, stitle) in enumerate(SPLITS):
        ax = axes[ri][ci]
        for si, (model, slabel, color) in enumerate(SERIES):
            xs, vals = [], []
            for pi, (prompt, _) in enumerate(PROMPTS):
                v = get(model, prompt, skey, mkey)
                if v is None:
                    continue
                xs.append(pi + (si - 0.5) * bar_w)
                vals.append(v)
            bars = ax.bar(xs, vals, width=bar_w - 0.03, color=color, zorder=3, label=slabel)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.3f}".lstrip("0") if v < 0.1 else f"{v:.2f}".lstrip("0"),
                            (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=8.6, color=INK,
                            fontweight="bold" if (mkey == "judged_traj_all" and v > 0) else "normal")
        ax.set_xticks(range(len(PROMPTS)))
        ax.set_xticklabels([p[1] for p in PROMPTS], fontsize=10)
        ax.set_ylim(0, 1.0 if ri == 0 else 0.25)
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

fig.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in SERIES],
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.945), fontsize=10)
fig.suptitle("Fine-tuning subsumes prompt engineering — and a worked example then hurts",
             fontsize=14, fontweight="bold", y=0.995)
fig.text(0.5, 0.052,
         "Qwen3-VL-8B, forced-JSON interface throughout; only fine-tuning and prompt aids vary. "
         "(In its native format the SFT baseline is higher still, .878 / .675 — see the interface figure.)",
         ha="center", fontsize=9, color=MUTED)
fig.text(0.5, 0.030,
         "Prompt aids barely move the base model (.247 -> .290 at best) and it never completes a trajectory.",
         ha="center", fontsize=9, color=MUTED)
fig.text(0.5, 0.008,
         "The task spec is the fine-tuned model's best prompt (.878, 16% of trajectories); adding a worked example "
         "collapses it (.868 -> .550, trajectories -> 0).",
         ha="center", fontsize=9, color=MUTED)
fig.tight_layout(rect=[0, 0.075, 1, 0.9])
fig.savefig(BASE / "qwen8b_comparison.png", dpi=200, facecolor="white")
print("wrote qwen8b_comparison.png")

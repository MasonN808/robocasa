# RoboTalk

**Learning Multi-Robot Communication and Coordination from Multimodal Demonstrations**

Dorian Benhamou Goldfajn\*, Mason Nakamura\*, Saaduddin Mahmud, Justin Svegliato, Kyle H. Wray, Shlomo Zilberstein
(\* equal contribution)

[Paper (arXiv:2609.23997)](https://arxiv.org/abs/2609.23997) ·
[Dataset](https://huggingface.co/datasets/DorianAtSchool/RoboTalk) ·
[Explorer](https://huggingface.co/spaces/DorianAtSchool/RoboTalk-Explorer) ·
[Models](https://huggingface.co/DorianAtSchool) ·
[PDF in this repo](docs/paper/RoboTalk_arxiv.pdf)

RoboTalk is a synthetic data-generation pipeline and dataset for training small
vision-language models (VLMs) to act as decentralized policies for two mobile
manipulators. It contains 7,950 two-robot trajectories across 53 kitchen tasks
adapted from [RoboCasa365](https://robocasa.ai). Each trajectory interleaves
both robots' camera observations, natural-language messages, skill-level tool
calls (perception, navigation, manipulation and communication) and one-sentence
rationales. Coordination follows a leader–follower planning protocol,
wait–release synchronization and exclusive-workspace rules, and a
finite-state machine (FSM) validates every trajectory.

This repository contains:
- the two-robot RoboCasa365 simulator and the FSM validator;
- the generation pipeline (Gemini 3 Flash with a bounded retry cascade);
- LoRA fine-tuning of Qwen3-VL-8B;
- closed-loop live-simulation evaluation;
- one command per paper figure (Figs. 3 and 5–7).

## Installation

Linux with Python 3.11. [uv](https://docs.astral.sh/uv/) installs the exact
versions in `uv.lock`:

```bash
git clone https://github.com/DorianAtSchool/robotalk.git
cd robotalk
uv sync --all-extras          # or pick extras: --extra train --extra gen --extra flash
source .venv/bin/activate

python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets   # RoboCasa365 assets: ~10 GB download, 23 GB on disk
cp .env.example .env          # optional: data/output roots, Gemini credentials, vLLM launcher
```

| Extra | For |
|---|---|
| (none) | Simulator, FSM validator, dataset tools, live-sim evaluation client |
| `train` | LoRA fine-tuning (`torch`, `transformers`, `peft`, `accelerate`) |
| `flash` | FlashAttention-2, as used for the paper's training runs (prebuilt wheel for Python 3.11) |
| `gen` | Trajectory generation and the Gemini evaluation cells (`google-genai`) |
| `dev` | `pytest` |

robosuite is installed from
[DorianAtSchool/robosuite](https://github.com/DorianAtSchool/robosuite) at the
commit used for the paper. That is upstream robosuite plus one fix that gives
each robot's mounting body a unique name, which scenes with two mobile
manipulators need.

> [!IMPORTANT]
> **GPU requirements.** `torch` comes from PyTorch's CUDA 12.8 package index,
> as in the paper's environment. Those builds support NVIDIA GPUs of compute
> capability 7.5–12.0 (Turing, Ampere, Hopper and Blackwell) and need a
> driver that supports CUDA 12.8 (version 570 or newer). They are **required
> on Blackwell GPUs** (e.g. RTX PRO 6000, B200): the default PyPI build of
> torch 2.7.1 fails there with "no kernel image is available". Volta (V100)
> and older GPUs are **not supported**. On those machines, point `torch` and
> `torchvision` at `https://download.pytorch.org/whl/cu126` in
> `pyproject.toml` (`[[tool.uv.index]]`). The paper trained on 4× B200 and
> evaluated on single RTX PRO 6000 GPUs.

**vLLM.** Open-weight models are evaluated through a local
[vLLM](https://github.com/vllm-project/vllm) 0.27.1 server. It pins its own
torch, so install it in a separate environment or run it from the official
container. Set `VLLM_LAUNCHER` to the command prefix, for example
`VLLM_LAUNCHER="apptainer exec --nv vllm-openai-v0.27.1.sif"`.

**FlashAttention.** The `flash` extra installs a prebuilt wheel for CUDA 12,
torch 2.7 and Python 3.11. Elsewhere, train with `--attn-implementation sdpa`.

## Quickstart

Load the dataset:

```python
from datasets import load_dataset

trajectories = load_dataset("DorianAtSchool/RoboTalk", "trajectories", split="train")
ticks = load_dataset("DorianAtSchool/RoboTalk", "ticks", split="train")
```

Replay demonstrations in the simulator. This downloads one task, rebuilds the
per-trajectory layout and replays two trajectories with the expert ("oracle")
policy:

```bash
hf download DorianAtSchool/RoboTalk --repo-type dataset --revision v1.1 \
    --include "data/trajectories.parquet" "raw/prepare_coffee/*" "media_archives/prepare_coffee__*" \
    --local-dir data/hf
python -m robotalk.release.materialize_training_layout \
    --hf-dir data/hf --output data/robotalk_rendered --tasks prepare_coffee
MUJOCO_GL=egl python -m robotalk.evaluation.live_sim_eval --backend oracle \
    --concurrent-expert-replay --manifest configs/eval/oracle_smoke.json \
    --tasks prepare_coffee --dataset-root data/robotalk_rendered --output-dir outputs/oracle
```

Episode outcomes go to `outputs/oracle/live_sim_trajectories.jsonl`, and
videos of each robot's cameras to `outputs/oracle/recordings/`.

Run a fine-tuned policy for one episode per task on two tasks. This needs a
GPU and vLLM:

```bash
python scripts/reproduce.py fig6 --models instruct_s30 \
    --tasks prepare_coffee,arrange_bread_bowl --episodes-per-task 1
```

## Results

Every number is closed-loop **error-free FSM success**: the episode reaches the
task goal with no rejected or failed tool call, within a budget of four times
the length of the task's example solution. Each model is evaluated on 430
episodes of the 43 training tasks and 100 episodes of the 10 held-out tasks
(10 per task, fixed cohort). Error bars in the paper are 95% Wilson
intervals.

`scripts/reproduce.py` runs a figure end to end:
- it downloads the dataset (v1.1) and the released adapters;
- it starts vLLM and evaluates every cell on both splits;
- it plots the figure into `outputs/figures/43_10/`.

Finished cells are skipped, so an interrupted run resumes when you rerun the
same command. `--dry-run` prints the commands without running them.
`--train` retrains the adapters instead of downloading them.
`scripts/slurm/reproduce.sbatch` wraps it for SLURM.

### Fig. 3: task phases

| Phase (stages) | 43 training tasks | 10 held-out tasks |
|---|---:|---:|
| Phase 2 (2–3) | 12 | 2 |
| Phase 3 (4–5) | 12 | 2 |
| Phase 4 (6+) | 19 | 6 |

```bash
python scripts/reproduce.py fig3        # seconds, CPU only
```

### Fig. 5: communication guidance, untuned models

| Condition | Gemini 3 Flash (train / held-out) | Qwen3-VL-8B-Instruct (train / held-out) |
|---|---:|---:|
| None | 7.7% / 5% | 0.2% / 0% |
| Unguided | 9.5% / 8% | 0.5% / 0% |
| Minimal | 43.3% / 34% | 1.2% / 0% |
| Intermediate | 46.0% / 46% | 1.6% / 0% |
| Full | 65.6% / 72% | 0.0% / 0% |

```bash
python scripts/reproduce.py fig5
```

Fig. 5 needs Gemini credentials (see `.env.example`) and one GPU for Qwen.
- **Workload:** 10 cells of 530 two-robot episodes each. The 5 Qwen cells run on one GPU, like the Fig. 6 cells; the 5 Gemini cells make 2,650 episodes' worth of API calls, for which no cost was recorded.
- **Gemini outcomes vary:** Gemini is not deterministic through the API, so expect rates within the confidence intervals rather than identical episodes.
- **Cohort difference:** the paper ran this ablation on an earlier 47/6 task split and regrouped the results to 43/10. A rerun on the 43/10 cohort draws different episodes for the 4 tasks that moved to the held-out split.

### Fig. 6: SFT scaling (trajectories per training task)

| Model | 30 | 60 | 90 | 120 | 150 |
|---|---:|---:|---:|---:|---:|
| Instruct, train tasks | 81.9% | 87.9% | 92.1% | 93.0% | 94.2% |
| Instruct, held-out tasks | 45% | 64% | 55% | 48% | 65% |
| Thinking + rationale, train tasks | 80.2% | 84.7% | 91.4% | 90.2% | 92.6% |
| Thinking + rationale, held-out tasks | 71% | 72% | 66% | 77% | 73% |

```bash
python scripts/reproduce.py fig6                         # released adapters
python scripts/reproduce.py fig6 --train                 # retrain first
python scripts/reproduce.py fig6 --models instruct_s30   # one cell
```

**Compute for Fig. 6:**
- **Evaluation:** about 1 hour per model on one GPU, with 4 simulator workers sharing the vLLM server. Thinking models take about 30% longer.
- **Retraining:** about 2 hours per 30 trajectories per task on 4× B200. That is 2 h for the 30/task models and 9.7 h for the 150/task models: about 59 h of wall time, or 235 B200 GPU-hours, for all ten adapters.
- **Variance:** Thinking models are evaluated with sampling (temperature 0.6), so their rates vary from run to run.

### Fig. 7: rationale ablation at 30 trajectories per task

| Model and targets | Train tasks | Held-out tasks |
|---|---:|---:|
| Instruct, tool calls | 81.9% | 45% |
| Instruct, rationale + tool calls | 82.6% | 58% |
| Thinking, tool calls† | 71.6% | 50% |
| Thinking, rationale + tool calls | 80.2% | 71% |

```bash
python scripts/reproduce.py fig7 [--train]
```

† Evaluated with `--recover-missing-open-tool-tag`, which accepts a tool call
whose opening `<tool_call>` tag is missing.

Exact settings for every cell are in `configs/experiments.yaml`:
training hyperparameters, decoding profiles, vLLM arguments, splits and
seeds. The adapters were trained on dataset v1.0.

**Dataset versions.** v1.1 replaces 95 trajectories (1.2%) that failed in
simulation; see the dataset card. Retraining on v1.1 should match the paper
within its confidence intervals.

## Repository structure

```text
robotalk/
  tasks/        53 task specifications, symbolic state, concurrent FSM validator
  tools/        the tool interface shared by generation, training and evaluation
  generation/   Gemini trajectory generation, observation insertion, rendering sweep
  training/     example construction, preprocessing, LoRA SFT
  evaluation/   live-simulation evaluation (vLLM, Gemini, oracle backends)
  analysis/     paper result aggregation and figures
  release/      Hugging Face export, training-layout rebuild, explorer Space
robocasa/       RoboCasa365 with two-robot support and the skill executor (docs/FORK_CHANGES.md)
configs/        experiment settings, task split, scale subsets, evaluation cohort
scripts/        reproduce.py and a SLURM wrapper
tests/          unit and simulator tests
docs/           paper PDF and technical notes
```

Technical notes:
- [concurrency semantics](docs/concurrency_semantics.md)
- [tool interface](docs/tool_interface.md)
- [data format](docs/data_format.md)
- [evaluation](docs/evaluation.md)

## Generating new data

Generation needs the `gen` extra and Gemini access, either `GOOGLE_API_KEY` or
Vertex AI credentials. It runs in three stages:
1. **Generate** symbolic trajectories with the production cascade.
2. **Insert observations.** This adds `get_image` calls and revalidates.
3. **Render** the trajectories in the simulator.

```bash
python -m robotalk.generation.raw.production_cascade --help
python -m robotalk.generation.image.cli --help
python -m robotalk.generation.sweep_trajectories --help
```

`robotalk.release.export_robotalk` converts the result to the published
dataset layout.

## Limitations

- Skills run through scripted simulator tools. RoboTalk covers high-level
  planning, communication and coordination, not low-level motor control.
- Success is judged by a symbolic FSM that abstracts some physical details. For
  example, placing the spatula in the pot counts as stirring in CheeseMixing.
- There are two robots, 53 tasks and a certified set of 60 kitchen scenes.
  Held-out generalization is measured on 10 task types.
- Each result comes from one training run per configuration. The confidence
  intervals do not capture variation across training seeds.
- Messages and rationales are written by Gemini and inherit its style.

## License

| Component | License |
|---|---|
| RoboTalk code | [Apache-2.0](LICENSE) |
| RoboCasa365 code in `robocasa/` | MIT, © the RoboCasa Team ([robocasa/LICENSE](robocasa/LICENSE)) |
| RoboCasa365 assets (downloaded separately) | CC BY 4.0 |
| Dataset and adapters | Apache-2.0 |

See [NOTICE](NOTICE). The trajectories were generated with Google Gemini, and
users are responsible for complying with the applicable model terms of
service.

## Citation

```bibtex
@article{robotalk2026,
  title   = {RoboTalk: Learning Multi-Robot Communication and Coordination from Multimodal Demonstrations},
  author  = {Benhamou Goldfajn, Dorian and Nakamura, Mason and Mahmud, Saaduddin and Svegliato, Justin and Wray, Kyle H. and Zilberstein, Shlomo},
  journal = {arXiv preprint arXiv:2609.23997},
  year    = {2026}
}
```

Please also cite RoboCasa365, RoboCasa and robosuite, on which the simulator
is built:

```bibtex
@inproceedings{robocasa365,
  title     = {RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots},
  author    = {Soroush Nasiriany and Sepehr Nasiriany and Abhiram Maddukuri and Yuke Zhu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}

@inproceedings{robocasa2024,
  title     = {RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots},
  author    = {Soroush Nasiriany and Abhiram Maddukuri and Lance Zhang and Adeet Parikh and Aaron Lo and Abhishek Joshi and Ajay Mandlekar and Yuke Zhu},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2024}
}

@article{robosuite2020,
  title   = {robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author  = {Yuke Zhu and Josiah Wong and Ajay Mandlekar and Roberto Mart{\'\i}n-Mart{\'\i}n and Abhishek Joshi and Soroush Nasiriany and Yifeng Zhu and Kevin Lin},
  journal = {arXiv preprint arXiv:2009.12293},
  year    = {2020}
}
```

## Acknowledgements

RoboTalk builds on [RoboCasa365](https://github.com/robocasa/robocasa) and
[robosuite](https://github.com/ARISE-Initiative/robosuite). This research was
supported in part by the National Science Foundation under Grant Nos. 2321786,
2326054 and 2416460, and by Schmidt Sciences under the AI Safety Science
program. Mason Nakamura was supported by an NSF Graduate Research Fellowship
under Grant No. 2439846. This work used Google Cloud through the CloudBank
project, which is supported by National Science Foundation grant No. 1925001.

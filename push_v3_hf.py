import sys
from huggingface_hub import HfApi

api = HfApi()
repo_id = "DorianAtSchool/qwen3vl-8b-robocasa-v3-active-observation"

api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
print("repo created/exists", flush=True)

local_dir = "/work/umass/shlomo_umass/dbenhamougol_umass/robocasa/training/bc_task_vlm/runs/qwen3vl-8b-v3-active-observation"
for fname in ["README.md", "adapter_config.json", "adapter_model.safetensors",
              "train_results.json", "eval_results.json", "final_metrics.json"]:
    api.upload_file(
        path_or_fileobj=f"{local_dir}/{fname}",
        path_in_repo=fname,
        repo_id=repo_id,
        repo_type="model",
    )
    print("uploaded", fname, flush=True)

print("Done:", f"https://huggingface.co/{repo_id}", flush=True)

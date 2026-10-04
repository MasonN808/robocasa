# RoboTalk Hugging Face export

This directory contains the reproducible exporter and Hugging Face Space for
publishing the canonical 7,950-trajectory RoboTalk dataset.

The publication representation has two layers:

- one searchable Parquet row per trajectory;
- one Parquet row per concurrent tick, with links to the latest observation
  available to each agent at that tick.

The original trajectory JSON remains the source of truth.  A later production
pass will additionally encode the observation streams as LeRobot v3 MP4
features.  The smoke export deliberately uses image files so tick alignment can
be reviewed before video encoding.

## Build the smoke export

```bash
/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python \
  training/bc_task_vlm/robotalk_hf/export_robotalk.py \
  --output training/bc_task_vlm/reports/robotalk_hf_smoke
```

The resulting `preview.html` is a local, dependency-free review surface. The
free static Space implementation is in `static_space/`; the exporter packages
it together with the smoke data under the output directory's `space/` folder.

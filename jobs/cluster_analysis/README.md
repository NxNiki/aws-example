# Cluster analysis (`jobs/cluster_analysis`)

K-means clustering workflows driven by YAML config. Core logic lives in **`ClusterAnalysisPipeline`** in [`src/bituslabs_ds/ml.py`](../../src/bituslabs_ds/ml.py). Run scripts from the repository root, for example:

```bash
poetry run python jobs/cluster_analysis/cluster_analysis_pipeline.py --config_file jobs/cluster_analysis/cluster_config-wucaishen.yaml
```

---

## Overview

- **One class, one config per project**: `ClusterAnalysisPipeline` loads a YAML file that defines S3 data sources, features, pipeline steps, and model parameters.
- **Configurable pipeline**: Elbow analysis, training, optional inference, attaching cluster labels, cluster stats, and optional S3 upload are toggled in YAML under `pipeline:`.
- **Feature-driven**: Normal vs skewed features, key/merge columns, and data loading (bucket, prefix, patterns, caches) are declared in the config—no hardcoded project lists in code.

---

## `ClusterAnalysisPipeline`

Initialize with the **path to a project YAML** (not a short name—the file carries `project_name`, `work_dir`, etc.):

```python
from bituslabs_ds.ml import ClusterAnalysisPipeline

pipeline = ClusterAnalysisPipeline("jobs/cluster_analysis/cluster_config-wucaishen.yaml")
data = pipeline.load_cluster_data(reload=False)
```

### Typical methods

| Area | Methods |
|------|---------|
| Data | `load_cluster_data`, `load_attach_data`, `load_raw_data`, `preprocess_data` |
| Features | `smart_feature_selection`, `feature_selection_by_variance`, `feature_selection_by_pca`, `get_transform_columns` |
| Model | `create_cluster_model`, `create_clustering_pipeline`, `elbow_method`, `cluster_analysis`, `apply_model` |
| Artifacts | `save_pipeline_model`, `load_trained_model`, `predict_clusters`, `attach_cluster_label`, `get_cluster_stats` |
| Viz | `plot_pca`, `plot_radar_chart`, elbow helpers |

Properties such as `normal_features`, `skewed_features`, `n_clusters`, and flags like `run_elbow_method` / `run_cluster_analysis` read from the loaded YAML.

---

## Configuration files

YAML configs in this folder (pick one per run):

| File | Purpose |
|------|---------|
| `cluster_config-wucaishen.yaml` | Wucaishen-style grouped stats |
| `cluster_config-deepdive.yaml` | Deepdive project |
| `cluster_config-ss01.yaml` | SS01 |
| `cluster_config-ss01_only_normalized.yaml` | SS01 (normalized variant) |

Each file sets `project_name`, `work_dir`, `data_loader` (cluster vs attach data on S3), `features`, `cluster_analysis`, `pipeline`, and `output` sections. Adjust buckets, prefixes, patterns, and columns to match your data layout.

### Pipeline switches (`pipeline:`)

Enable or disable steps without editing code, for example:

```yaml
pipeline:
  elbow_method: true
  cluster_analysis: true
  apply_model: false          # score another ai_group with the trained model (two-phase configs)
  attach_cluster_label: true
  get_cluster_stats: true
  upload_result_to_s3: false
```

For two-phase configs (see below) the switches are keyed by group, e.g.
`pipeline.train` (fit) vs `pipeline.inference` (apply).

---

## Main entrypoint: `cluster_analysis_pipeline.py`

Runs profiling, feature selection, elbow (if enabled), clustering, optional test inference, attach labels, stats, and optional upload—according to your YAML.

```bash
poetry run python jobs/cluster_analysis/cluster_analysis_pipeline.py \
  --config_file jobs/cluster_analysis/cluster_config-deepdive.yaml
```

The default inside the script points at a specific `cluster_config-*.yaml` if you omit `--config_file`; **prefer passing `--config_file` explicitly** so the run is unambiguous.

Logs are written under `jobs/log/` (see `setup_logging` in the script).

---

## Attach output (`attach_cluster_label`) & `spin_id` flow-through

When `pipeline.attach_cluster_label: true`, the trained cluster labels are joined back onto the
per-bet **attach (enriched)** data and written as one parquet per cluster:

```
<work_dir>/<project_name>/<model>/output/enriched_data_cluster_{0..k-1}.parquet
```

- **Which per-bet columns are carried through** is controlled by
  `data_loader.attach_data.columns_to_read`. `spin_id` is included there, so every labeled bet
  keeps its spin identifier — i.e. a cluster label can be joined back to raw spin-level events
  downstream.
- **Row accounting**: `load_attach_data` keeps only `merge_on` groups with exactly `bin_size`
  rows (complete bins), and the label join is an inner merge on `merge_on`. Rows are therefore
  conserved — no duplication; any group with no label is dropped (and logged).

**ss03 end-to-end verification (2026-06-05):** with `spin_id` in `columns_to_read` and
`bin_size: 50`, an attach run (`reload=True`) produced 3 cluster files totalling
**10,442,300** rows (= 208,846 complete groups × 50 bets), `spin_id` present in every file, and
**0 groups dropped** in the label merge (`attach keys == label keys == 208,846`).

---

## Two-phase clustering: train one ai_group, score another

Some projects (e.g. ss03) train KMeans on one ai_group (`Default`) and then apply that
frozen model to a second group (`AI`) for an apples-to-apples comparison. This is driven
by a single config with a `groups:` mapping plus a `--group` switch — there are **no two
files to keep in sync**, so `top_features` / `n_clusters` (which resolve the saved model
path) can't drift between the train and apply runs.

- `data_loader` holds the shared schema (`data_types`, `columns_to_read`, `bin_size`,
  thresholds, `merge_on`). `groups.<name>` overrides only `prefix` / `row_filters` /
  `local_cache` (and an `output_tag`), deep-merged over the shared block.
- `merge_on` includes `ai_group`, so Default and AI labels coexist in the shared
  `cluster_labels.parquet` without colliding.
- `cluster_analysis` (training) persists `models/feature_order.json` (ordered feature
  list) and `models/clip_bounds.json` (per-column clip bounds fitted on the training
  group). `apply_model` reloads both — it does **not** re-run feature selection or
  re-fit clipping on the new group — slices to `top_features`, clips with the **frozen
  training bounds**, then `predict`s with the saved pipeline and writes labels. This keeps
  every preprocessing step a pure transform, so AI is scored in the exact feature space
  the model was trained on (a model trained before this sidecar existed falls back to
  recomputing clip bounds on the new group, with a warning).
- `attach_cluster_label` tags per-cluster output by group via `output_tag`:
  `enriched_data_cluster_{k}.parquet` (empty tag) vs `enriched_data_ai_cluster_{k}.parquet`.

Each run writes to its own folder `work_dir/<run_id>/` (default
`run_id = result_<YYYY-MM-DD_HH-MM>`) so successive runs don't overwrite each other.
Because the inference run reads the train run's model, **pass the train run's `run_id`
to the inference run via `--run-id`** (the train run logs its id as `run_id for this run: ...`):

```bash
# Phase 1 — train group (elbow -> pick k & top_n -> fit -> attach). Note the logged run_id.
poetry run python jobs/cluster_analysis/cluster_analysis_pipeline.py \
  --config_file jobs/cluster_analysis/cluster_config-ss03.yaml --group train

# Phase 2 — inference group: apply the train-group model (predict -> attach).
# Reuse the train run's run_id so it finds the trained model.
poetry run python jobs/cluster_analysis/cluster_analysis_pipeline.py \
  --config_file jobs/cluster_analysis/cluster_config-ss03.yaml --group inference \
  --run-id result_2026-06-17_10-18
```

The inference run requires the train run's artifacts (`feature_order.json`, `clip_bounds.json`,
and the `kmeans_model_top{N}_features_k_{K}.pkl` pickle) under the shared
`work_dir/<run_id>/<model>/models/`, so run phase 1 first, reuse its `run_id`, and keep
`top_features` / `n_clusters` unchanged between the two. (Data caches live directly under
`work_dir`, shared across run_ids, so a new run_id does not re-download from S3.)

## Other scripts in this folder

| Script | Role |
|--------|------|
| `cluster_analysis_03_fit_kmeans.py`, `cluster_analysis_03_fit_kmeans_deepdive.py` | Older notebooks-style scripts for applying a saved model to local CSVs; paths and imports may need to match your checkout |
| `pipeline_example.py` | Examples around pipeline-style usage |
| `reindex_cluster_label.py` | Reindex cluster labels |
| `model_training_gail.py` | Reserved / empty stub |

---

## Troubleshooting

- **Missing config**: Pass a valid `--config_file` path; ensure `project_name`, `data_loader`, and `features` match your S3 layout.
- **Caches**: Cluster/attach data can be cached locally under paths derived from `work_dir` and the YAML `local_cache` fields; use `reload=True` in code if you need a fresh pull.
- **Debug logging**: `import logging` and `logging.basicConfig(level=logging.DEBUG)` for verbose output from `bituslabs_ds.ml`.

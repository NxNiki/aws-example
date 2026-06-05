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
| Model | `create_cluster_model`, `create_clustering_pipeline`, `elbow_method`, `cluster_analysis`, `model_inference` |
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
  test_model: false
  attach_cluster_label: true
  get_cluster_stats: true
  upload_result_to_s3: false
```

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

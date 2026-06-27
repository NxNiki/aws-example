# jobs/

ETL pipelines, scheduled reports, and ad-hoc analyses. Everything here imports
from `src/bituslabs_ds` (run with `poetry run python jobs/<script>.py` from the
repo root).

## Layout

| Path | What it is |
|------|------------|
| `run_scheduled_etl_jobs.py` | **The scheduled-jobs registry + orchestrator.** EventBridge → ECS Fargate runs this nightly; it executes the registered jobs in order (see below). |
| `build_rag_index.py` | Rebuilds the RAG service's Confluence index. Baked into the rag_service image; also run by a scheduled Fargate task. |
| `operation_daily_report/` | Daily operation report (Slack) + weekly-report ETL. |
| `ss01_wucaishen/`, `ss02_deepdive/`, `ss03_mahjiang_streak/`, `fish_hunter/` | Per-game ETL + analyses. Each contains the game's scheduled `etl_*` script plus ad-hoc studies. |
| `cluster_analysis/` | User-clustering pipeline (K-means/GAIL) + its `cluster_config-*.yaml`. |
| `risk_control/` | Risk scoring / IP analyses. |
| `simulation_report/` | Simulation result analyses + `simulation_analysis_config-*.yaml`. |
| `analyses/` | One-off EDA and data-prep scripts not tied to a game directory. Run them from inside `jobs/analyses/` (some use cwd-relative sibling paths, e.g. `data_process_wucaishen_submit.py` uploads `data_process_wucaishen.py`). |
| `examples/` | SageMaker / DataLoader usage examples. |
| `output*/` | Local result caches written by analyses (gitignored). Safe to delete; scripts regenerate them. |

## ⚠️ Scheduled job paths are FROZEN

The EventBridge rules in AWS bake `jobs/...` paths into their ECS task command
overrides (`infra/etl/setup_schedule.sh`, `infra/operation_report/
setup_ecs_schedule.sh`). **Moving or renaming any path below breaks the
nightly schedule** until the rules are re-created:

- `operation_daily_report/run_daily_report.py`
- `ss01_wucaishen/etl_game_stats_daily_by_user_group.py`
- `ss02_deepdive/etl_game_stats_daily_by_user_group.py`
- `ss03_mahjiang_streak/etl_game_stats_daily_by_user_group.py`
- `fish_hunter/etl_game_stats_daily_by_user.py`
- `operation_daily_report/etl_weekly_report_all_games.py`
- `run_scheduled_etl_jobs.py` (the orchestrator itself)
- `build_rag_index.py` (rag_service image + scheduled index task)

To change the registry (add/remove/reorder jobs), edit
`run_scheduled_etl_jobs.py` — the EventBridge rule only knows the orchestrator.

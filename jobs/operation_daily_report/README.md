# Operation Daily Report

Single entry point to generate the daily report. Outputs go to `output/` in this folder.

**Cached files** in `output/` (not refreshed by default): `stats_by_day_pa.parquet`, `fishhunter_product_id_pa.parquet`, `slot_product_id_pa.parquet`. Use `--reload-cached-etl` to force refresh. If missing, the corresponding ETL runs automatically.

## What it does

1. **ETL game stats daily by group (HG)** – Redshift → `output/stats_by_date.parquet`
2. **ETL game stats daily by group PA** – Athena → `output/stats_by_day_pa.parquet` (skipped by default; PA is static year-ago comparison)
3. **Display daily report** – SS01 metrics table (Control vs AI vs PA) for last N days
4. **Check PID difference** – Redshift vs Athena PIDs for fishhunter, ss01, ss03 (outputs: `fishhunter_*`, `ss01_*`, `ss03_*`, shared `slot_product_id_pa.parquet` for ss01/ss03)

The final output is wrapped in "FINAL REPORT FOR SUBMISSION" / "END OF REPORT" for easy copy-paste.

## Usage

```bash
poetry run python jobs/operation_daily_report/run_daily_report.py
poetry run python jobs/operation_daily_report/run_daily_report.py --bastion-ip 13.215.212.244
poetry run python jobs/operation_daily_report/run_daily_report.py --lookback-days 3
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--bastion-ip IP` | — | Bastion IP for Redshift tunnel |
| `--skip-hg-etl` | off | Skip HG ETL (step 1) |
| `--skip-pa-etl` | on | Skip PA ETL (step 2). Use `--no-skip-pa-etl` to run when refreshing year-ago data |
| `--skip-report` | off | Skip display report (step 3) |
| `--skip-pid-check` | off | Skip PID diff check (step 4) |
| `--lookback-days N` | 7 | Number of days for daily report (step 3) |
| `--reload-cached-etl` | off | Force re-run ETL for stats_by_day_pa, fishhunter_product_id_pa, slot_product_id_pa |

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `jobs/operation_daily_report/run_daily_report.py`: removed unused `--bastion-ip`, `--slack-channel`, `--skip-hg-etl`, `--skip-pa-etl`/`--no-skip-pa-etl` flags; replaced with `--run-pa-etl` (default off). HG ETL always runs; downstream scripts use their own bastion-IP defaults; Slack channel comes from `SLACK_CHANNEL_ID`.

### Added
- `CONTRIBUTING.md` — branch model, merge workflow, PR template, tag/release process, per-workflow ASCII diagrams (incl. rebase before/after).
- `CHANGELOG.md` — this file.

---

Releases prior to this changelog (`0.1.0`–`0.1.5`) are recorded in git tags only.

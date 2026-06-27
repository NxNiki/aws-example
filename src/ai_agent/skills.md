# Agent skills

Multi-step procedures for operating the dashboard. Each skill composes the
action tools (see the action contract above). Loaded into the agent's system
prompt at chat time; also served read-only at `GET /api/agent/skills`.

## update_report

Use when the user asks to update/refresh a report or shift it to a new period
(e.g. "update the report with May's data").

1. `navigate_tab("report")` — so the user can watch the regeneration.
2. If the user named a saved report and it isn't already loaded:
   `load_report_spec(name)` (name must be in the report section's `specs`).
3. `set_report_period(date_from, date_to)` with the requested window
   (inclusive ISO dates; resolve phrases like "May 2026" to 2026-05-01 →
   2026-05-31).
4. `regenerate_report()` — re-fetches inheriting figures and refreshes stale
   prose. Hand-written descriptions are preserved automatically.
5. Only call `save_report_spec(name)` if the user explicitly asked to save.
   Saving overwrites: if `name` already exists in `specs`, confirm with
   `ask_user` first.

## show_metric

Use when the user asks to SEE a metric or trend on the dashboard.

1. `load_config(config_id)` if the requested game differs from `configId`.
2. `navigate_tab("stats-by-date")` (or `stats-by-group` / `stats-deepdive`
   for distribution / correlation questions).
3. `set_date_range(date_from, date_to)` / `set_granularity(g)` if asked.
4. `set_metrics(panel_id, left_metrics, right_metrics)` — pick the panel
   whose `available_metrics` contain the requested metric; never invent names.

## edit_report_figure

Use when the user asks to change, add, or remove a figure in the report.

- Figure ids come from the report section of the DASHBOARD STATE.
- `patch_report_figure(figure_id, patch_json)`: `patch_json` contains ONLY the
  fields to change (`title`, `description`, `inherit_period`, `source`).
  Fields are shallow-merged, EXCEPT `source`, which replaces the old source
  wholesale — when changing anything inside `source`, send the complete source
  object (copy the figure's current source from the state and modify it).
- `description_preview` / `summary_preview` in the DASHBOARD STATE are
  TRUNCATED previews. Never copy them into a patch — omit `description` to
  keep the existing text, or write entirely new text.
- After source changes, the figure re-renders automatically and stale
  generated prose refreshes; call `regenerate_report()` only for report-wide
  period shifts, not after a single patch.
- `add_report_figure(title, source_json)` / `remove_report_figure(figure_id)`
  to add or drop figures. New figures default to `inherit_period: true`.

`source_json` must be a complete object of one of these shapes (use config /
panel / metric names from the DASHBOARD STATE; `cohort_selection` maps group
column → selected values, `{}` = all):

stats-by-date (time-series panel):

```json
{"kind": "stats-by-date", "config": "fishhunter", "granularity": "day",
 "date_from": "2026-05-01", "date_to": "2026-05-31", "cohort_selection": {},
 "panel_id": "group1", "left": ["num_active_users"], "right": [],
 "log": false, "threshold": 100}
```

stats-by-group (distribution across cohorts / up to 3 date ranges):

```json
{"kind": "stats-by-group", "config": "fishhunter", "granularity": "day",
 "ranges": [{"start": "2026-05-01", "end": "2026-05-31"}],
 "cohort_selection": {}, "panel_id": "group1", "metric": "num_active_users",
 "mode": "bar", "clip": {"enabled": false, "min": null, "max": null}}
```

stats-deepdive (histogram / heatmap / scatter over metric sets):

```json
{"kind": "stats-deepdive", "config": "fishhunter", "granularity": "day",
 "ranges": [{"start": "2026-05-01", "end": "2026-05-31"}],
 "cohort_selection": {}, "panel": "derived", "mode": "histogram",
 "metrics": ["rtp", "total_bet"], "nbins": 30, "normalize": true,
 "outliers_std": 3, "scatter_log_x": false, "scatter_log_y": false,
 "log_y": false, "clip": {"enabled": false, "min": null, "max": null}}
```

# Report Agent — Design Spec

The dashboard's Report tab lets a user assemble an analytical report
from rendered figures, attach LLM-generated descriptions and an overall
summary, and publish a snapshot to a Confluence page.

The interaction is one-way: the dashboard owns the report, and the
Confluence doc is a pushed snapshot. Nothing reads the doc back.


## End-to-end workflow

1. **Render figures** in any of the dashboard's stats tabs.
2. **Add to report.** A button under each figure stages the figure in
   the Report tab — captures the Plotly figure dict, extracts a JSON
   data summary for the LLM, and pushes an entry into an in-memory
   store. No network call yet.
3. **Open the Report tab.** Top row: Confluence doc URL input,
   language dropdown, **Export to Doc** button. Below: a summary
   panel and one panel per staged figure with its image, description
   text, **Generate description** and **Remove figure** buttons.
4. **Generate descriptions.** Per-figure button calls the LLM with the
   figure's data summary and returns 2–3 paragraphs in the selected
   language. Re-clicking regenerates.
5. **Generate summary.** Top-level button calls the LLM with every
   staged figure's description + data summary and returns a
   2–3-paragraph overview.
6. **Export to Doc.** Renders each figure to PNG, attaches the PNGs to
   the target Confluence page, and writes a `Dashboard Report` region
   into the page body. Subsequent exports replace the region in place.

Steps 2–5 are local to the dashboard; only step 6 hits Confluence.


## Report tab state

Two `dcc.Store` values in the tab:

- `report-figures`: list of staged-figure entries.
- `report-summary`: most recently generated summary text.

Each `report-figures` entry:

```jsonc
{
  "id": "uuid4 hex",
  "graph_id": "date-g1-plot",        // source figure id in the dashboard
  "tab": "tab-date",                 // dashboard tab the figure came from
  "config_name": "ss03",             // dashboard config in effect
  "fig_dict": { ... },               // full Plotly figure dict
  "data_summary": { ... },           // compact JSON for LLM prompts
  "description": "...",              // LLM prose, may be empty
  "export_width": 1800,              // PNG dimensions used at Export time
  "export_height": 430
}
```

Removing a figure filters by `id`. Re-running Generate description
rewrites that entry's `description`. Generate summary rewrites
`report-summary`.


## Data summary

The LLM never sees the rendered PNG — it sees a JSON data summary
extracted from the Plotly figure dict by
`report_agent/data_summary.py:extract_data_summary`. That's what lets
descriptions quote exact values instead of guessing from a rendering.

Each summary contains:

- One entry per trace: `name`, `type`, `mode` (when set), downsampled
  `x` and `y`, `error_y.upper` / `error_y.lower` bounds when present,
  and a non-default `yaxis` reference.
- Layout title, x-axis title, primary and secondary y-axis titles, and
  any explicit ranges.

Traces over 400 points are downsampled by uniform-stride sampling.
Floats round to 4 decimals; NaN becomes `null`.


## Languages

The language dropdown controls both Generate description and Generate
summary. Codes:

| Code | Display |
| --- | --- |
| `en` | English |
| `zh-Hans` | Chinese (Simplified) |
| `zh-Hant` | Chinese (Traditional) |

The selected language is passed as a system-prompt hint. The model
itself comes from `chat_agent._build_llm()` and respects the same
`CHAT_PROVIDER` / `CHAT_MODEL` configuration as the AI Assistant chat
panel.


## Export contract

`report_agent/exporter.py:export_report` produces an idempotent
snapshot:

1. Resolve the page URL to a Confluence page ID via
   `confluence_client.extract_page_id_from_url`.
2. For each figure in the store:
   - Render the figure dict to PNG via kaleido at
     `export_width` × `export_height`.
   - Upload the PNG as an attachment with a fresh
     `report_<ts>_<short uuid>.png` filename.
   - Build a `<h2>Figure N</h2>` block followed by the image and the
     description as `<p>` paragraphs.
3. Assemble the full region:

   ```html
   <h1>Dashboard Report</h1>
   <h2>Summary</h2>
   <p>...summary paragraphs...</p>
   <h2>Figure 1</h2>
   <ac:image ac:width="800"><ri:attachment ri:filename="..."/></ac:image>
   <p>...description...</p>
   <h2>Figure 2</h2>
   ...
   ```

4. Splice into the page body:
   - If `<h1>Dashboard Report</h1>` already exists, replace everything
     from that heading to the end of the page body.
   - Otherwise, append the region to the end.

   Anything **above** the `Dashboard Report` heading is preserved
   across exports. Anything below is owned by the exporter and
   overwritten on each click.

5. Save the page via `confluence_client.update_page_storage` with the
   captured `version.number`.


## Modules

| Concern | Module |
| --- | --- |
| Plotly figure → JSON data summary for LLM prompts | `src/dashboards/report_agent/data_summary.py` |
| LLM helpers for per-figure description and overall summary | `src/dashboards/report_agent/description.py` |
| Render PNGs, attach, write the snapshot region | `src/dashboards/report_agent/exporter.py` |
| Report tab layout + 7 callbacks (add, render, describe, remove, summarize, render summary, export) | `src/dashboards/game_stats_monitor.py` (`_layout_report_tab`, `_register_report_tab_callbacks`) |
| Confluence read / attach / update API wrapper | `src/dashboards/confluence_client.py` |
| LLM factory used by description/summary | `src/dashboards/chat_agent.py:_build_llm` |


## Authentication

The Confluence client uses a single API token provided via env var or
AWS Secrets Manager: `CONFLUENCE_URL`, `CONFLUENCE_EMAIL`,
`CONFLUENCE_TOKEN`. The token must have write scope on the target
space for Export to Doc to succeed.


## Future extensions

- **Vision-based figure interpretation** when the data summary alone
  is insufficient (e.g. heatmaps with geometric structure lost in a
  flat array). Pass the rendered PNG to a vision-capable model.
- **Google Docs backend.** Re-implement the read/attach/update layer
  against the Google Docs API; the heading-bracketed snapshot model
  ports cleanly.
- **Drag-to-reorder figure panels** in the Report tab.
- **Templates.** Save a Report tab snapshot (figures + descriptions +
  summary) to S3 so a recurring weekly report can be regenerated from
  a saved skeleton.
- **Per-figure language overrides.** Today the dropdown is global; one
  figure in English and another in Chinese isn't supported.

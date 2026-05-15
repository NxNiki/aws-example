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
   language dropdown, **Export to Doc** button. Below: a References
   panel, a summary panel, and one panel per staged figure with its
   image, an editable description textarea, and **Generate
   description** / **Remove figure** buttons.
4. **(Optional) Attach references.** Click **Add reference** to attach
   Confluence URLs (or external URLs); each row's URL is fetched on
   blur and shows a green ✓ + page title, red ✗ + error, or grey ↗
   for external links. Reference bodies are inlined in subsequent
   LLM prompts and listed under the exported doc's References
   section.
5. **Generate descriptions.** Per-figure button calls the LLM with the
   figure's data summary, the textarea contents (treated as a *prior
   draft to refine*), any `/prompt` instruction lines, and the
   reference bodies. Returns 2–3 paragraphs in the selected language.
   Re-clicking regenerates.
6. **Edit and iterate.** Type directly into the description textarea
   or insert `/prompt ...` lines to push instructions into the next
   regenerate. Same for the summary textarea. Edits persist on blur.
7. **Generate summary.** Top-level button calls the LLM with every
   staged figure's description + data summary, the summary textarea
   contents, any `/prompt` lines in it, and the references. Returns
   a 2–3-paragraph overview.
8. **Export to Doc.** Renders each figure to PNG, attaches the PNGs
   to the target Confluence page, and writes a `Dashboard Report`
   region into the page body — including a trailing References
   section listing every URL. Subsequent exports replace the region
   in place.

Steps 2–7 are local to the dashboard; the only Confluence calls
before step 8 are the on-blur reference fetches in step 4.


## Report tab state

Three `dcc.Store` values in the tab:

- `report-figures`: list of staged-figure entries.
- `report-summary`: most recently generated summary text.
- `report-references`: list of user-curated reference entries.

Each `report-figures` entry:

```jsonc
{
  "id": "uuid4 hex",
  "graph_id": "date-g1-plot",        // source figure id in the dashboard
  "tab": "tab-date",                 // dashboard tab the figure came from
  "config_name": "ss03",             // dashboard config in effect
  "fig_dict": { ... },               // full Plotly figure dict
  "data_summary": { ... },           // compact JSON for LLM prompts
  "description": "...",              // LLM prose, may be empty (user-editable)
  "export_width": 1800,              // PNG dimensions used at Export time
  "export_height": 430
}
```

Each `report-references` entry:

```jsonc
{
  "id": "uuid4 hex",
  "url": "https://yourcompany.atlassian.net/wiki/...",
  "title": "Metric definitions",   // populated by blur-time fetch
  "status": "ok"                   // "ok" | "error" | "external"
  // "error": "Page fetch failed: 404"   // present when status=="error"
}
```

Removing a figure filters by `id`. Re-running Generate description
rewrites that entry's `description`. Generate summary rewrites
`report-summary`.


## Editing descriptions and the summary

Descriptions and the overall summary render as `dcc.Textarea` controls,
not read-only HTML. The user's editing surface is the dashboard;
Confluence is still snapshot-only (the export pushes the latest
textarea contents and never reads back).

### LLM lives in the ai_agent service

LLM generation runs in the **ai_agent** service, not in the dashboard
process. The dashboard sends `POST` requests to two endpoints on
`{CHAT_API_URL}`:

* `POST /api/report/description` — one figure's description
* `POST /api/report/summary` — overall report summary

Both wrap the same `generate_description` / `generate_summary` functions
in `dashboards/report_agent/description.py`, which is mounted into the
ai_agent Docker image. The dashboard image no longer needs LangChain /
LangGraph — `pyproject.toml` keeps the `llm` group (ai_agent-only) and
a separate `confluence` group (shared with the dashboard for reference
fetching + Confluence PNG export).

**Reference fetching stays in the dashboard.** Each Generate request
sends *pre-fetched* reference bodies (via `references.load_references`)
to the ai_agent, so the ai_agent never does Confluence I/O on the
hot path. This also keeps the per-session reference cache local to
the dashboard process where it belongs.

Prompt and behavior tweaks (system prompts, the four-case logic) now
only require redeploying ai_agent — no dashboard rebuild needed.

- Edit prose directly in the textarea. On blur the value persists into
  `report-figures[i].description` or `report-summary` via dedicated
  callbacks (`persist_description_edits`, `persist_summary_edits`).
- Click **Generate description** or **Generate summary** to regenerate.
  The current textarea value is sent to the LLM as a *Prior draft —
  refine, preserving any user edits*. The system prompt instructs the
  model to keep user-shaped tweaks intact when possible.
- The Generate callbacks read the **live** textarea value (via Dash
  `State`) so unblurred edits and just-typed `/prompt` lines reach
  the LLM even if Dash schedules the blur-time persist callback after
  the regenerate one.

### `/prompt` syntax

Lines that start with `/prompt` (case-insensitive, leading whitespace
allowed, one per line) are user instructions, not prose to preserve.
A legacy form with a trailing colon (`/prompt:`) is also accepted —
both parse identically. The trigger must be at the **start of the
line** (after optional indentation); mid-line occurrences of `/prompt`
in prose are not stripped.

### Generate behavior (four cases)

Clicking Generate is intent-driven — it does not blindly overwrite
manual edits. The parser in `description.py:_split_at_first_prompt`
inspects the textarea and the regenerate branches on its state:

| Textarea state | What happens | Status |
| --- | --- | --- |
| **Empty / whitespace only** | LLM produces a first draft from the data summary. | `STATUS_GENERATED` |
| **Non-empty, no `/prompt` lines** | **No LLM call.** The textarea is left exactly as-is. A small status hint tells the user nothing happened and how to trigger a regenerate. | `STATUS_NO_INSTRUCTIONS` |
| **Non-empty, `/prompt` at the very start (no prose above it)** | Full regenerate from scratch using the `/prompt` instructions + data summary. | `STATUS_REGENERATED` |
| **Non-empty, `/prompt` after some prose** | The prose *before* the first `/prompt` is preserved byte-for-byte (the LLM never sees it as something to rewrite — it's passed as read-only context). The LLM generates 1–2 additional paragraphs based on the instructions, and the caller concatenates: `preserved + "\n\n" + new_paragraphs`. | `STATUS_APPENDED` |

The append case uses a separate `_DESCRIPTION_APPEND_SYSTEM_PROMPT` /
`_SUMMARY_APPEND_SYSTEM_PROMPT` that explicitly forbids the model from
echoing or rewriting the preserved prose, so manual edits in that
region are guaranteed safe.

```text
The DAU spike on Apr 15 is consistent with the marketing push.

/prompt also compare with the Q1 baseline
/prompt emphasize the cohort that re-activated
```

In this example, the first paragraph is preserved verbatim; the LLM
generates new paragraphs covering the Q1 baseline comparison and the
re-activated cohort, which are appended below.


## References

Users can attach Confluence URLs (or external URLs) to ground the
agent's descriptions on existing documentation. Click **Add reference**
in the References panel; each row is one URL with a remove button and
an inline status pill.

### Per-row status

After blur, the URL is resolved synchronously via
`report_agent/references.py:load_references`:

| Status | Pill | Meaning |
| --- | --- | --- |
| `ok` | green ✓ + page title | Confluence page fetched; body inlined into LLM prompts. |
| `error` | red ✗ + error message | Fetch failed (bad URL, missing permission). Still listed in the export References section. |
| `external` | grey ↗ "external link" | URL doesn't match `/wiki/`. Listed at export but never fetched. |
| _(absent)_ | empty | URL not yet set or just cleared. |

Fetched bodies are stripped to plain text and truncated at 4000 chars
per doc. A process-level cache (`references._REFERENCE_CACHE`)
short-circuits repeat fetches across successive Generate / Export
clicks. The cache lives for the dashboard process; restart the
dashboard to pick up upstream edits.

### Where references land

- **At Generate time** — fetched bodies are inlined in the LLM prompt
  as a `# Reference materials` section, ordered by the row order in
  the panel. The system prompt instructs the model to ground
  contextual claims on this material and not invent metrics that
  aren't present.
- **At Export time** — every URL (regardless of fetch outcome) is
  listed at the bottom of the Confluence snapshot under
  `<h2>References</h2>` as `<a href>` links titled by the fetched
  page title, falling back to the URL itself for `external` rows.


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
   <h2>References</h2>
   <ul>
     <li><a href="...">Title 1</a></li>
     <li><a href="...">Title 2</a></li>
   </ul>
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
| Plotly figure → JSON data summary for LLM prompts (incl. heatmap z, Plotly 6.x binary-array decoding) | `src/dashboards/report_agent/data_summary.py` |
| LLM helpers for per-figure description and overall summary; `/prompt` parsing; references plumbing | `src/ai_agent/report_agent/description.py` |
| User-curated reference loader: URL → Confluence page → stripped, truncated body; cached | `src/dashboards/report_agent/references.py` |
| Render PNGs, attach, write the snapshot region (with trailing References section) | `src/dashboards/report_agent/exporter.py` |
| Report tab layout + callbacks (add/render/describe/remove figure, summarize, persist edits, add/remove/persist/render reference, export) | `src/dashboards/game_stats_monitor.py` (`_layout_report_tab`, `_register_report_tab_callbacks`) |
| Confluence read / attach / update API wrapper, tinyurl decoding, HTTP redirect fallback | `src/dashboards/confluence_client.py` |
| LLM factory used by description/summary; auto-mode provider pick | `src/ai_agent/chat_agent.py:_build_llm` |


## Authentication

The Confluence client uses a single API token provided via env var or
AWS Secrets Manager: `CONFLUENCE_URL`, `CONFLUENCE_EMAIL`,
`CONFLUENCE_TOKEN`. The token must have write scope on the target
space for Export to Doc to succeed.


## Future extensions

- **Reference cache invalidation.** Currently the references cache
  lives for the dashboard process; a per-row "Refresh" button or a
  cache-bust toggle would let users pick up upstream edits without a
  restart.
- **`dcc.Loading` spinner during reference fetches.** First-fetch
  latency on a Confluence page can be 500ms–2s; today the row sits
  blank during that window.
- **Auto-discovered references via the RAG service.** Skipped for v1
  in favour of user-curated only. If users want the agent to pull in
  related context automatically, plumb `rag_service.client` into
  `description.py` and let it inject top-K passages alongside the
  user-curated docs.
- **Vision-based figure interpretation** when the data summary alone
  is insufficient (e.g. heatmaps with geometric structure lost in a
  flat array). Pass the rendered PNG to a vision-capable model.
- **Google Docs backend.** Re-implement the read/attach/update layer
  against the Google Docs API; the heading-bracketed snapshot model
  ports cleanly.
- **Drag-to-reorder figure panels** in the Report tab.
- **Templates.** Save a Report tab snapshot (figures + descriptions +
  summary + references) to S3 so a recurring weekly report can be
  regenerated from a saved skeleton.
- **Per-figure language overrides.** Today the dropdown is global; one
  figure in English and another in Chinese isn't supported.

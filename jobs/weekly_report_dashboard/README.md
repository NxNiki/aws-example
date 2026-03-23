# Weekly Report Dashboard

Dash app for weekly game reporting. It supports:

- default CSV loading
- Redshift fetch over SSH tunnel
- trend charts and weekly summaries
- Gmail draft generation

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure

Copy `.env.example` to `.env` and fill in the values you need.

```bash
cp .env.example .env
```

Notes:

- If you only want local CSV mode, you only need `DEFAULT_CSV_PATH`.
- If you want Gmail draft support, set `GMAIL_CREDENTIALS_PATH`.
- If you want Redshift fetch support, fill in all `REDSHIFT_*` and `SSH_*` values.

## 3. Run

```bash
python3 weekly_report_dashboard.py
```

Or on macOS, double-click:

- `Open Weekly Report.command`
- `Stop Weekly Report.command`

## 4. Deploy to Render

This app can run without a local CSV as long as Redshift is configured.

Render setup:

1. Connect the GitHub repo in Render.
2. Use the included `render.yaml`.
3. Set the required environment variables in Render:
   - `REDSHIFT_HOST`
   - `REDSHIFT_PORT`
   - `REDSHIFT_DATABASE`
   - `REDSHIFT_USER`
   - `REDSHIFT_PASSWORD`
   - `SSH_TUNNEL_HOST`
   - `SSH_TUNNEL_PORT`
   - `SSH_TUNNEL_USER`
   - `SSH_PRIVATE_KEY_CONTENT` or `SSH_PRIVATE_KEY_BASE64`
   - optional: `GMAIL_CREDENTIALS_PATH`, `GMAIL_TOKEN_PATH`

Notes:

- On cloud, the app can start with no default CSV.
- Users can fetch data directly from Redshift from the page.
- Gmail draft creation should only be enabled after cloud-safe OAuth setup is ready.
- For Render, do not use your local pem path. Put the bastion private key into a secret env var:
  - `SSH_PRIVATE_KEY_CONTENT`: paste the full key text
  - or `SSH_PRIVATE_KEY_BASE64`: paste a base64-encoded key

## 5. Share with others

Before pushing to GitHub:

- keep `.env` out of git
- keep `credentials.json`, `token.json`, and `*.pem` out of git
- never commit real passwords or keys into the codebase

Other users can run the same app by:

1. cloning the repo
2. installing `requirements.txt`
3. creating their own `.env`
4. providing their own Gmail and Redshift credentials if needed

## 6. Feature behavior

- Without Redshift config, the app still works with the default CSV.
- Without Gmail config, the dashboard still works, but Gmail draft creation is disabled.

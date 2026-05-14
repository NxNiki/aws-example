# AI Agent — Design Spec

The AI agent is a FastAPI service that answers questions about dashboard
columns, user groups, and game metrics using a LangChain ReAct agent
over OpenAI or Gemini. It exposes two surfaces:

* **`POST /api/chat`** — JSON API used by the Dash dashboard's embedded
  chat panel. One-shot request, history passed in by the caller.
* **`POST /slack/events`** *(optional)* — Slack Event Subscriptions
  endpoint. Responds only to `app_mention` events (the bot must be
  @-tagged), replies in-thread, and pulls the prior 20 thread messages
  back as conversation history.

Both surfaces share the same `chat()`/`achat()` core in
`dashboards.chat_agent`, so they have identical tool access (column
lookup, group lookup, RAG search via `rag_service.client`, Confluence
fetch via `dashboards.confluence_client`).

Deploy / build steps live in [`infra/README_AI_AGENT.md`](../infra/README_AI_AGENT.md);
this doc focuses on architecture and the Slack-side setup.


## Slack Bot

### Why @-mention only

The Slack handler subscribes to `app_mention` and nothing else. DMs and
regular channel chatter are ignored on purpose:

* The bot is a heavyweight LLM caller — every message would cost a few
  cents and a few seconds. Limiting to explicit @-mentions keeps the
  signal-to-noise ratio high.
* It also constrains the blast radius if the signing-secret check ever
  fails — an attacker can't impersonate a typing user and silently
  trigger LLM runs.

If DM support is needed later, it's a 4-line change: subscribe to
`message.im` in the Slack app config and add a second handler in
`slack_handler.py` that calls the same `_respond_to_mention()` body.


### Request flow

```
Slack user types "@ds-agent <question>"
        │
        ▼
  Slack sends POST /slack/events
        │  (X-Slack-Signature: HMAC-SHA256 of body w/ signing_secret)
        ▼
  CloudFront (HTTPS frontdoor, see below)
        │
        ▼
  ALB :80  ──►  ECS task (ai-chat-agent)
                  │
                  ▼
           slack_bolt.AsyncApp
              · verifies signature
              · routes by event type
              · ack returns 200 within 3s
              · asyncio.create_task(_respond_to_mention(...))
                          │
                          ▼
                  achat() → LangChain agent → LLM + tools
                          │
                          ▼
                  client.chat_postMessage(channel, thread_ts, text)
```

The ack-then-process pattern is critical: Slack disconnects after 3
seconds and retries the event up to 3 times. The handler returns
immediately so Bolt can send the 200; the actual LLM call (10–30s)
runs as a background task and posts its reply when ready. An in-memory
LRU keyed on `event_id` swallows Slack's retries during cold starts.


### Why CloudFront sits in front

Slack's Event Subscriptions API requires **HTTPS** for the Request URL.
The ai-chat-agent ALB is HTTP-only (port 80, no ACM cert). Rather than
provision a domain + cert just for this, we put a CloudFront
distribution in front with the default `*.cloudfront.net` certificate.

```
Slack (HTTPS) ──► CloudFront (https://d1abc234.cloudfront.net)
                       │  origin: http-only
                       ▼
                  ALB (port 80) ──► ECS task
```

CloudFront also gives us free DDoS sponging at the edge, which is nice
for a public webhook endpoint.

#### Distribution configuration

If the distribution is ever deleted / needs to be recreated:

| Setting | Value | Reason |
|---|---|---|
| Origin domain | `ai-chat-agent-alb-523741417.us-west-2.elb.amazonaws.com` | the ai_agent ALB |
| Origin protocol | HTTP only, port 80 | ALB has no HTTPS listener |
| Viewer protocol policy | Redirect HTTP to HTTPS | Slack requires HTTPS; redirect handles any stray HTTP requests |
| Allowed HTTP methods | GET, HEAD, OPTIONS, PUT, POST, PATCH, DELETE | Slack POSTs events; must allow POST |
| Cache policy | CachingDisabled (managed) | this is a stateful API, never cache |
| Origin request policy | AllViewer (managed) | forward `X-Slack-*` headers, body, and query strings unchanged so the HMAC signature stays valid |
| Price class | North America + Europe | cheapest tier; Slack's servers are mostly US-based |
| WAF | none | low-traffic endpoint, signing-secret check is sufficient |
| Custom domain (CNAME) | none | the default `*.cloudfront.net` hostname is fine for webhook use |

Creating it via AWS CLI requires `cloudfront:CreateDistribution`
permission, which a non-admin IAM user typically lacks. Easiest path is
the console (CloudFront → *Create distribution*) — the form has all the
fields above.

After creation, the *Distribution domain name* (e.g.
`d1abc234.cloudfront.net`) is what gets pasted into Slack as
`https://<that-domain>/slack/events`.

CloudFront takes 5–15 minutes to deploy after creation. The distribution
ID and domain name don't change once created; they survive ALB
re-deploys.


### Slack app setup (one-time)

Detailed step-by-step is in [`infra/README_AI_AGENT.md`](../infra/README_AI_AGENT.md)
under *Slack Bot (HTTPS Events)*. Summary:

1. Create Slack app → add scopes `app_mentions:read`, `chat:write`,
   `channels:history`.
2. Event Subscriptions → URL (see *Building the Request URL* below),
   subscribe to `app_mention`.
3. Install to workspace; copy Bot Token (`xoxb-…`) and Signing Secret.
4. `aws secretsmanager put-secret-value …` to add
   `SLACK_BOT_TOKEN` + `SLACK_SIGNING_SECRET` to the
   `ai-dashboard_ai_agent` secret.
5. Redeploy the ECS service so the new task picks the secret up on
   cold start.

### Building the Request URL

The Request URL goes into the Slack app → *Event Subscriptions* →
*Enable Events* → *Request URL* field. The format is:

```
https://<cloudfront-distribution-domain>/slack/events
```

To find the CloudFront distribution domain:

1. AWS Console → CloudFront → *Distributions*.
2. Locate the distribution with description `ai-agent-slack-https`
   (or whatever you named it; origin is the ai-chat-agent ALB).
3. The value under the **Domain name** column (also shown on the
   detail page under *Details → Distribution domain name*) is what
   you want. Format: `d<random>.cloudfront.net`, e.g.
   `d1rd1jgagdktb.cloudfront.net`.
4. Don't confuse it with the **ARN** or **ID** — those look like
   `arn:aws:cloudfront::…:distribution/E20IB9JUNG29EQ` and
   `E20IB9JUNG29EQ`. Neither works as a Slack URL.

Worked example:

```
https://d1rd1jgagdktb.cloudfront.net/slack/events
```

When you paste and save, Slack sends a one-time `url_verification`
POST. `slack-bolt` answers the challenge automatically — you'll see
a green ✓ *Verified* within a few seconds.

**Pre-flight checks before pasting:**

* CloudFront distribution status is **Deployed**, not *Deploying*
  (otherwise the verification POST hits a 5xx).
* `SLACK_SIGNING_SECRET` in Secrets Manager matches the one in the
  Slack app's *Basic Information → App Credentials* (otherwise the
  HMAC check fails and Bolt returns 401).
* The latest ECS task is running an image that includes `aiohttp`
  (commit `81f752f` or later) — earlier images crash on
  `slack_bolt.async_app` import.

**If verification fails:** check the ai_agent CloudWatch log group
(`/ecs/ai-chat-agent`) for the verification POST. The Slack error
message names the failure (URL didn't respond, response invalid,
SSL error, etc.) — match it against the *Troubleshooting* table
below.


## Trouble­shooting

| Symptom | Likely cause |
|---|---|
| Slack *Event Subscriptions* URL shows red ✗ | Signing secret in Secrets Manager doesn't match Slack's; or app wasn't reinstalled after adding scopes |
| `ModuleNotFoundError: No module named 'aiohttp'` at task startup | `aiohttp` missing from `ai_agent` Poetry group — slack-bolt's async path needs it explicitly |
| `GET / 404` lines in logs | Harmless. Browser / probe hitting the ALB root, which has no handler. Real health check is `/health`. |
| @mention in channel but no reply, no errors in logs | Slack isn't delivering events. Re-check the Event Subscriptions URL verification status; confirm the app was reinstalled after scope changes |
| First @mention after idle is very slow | Expected — ECS scale-to-zero cold-start takes 30–60s. Slack retries the event up to 3 times during this window; the handler dedupes on `event_id` |
| Reply posts in wrong thread / starts a new thread when it shouldn't | Check `thread_ts` vs `ts` logic in `slack_handler._respond_to_mention` — should reply to `event["thread_ts"] or event["ts"]` |

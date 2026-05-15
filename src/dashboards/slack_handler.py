"""
Slack bot integration for the dashboard AI agent.

Mounts an HTTPS-events endpoint on the chat_api FastAPI app. The bot
responds *only* to ``app_mention`` events (when explicitly @-tagged) —
DMs and regular channel messages are ignored. Replies post in a thread
on the mentioning message so channel noise stays bounded; if the mention
is already in a thread, the reply joins that thread.

Configuration
-------------
Two secrets are required (read via ``dashboards.secrets.get_secret``,
which falls back to AWS Secrets Manager):

* ``SLACK_BOT_TOKEN``      — xoxb-...  (Bot User OAuth Token)
* ``SLACK_SIGNING_SECRET`` — 32-char hex signing secret

If either is missing, ``build_slack_handler()`` returns ``None`` and the
chat_api silently skips mounting the Slack route, so local dev without
Slack credentials continues to work.

Cold-start retries
------------------
When the ai_agent task is scaled to zero, Slack's 3-retry delivery
policy will fire several event copies at the ALB while the task is
spinning up. A tiny in-memory LRU keyed on ``event_id`` drops the
duplicates *within the lifetime of a single task*. A fresh task starts
with an empty cache, which is acceptable because Slack's retry window
is short and bounded.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from dashboards.secrets import get_secret

logger = logging.getLogger(__name__)


_EVENT_ID_CACHE_SIZE = 512
_seen_event_ids: "OrderedDict[str, None]" = OrderedDict()


def _seen(event_id: str) -> bool:
    if not event_id:
        return False
    if event_id in _seen_event_ids:
        _seen_event_ids.move_to_end(event_id)
        return True
    _seen_event_ids[event_id] = None
    if len(_seen_event_ids) > _EVENT_ID_CACHE_SIZE:
        _seen_event_ids.popitem(last=False)
    return False


# Slack user-ID mentions look like ``<@U01ABCDEFG>`` (users) or
# ``<@W01ABCDEFG>`` (Enterprise Grid). Strip every occurrence so a
# mention anywhere in the message doesn't leak into the LLM prompt.
_MENTION_RE = re.compile(r"<@[UW][A-Z0-9]+>")


def _strip_mentions(text: str) -> str:
    if not text:
        return ""
    return _MENTION_RE.sub("", text).strip()


async def _fetch_thread_history(client: Any, channel: str, thread_ts: str) -> List[Dict[str, str]]:
    """Pull prior thread messages so follow-up @mentions have context.

    Returns the history in ``achat()``'s expected ``[{role, content}, ...]``
    shape, excluding the message that triggered this handler (the caller
    passes it as ``user_message``).
    """
    try:
        resp = await client.conversations_replies(channel=channel, ts=thread_ts, limit=20)
    except Exception:
        logger.warning("Slack: thread history fetch failed", exc_info=True)
        return []
    msgs = resp.get("messages", []) or []
    history: List[Dict[str, str]] = []
    for m in msgs[:-1]:
        text = _strip_mentions(m.get("text", ""))
        if not text:
            continue
        role = "assistant" if m.get("bot_id") else "user"
        history.append({"role": role, "content": text})
    return history


_THINKING_REACTION = "eyes"


async def _add_thinking_reaction(client: Any, channel: str, ts: str) -> None:
    """Drop an :eyes: reaction on the user's mention so they know the bot
    saw them while the LLM call is still running.

    Errors are swallowed — the reaction is a nice-to-have UX cue, not a
    correctness requirement. Common failure: the Slack app lacks
    ``reactions:write`` (returns ``missing_scope``); the user reaction
    feature stays dark until the scope is added and the app reinstalled.
    """
    try:
        await client.reactions_add(channel=channel, timestamp=ts, name=_THINKING_REACTION)
    except Exception:
        logger.warning("Slack: reactions_add failed (continuing without ack reaction)", exc_info=True)


async def _respond_to_mention(event_id: str, event: Dict[str, Any], client: Any) -> None:
    """Background-task body: call achat and post the reply in-thread."""
    if _seen(event_id):
        logger.info("Slack: duplicate event_id %s, skipping", event_id)
        return

    channel = event.get("channel")
    user_text = _strip_mentions(event.get("text", ""))
    # If the mention is inside an existing thread, ``thread_ts`` is set;
    # otherwise the new reply starts a thread anchored on ``ts``.
    parent_ts = event.get("thread_ts")
    mention_ts = event.get("ts")
    thread_ts = parent_ts or mention_ts

    if not channel or not user_text:
        logger.info("Slack: empty mention or missing channel; skipping")
        return

    # Immediate visual ack on the user's message so they know the bot picked
    # up the @mention — the actual reply arrives 10-30s later (or longer on
    # cold start). Fire-and-forget; failures don't block the LLM call.
    if mention_ts:
        await _add_thinking_reaction(client, channel, mention_ts)

    history: List[Dict[str, str]] = []
    if parent_ts:
        history = await _fetch_thread_history(client, channel, parent_ts)

    from dashboards.chat_agent import achat

    try:
        reply = await achat(user_message=user_text, history=history)
    except Exception as exc:
        logger.exception("Slack: achat failed")
        reply = f":warning: Sorry, I hit an error: `{exc}`"

    try:
        await client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=reply)
    except Exception:
        logger.exception("Slack: chat_postMessage failed")


def build_slack_handler() -> Optional[Any]:
    """Return an ``AsyncSlackRequestHandler`` or ``None`` if not configured.

    The caller (``chat_api.create_chat_app``) checks the return value and
    only mounts the ``/slack/events`` route when secrets are present, so
    a local dev environment without Slack tokens can still boot.
    """
    bot_token = get_secret("SLACK_BOT_TOKEN")
    signing_secret = get_secret("SLACK_SIGNING_SECRET")
    if not bot_token or not signing_secret:
        logger.info(
            "Slack secrets not configured; skipping Slack bot mount. "
            "Set SLACK_BOT_TOKEN + SLACK_SIGNING_SECRET to enable."
        )
        return None

    from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
    from slack_bolt.async_app import AsyncApp

    bolt_app = AsyncApp(token=bot_token, signing_secret=signing_secret)

    @bolt_app.event("app_mention")
    async def _on_app_mention(body: Dict[str, Any], event: Dict[str, Any], client: Any) -> None:
        # Return immediately so Bolt's ASGI adapter can send the 200 OK
        # inside Slack's 3-second deadline. The LLM call may take 10-30 s.
        event_id = str(body.get("event_id") or "")
        asyncio.create_task(_respond_to_mention(event_id, event, client))

    # Subscribing to nothing else means non-@mention events (DMs,
    # regular channel chatter) never reach this app even if the Slack
    # app config accidentally enables them — Bolt routes by event type
    # and unhandled types are no-ops.

    logger.info("Slack bot ready: responds to app_mention only")
    return AsyncSlackRequestHandler(bolt_app)

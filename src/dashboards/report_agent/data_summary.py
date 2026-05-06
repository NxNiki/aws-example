"""
Extract a compact, JSON-serializable summary of a Plotly figure dict.

The summary is what the LLM sees when generating a description: trace names,
x/y values (downsampled if very long), error-bar bounds, axis titles, and the
overall layout title. Keeping it compact protects the prompt from blowing up
on figures with thousands of points while preserving enough numeric detail
that the description references exact values rather than guessing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# Hard cap on how many points per trace land in the summary. Large enough
# to keep daily series for a year intact, small enough that a multi-trace
# figure stays under typical LLM context budgets.
_MAX_POINTS_PER_TRACE = 400


def _round_value(value: Any) -> Any:
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        return round(value, 4)
    return value


def _downsample(values: Optional[Sequence[Any]], cap: int = _MAX_POINTS_PER_TRACE) -> Optional[List[Any]]:
    if values is None:
        return None
    items = list(values)
    if len(items) <= cap:
        return [_round_value(v) for v in items]
    step = len(items) / cap
    sampled = [items[min(int(i * step), len(items) - 1)] for i in range(cap)]
    return [_round_value(v) for v in sampled]


def _trace_summary(trace: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "name": trace.get("name") or "(unnamed)",
        "type": trace.get("type") or "scatter",
    }
    mode = trace.get("mode")
    if mode:
        out["mode"] = mode
    x = _downsample(trace.get("x"))
    y = _downsample(trace.get("y"))
    if x is not None:
        out["x"] = x
    if y is not None:
        out["y"] = y

    err_y = trace.get("error_y") or {}
    if err_y.get("array") is not None or err_y.get("arrayminus") is not None:
        upper = _downsample(err_y.get("array"))
        lower = _downsample(err_y.get("arrayminus")) if err_y.get("arrayminus") is not None else upper
        out["error_y"] = {"upper": upper, "lower": lower}

    if "yaxis" in trace and trace["yaxis"] != "y":
        out["yaxis"] = trace["yaxis"]
    return out


def _axis_summary(axis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(axis, dict):
        return {}
    title = (axis.get("title") or {}).get("text") if isinstance(axis.get("title"), dict) else axis.get("title")
    out: Dict[str, Any] = {}
    if title:
        out["title"] = str(title)
    rng = axis.get("range")
    if isinstance(rng, list) and len(rng) == 2:
        out["range"] = [_round_value(rng[0]), _round_value(rng[1])]
    return out


def extract_data_summary(fig_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a JSON-serializable summary of a Plotly figure dict for LLM prompts.

    Empty/None figures return an empty dict so the caller can detect "nothing
    rendered yet" without raising.
    """
    if not isinstance(fig_dict, dict):
        return {}
    data = fig_dict.get("data") or []
    layout = fig_dict.get("layout") or {}
    if not data:
        return {}

    title = layout.get("title")
    title_text = title.get("text") if isinstance(title, dict) else title
    summary: Dict[str, Any] = {
        "traces": [_trace_summary(t) for t in data if isinstance(t, dict)],
    }
    if title_text:
        summary["title"] = str(title_text)
    x_axis = _axis_summary(layout.get("xaxis"))
    y_axis = _axis_summary(layout.get("yaxis"))
    y2_axis = _axis_summary(layout.get("yaxis2"))
    if x_axis:
        summary["xaxis"] = x_axis
    if y_axis:
        summary["yaxis"] = y_axis
    if y2_axis:
        summary["yaxis2"] = y2_axis
    return summary

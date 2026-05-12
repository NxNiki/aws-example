"""
Extract a compact, JSON-serializable summary of a Plotly figure dict.

The summary is what the LLM sees when generating a description: trace names,
x/y values (downsampled if very long), error-bar bounds, axis titles, and the
overall layout title. Keeping it compact protects the prompt from blowing up
on figures with thousands of points while preserving enough numeric detail
that the description references exact values rather than guessing.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Hard cap on how many points per trace land in the summary. Large enough
# to keep daily series for a year intact, small enough that a multi-trace
# figure stays under typical LLM context budgets.
_MAX_POINTS_PER_TRACE = 400


def _decode_plotly_array(value: Any) -> Any:
    """Materialise Plotly 6.x's binary array dict back into a Python list.

    When a Figure containing numpy arrays travels through ``dcc.Store`` or a
    callback ``State``, Plotly serialises arrays as
    ``{"dtype": <np dtype>, "bdata": <base64 bytes>, "shape": "n[, m]"}`` to
    save bandwidth. Without decoding we end up iterating that dict's *keys*
    and the LLM sees the literal strings "dtype", "bdata", "shape" as values.

    Returns the input unchanged if it's not in the binary-array shape.
    """
    if not isinstance(value, dict):
        return value
    if "bdata" not in value or "dtype" not in value:
        return value
    try:
        raw = base64.b64decode(value["bdata"])
        arr = np.frombuffer(raw, dtype=np.dtype(value["dtype"]))
        shape_str = str(value.get("shape") or "")
        if shape_str:
            shape = tuple(int(s.strip()) for s in shape_str.split(",") if s.strip())
            if shape:
                arr = arr.reshape(shape)
        return arr.tolist()
    except (ValueError, TypeError, binascii.Error):
        return value


def _round_value(value: Any) -> Any:
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        return round(value, 4)
    return value


def _downsample(values: Optional[Sequence[Any]], cap: int = _MAX_POINTS_PER_TRACE) -> Optional[List[Any]]:
    if values is None:
        return None
    items = _decode_plotly_array(values)
    if not isinstance(items, list):
        items = list(items)
    if len(items) <= cap:
        return [_round_value(v) for v in items]
    step = len(items) / cap
    sampled = [items[min(int(i * step), len(items) - 1)] for i in range(cap)]
    return [_round_value(v) for v in sampled]


def _round_matrix(matrix: Any) -> List[List[Any]]:
    rows = _decode_plotly_array(matrix)
    if not isinstance(rows, list):
        try:
            rows = list(rows)
        except TypeError:
            return []
    out: List[List[Any]] = []
    for row in rows:
        if hasattr(row, "__iter__") and not isinstance(row, (str, bytes)):
            out.append([_round_value(c) for c in row])
        else:
            out.append([_round_value(row)])
    return out


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

    # Heatmap-style traces carry their values in ``z`` (a 2-D array). Without
    # this the LLM only sees axis labels and has to guess at the cells —
    # which produced the "lacks Z-values" bug on the correlation matrix.
    z = trace.get("z")
    if z is not None:
        out["z"] = _round_matrix(z)

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

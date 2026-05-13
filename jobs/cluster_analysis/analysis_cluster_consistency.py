"""Cross-method cluster-label consistency dashboard.

Reads the shared ``cluster_labels.parquet`` produced by
:py:meth:`ClusterAnalysisPipeline.cluster_analysis` (one column per
``<model>-n_features_<N>-k_<K>`` run) and writes an interactive HTML that lets the user
pick any two label columns and view a contingency matrix (heatmap with counts / row %
/ column % / total %), plus permutation-invariant agreement metrics (ARI, NMI).

Extending: each analysis section is a self-contained block in the HTML template + a
render function in the JS. Add a new analysis by:
  1. Adding a new key to the ``bootstrap`` dict in :func:`build_html`.
  2. Adding a corresponding ``<h2>``/filters/chart block in ``cluster_consistency.html``.
  3. Adding a ``render*`` function in ``cluster_consistency.js`` wired to that block.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.utils import load_config

logger = logging.getLogger(__name__)

# Same convention as jobs/risk_control and jobs/simulation_report: HTML / JS templates
# live in `templates/` next to this script; substitution uses `__PLACEHOLDER__` tokens.
HTML_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _json_for_html_embed(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _render_html_from_template(template_name: str, replacements: Dict[str, str]) -> str:
    path = HTML_TEMPLATES_DIR / template_name
    text = path.read_text(encoding="utf-8")
    for key, val in replacements.items():
        text = text.replace(key, val)
    return text


def resolve_paths(config: dict) -> Tuple[Path, Path]:
    """Mirror ``ClusterAnalysisPipeline.cluster_labels_file_path`` derivation."""
    work_dir = LOCAL_ROOT / "jobs" / config["work_dir"]
    project_dir = work_dir / config["project_name"]
    labels_path = project_dir / "cluster_labels.parquet"
    return labels_path, project_dir


def discover_label_columns(df: pd.DataFrame, merge_cols: List[str]) -> List[str]:
    return [c for c in df.columns if c not in merge_cols]


def compute_pair(df: pd.DataFrame, col_a: str, col_b: str) -> Dict[str, Any]:
    """Crosstab + permutation-invariant agreement metrics for one (a, b) pair."""
    # When col_a == col_b (matrix diagonal) `df[[col_a, col_b]]` would create two
    # identical columns sharing the same name; reading `sub[col_a]` then returns a
    # 2-column DataFrame and crosstab fails.
    cols = [col_a] if col_a == col_b else [col_a, col_b]
    sub = df[cols].dropna()
    if sub.empty:
        return {"x": [], "y": [], "z": [], "total": 0, "ari": None, "nmi": None}
    sub = sub.astype(int)
    a = sub[col_a]
    b = a if col_a == col_b else sub[col_b]
    ct = pd.crosstab(a, b)  # rows = a, cols = b
    return {
        "y": [int(v) for v in ct.index.tolist()],
        "x": [int(v) for v in ct.columns.tolist()],
        "z": ct.values.tolist(),
        "total": int(sub.shape[0]),
        "ari": float(adjusted_rand_score(a, b)),
        "nmi": float(normalized_mutual_info_score(a, b)),
    }


def build_html(
    df: pd.DataFrame,
    label_columns: List[str],
    output_html: Path,
    title: str,
) -> None:
    if not label_columns:
        raise ValueError("no label columns found in cluster_labels.parquet")

    crosstabs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for a in label_columns:
        crosstabs[a] = {}
        for b in label_columns:
            crosstabs[a][b] = compute_pair(df, a, b)
    logger.info(f"computed {len(label_columns) ** 2} crosstabs across {len(label_columns)} label columns")

    bootstrap = {
        "label_columns": label_columns,
        "default_a": label_columns[0],
        "default_b": label_columns[1] if len(label_columns) > 1 else label_columns[0],
        "crosstabs": crosstabs,
        "contingency_div_id": "contingency-heatmap",
    }
    label_options = "".join(f'<option value="{c}">{c}</option>' for c in label_columns)
    script_body = (HTML_TEMPLATES_DIR / "cluster_consistency.js").read_text(encoding="utf-8")

    full_html = _render_html_from_template(
        "cluster_consistency.html",
        {
            "__TITLE__": title,
            "__N_ROWS__": str(df.shape[0]),
            "__N_LABELS__": str(len(label_columns)),
            "__LABEL_OPTIONS__": label_options,
            "__BOOTSTRAP_JSON__": _json_for_html_embed(bootstrap),
            "__SCRIPT__": script_body,
        },
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(full_html)
    logger.info(f"cluster consistency html written to {output_html}")


def main(config_path: str) -> None:
    setup_logging(
        f"{LOCAL_ROOT}/jobs/log",
        log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log",
    )

    config = load_config(config_path)
    labels_path, project_dir = resolve_paths(config)
    if not labels_path.exists():
        raise FileNotFoundError(f"cluster_labels.parquet not found at {labels_path}. Run cluster_analysis first.")

    df = pd.read_parquet(labels_path)
    merge_on = config["data_loader"]["merge_on"]
    merge_cols = [merge_on] if isinstance(merge_on, str) else list(merge_on)
    label_columns = discover_label_columns(df, merge_cols)
    logger.info(f"found {len(label_columns)} label column(s) in {labels_path}: {label_columns}")
    if len(label_columns) < 2:
        logger.warning(
            f"only {len(label_columns)} label column(s); need at least 2 for a meaningful comparison. "
            f"Continuing — the dropdowns will just show the same column on both axes."
        )

    output_html = project_dir / "cluster_consistency.html"
    title = f"Cluster Label Consistency — {config['project_name']}"
    build_html(df, label_columns, output_html, title)
    print(f"wrote: {output_html}")


if __name__ == "__main__":

    # project = "deepdive"
    # project = "wucaishen"
    # project = "ss01"
    # project = "ss01_only_normalized"
    project = "ss03"

    current_path = os.path.abspath(os.path.dirname(__file__))
    config_path = f"{current_path}/cluster_config-{project}.yaml"
    main(config_path)

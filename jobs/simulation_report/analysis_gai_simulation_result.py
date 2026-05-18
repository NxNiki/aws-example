"""
In the previous steps, we trained knn cluster model to identify 3 groups of bet behaviors for the data of 2024.
The cluster model was applied to data in month 1, 2, 3 of 2025.
GAI model was trained with data in 2024 and applied to data in 2025 to simulate gaming behaviors.
The GAI model is deployed to generate simulation data.

This script gets simulation data from the GAI model (slot machine) and compares game metrics such as bet, profit, etc.
between gamers from different clusters and using different math tables.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Literal, Set, Tuple, Union, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from matplotlib import font_manager, rcParams
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET, setup_logging
from bituslabs_ds.dashboard_utils import bootstrap_worker
from bituslabs_ds.eda import Anova, DataProfiler, DataVisualizer, split_column_by_threshold
from bituslabs_ds.s3_utils import list_s3_files, read_files, write_df_to_s3

logger = logging.getLogger(__name__)

# Same convention as jobs/risk_control: HTML / JS templates live in `templates/` next
# to this script. Substitution uses `__PLACEHOLDER__` tokens (no jinja runtime).
HTML_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.expand_frame_repr", False)


def _json_for_html_embed(obj: Any) -> str:
    """Serialize JSON for embedding in HTML; escape `</` so a stray closing tag in the
    payload can't break out of the surrounding <script> context."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _render_html_from_template(template_name: str, replacements: Dict[str, str]) -> str:
    path = HTML_TEMPLATES_DIR / template_name
    text = path.read_text(encoding="utf-8")
    for key, val in replacements.items():
        text = text.replace(key, val)
    return text


fontP = font_manager.FontProperties()
fontP.set_family("Heiti TC")
fontP.set_size(12)
rcParams["font.family"] = ["Heiti TC"]


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_file}:")
    logger.info(f"config: \n {pformat(config)} \n")
    return config


def load_data(config, reload: bool = False) -> pd.DataFrame:

    local_output = f"{LOCAL_ROOT}/jobs/{config['work_dir']}/{config['data_loader']['local_cache']}"
    s3_files: List[str] = []
    if reload or not os.path.exists(local_output):
        # Honor the per-project bucket override in the yaml (e.g. ss03 reads from
        # slotmachine-ai-game-table) instead of the team-wide S3_BUCKET constant.
        input_bucket = config["data_loader"].get("bucket", S3_BUCKET)
        s3_files += list_s3_files(
            input_bucket,
            prefix=config["data_loader"]["prefix"],
            pattern=config["data_loader"]["pattern"],
        )

    data = cast(
        pd.DataFrame,
        read_files(
            s3_files, local_cache_path=local_output, reload=reload, columns=config["data_loader"]["columns_to_read"]
        ),
    )
    # Pull the cluster digit out of player_id (e.g. "cluster2_user_007" -> "2") and prefix
    # it dynamically, so the dashboard handles 1-cluster runs (ss03) and N-cluster runs
    # (ss01/ss02) without a hardcoded label map. Direct assignment avoids the chained
    # `df[col].replace(..., inplace=True)` pattern, which under pandas 2.x copy-on-write
    # semantics can silently no-op and leave cluster_index with the raw "clusterN" tokens.
    cluster_digit = data["player_id"].str.extract(r"cluster(\d+)", expand=False)
    data["cluster_index"] = "user group: " + cluster_digit

    print(data.shape)
    # data = data[data["total_spins"]>40]
    # print(data.shape)

    data = split_column_by_threshold(
        data,
        columns=["base_game_win", "free_game_win", "big_win_count", "free_spins_count"],
        threshold=[0, 0, 0, 9],
    )

    return data


def process_data(data, config, output_path) -> pd.DataFrame:
    data = data.copy()
    transformer = PowerTransformer()

    metrics = config["radar_plot_columns"]
    data[metrics] = transformer.fit_transform(data[metrics])

    os.makedirs(output_path, exist_ok=True)
    joblib.dump(transformer, f"{output_path}/power_transformer.joblib")

    scaler_name = config["data_loader"]["scaler"]
    if scaler_name == "standard":
        scaler: Union[StandardScaler, MinMaxScaler] = StandardScaler()
    elif scaler_name == "minMax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"unsupported scaler: {scaler_name}")

    data[metrics] = scaler.fit_transform(data[metrics])
    # Save scaler mean and std to a CSV file
    if isinstance(scaler, StandardScaler):
        scaler_stats = pd.DataFrame({"metric": metrics, "mean": scaler.mean_, "std": scaler.scale_})
        scaler_stats.to_csv(f"{output_path}/scaler_stats.csv", index=False)
    # Also save the scaler model for later use
    scaler_filename = f"{output_path}/scaler_model.joblib"
    joblib.dump(scaler, scaler_filename)

    return data


def make_radar_plot(
    data: pd.DataFrame,
    group: str,
    config: Dict,
    agg_method: Literal["mean", "median"],
    title: str,
    output_path: str,
) -> None:
    """
    Generate and save a radar plot for specified metrics, aggregated over groups.

    Args:
        data (pd.DataFrame): Input dataframe containing the data to plot.
        group (str): Categorical column name in `data` to group by, curves will be plotted for each unique value.
        metrics (List[str]): List of columns containing the numeric metrics to plot on the radar chart's axes.
        agg_method (Literal["mean", "median"]): Method to aggregate `metrics` within each group ('mean' or 'median').
        title (str): Title for the radar plot. If empty or None, a default title is constructed.
        output_path (str): Path to save the resulting radar plot figure (e.g., "output.png").
    """

    metrics = config["radar_plot_columns"]
    metrics_rename = config["radar_plot_rename"]

    # Compute mean values per group for the metrics
    if agg_method == "mean":
        grouped = data.groupby(group)[metrics].mean().reset_index()
    elif agg_method == "median":
        grouped = data.groupby(group)[metrics].median().reset_index()
    else:
        raise ValueError(f"unsupported agg_method: {agg_method}")

    # Radar plot setup
    labels = metrics_rename
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for _, row in grouped.iterrows():
        values = row[metrics].tolist()
        values += values[:1]  # close the polygon
        ax.plot(angles, values, label=row[group])
        ax.fill(angles, values, alpha=0.25)

    # Add labels and aesthetics
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    for i, label in enumerate(ax.get_xticklabels()):
        if (i + len(labels) // 4) // (len(labels) // 2) % 2 == 0:
            label.set_horizontalalignment("left")
        else:
            label.set_horizontalalignment("right")

        if i // (len(labels) // 2) == 0:
            label.set_verticalalignment("bottom")
        else:
            label.set_verticalalignment("top")

        label.set_y(label.get_position()[1] + 0.05)

    # --- Set y-axis (radial axis) scale for the radar plot. ---
    if config["data_loader"]["scaler"] == "minMax":
        ax.set_ylim(0, 1)
        num_rgrids = 5
        grid_values = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num_rgrids)
        ax.set_yticks(grid_values)
        ax.set_yticklabels([f"{v:.2f}" for v in grid_values], fontsize=10)

    if title is None or len(title) == 0:
        title = f"Radar plot by {group}"
    ax.set_title(title, fontsize=14, pad=15, fontproperties=fontP)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def _build_radar_figure(
    radar_data: pd.DataFrame,
    raw_radar_data: pd.DataFrame,
    metrics: List[str],
    metric_labels: List[str],
    aggs: List[str],
    math_tables: List[str],
    clusters: List[str],
    radial_range: Union[List[float], None],
) -> tuple:
    """
    Build a Plotly radar figure with one Scatterpolar trace per
    (agg, math_table, cluster, data_mode) combo. ``data_mode`` is "transformed" (power
    transform + scaler applied) or "raw" (load_data values) so the HTML page can switch
    between the two via a top-level toggle.

    Returns (fig, trace_keys) where trace_keys is a parallel list of
    [agg, math, cluster, data_mode] used by the JS visibility map.
    """
    theta = list(metric_labels) + [metric_labels[0]]
    fig = go.Figure()
    trace_keys: List[List[str]] = []
    data_modes = (("transformed", radar_data), ("raw", raw_radar_data))
    for agg in aggs:
        for data_mode, src in data_modes:
            for math in math_tables:
                for cluster in clusters:
                    sub = cast(
                        pd.DataFrame,
                        src[(src["machine_id"] == math) & (src["cluster_index"] == cluster)],
                    )
                    if sub.empty:
                        # Keep trace count consistent with trace_keys so the JS visibility map lines up.
                        values: List[float] = [0.0] * len(metrics)
                    else:
                        agg_series = sub[metrics].mean() if agg == "mean" else sub[metrics].median()
                        values = cast(pd.Series, agg_series).tolist()
                    r_vals = values + [values[0]]
                    trace_name = f"{math}, {cluster}"
                    fig.add_trace(
                        go.Scatterpolar(
                            r=r_vals,
                            theta=theta,
                            fill="toself",
                            name=trace_name,
                            legendgroup=trace_name,
                            visible=False,
                        )
                    )
                    trace_keys.append([agg, str(math), str(cluster), data_mode])

    polar: Dict[str, Dict[str, Any]] = {"radialaxis": {"visible": True}}
    if radial_range is not None:
        polar["radialaxis"]["range"] = radial_range
    fig.update_layout(polar=polar, height=600, margin=dict(l=40, r=40, t=40, b=40), showlegend=True)
    return fig, trace_keys


def _build_boxplot_figure(
    boxplot_data: pd.DataFrame,
    raw_boxplot_data: pd.DataFrame,
    metrics: List[str],
    factors: List[str],
    n_boot: int = 500,
) -> Tuple[go.Figure, List[Dict[str, str]]]:
    """
    Build a Plotly figure containing BOTH a box-plot trace set and a bar-with-CI trace set
    for every (metric, data_mode) pair. ``data_mode`` is "transformed" (post
    ``transform_skewed_columns`` + PCA augmentation) or "raw" (load_data values).
    Derived columns (e.g. ``compound_metric_pca*``) are absent from raw and fall back to
    the transformed values, so the toggle is a no-op for them — as called out by the user.

    Bars show mean ± 95 % bootstrap CI using ``n_boot`` resamples per cell, reusing the
    same ``bootstrap_worker`` helper as the operational game-stats dashboard.

    Returns (fig, trace_keys), where
    trace_keys[i] = {"metric": ..., "mode": "box"|"bar", "data_mode": "transformed"|"raw"}
    is parallel to fig.data so JS can flip visibility cleanly.
    """
    fig = go.Figure()
    trace_keys: List[Dict[str, str]] = []

    two_factor = len(factors) >= 2
    if two_factor:
        x_col, color_col = factors[0], factors[1]
        color_levels = sorted(boxplot_data[color_col].dropna().unique().tolist(), key=str)
    else:
        x_col = factors[0]
        color_col = None
        color_levels = [None]
    x_levels = sorted(boxplot_data[x_col].dropna().unique().tolist(), key=str)

    def _resolve_source(metric: str, data_mode: str) -> pd.DataFrame:
        if data_mode == "raw" and metric in raw_boxplot_data.columns:
            return raw_boxplot_data
        return boxplot_data

    def _bar_arrays(metric: str, level, src: pd.DataFrame):
        x_vals: List = []
        means: List[float] = []
        err_upper: List[float] = []
        err_lower: List[float] = []
        for xv in x_levels:
            mask = src[x_col] == xv
            if two_factor:
                mask = mask & (src[color_col] == level)
            arr = src.loc[mask, metric].dropna().to_numpy()
            if len(arr) == 0:
                continue
            mean_val = float(np.mean(arr))
            lo, hi = bootstrap_worker(arr, n_boot=n_boot)
            x_vals.append(xv)
            means.append(mean_val)
            err_upper.append(hi - mean_val if np.isfinite(hi) else 0.0)
            err_lower.append(mean_val - lo if np.isfinite(lo) else 0.0)
        return x_vals, means, err_upper, err_lower

    for metric in metrics:
        for data_mode in ("transformed", "raw"):
            src = _resolve_source(metric, data_mode)
            # Box traces.
            if two_factor:
                for level in color_levels:
                    sub = src[src[color_col] == level]
                    fig.add_trace(
                        go.Box(
                            y=sub[metric],
                            x=sub[x_col],
                            name=f"{color_col}={level}",
                            legendgroup=str(level),
                            boxpoints="outliers",
                            visible=False,
                        )
                    )
                    trace_keys.append({"metric": metric, "mode": "box", "data_mode": data_mode})
            else:
                fig.add_trace(
                    go.Box(
                        y=src[metric],
                        x=src[x_col],
                        name=metric,
                        boxpoints="outliers",
                        visible=False,
                    )
                )
                trace_keys.append({"metric": metric, "mode": "box", "data_mode": data_mode})

            # Bar-with-CI traces (mean ± 95% bootstrap CI).
            if two_factor:
                for level in color_levels:
                    x_vals, means, eup, elo = _bar_arrays(metric, level, src)
                    fig.add_trace(
                        go.Bar(
                            x=x_vals,
                            y=means,
                            error_y=dict(type="data", array=eup, arrayminus=elo, symmetric=False),
                            name=f"{color_col}={level}",
                            legendgroup=str(level),
                            visible=False,
                        )
                    )
                    trace_keys.append({"metric": metric, "mode": "bar", "data_mode": data_mode})
            else:
                x_vals, means, eup, elo = _bar_arrays(metric, None, src)
                fig.add_trace(
                    go.Bar(
                        x=x_vals,
                        y=means,
                        error_y=dict(type="data", array=eup, arrayminus=elo, symmetric=False),
                        name=metric,
                        visible=False,
                    )
                )
                trace_keys.append({"metric": metric, "mode": "bar", "data_mode": data_mode})

    fig.update_layout(
        boxmode="group" if two_factor else "overlay",
        barmode="group",
        xaxis_title=factors[0],
        yaxis_title=metrics[0] if metrics else "",
        height=550,
        margin=dict(l=60, r=40, t=60, b=40),
        showlegend=two_factor,
    )
    return fig, trace_keys


def build_simulation_dashboard_html(
    radar_data: pd.DataFrame,
    raw_radar_data: pd.DataFrame,
    radar_metrics: List[str],
    radar_metric_labels: List[str],
    boxplot_data: pd.DataFrame,
    raw_boxplot_data: pd.DataFrame,
    boxplot_metrics: List[str],
    significant_metrics: Set[str],
    factors: List[str],
    output_html: str,
    title: str,
    scaler_kind: str,
    n_boot: int = 500,
) -> None:
    """
    Render an interactive HTML dashboard combining the radar plot (per agg / math_table / cluster)
    and per-metric distribution plots into one page. Works for both 1-cluster (single-factor ANOVA)
    and multi-cluster (two-factor ANOVA) runs.

    Radar: math_table and cluster are exposed as **multi-select checkboxes** so several curves
    can be overlaid at once. Agg method (mean/median) is a single-select dropdown.

    Distribution panel: a metric dropdown picks the variable; a display toggle switches between
    a **box plot** and a **bar plot showing mean ± 95 % bootstrap CI** (``n_boot`` resamples).

    A page-level **Values** toggle (Transformed / Raw) switches both plots between
    post-transform / scaled values and the original load_data values. The radar's polar
    axis switches to autorange in raw mode (the transformed [0, 1] cap from a minMax
    scaler would clip raw values). PCA features in the box panel are not affected by the
    toggle (they only exist in transformed form).
    """
    aggs: List[str] = ["mean", "median"]
    math_tables = sorted(radar_data["machine_id"].dropna().unique().tolist(), key=str)
    clusters = sorted(radar_data["cluster_index"].dropna().unique().tolist(), key=str)

    radial_range = [0.0, 1.0] if scaler_kind == "minMax" else None
    radar_fig, trace_keys = _build_radar_figure(
        radar_data,
        raw_radar_data,
        radar_metrics,
        radar_metric_labels,
        aggs,
        math_tables,
        [str(c) for c in clusters],
        radial_range,
    )

    radar_div_id = "radar-plot"
    # plotly's stubs type `include_plotlyjs` as bool, but the runtime accepts "cdn"/"directory"/etc.
    radar_html = pio.to_html(radar_fig, include_plotlyjs="cdn", full_html=False, div_id=radar_div_id)  # type: ignore[arg-type]

    box_div_id = "box-plot"
    box_trace_keys: List[Dict[str, str]] = []
    has_box = bool(boxplot_metrics and factors)
    if has_box:
        box_fig, box_trace_keys = _build_boxplot_figure(
            boxplot_data, raw_boxplot_data, boxplot_metrics, factors, n_boot=n_boot
        )
        box_html = pio.to_html(box_fig, include_plotlyjs=False, full_html=False, div_id=box_div_id)
    else:
        box_html = "<p><em>No boxplot metrics or factors available.</em></p>"

    def _options(values: List[str]) -> str:
        return "".join(f'<option value="{v}">{v}</option>' for v in values)

    def _metric_options(values: List[str], significant: Set[str]) -> str:
        parts = []
        for v in values:
            prefix = "★ " if v in significant else ""
            parts.append(f'<option value="{v}">{prefix}{v}</option>')
        return "".join(parts)

    def _checkboxes(name: str, values: List[str]) -> str:
        return "".join(
            f'<label class="cb"><input type="checkbox" name="{name}" value="{v}" checked> {v}</label>' for v in values
        )

    agg_options = _options(aggs)
    math_boxes = _checkboxes("math", [str(m) for m in math_tables])
    cluster_boxes = _checkboxes("cluster", [str(c) for c in clusters])
    box_metric_options = _metric_options(boxplot_metrics, significant_metrics) if has_box else ""

    # Derived metrics (e.g. compound_metric_pca*) only exist in transformed data; the
    # box panel silently falls back to transformed values for them in raw mode.
    raw_passthrough_metrics = [m for m in boxplot_metrics if m not in raw_boxplot_data.columns]
    sig_note = (
        f"<p><em>Metrics with at least one significant post-hoc pair are prefixed with ★ "
        f"({len(significant_metrics)} of {len(boxplot_metrics)} marked).</em></p>"
    )
    raw_note = (
        f'<p class="meta"><em>Raw mode falls back to transformed values for derived metrics '
        f"({', '.join(raw_passthrough_metrics)}).</em></p>"
        if raw_passthrough_metrics
        else ""
    )

    if has_box:
        box_filters_html = (
            f'<div class="filters">'
            f'  <label>Metric: <select id="boxMetricSelect">{box_metric_options}</select></label>'
            f"  <label>Display:"
            f'    <label class="cb"><input type="radio" name="boxMode" value="box" checked> Box plot</label>'
            f'    <label class="cb"><input type="radio" name="boxMode" value="bar"> '
            f"Bar (mean ± 95% CI, {n_boot} boot)</label>"
            f"  </label>"
            f"</div>"
        )
    else:
        box_filters_html = ""

    meta_text = (
        f"Factors: {', '.join(factors) if factors else '(none)'}"
        f" &nbsp;|&nbsp; Math tables: {len(math_tables)}"
        f" &nbsp;|&nbsp; Clusters: {len(clusters)}"
    )
    bootstrap = {
        "radar_trace_keys": trace_keys,
        "box_trace_keys": box_trace_keys,
        "transformed_radial_range": radial_range,
        "radar_div_id": radar_div_id,
        "box_div_id": box_div_id,
    }
    script_body = (HTML_TEMPLATES_DIR / "simulation_dashboard.js").read_text(encoding="utf-8")

    full_html = _render_html_from_template(
        "simulation_dashboard.html",
        {
            "__TITLE__": title,
            "__META__": meta_text,
            "__SCALER_KIND__": scaler_kind,
            "__AGG_OPTIONS__": agg_options,
            "__MATH_BOXES__": math_boxes,
            "__CLUSTER_BOXES__": cluster_boxes,
            "__RADAR_HTML__": radar_html,
            "__SIG_NOTE__": sig_note,
            "__RAW_NOTE__": raw_note,
            "__BOX_FILTERS_HTML__": box_filters_html,
            "__BOX_HTML__": box_html,
            "__BOOTSTRAP_JSON__": _json_for_html_embed(bootstrap),
            "__SCRIPT__": script_body,
        },
    )

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(full_html)
    logger.info(f"Dashboard HTML written to {output_html}")


def main(config_file):

    project_name = os.path.basename(config_file).replace(".yaml", "")
    time_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    setup_logging(LOCAL_ROOT / "jobs/log", f"simulation_analysis_{project_name}_{time_tag}.log")

    config = load_config(config_file)
    output_path = f"{LOCAL_ROOT}/jobs/{config['work_dir']}/radar_plot_{config['data_loader']['scaler']}"

    data = load_data(config, reload=False)
    data_profiler = DataProfiler(data, skewness_threshold=1.5)

    # Snapshot untouched values for the dashboard's raw-vs-transformed toggle BEFORE
    # process_data (which fits PowerTransformer + scaler on radar metrics) or
    # transform_skewed_columns (which mutates the profiler's frame in place).
    raw_radar_data = cast(
        pd.DataFrame,
        data[["machine_id", "cluster_index"] + list(config["radar_plot_columns"])].copy(),
    )
    raw_boxplot_data = data.copy()

    # # Radar plot for each math table (3 clusters in one plot)
    data = process_data(data, config, output_path=output_path)
    agg_methods: List[Literal["mean", "median"]] = ["mean", "median"]
    for math_table in data["machine_id"].unique():
        logger.info(f"make radar plot for {math_table}")
        for agg_method in agg_methods:
            make_radar_plot(
                cast(pd.DataFrame, data[data["machine_id"] == math_table]),
                group="cluster_index",
                config=config,
                agg_method=agg_method,
                title=f"数学表: {math_table}",
                output_path=f"{output_path}/radar_plot_{math_table}_{agg_method}.png",
            )

    data_profiler.transform_skewed_columns(pos_suffix="", neg_suffix="")
    output_path = f"{LOCAL_ROOT}/jobs/{config['work_dir']}/anova_result"
    os.makedirs(output_path, exist_ok=True)

    # viz = DataVisualizer(data_profiler)

    # # Plot distributions with distribution statistics
    # viz.create_figure(
    #     layout_cols=data_profiler.processed_numerical_columns, group_col="cluster_index", n_cols=5, fig_size=(7, 3.5)
    # )
    # viz.add_histogram(show_distribution_stats=True, kde=True)
    # viz.figure.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.97, wspace=0.2, hspace=0.25)
    # viz.display()
    # viz.save(f"{output_path}/gai_simulation_distribution_transformed.png")

    # # Plot correlation heatmap
    # viz.create_figure(fig_title="Correlation for: cluster 2", fig_size=(12, 7))
    # viz.add_correlation_heatmap(cbar="Red")
    # viz.figure.subplots_adjust(left=0.15, bottom=0.15, top=0.90, right=0.97)
    # viz.display()
    # viz.save(f"{output_path}/gai_simulation_correlation.png")

    # # Plot correlation heatmap for machine_id
    # viz.create_figure(layout_cols=["machine_id"], fig_title="Correlation for: cluster 2", fig_size=(12, 7))
    # viz.add_correlation_heatmap(cbar="Red")
    # viz.figure.subplots_adjust(left=0.15, bottom=0.15, top=0.90, right=0.97, wspace=0.25, hspace=0.35)
    # viz.display()
    # viz.save(f"{output_path}/gai_simulation_correlation_machine_id.png")

    # ANOVA:
    anova_between_vars = ["machine_id", "cluster_index"]
    # anova_between_vars = ["machine_id"]

    data_profiler.augment_columns(columns=config["pca_columns"], col_name="compound_metric", method=["pca"])

    anova_dv = [
        "total_bet",
        "total_spins",
        "total_profit",
        "compound_metric_pca1",
        "compound_metric_pca2",
        "compound_metric_pca3",
    ]

    res = data_profiler.show_group_stats(
        group_cols=["machine_id", "cluster_index"],
        var_columns=anova_dv,
        transpose=False,
        agg_stats=["mean", "median", "std"],
    )
    res.to_csv(f"{output_path}/report_selected_features.csv")

    res = data_profiler.show_group_stats(
        group_cols=["machine_id", "cluster_index"],
        var_columns=anova_dv,
        transpose=False,
    )
    res.to_csv(f"{output_path}/report_all_features.csv")

    anova = Anova(data_profiler.df, between_vars=anova_between_vars, var_columns=anova_dv)
    anova_res = anova.run_anova()
    write_df_to_s3(anova_res, S3_BUCKET, key=f"ds-data-kmeans/{project_name}/anova_result.csv")

    anova.run_post_hoc_analysis(anova_dv, effects="main", p_thresh=0.05)

    # Use the between_vars Anova actually kept (degenerate single-level factors are dropped in __init__)
    # so the branch below stays consistent with reality, e.g. for ss03 where only one cluster exists.
    if len(anova.between_vars) > 1:
        anova.run_post_hoc_analysis(anova_dv, effects="interaction", group_var="cluster_index", p_thresh=0.05)
        anova.show_box_plot(
            x_col="cluster_index",
            group_col="machine_id",
            output_path=output_path,
            fig_title="boxplot_two_factors",
            stripplot_kws={"size": 2},
            effects="interaction",
        )
    else:
        anova.show_box_plot(
            x_col=anova.between_vars[0],
            output_path=output_path,
            fig_title=f"boxplot_{anova.between_vars[0]}",
            effects="main",
        )

    # Interactive dashboard combining the radar plot (filterable by agg method / math table / cluster)
    # with per-metric box plots. Metrics that produced any significant post-hoc pair are marked with ★.
    significant_metrics: Set[str] = set(anova.post_hoc_report.get("variable", []))
    build_simulation_dashboard_html(
        radar_data=data,
        raw_radar_data=raw_radar_data,
        radar_metrics=config["radar_plot_columns"],
        radar_metric_labels=config["radar_plot_rename"],
        boxplot_data=data_profiler.df,
        raw_boxplot_data=raw_boxplot_data,
        boxplot_metrics=anova_dv,
        significant_metrics=significant_metrics,
        factors=anova.between_vars,
        output_html=f"{output_path}/simulation_dashboard.html",
        title=f"Simulation Analysis: {project_name}",
        scaler_kind=config["data_loader"]["scaler"],
    )


if __name__ == "__main__":

    # project = "ss01"
    project = "ss03"
    # project = "deepdive"
    # project = "wucaishen"

    current_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(description="simulation analysis pipeline")
    parser.add_argument(
        "--config_file", default=f"{current_path}/simulation_analysis_config-{project}.yaml", help="Project to analyze"
    )
    args = parser.parse_args()

    main(args.config_file)

/* Template file. Python replaces each "__TOKEN__" (quotes included) with the JSON value.
 * After substitution, every const below holds its real runtime type (array, number, null, string). */
const LINE_RAW = "__LINE_RAW__";
const LINE_METRICS = "__LINE_METRICS__";
const LINE_CUTOFF = "__LINE_CUTOFF__";

const RTP_DATA = "__RTP_DATA__";
const RTP_GROUPS = "__RTP_GROUPS__";
const RTP_GROUP_KEY = "__RTP_GROUP_KEY__";
const RTP_BINS = "__RTP_BINS__";

/* ---- Per-arm RTP histograms ---- */
(function renderRtpHistograms() {
  const host = document.getElementById("rtp-hist-container");
  if (!host) return;
  if (!RTP_DATA.length) {
    host.innerHTML = "<p><em>No base-game RTP values to plot.</em></p>";
    return;
  }
  const arms = ReportUtils.uniqueSortedBy(RTP_DATA, function (r) { return r.ab_arm; });
  arms.forEach(function (arm, idx) {
    const armRows = RTP_DATA.filter(function (r) { return r.ab_arm === arm; });
    const panel = document.createElement("div");
    panel.style.marginBottom = "1.5rem";

    const h = document.createElement("h2");
    h.style.fontSize = "1.05rem";
    h.textContent = "ab_arm: " + arm + " (n = " + armRows.length + " user-days)";
    panel.appendChild(h);

    const gd = document.createElement("div");
    gd.id = "rtp-hist-" + idx;
    gd.style.width = "100%";
    gd.style.minHeight = "360px";
    panel.appendChild(gd);
    host.appendChild(panel);

    let traces;
    if (RTP_GROUP_KEY && RTP_GROUPS.length) {
      traces = RTP_GROUPS.map(function (g) {
        return {
          type: "histogram",
          name: String(g),
          x: armRows.filter(function (r) { return r[RTP_GROUP_KEY] === g; }).map(function (r) { return r.rtp; }),
          nbinsx: RTP_BINS,
          opacity: 0.55,
          histfunc: "count",
        };
      });
    } else {
      traces = [{
        type: "histogram",
        name: arm,
        x: armRows.map(function (r) { return r.rtp; }),
        nbinsx: RTP_BINS,
        histfunc: "count",
      }];
    }

    const layout = {
      title: "Per-user-day base-game RTP — " + arm,
      xaxis: { title: "Base-game RTP (mean)" },
      yaxis: { title: "User-days (log scale)", type: "log" },
      barmode: "overlay",
      showlegend: Boolean(RTP_GROUP_KEY),
      legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1 },
    };
    Plotly.newPlot(gd.id, traces, layout, { responsive: true });
  });
})();

/* ---- Daily metrics line chart ---- */
(function renderLineChart() {
  const sel = document.getElementById("metric-select");
  const chartEl = document.getElementById("line-chart");
  if (!sel || !chartEl) return;
  LINE_METRICS.forEach(function (m) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    sel.appendChild(opt);
  });
  const defaultMetric = LINE_METRICS.indexOf("ratio_users_incentivized") >= 0
    ? "ratio_users_incentivized"
    : LINE_METRICS[0];
  if (LINE_METRICS.length) {
    sel.value = defaultMetric;
  }

  function layoutFor(metric) {
    return {
      title: metric,
      xaxis: { title: "activity_date" },
      yaxis: { title: metric },
      hovermode: "closest",
      legend: { orientation: "h" },
      shapes: ReportUtils.cutoffShape(LINE_CUTOFF),
    };
  }

  function tracesFor(metric) {
    const arms = ReportUtils.uniqueSortedBy(LINE_RAW, function (r) { return r.ab_arm; });
    return arms.map(function (arm) {
      const rows = LINE_RAW.filter(function (r) { return r.ab_arm === arm; }).sort(function (a, b) {
        return String(a.activity_date).localeCompare(String(b.activity_date));
      });
      return {
        type: "scatter",
        mode: "lines+markers",
        name: arm,
        x: rows.map(function (r) { return r.activity_date; }),
        y: rows.map(function (r) { return r[metric]; }),
      };
    });
  }

  function redraw() {
    const metric = sel.value;
    Plotly.newPlot("line-chart", tracesFor(metric), layoutFor(metric), { responsive: true });
  }

  if (LINE_METRICS.length) {
    sel.addEventListener("change", redraw);
    redraw();
  } else {
    chartEl.innerHTML = "<p><em>No numeric metrics after aggregating by day and AB arm.</em></p>";
  }
})();

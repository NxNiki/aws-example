/* Template file. Python replaces each "__TOKEN__" (quotes included) with the JSON value. */
const RAW = "__RAW__";
const DEFAULT_BINS = "__DEFAULT_BINS__";

(function () {
  const binsInput = document.getElementById("bet-bins");
  const thresholdInput = document.getElementById("bet-threshold");
  const logCheck = document.getElementById("bet-log");
  const note = document.getElementById("threshold-note");
  const host = document.getElementById("bet-hist-container");
  const COLOR_BELOW = "#27ae60";
  const COLOR_AT_ABOVE = "#2980b9";
  if (!RAW.length) {
    host.innerHTML = "<p><em>No rows to plot.</em></p>";
    return;
  }
  const groups = ReportUtils.uniqueSortedBy(RAW, function (r) { return r.ab_arm; });
  const plotIds = [];

  function draw() {
    const nbins = Math.max(1, parseInt(binsInput.value, 10) || DEFAULT_BINS);
    const T = parseFloat(thresholdInput.value);
    const useLog = logCheck.checked;
    const Tvalid = !isNaN(T) && isFinite(T);
    ReportUtils.purgePlots(plotIds);
    plotIds.length = 0;
    host.innerHTML = "";

    const summaryParts = [];
    if (Tvalid) {
      summaryParts.push("Threshold: <code>num_bets &lt; " + T + "</code>. ");
    }

    groups.forEach(function (g, idx) {
      const x = RAW.filter(function (r) { return r.ab_arm === g; }).map(function (r) { return r.num_bets; });
      const n = x.length;
      let xBelow = [];
      let xAbove = [];
      if (Tvalid) {
        xBelow = x.filter(function (v) { return v < T; });
        xAbove = x.filter(function (v) { return v >= T; });
      }
      const belowCount = Tvalid ? xBelow.length : 0;
      const propBelow = n > 0 && Tvalid ? (100 * belowCount / n) : null;

      const panel = document.createElement("div");
      panel.style.marginBottom = "2rem";
      const h = document.createElement("h2");
      h.style.fontSize = "1.05rem";
      h.appendChild(document.createTextNode("ab_arm: " + g + " (n = " + n + " user-days)"));
      if (propBelow !== null) {
        h.appendChild(document.createTextNode(" — "));
        const s = document.createElement("span");
        s.innerHTML = "P(<code>num_bets</code> &lt; " + T + ") = <strong>" + propBelow.toFixed(2) + "%</strong> (" + belowCount + "/" + n + ")";
        h.appendChild(s);
      } else {
        h.appendChild(document.createTextNode(" — set a numeric threshold to color bars and see proportion below T."));
      }
      panel.appendChild(h);
      const gd = document.createElement("div");
      const gid = "bet-hist-" + idx;
      gd.id = gid;
      gd.style.width = "100%";
      gd.style.minHeight = "360px";
      panel.appendChild(gd);
      host.appendChild(panel);
      plotIds.push(gid);

      let data;
      const bingroup = "bg-" + idx;
      if (Tvalid) {
        data = [
          {
            type: "histogram",
            name: "num_bets < " + T,
            x: xBelow,
            nbinsx: nbins,
            bingroup: bingroup,
            histfunc: "count",
            marker: { color: COLOR_BELOW, line: { width: 1, color: "#1e8449" } },
            opacity: 0.88,
          },
          {
            type: "histogram",
            name: "num_bets ≥ " + T,
            x: xAbove,
            nbinsx: nbins,
            bingroup: bingroup,
            histfunc: "count",
            marker: { color: COLOR_AT_ABOVE, line: { width: 1, color: "#1f618d" } },
            opacity: 0.88,
          },
        ];
      } else {
        data = [{
          type: "histogram",
          name: "all",
          x: x,
          nbinsx: nbins,
          histfunc: "count",
          marker: { color: "#7f8c8d", line: { width: 1, color: "#333" } },
          opacity: 0.85,
        }];
      }

      const shapes = Tvalid ? [{
        type: "line",
        xref: "x",
        yref: "paper",
        x0: T,
        x1: T,
        y0: 0,
        y1: 1,
        line: { color: "#555", width: 1.5, dash: "dash" },
      }] : [];

      const layout = {
        title: "Frequency of user-days by num_bets",
        barmode: Tvalid ? "stack" : "overlay",
        xaxis: { title: "num_bets (bets before first FREE that day)" },
        yaxis: {
          title: "Count of user-days (frequency)",
          type: useLog ? "log" : "linear",
        },
        showlegend: Tvalid,
        legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1 },
        shapes: shapes,
      };
      Plotly.newPlot(gd, data, layout, { responsive: true });

      if (propBelow !== null) {
        summaryParts.push(
          "<strong>" + g + "</strong>: " + propBelow.toFixed(2) + "% &lt; " + T + " (" + belowCount + "/" + n + ")"
        );
      }
    });

    if (note) {
      note.innerHTML = Tvalid
        ? summaryParts.join(" &nbsp;|&nbsp; ")
        : ('Enter a numeric threshold to split each bar into '
          + '<span style="color:' + COLOR_BELOW + '">num_bets &lt; T</span> vs '
          + '<span style="color:' + COLOR_AT_ABOVE + '">num_bets ≥ T</span>, '
          + 'and to show P(num_bets &lt; T).');
    }
  }

  binsInput.addEventListener("change", draw);
  binsInput.addEventListener("input", draw);
  thresholdInput.addEventListener("change", draw);
  thresholdInput.addEventListener("input", draw);
  logCheck.addEventListener("change", draw);
  draw();
})();

(function (global) {
  "use strict";

  function uniqueSortedBy(arr, keyFn) {
    var seen = new Set();
    var out = [];
    for (var i = 0; i < arr.length; i++) {
      var k = keyFn(arr[i]);
      if (!seen.has(k)) {
        seen.add(k);
        out.push(k);
      }
    }
    return out.sort();
  }

  function purgePlots(ids) {
    if (typeof Plotly === "undefined") return;
    ids.forEach(function (id) {
      var el = (typeof id === "string") ? document.getElementById(id) : id;
      if (el) { Plotly.purge(el); }
    });
  }

  function cutoffShape(cutoffValue) {
    if (cutoffValue === null || cutoffValue === undefined || cutoffValue === "") {
      return [];
    }
    return [{
      type: "line",
      xref: "x",
      yref: "paper",
      x0: cutoffValue,
      x1: cutoffValue,
      y0: 0,
      y1: 1,
      line: { color: "#555", width: 2, dash: "dash" }
    }];
  }

  global.ReportUtils = {
    uniqueSortedBy: uniqueSortedBy,
    purgePlots: purgePlots,
    cutoffShape: cutoffShape
  };
})(window);

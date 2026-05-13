(function () {
  var bootstrap = JSON.parse(document.getElementById('dashboard-bootstrap').textContent);
  var crosstabs = bootstrap.crosstabs || {};
  var divId = bootstrap.contingency_div_id;

  var labelASelect = document.getElementById('labelASelect');
  var labelBSelect = document.getElementById('labelBSelect');
  var normSelect = document.getElementById('normSelect');
  var overlapMeta = document.getElementById('overlap-meta');

  if (bootstrap.default_a) labelASelect.value = bootstrap.default_a;
  if (bootstrap.default_b) labelBSelect.value = bootstrap.default_b;

  function rowSums(z) {
    return z.map(function (row) { return row.reduce(function (a, b) { return a + b; }, 0); });
  }
  function colSums(z) {
    if (!z.length) return [];
    return z[0].map(function (_, j) {
      return z.reduce(function (a, row) { return a + row[j]; }, 0);
    });
  }
  function normalize(z, mode) {
    if (mode === 'none' || !z.length) return z;
    var rs = rowSums(z);
    var cs = colSums(z);
    var total = rs.reduce(function (a, b) { return a + b; }, 0);
    return z.map(function (row, i) {
      return row.map(function (v, j) {
        var denom = mode === 'row' ? rs[i] : mode === 'col' ? cs[j] : total;
        return denom ? v / denom : 0;
      });
    });
  }

  function renderContingency() {
    var a = labelASelect.value;
    var b = labelBSelect.value;
    var mode = normSelect.value;
    var ct = (crosstabs[a] || {})[b];
    if (!ct || ct.total === 0) {
      Plotly.purge(divId);
      overlapMeta.textContent = 'No overlapping rows for this label pair.';
      return;
    }
    var zNorm = normalize(ct.z, mode);
    var text = zNorm.map(function (row, i) {
      return row.map(function (v, j) {
        return mode === 'none'
          ? ct.z[i][j].toString()
          : (v * 100).toFixed(1) + '%\n(' + ct.z[i][j] + ')';
      });
    });
    var data = [{
      type: 'heatmap',
      z: zNorm,
      x: ct.x.map(String),
      y: ct.y.map(String),
      text: text,
      texttemplate: '%{text}',
      colorscale: 'Blues',
      hovertemplate: 'A=%{y}, B=%{x}<br>value: %{z}<extra></extra>',
      showscale: true
    }];
    var layout = {
      xaxis: { title: 'B: ' + b, type: 'category' },
      yaxis: { title: 'A: ' + a, type: 'category', autorange: 'reversed' },
      margin: { l: 80, r: 40, t: 40, b: 100 }
    };
    Plotly.react(divId, data, layout, { responsive: true });
    overlapMeta.textContent =
      'Overlapping rows: ' + ct.total +
      ' (rows where both labels are non-null). ' +
      'Adjusted Rand index: ' + (ct.ari !== undefined ? ct.ari.toFixed(3) : 'n/a') +
      '. Normalized MI: ' + (ct.nmi !== undefined ? ct.nmi.toFixed(3) : 'n/a') + '.';
  }

  labelASelect.addEventListener('change', renderContingency);
  labelBSelect.addEventListener('change', renderContingency);
  normSelect.addEventListener('change', renderContingency);
  renderContingency();
})();

(function () {
  var bootstrap = JSON.parse(document.getElementById('dashboard-bootstrap').textContent);
  var radarTraceKeys = bootstrap.radar_trace_keys || [];
  var boxTraceKeys = bootstrap.box_trace_keys || [];
  var transformedRadialRange = bootstrap.transformed_radial_range;
  var radarDivId = bootstrap.radar_div_id;
  var boxDivId = bootstrap.box_div_id;

  function getChecked(name) {
    return Array.prototype.slice.call(
      document.querySelectorAll('input[name="' + name + '"]:checked')
    ).map(function (cb) { return cb.value; });
  }

  function getDataMode() {
    var el = document.querySelector('input[name="dataMode"]:checked');
    return el ? el.value : 'transformed';
  }

  function applyRadialRange(dataMode) {
    // Raw values can exceed the [0, 1] minMax cap, so always autorange in raw mode.
    if (dataMode === 'raw' || transformedRadialRange === null) {
      Plotly.relayout(radarDivId, { 'polar.radialaxis.autorange': true });
    } else {
      Plotly.relayout(radarDivId, {
        'polar.radialaxis.autorange': false,
        'polar.radialaxis.range': transformedRadialRange
      });
    }
  }

  function updateRadar() {
    var agg = document.getElementById('aggSelect').value;
    var maths = getChecked('math');
    var clusters = getChecked('cluster');
    var dataMode = getDataMode();
    var vis = radarTraceKeys.map(function (k) {
      return (k[0] === agg)
        && (maths.indexOf(k[1]) !== -1)
        && (clusters.indexOf(k[2]) !== -1)
        && (k[3] === dataMode);
    });
    Plotly.restyle(radarDivId, { visible: vis });
    applyRadialRange(dataMode);
  }

  document.getElementById('aggSelect').addEventListener('change', updateRadar);
  Array.prototype.forEach.call(
    document.querySelectorAll('input[name="math"], input[name="cluster"]'),
    function (el) { el.addEventListener('change', updateRadar); }
  );
  Array.prototype.forEach.call(document.querySelectorAll('.cb-row button'), function (btn) {
    btn.addEventListener('click', function () {
      var group = btn.getAttribute('data-group');
      var checked = btn.getAttribute('data-action') === 'all';
      Array.prototype.forEach.call(
        document.querySelectorAll('input[name="' + group + '"]'),
        function (cb) { cb.checked = checked; }
      );
      updateRadar();
    });
  });
  updateRadar();

  var boxSel = document.getElementById('boxMetricSelect');

  function updateBox() {
    if (!boxSel || !boxTraceKeys.length) return;
    var metric = boxSel.value;
    var modeEl = document.querySelector('input[name="boxMode"]:checked');
    var mode = modeEl ? modeEl.value : 'box';
    var dataMode = getDataMode();
    var vis = boxTraceKeys.map(function (k) {
      return k.metric === metric && k.mode === mode && k.data_mode === dataMode;
    });
    Plotly.restyle(boxDivId, { visible: vis });
    Plotly.relayout(boxDivId, { 'yaxis.title.text': metric });
  }

  if (boxSel) {
    boxSel.addEventListener('change', updateBox);
    Array.prototype.forEach.call(document.querySelectorAll('input[name="boxMode"]'), function (el) {
      el.addEventListener('change', updateBox);
    });
    updateBox();
  }

  // Data-mode (raw vs transformed) drives both plots.
  Array.prototype.forEach.call(document.querySelectorAll('input[name="dataMode"]'), function (el) {
    el.addEventListener('change', function () {
      updateRadar();
      updateBox();
    });
  });
})();

/* Predict & Visualize tab: drawing canvas, live tracing, prediction card, network view,
 * neuron inspector, and the "teach the network" flow. */
(function (S) {
  'use strict';
  const { $, el, api, clear } = S.dom;
  const U = S.util;
  const SIZE = 280, BRUSH = 20, LIVE_MS = 110;
  const st = { ctx: null, drawing: false, last: null, ink: false, label: null, pixels: null, seq: 0,
               timer: null, view: null, modelId: null, weights: new Map(), canPromoteUndo: false, hoverSeq: 0 };

  // ----- drawing ---------------------------------------------------------------------------
  function resetCanvas() { st.ctx.fillStyle = '#000'; st.ctx.fillRect(0, 0, SIZE, SIZE); st.ctx.fillStyle = '#fff'; }
  function pt(e) {
    const r = $('canvas').getBoundingClientRect();
    return { x: (e.clientX - r.left) * (SIZE / (r.width || SIZE)), y: (e.clientY - r.top) * (SIZE / (r.height || SIZE)) };
  }
  function dot(p) { st.ctx.beginPath(); st.ctx.arc(p.x, p.y, BRUSH / 2, 0, Math.PI * 2); st.ctx.fill(); }
  function seg(a, b) { st.ctx.beginPath(); st.ctx.moveTo(a.x, a.y); st.ctx.lineTo(b.x, b.y); st.ctx.stroke(); }

  function onDown(e) {
    if (e.button > 0) return;
    try { $('canvas').setPointerCapture(e.pointerId); } catch (err) { /* not supported / synthetic event */ }
    st.drawing = true; st.last = pt(e); dot(st.last); st.ink = true;
    schedule(false); e.preventDefault();
  }
  function onMove(e) {
    if (!st.drawing) return;
    const evs = (e.getCoalescedEvents && e.getCoalescedEvents().length) ? e.getCoalescedEvents() : [e];
    evs.forEach((ev) => { const p = pt(ev); seg(st.last, p); st.last = p; });
    schedule(false);
  }
  function onUp() {
    if (!st.drawing) return;
    st.drawing = false;
    schedule(true);
  }

  function schedule(final) {
    if (!final && !$('live').checked) return;
    if (final) { clearTimeout(st.timer); st.timer = null; if ($('live').checked) run(false); return; }
    if (st.timer) return;
    st.timer = setTimeout(() => { st.timer = null; run(false); }, LIVE_MS);
  }

  function setHint(text, isError) {
    const h = $('hint'); h.textContent = text; h.classList.toggle('error-text', !!isError);
  }
  const DEFAULT_HINT = 'Draw a large digit. Enter predicts, C clears. With Live on, neurons react while you draw.';

  function currentPixels() {
    if (!st.ink) { setHint('Draw a digit first.', true); return null; }
    const d = st.ctx.getImageData(0, 0, SIZE, SIZE).data;
    const gray = new Float64Array(SIZE * SIZE);
    for (let i = 0; i < gray.length; i++) gray[i] = d[i * 4];
    const r = window.SynapsePre.normalizeToMnist(gray, SIZE, SIZE);
    if (r.error) { setHint(r.error === 'too_small' ? 'Draw a bigger digit: it should fill most of the canvas.' : 'The canvas is empty.', true); return null; }
    setHint(DEFAULT_HINT, false);
    st.pixels = r.pixels;
    drawPreview(r.pixels);
    return r.pixels;
  }
  function drawPreview(px) {
    const c = $('preview').getContext('2d'), img = c.createImageData(28, 28);
    for (let i = 0; i < 784; i++) { img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = px[i]; img.data[i * 4 + 3] = 255; }
    c.putImageData(img, 0, 0);
  }

  // ----- requests --------------------------------------------------------------------------
  async function run(animate) {
    const pix = currentPixels();
    if (!pix) return null;
    const my = ++st.seq, t0 = performance.now();
    try {
      const res = await api('/trace', { json: { pixels: Array.from(pix), label: st.label == null ? undefined : st.label, edges: 80 } });
      if (my !== st.seq) return null;
      apply(res, pix, animate, performance.now() - t0);
      return res;
    } catch (e) {
      if (my === st.seq) showError(e);
      return null;
    }
  }

  function apply(res, pix, animate, rtt) {
    if (res.model_id !== st.modelId) { st.modelId = res.model_id; st.weights.clear(); }
    $('model-badge').textContent = res.model_id === 'release' ? 'release model' : 'model ' + res.model_id;
    showPrediction(res.prediction);
    $('latency').textContent = U.fmtMs(res.prediction.processing_ms) + ' · round trip ' + U.fmtMs(rtt);
    st.view.update(pix, res.trace, { animate: !!animate });
    $('btn-save-sample').disabled = st.label == null;
  }

  function showError(e) {
    const w = $('result-warning');
    w.hidden = false; w.className = 'warning error';
    w.textContent = e.message || String(e);
  }

  function showPrediction(p) {
    const digit = $('result-digit'), w = $('result-warning');
    const ok = p.status === 'ok';
    digit.textContent = ok ? String(p.digit) : '?';
    digit.classList.toggle('dim', !ok);
    $('result-label').textContent = ok ? 'Looks like a ' + p.digit : 'Not sure what this is';
    $('result-conf').textContent = U.fmtPct(p.confidence, 1) + ' confident' + (p.threshold ? ' · reject threshold ' + U.fmtPct(p.threshold, 1) : '');
    if (ok) { w.hidden = true; }
    else {
      w.hidden = false; w.className = 'warning';
      w.textContent = 'Low confidence. Best guess is ' + p.top_guess + ' (' + U.fmtPct(p.confidence, 1) + '), below the reject threshold. '
        + 'This may not be a clean digit from 1 to 9.';
    }
    const rows = $('scores').children;
    p.digits.forEach((d, i) => {
      const row = rows[i]; if (!row) return;
      row.classList.toggle('top', d === p.top_guess);
      row.querySelector('.score-fill').style.width = (p.scores[i] * 100).toFixed(1) + '%';
      row.querySelector('.score-pct').textContent = U.fmtPct(p.scores[i], 1);
    });
  }

  function buildScores(digits) {
    const box = $('scores'); clear(box);
    digits.forEach((d) => box.appendChild(el('div', { class: 'score-row' }, [
      el('span', { text: String(d) }), el('div', { class: 'score-track' }, [el('div', { class: 'score-fill' })]),
      el('span', { class: 'score-pct', text: '0%' })])));
  }

  // ----- inspector -------------------------------------------------------------------------
  async function getWeights(layer) {
    const key = st.modelId + ':' + layer;
    if (!st.weights.has(key)) {
      const w = await api('/model/weights/' + layer);
      st.weights.set(key, { shape: w.shape, scales: w.scales, bias: w.bias, q: U.b64ToInt8(w.data) });
    }
    return st.weights.get(key);
  }

  async function inspect(info) {
    const title = $('inspect-title'), body = $('inspect-body'), tile = $('inspect-tile');
    const my = ++st.hoverSeq;
    if (!info) {
      title.textContent = 'Hover a neuron';
      body.textContent = 'See its activation, error signal and the weight pattern it detects.';
      tile.getContext('2d').clearRect(0, 0, tile.width, tile.height);
      return;
    }
    title.textContent = info.isOutput ? 'Output neuron for digit ' + info.digit : 'Hidden ' + (info.layer + 1) + ' · neuron #' + info.index;
    const parts = [info.isOutput ? 'probability ' + U.fmtPct(info.activation, 1) + ' (logit ' + U.fmtNum(info.logit, 2) + ')'
                                 : 'activation ' + U.fmtNum(info.activation, 3) + (info.activation === 0 ? ' (inactive)' : '')];
    if (info.delta != null) parts.push('error signal δ = ' + U.fmtNum(info.delta, 4));
    body.textContent = parts.join(' · ');
    try {
      const w = await getWeights(info.layer);
      if (my !== st.hoverSeq) return;
      const row = U.dequantRow(w.q, w.scales, info.index, w.shape[1]);
      const vmax = U.maxAbs(row);
      if (info.layer === 0) S.charts.drawTile(tile, row, 28, 28, vmax);
      else {
        const g = U.gridLayout(w.shape[1], 128, 320), padded = new Float32Array(g.cols * g.rows);
        padded.set(row);
        S.charts.drawTile(tile, padded, g.cols, g.rows, vmax);
      }
      body.textContent += ' · bias ' + U.fmtNum(w.bias[info.index], 3);
    } catch (e) { /* weights unavailable: keep text only */ }
  }

  // ----- direction / label controls ----------------------------------------------------------
  function setMode(mode) {
    $('mode-forward').classList.toggle('active', mode === 'forward');
    $('mode-backward').classList.toggle('active', mode === 'backward');
    st.view.setMode(mode);
    legend(mode);
  }
  function legend(mode) {
    const box = $('viz-legend'); clear(box);
    const add = (color, text) => { const s = el('span', {}, [el('span', { class: 'sw' }), text]); s.firstChild.style.background = color; box.appendChild(s); };
    if (mode === 'forward') {
      add('#22d3ee', 'bright = neuron active (ReLU output)'); add('#22d3ee', 'cyan line = excites the next neuron'); add('#f472b6', 'pink line = inhibits it');
    } else {
      add('#f97316', 'orange: δ > 0, this neuron pushed the loss up'); add('#3b82f6', 'blue: δ < 0, it should grow');
      add('#f87171', 'red/blue lines = error flowing back'); add('#34d399', 'dashed green ring = true digit');
    }
  }

  function buildLabelButtons(digits) {
    const box = $('label-buttons'); clear(box);
    digits.forEach((d) => {
      const b = el('button', { class: 'digit-btn', type: 'button', 'data-digit': String(d), 'aria-pressed': 'false', text: String(d) });
      b.addEventListener('click', () => setLabel(st.label === d ? null : d));
      box.appendChild(b);
    });
  }
  function setLabel(d) {
    st.label = d;
    Array.from($('label-buttons').children).forEach((b) => {
      const on = Number(b.dataset.digit) === d;
      b.classList.toggle('sel', on); b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    $('btn-save-sample').disabled = d == null || !st.pixels;
    if (d != null) setMode('backward');
    if (st.ink) run(true);
  }

  async function saveSample() {
    if (st.label == null || !st.pixels) return;
    const msg = $('fb-status');
    try {
      const r = await api('/feedback', { json: { pixels: Array.from(st.pixels), label: st.label } });
      msg.textContent = (r.duplicate ? 'Already saved. ' : 'Saved as ' + st.label + '. ') + r.count + ' drawing' + (r.count === 1 ? '' : 's') + ' in your training set.';
      if (S.train && S.train.refreshFeedback) S.train.refreshFeedback();
    } catch (e) { msg.textContent = e.message; }
  }

  async function refreshUndo() {
    try {
      const v = await api('/model/versions');
      st.canPromoteUndo = v.history.length > 0;
      $('btn-undo').hidden = !st.canPromoteUndo;
    } catch (e) { /* ignore */ }
  }
  async function undo() {
    try {
      await api('/model/rollback', { method: 'POST', json: {} });
      $('fb-status').textContent = 'Rolled back to the previous model.';
      await refreshUndo();
      if (st.ink) run(true);
      if (S.model) S.model.refresh();
    } catch (e) { $('fb-status').textContent = e.message; }
  }

  function clearAll() {
    resetCanvas(); st.ink = false; st.pixels = null; st.seq++;
    clearTimeout(st.timer); st.timer = null;
    $('result-digit').textContent = '–'; $('result-digit').classList.remove('dim');
    $('result-label').textContent = 'Draw a digit to begin'; $('result-conf').textContent = '';
    $('result-warning').hidden = true; $('latency').textContent = '–';
    Array.from($('scores').children).forEach((r) => { r.classList.remove('top'); r.querySelector('.score-fill').style.width = '0'; r.querySelector('.score-pct').textContent = '0%'; });
    $('preview').getContext('2d').clearRect(0, 0, 28, 28);
    $('btn-save-sample').disabled = true;
    setHint(DEFAULT_HINT, false);
    st.view.trace = null; st.view.renderEmpty();
  }

  function onKey(e) {
    if ($('tab-predict').hidden) return;
    const tag = (e.target && e.target.tagName) || '';
    if (['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].indexOf(tag) >= 0) return;
    if (e.key === 'Enter') { e.preventDefault(); run(true); }
    else if (e.key === 'c' || e.key === 'C') clearAll();
  }

  function init(digits) {
    digits = digits || [1, 2, 3, 4, 5, 6, 7, 8, 9];
    const canvas = $('canvas');
    st.ctx = canvas.getContext('2d', { willReadFrequently: true });
    st.ctx.fillStyle = '#fff'; st.ctx.strokeStyle = '#fff'; st.ctx.lineWidth = BRUSH; st.ctx.lineCap = 'round'; st.ctx.lineJoin = 'round';
    resetCanvas();
    canvas.addEventListener('pointerdown', onDown); canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerup', onUp); canvas.addEventListener('pointercancel', onUp);
    $('btn-predict').addEventListener('click', () => run(true));
    $('btn-clear').addEventListener('click', clearAll);
    $('btn-replay').addEventListener('click', () => st.view.play());
    $('mode-forward').addEventListener('click', () => setMode('forward'));
    $('mode-backward').addEventListener('click', () => setMode('backward'));
    $('btn-save-sample').addEventListener('click', saveSample);
    $('btn-undo').addEventListener('click', undo);
    $('btn-finetune').addEventListener('click', () => { S.train.startFinetune({ inline: true }); });
    document.addEventListener('keydown', onKey);
    st.view = new S.NetworkView($('netviz'), { onHover: inspect });
    buildScores(digits); buildLabelButtons(digits); legend('forward'); setHint(DEFAULT_HINT, false);
    refreshUndo();
  }

  S.predict = { init, run, clearAll, setLabel, setMode, refreshUndo, st, saveSample };
})(window.Synapse = window.Synapse || {});

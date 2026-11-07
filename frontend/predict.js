(function (S) {
  'use strict';
  const { $, el, api, clear } = S.dom;
  const U = S.util;
  const SIZE = 280, BRUSH = 20, LIVE_MS = 110;
  const PAPER = '#fffdf6', INK = '#14171f';
  const st = { ctx: null, drawing: false, last: null, ink: false, label: null, pixels: null, seq: 0,
               timer: null, view: null, modelId: null, weights: new Map(), hoverSeq: 0 };

  function resetCanvas() {
    st.ctx.fillStyle = PAPER; st.ctx.fillRect(0, 0, SIZE, SIZE);
    st.ctx.fillStyle = INK;
  }
  function pt(e) {
    const r = $('canvas').getBoundingClientRect();
    return { x: (e.clientX - r.left) * (SIZE / (r.width || SIZE)), y: (e.clientY - r.top) * (SIZE / (r.height || SIZE)) };
  }
  function dot(p) { st.ctx.beginPath(); st.ctx.arc(p.x, p.y, BRUSH / 2, 0, Math.PI * 2); st.ctx.fill(); }
  function seg(a, b) { st.ctx.beginPath(); st.ctx.moveTo(a.x, a.y); st.ctx.lineTo(b.x, b.y); st.ctx.stroke(); }

  function onDown(e) {
    if (e.button > 0) return;
    try { $('canvas').setPointerCapture(e.pointerId); } catch (err) { st.drawing = st.drawing; }
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

  const DEFAULT_HINT = 'Draw one digit, big enough to fill most of the board. Enter reads it, C wipes it.';
  function setHint(text, isError) {
    const h = $('hint'); h.textContent = text; h.classList.toggle('error-text', !!isError);
  }

  function currentPixels() {
    if (!st.ink) { setHint('Draw a digit first.', true); return null; }
    const d = st.ctx.getImageData(0, 0, SIZE, SIZE).data;
    const gray = new Float64Array(SIZE * SIZE);
    for (let i = 0; i < gray.length; i++) gray[i] = 255 - d[i * 4];
    const r = window.SynapsePre.normalizeToMnist(gray, SIZE, SIZE);
    if (r.error) {
      setHint(r.error === 'too_small' ? 'Draw a bigger digit: it should fill most of the board.' : 'The board is empty.', true);
      return null;
    }
    setHint(DEFAULT_HINT, false);
    st.pixels = r.pixels;
    drawPreview(r.pixels);
    return r.pixels;
  }

  function drawPreview(px) {
    const c = $('preview').getContext('2d'), img = c.createImageData(28, 28);
    const bg = [251, 248, 240], fg = [28, 34, 48];
    for (let i = 0; i < 784; i++) {
      const t = px[i] / 255;
      img.data[i * 4] = bg[0] + (fg[0] - bg[0]) * t;
      img.data[i * 4 + 1] = bg[1] + (fg[1] - bg[1]) * t;
      img.data[i * 4 + 2] = bg[2] + (fg[2] - bg[2]) * t;
      img.data[i * 4 + 3] = 255;
    }
    c.putImageData(img, 0, 0);
  }

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
      if (my === st.seq) {
        if (e.code === 'rate_limited') setHint('Reading faster than the server allows. Give it a second.', true);
        else showError(e);
      }
      return null;
    }
  }

  function apply(res, pix, animate, rtt) {
    if (res.model_id !== st.modelId) { st.modelId = res.model_id; st.weights.clear(); }
    $('model-badge').textContent = res.model_id === 'release' ? 'release model' : res.model_id;
    showPrediction(res.prediction);
    $('latency').textContent = U.fmtMs(res.prediction.processing_ms) + ', round trip ' + U.fmtMs(rtt);
    st.view.update(pix, res.trace, { animate: !!animate });
    $('btn-save-sample').disabled = st.label == null;
  }

  function showError(e) {
    const w = $('result-warning');
    w.hidden = false; w.className = 'alert error';
    w.textContent = e.message || String(e);
  }

  function showPrediction(p) {
    const digit = $('result-digit'), w = $('result-warning');
    const ok = p.status === 'ok';
    digit.textContent = ok ? String(p.digit) : '?';
    digit.classList.toggle('dim', !ok);
    $('result-label').textContent = ok ? 'Reads as a ' + p.digit + '.' : 'Not sure what this is.';
    $('result-conf').textContent = U.fmtPct(p.confidence, 1) + ' confident' + (p.threshold ? ', says "not sure" below ' + U.fmtPct(p.threshold, 1) : '');
    if (ok) w.hidden = true;
    else {
      w.hidden = false; w.className = 'alert';
      w.textContent = 'Its best guess is ' + p.top_guess + ' at ' + U.fmtPct(p.confidence, 1) + ', which is under the confidence it needs. This may not be a clean digit from 1 to 9.';
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

  async function getWeights(layer) {
    const key = st.modelId + ':' + layer;
    if (!st.weights.has(key)) {
      const w = await api('/model/weights/' + layer);
      st.weights.set(key, { shape: w.shape, scales: w.scales, bias: w.bias, q: U.b64ToInt8(w.data) });
    }
    return st.weights.get(key);
  }

  function topK(items, k) {
    return items.sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, k);
  }

  function unitEdges(info, ws) {
    const view = st.view, tr = view.trace, back = view.mode === 'backward' && view.hasBackward;
    const l = info.layer, j = info.index, out = [];
    const act = (layer) => (layer < 0 ? Array.from(st.pixels, (v) => v / 255) : tr.layers[layer].activation);
    const delta = (layer) => tr.backward.deltas[tr.layers[layer].name];
    const Wl = ws[l], cols = Wl.shape[1];
    const incoming = [];
    const row = U.dequantRow(Wl.q, Wl.scales, j, cols);
    const src = act(l - 1);
    for (let i = 0; i < cols; i++) incoming.push({ fromLayer: l - 1, fromIdx: i, toLayer: l, toIdx: j, v: back ? row[i] * delta(l)[j] : row[i] * src[i] });
    topK(incoming, 12).forEach((e) => out.push(e));
    if (ws[l + 1]) {
      const Wn = ws[l + 1], outgoing = [];
      for (let k = 0; k < Wn.shape[0]; k++) {
        const w = Wn.q[k * Wn.shape[1] + j] * Wn.scales[k];
        outgoing.push({ fromLayer: l, fromIdx: j, toLayer: l + 1, toIdx: k, v: back ? w * delta(l + 1)[k] : w * act(l)[j] });
      }
      topK(outgoing, 12).forEach((e) => out.push(e));
    }
    const m = Math.max.apply(null, out.map((e) => Math.abs(e.v)).concat([1e-12]));
    out.forEach((e) => { e.s = Math.abs(e.v) / m; });
    return out;
  }

  async function inspect(info) {
    const title = $('inspect-title'), body = $('inspect-body'), tile = $('inspect-tile');
    const my = ++st.hoverSeq;
    if (!info) {
      st.view.setFocus(null);
      title.textContent = 'Point at a unit';
      body.textContent = 'You get its activation, its error signal, and the picture of weights it looks for.';
      tile.getContext('2d').clearRect(0, 0, tile.width, tile.height);
      return;
    }
    title.textContent = info.isOutput ? 'Output unit for the digit ' + info.digit : 'Hidden layer ' + (info.layer + 1) + ', unit ' + info.index;
    const parts = [info.isOutput ? 'probability ' + U.fmtPct(info.activation, 1) + ', logit ' + U.fmtNum(info.logit, 2)
                                 : 'activation ' + U.fmtNum(info.activation, 3) + (info.activation === 0 ? ' (silent)' : '')];
    if (info.delta != null) parts.push('error signal ' + U.fmtNum(info.delta, 4));
    body.textContent = parts.join('; ');
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
      body.textContent += '; bias ' + U.fmtNum(w.bias[info.index], 3);
      const ws = {};
      ws[info.layer] = w;
      if (info.layer + 1 < st.view.trace.layers.length) ws[info.layer + 1] = await getWeights(info.layer + 1);
      if (my === st.hoverSeq && st.view.hover) st.view.setFocus(unitEdges(info, ws));
    } catch (e) { body.textContent += ''; }
  }

  function setMode(mode) {
    $('mode-forward').classList.toggle('active', mode === 'forward');
    $('mode-backward').classList.toggle('active', mode === 'backward');
    st.view.setMode(mode);
    legend(mode);
  }
  function legend(mode) {
    $('viz-legend').textContent = mode === 'backward'
      ? 'In the error view, red-brown dots are units that pushed the loss up, blue dots are units that should have been stronger, and the pen lines carry the error back.'
      : '';
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
      msg.textContent = (r.duplicate ? 'Already saved. ' : 'Saved as ' + st.label + '. ') + r.count + ' drawing' + (r.count === 1 ? '' : 's') + ' saved so far.';
      if (S.train && S.train.refreshFeedback) S.train.refreshFeedback();
    } catch (e) { msg.textContent = e.message; }
  }

  async function refreshUndo() {
    try {
      const v = await api('/model/versions');
      $('btn-undo').hidden = v.history.length === 0;
    } catch (e) { $('btn-undo').hidden = true; }
  }
  async function undo() {
    try {
      await api('/model/rollback', { method: 'POST', json: {} });
      $('fb-status').textContent = 'Went back to the previous network.';
      await refreshUndo();
      if (st.ink) run(true);
      if (S.model) S.model.refresh();
    } catch (e) { $('fb-status').textContent = e.message; }
  }

  function clearAll() {
    resetCanvas(); st.ink = false; st.pixels = null; st.seq++;
    clearTimeout(st.timer); st.timer = null;
    $('result-digit').textContent = '–'; $('result-digit').classList.remove('dim');
    $('result-label').textContent = 'Nothing drawn yet.'; $('result-conf').textContent = '';
    $('result-warning').hidden = true; $('latency').textContent = '–';
    Array.from($('scores').children).forEach((r) => {
      r.classList.remove('top'); r.querySelector('.score-fill').style.width = '0'; r.querySelector('.score-pct').textContent = '0%';
    });
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
    st.ctx.strokeStyle = INK; st.ctx.lineWidth = BRUSH; st.ctx.lineCap = 'round'; st.ctx.lineJoin = 'round';
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

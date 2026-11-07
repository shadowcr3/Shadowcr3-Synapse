(function (S) {
  'use strict';
  const { $, el, api, clear } = S.dom;
  const U = S.util, C = S.charts;
  const MAX_PTS = 600;
  const st = { since: 0, timer: null, busy: false, ready: false, jobId: null, kind: null, inline: false, total: 0, best: null,
               hist: { tl: [], vl: [], ta: [], va: [] }, batch: { x: [], y: [] }, grad: {}, names: [],
               init: null, cur: null, showChange: false, probeView: null, sweep: [], redraw: false, ftConfirm: null };

  const setMsg = (id, text, isErr) => { const n = $(id); n.textContent = text || ''; n.classList.toggle('error-text', !!isErr); };

  function resetLive(ev) {
    st.jobId = ev.job_id; st.kind = ev.kind; st.best = null; st.total = 0;
    st.hist = { tl: [], vl: [], ta: [], va: [] }; st.batch = { x: [], y: [] }; st.grad = {}; st.sweep = []; st.init = st.cur = null;
    $('job-result').hidden = true;
    $('sweep-panel').hidden = ev.kind !== 'sweep';
    $('job-progress').value = 0;
    clear($('weight-tiles')); clear($('weight-delta'));
    redrawCharts();
  }

  function decodeTiles(t) {
    const q = U.b64ToInt8(t.data), rows = [];
    for (let i = 0; i < t.n; i++) rows.push(U.dequantRow(q, t.scales, i, 784));
    return rows;
  }

  function onInit(ev) {
    st.total = ev.total_epochs || 0; st.names = ev.names || [];
    if (ev.tiles) { st.init = st.cur = decodeTiles(ev.tiles); drawTiles(); }
    if (ev.probe) probe(ev.probe, false);
    header();
  }

  function probe(p, animate) {
    if (!st.probeView) return;
    st.probeView.update(Uint8Array.from(p.pixels), Object.assign({ digits: p.trace.digits || [1, 2, 3, 4, 5, 6, 7, 8, 9] }, p.trace), { animate: animate });
  }

  function push(arr, v) { arr.push(v); if (arr.length > MAX_PTS) arr.splice(0, arr.length - MAX_PTS); }

  function onBatch(ev) {
    const frac = (ev.step_in_epoch || 0) / (ev.steps_per_epoch || 1);
    push(st.batch.x, (ev.epoch - 1) + frac); push(st.batch.y, ev.loss);
    (ev.grad_norms || []).forEach((g, i) => {
      const name = st.names[i] || ('p' + i);
      if (name.charAt(0) !== 'W') return;
      (st.grad[name] = st.grad[name] || { x: [], y: [] });
      push(st.grad[name].x, ev.step); push(st.grad[name].y, g);
    });
    progress(ev.step / ev.total_steps, ev);
    st.redraw = true; scheduleRedraw();
  }

  function onEpoch(ev) {
    st.hist.tl.push(ev.train_loss); st.hist.vl.push(ev.val_loss); st.hist.ta.push(ev.train_acc); st.hist.va.push(ev.val_acc);
    st.best = ev.best_epoch;
    if (ev.tiles) { st.cur = decodeTiles(ev.tiles); drawTiles(); }
    if (ev.weight_stats) {
      const rows = [];
      ev.names.forEach((n, i) => {
        if (n.charAt(0) !== 'W') return;
        const where = i === 0 ? 'in to h1' : i === ev.names.length - 2 ? 'h to out' : 'h to h';
        rows.push({ label: n + ', ' + where, value: ev.weight_stats[i].rel_change, text: U.fmtPct(ev.weight_stats[i].rel_change, 2) });
      });
      C.bars($('weight-delta'), rows);
    }
    if (ev.probe) probe(ev.probe, true);
    progress(ev.epoch / ev.epochs, ev);
    redrawCharts();
  }

  function onSweepArch(ev) {
    st.sweep.push(ev.result);
    renderSweep(st.sweep);
    $('sweep-panel').hidden = false;
  }

  let rafId = 0;
  function scheduleRedraw() {
    if (!rafId) rafId = requestAnimationFrame(() => { rafId = 0; if (st.redraw) { st.redraw = false; redrawCharts(); } });
  }

  function redrawCharts() {
    const e = st.hist.tl.map((_, i) => i + 1);
    C.lineChart($('chart-loss'), [
      { name: 'batch', cls: 'line-faint', xs: st.batch.x, ys: st.batch.y },
      { name: 'train', cls: 'line-train', xs: e, ys: st.hist.tl }, { name: 'validation', cls: 'line-val', xs: e, ys: st.hist.vl }],
      { markX: st.best, label: 'loss chart' });
    C.lineChart($('chart-acc'), [{ name: 'train', cls: 'line-train', xs: e, ys: st.hist.ta }, { name: 'validation', cls: 'line-val', xs: e, ys: st.hist.va }],
      { markX: st.best, label: 'accuracy chart' });
    const gs = Object.keys(st.grad).sort().map((k) => ({ name: k, xs: st.grad[k].x, ys: st.grad[k].y }));
    C.lineChart($('chart-grad'), gs, { yLog: true, label: 'gradient size per layer' });
  }

  function drawTiles() {
    const box = $('weight-tiles');
    if (!st.cur) return;
    if (box.children.length !== st.cur.length) {
      clear(box);
      st.cur.forEach((_, i) => box.appendChild(el('canvas', { title: 'hidden unit ' + i, 'aria-label': 'weights of hidden unit ' + i })));
    }
    const vals = st.cur.map((row, i) => (st.showChange && st.init ? row.map((v, k) => v - st.init[i][k]) : row));
    let vmax = 0; vals.forEach((r) => { vmax = Math.max(vmax, U.maxAbs(r)); });
    vals.forEach((r, i) => C.drawTile(box.children[i], Float32Array.from(r), 28, 28, vmax || 1));
  }

  function header(extra) {
    $('job-header').textContent = st.kind ? (st.kind + (extra ? ', ' + extra : '')) : 'nothing running';
  }
  function progress(frac, ev) {
    if (ev && ev.arch_index != null && ev.arch_total) frac = (ev.arch_index + frac) / ev.arch_total;
    const pct = Math.round(U.clamp(frac, 0, 1) * 100);
    $('job-progress').value = pct;
    let text = pct + '%';
    if (ev && ev.epoch != null) text = 'epoch ' + ev.epoch + ' of ' + (ev.epochs || st.total) + ', ' + text;
    if (ev && ev.arch_index != null) text = 'layout ' + (ev.arch_index + 1) + ' of ' + ev.arch_total + ', ' + text;
    header(text);
    if (st.inline) { $('ft-bar').value = pct; $('ft-text').textContent = text; }
  }

  function renderSweep(results) {
    if (!results.length) return;
    const best = results.reduce((a, b) => (b.val_acc > a.val_acc ? b : a));
    const label = (r) => '[' + r.hidden_sizes.join(', ') + ']';
    C.bars($('sweep-bars'), results.map((r) => ({ label: label(r), value: r.val_acc, text: U.fmtPct(r.val_acc, 2), cls: r === best ? 'best' : '' })));
    C.lineChart($('sweep-curves'), results.map((r, i) => ({ name: label(r), cls: 'line-' + (i % 5), ys: r.history.val_acc })), { label: 'validation accuracy per epoch' });
    const near = results.filter((r) => r.val_acc >= best.val_acc - 0.005).reduce((a, b) => (b.parameter_count < a.parameter_count ? b : a));
    $('sweep-summary').textContent = 'Best layout: ' + label(best) + ' at ' + U.fmtPct(best.val_acc, 2) + ' with ' + U.fmtInt(best.parameter_count) + ' parameters. '
      + (near !== best ? 'The smallest within half a point of it is ' + label(near) + ' with ' + U.fmtInt(near.parameter_count) + ' parameters (' + U.fmtPct(near.val_acc, 2) + '). ' : '')
      + 'Past a point, more units mostly buy more overfitting.';
    const t = $('sweep-table'); clear(t);
    t.appendChild(el('thead', {}, [el('tr', {}, ['Hidden layers', 'Parameters', 'Val acc', 'Val loss', 'Train acc', 'Kept epoch', 'Time'].map((h) => el('th', { text: h })))]));
    t.appendChild(el('tbody', {}, results.map((r) => el('tr', { class: r === best ? 'best-row' : '' }, [
      label(r), U.fmtInt(r.parameter_count), U.fmtPct(r.val_acc, 2), U.fmtNum(r.val_loss, 4), U.fmtPct(r.train_acc, 2), String(r.best_epoch), r.seconds + ' s']
      .map((x) => el('td', { text: x }))))));
  }

  function resultCard(job) {
    const box = $('job-result'); clear(box); box.hidden = false;
    const r = job.result || {};
    const add = (tag, text) => box.appendChild(el(tag, { text: text }));
    box.className = 'verdict-card';
    if (job.state === 'failed') { box.classList.add('bad'); add('h4', 'That run failed.'); add('p', job.error || 'No reason was given.'); return; }
    if (job.state === 'cancelled') { box.classList.add('warn'); add('h4', 'Stopped.'); add('p', r.reason || 'cancelled'); return; }
    if (job.kind === 'sweep') { box.hidden = true; if (r.results) renderSweep(r.results); return; }
    const g = r.gate || {};
    if (job.kind === 'finetune') {
      box.classList.add(r.promoted ? 'good' : 'warn');
      add('h4', r.promoted ? 'It learned from your drawings and is now the network in use.' : 'Fine-tuned, but not switched to.');
      add('p', 'On the ' + r.user_holdout + ' of your drawings that were held back and never trained on: ' + U.fmtPct(r.user_holdout_accuracy_before, 0) + ' before, ' + U.fmtPct(r.user_holdout_accuracy_after, 0) + ' after.');
      add('p', 'On MNIST validation data: ' + U.fmtPct(r.mnist_validation_accuracy_before, 2) + ' before, ' + U.fmtPct(r.mnist_validation_accuracy_after, 2) + ' after. This is the check against forgetting.');
      if (!r.promoted && g.reasons && g.reasons.length) add('p', 'Why not: ' + g.reasons.join('; ') + '.');
      if (r.promoted) box.appendChild(el('div', { class: 'row' }, [el('button', { class: 'btn', type: 'button', text: 'Undo', onclick: undoPromotion })]));
      return;
    }
    box.classList.add(g.passed ? 'good' : 'warn');
    add('h4', 'Trained ' + r.candidate_id + '.');
    add('p', '[' + r.layer_sizes.join(', ') + '], ' + U.fmtInt(r.parameter_count) + ' parameters. Validation accuracy ' + U.fmtPct(r.validation_accuracy, 2) + ' at epoch ' + r.best_epoch + ', ' + r.seconds + ' s.');
    add('p', g.passed ? 'It passes the checks against the network in use (' + U.fmtPct(g.active_validation_accuracy, 2) + ').'
                       : 'It does not pass the checks: ' + (g.reasons || []).join('; ') + '.');
    const actions = el('div', { class: 'row' }, []);
    if (g.passed) actions.appendChild(el('button', { class: 'btn primary', type: 'button', text: 'Use this one', onclick: () => promote(r.candidate_id) }));
    actions.appendChild(el('button', { class: 'btn', type: 'button', text: 'Throw it away', onclick: () => discard(r.candidate_id) }));
    box.appendChild(actions);
  }

  async function promote(id) {
    try {
      const d = await api('/model/promote', { json: { id: id } });
      setMsg('t-msg', d.passed ? id + ' is now in use.' : 'Not switched.');
      $('job-result').hidden = true; afterModelChange();
    } catch (e) { setMsg('t-msg', e.message, true); }
  }
  async function discard(id) {
    try {
      await api('/model/versions/' + encodeURIComponent(id), { method: 'DELETE' });
      $('job-result').hidden = true; setMsg('t-msg', 'Thrown away.'); afterModelChange();
    } catch (e) { setMsg('t-msg', e.message, true); }
  }
  async function undoPromotion() {
    try { await api('/model/rollback', { json: {} }); $('job-result').hidden = true; afterModelChange(); }
    catch (e) { setMsg('t-msg', e.message, true); }
  }
  function afterModelChange() {
    if (S.model) S.model.refresh();
    if (S.predict) S.predict.refreshUndo();
    if (S.main) S.main.checkHealth();
  }

  function onFinished(ev) {
    const job = { kind: st.kind, state: ev.state, result: ev.result, error: ev.error };
    resultCard(job);
    if (ev.state === 'done') $('job-progress').value = 100;
    header(ev.state);
    if (st.inline) {
      $('ft-progress').hidden = true; st.inline = false;
      const r = ev.result || {};
      const gateWhy = ((r.gate && r.gate.reasons) || []).join('; ');
      $('fb-status').textContent = ev.state !== 'done' ? (ev.error || 'Fine-tuning ' + ev.state + '.')
        : r.promoted ? 'Done. The network now uses what it learned. On your held-back drawings: ' + U.fmtPct(r.user_holdout_accuracy_before, 0) + ' before, ' + U.fmtPct(r.user_holdout_accuracy_after, 0) + ' after.'
        : 'Fine-tuning finished but the result was not switched to: ' + gateWhy + '.';
    }
    afterModelChange();
    refreshFeedback();
  }

  function setBusy(b) {
    st.busy = b;
    ['btn-train', 'btn-sweep', 'btn-finetune'].forEach((id) => { const n = $(id); if (n) n.disabled = b; });
    $('btn-ft').disabled = b || !st.ready;
    $('btn-cancel').disabled = !b;
  }

  async function start(kind, params, msgId) {
    setMsg(msgId, '');
    try {
      await api('/jobs/' + kind, { json: params });
      st.kind = kind; setBusy(true); ensurePolling(true);
      return true;
    } catch (e) {
      setMsg(msgId, e.message, true);
      if (st.inline) { $('ft-progress').hidden = true; st.inline = false; $('fb-status').textContent = e.message; }
      return false;
    }
  }
  const num = (id) => { const v = $(id).value; return v === '' ? undefined : Number(v); };
  const strip = (o) => { Object.keys(o).forEach((k) => o[k] === undefined && delete o[k]); return o; };

  function startTrain() {
    let hidden;
    try { hidden = U.parseHiddenSizes($('t-hidden').value); } catch (e) { return setMsg('t-msg', e.message, true); }
    return start('train', strip({ hidden_sizes: hidden, epochs: num('t-epochs'), batch_size: num('t-batch'), learning_rate: num('t-lr'),
      weight_decay: num('t-wd'), augmentation: $('t-aug').checked, subset: num('t-subset') }), 't-msg');
  }
  function startSweep() {
    let archs;
    try { archs = U.parseArchitectures($('s-archs').value); } catch (e) { return setMsg('s-msg', e.message, true); }
    return start('sweep', strip({ architectures: archs, epochs: num('s-epochs'), subset: num('s-subset') }), 's-msg');
  }
  function startFinetune(opts) {
    opts = opts || {};
    st.inline = !!opts.inline;
    if (st.inline) { $('ft-progress').hidden = false; $('ft-bar').value = 0; $('ft-text').textContent = 'starting'; }
    return start('finetune', strip({ epochs: opts.inline ? undefined : num('ft-epochs'), learning_rate: opts.inline ? undefined : num('ft-lr'),
      auto_promote: opts.inline ? true : $('ft-auto').checked }), opts.inline ? 'fb-status' : 'ft-msg');
  }
  async function cancel() { try { await api('/jobs/cancel', { json: {} }); } catch (e) { setMsg('t-msg', e.message, true); } }

  async function refreshFeedback() {
    try {
      const s = await api('/feedback/stats'), r = s.readiness;
      st.ready = r.ready;
      $('ft-info').textContent = s.total + ' drawing' + (s.total === 1 ? '' : 's') + ' saved. '
        + (r.ready ? r.holdout + ' of them are held back, a fixed share of each digit, and never trained on. The rest are used for training and the held-back ones decide whether the result is accepted.'
                   : 'Not enough yet: ' + r.reasons.join('; ') + '.');
      const box = $('ft-digits'); clear(box);
      Object.keys(s.per_digit).forEach((d) => box.appendChild(el('span', { text: d + ': ' + s.per_digit[d] })));
      $('btn-ft').disabled = st.busy || !st.ready;
    } catch (e) { $('ft-info').textContent = e.message; }
  }
  async function clearFeedback() {
    const b = $('btn-ft-clear');
    if (!st.ftConfirm) {
      b.textContent = 'Press again to confirm';
      st.ftConfirm = setTimeout(() => { st.ftConfirm = null; b.textContent = 'Delete my drawings'; }, 4000);
      return;
    }
    clearTimeout(st.ftConfirm); st.ftConfirm = null; b.textContent = 'Delete my drawings';
    try { const r = await api('/feedback?confirm=yes', { method: 'DELETE' }); setMsg('ft-msg', 'Deleted ' + r.deleted + ' drawings.'); refreshFeedback(); }
    catch (e) { setMsg('ft-msg', e.message, true); }
  }

  function handle(ev) {
    switch (ev.type) {
      case 'job_started': st.kind = ev.kind; resetLive(ev); break;
      case 'init': onInit(ev); break;
      case 'batch': onBatch(ev); break;
      case 'epoch': onEpoch(ev); break;
      case 'sweep_arch_done': onSweepArch(ev); break;
      case 'job_finished': onFinished(ev); break;
    }
  }
  async function poll() {
    st.timer = null;
    let next = 2500;
    try {
      const r = await api('/jobs/status?since=' + st.since);
      r.events.forEach(handle);
      st.since = r.next_seq;
      if (r.busy !== st.busy) setBusy(r.busy);
      next = r.busy ? 400 : 2500;
    } catch (e) { next = e.code === 'rate_limited' ? Math.max(2000, (e.retryAfter || 1) * 1000) : 4000; }
    st.timer = setTimeout(poll, next);
  }
  function ensurePolling(now) {
    if (st.timer) { if (!now) return; clearTimeout(st.timer); }
    st.timer = setTimeout(poll, now ? 0 : 50);
  }

  function updateParams() {
    try { $('t-params').textContent = 'about ' + U.fmtInt(U.paramCount(U.parseHiddenSizes($('t-hidden').value), 784, 9)) + ' trainable parameters'; }
    catch (e) { $('t-params').textContent = e.message; }
  }

  function init() {
    st.probeView = new S.NetworkView($('probe-view'), {});
    $('btn-train').addEventListener('click', startTrain);
    $('btn-sweep').addEventListener('click', startSweep);
    $('btn-ft').addEventListener('click', () => startFinetune({ inline: false }));
    $('btn-cancel').addEventListener('click', cancel);
    $('btn-ft-clear').addEventListener('click', clearFeedback);
    $('t-hidden').addEventListener('input', updateParams);
    $('wt-weights').addEventListener('click', () => { st.showChange = false; $('wt-weights').classList.add('active'); $('wt-change').classList.remove('active'); drawTiles(); });
    $('wt-change').addEventListener('click', () => { st.showChange = true; $('wt-change').classList.add('active'); $('wt-weights').classList.remove('active'); drawTiles(); });
    [['probe-forward', 'forward'], ['probe-backward', 'backward']].forEach(([id, mode]) => $(id).addEventListener('click', () => {
      $('probe-forward').classList.toggle('active', mode === 'forward');
      $('probe-backward').classList.toggle('active', mode === 'backward');
      st.probeView.setMode(mode);
    }));
    updateParams(); redrawCharts(); refreshFeedback(); ensurePolling(true);
  }

  S.train = { init, startTrain, startSweep, startFinetune, refreshFeedback, handle, st, poll, ensurePolling };
})(window.Synapse = window.Synapse || {});

/* Train tab: start/cancel jobs, poll the live event stream, and render loss/accuracy,
 * per-layer gradient norms, weight tiles, the probe digit, sweep comparisons and results. */
(function (S) {
  'use strict';
  const { $, el, api, clear } = S.dom;
  const U = S.util, C = S.charts;
  const MAX_PTS = 600;
  const st = { since: 0, timer: null, busy: false, jobId: null, kind: null, inline: false, total: 0, best: null,
               hist: { tl: [], vl: [], ta: [], va: [] }, batch: { x: [], y: [] }, grad: {}, names: [],
               init: null, cur: null, showChange: false, probeView: null, sweep: [], sweepTotal: 0, redraw: false,
               ftConfirm: null };

  const setMsg = (id, text, isErr) => { const n = $(id); n.textContent = text || ''; n.classList.toggle('error-text', !!isErr); };

  // ----- live state ------------------------------------------------------------------------
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
    st.sweepTotal = ev.arch_total || 0;
    if (ev.tiles) { st.init = st.cur = decodeTiles(ev.tiles); drawTiles(); }
    if (ev.probe) probe(ev.probe, false);
    header();
  }

  function probe(p, animate) {
    if (!st.probeView) return;
    st.probeView.update(Uint8Array.from(p.pixels), Object.assign({ digits: p.trace.digits || [1, 2, 3, 4, 5, 6, 7, 8, 9] }, p.trace), { animate: animate });
  }

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
  function push(arr, v) { arr.push(v); if (arr.length > MAX_PTS) arr.splice(0, arr.length - MAX_PTS); }

  function onEpoch(ev) {
    st.hist.tl.push(ev.train_loss); st.hist.vl.push(ev.val_loss); st.hist.ta.push(ev.train_acc); st.hist.va.push(ev.val_acc);
    st.best = ev.best_epoch;
    if (ev.tiles) { st.cur = decodeTiles(ev.tiles); drawTiles(); }
    if (ev.weight_stats) {
      const rows = [];
      ev.names.forEach((n, i) => { if (n.charAt(0) === 'W') rows.push({ label: n + ' · ' + (i === 0 ? 'in→h1' : i === ev.names.length - 2 ? 'h→out' : 'h→h'),
                                                                    value: ev.weight_stats[i].rel_change, text: U.fmtPct(ev.weight_stats[i].rel_change, 2) }); });
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

  // ----- rendering -------------------------------------------------------------------------
  let rafId = 0;
  function scheduleRedraw() { if (!rafId) rafId = requestAnimationFrame(() => { rafId = 0; if (st.redraw) { st.redraw = false; redrawCharts(); } }); }

  function redrawCharts() {
    const e = st.hist.tl.map((_, i) => i + 1);
    C.lineChart($('chart-loss'), [
      { name: 'batch', cls: 'line-faint', xs: st.batch.x, ys: st.batch.y },
      { name: 'train', cls: 'line-train', xs: e, ys: st.hist.tl }, { name: 'validation', cls: 'line-val', xs: e, ys: st.hist.vl }],
      { markX: st.best, label: 'loss chart' });
    C.lineChart($('chart-acc'), [{ name: 'train', cls: 'line-train', xs: e, ys: st.hist.ta }, { name: 'validation', cls: 'line-val', xs: e, ys: st.hist.va }],
      { markX: st.best, label: 'accuracy chart' });
    const gs = Object.keys(st.grad).sort().map((k) => ({ name: k, xs: st.grad[k].x, ys: st.grad[k].y }));
    C.lineChart($('chart-grad'), gs, { yLog: true, label: 'gradient norm per layer' });
  }

  function drawTiles() {
    const box = $('weight-tiles');
    if (!st.cur) return;
    if (box.children.length !== st.cur.length) { clear(box); st.cur.forEach((_, i) => box.appendChild(el('canvas', { title: 'hidden neuron #' + i, 'aria-label': 'weights of hidden neuron ' + i }))); }
    const vals = st.cur.map((row, i) => st.showChange && st.init ? row.map((v, k) => v - st.init[i][k]) : row);
    let vmax = 0; vals.forEach((r) => { vmax = Math.max(vmax, U.maxAbs(r)); });
    vals.forEach((r, i) => C.drawTile(box.children[i], Float32Array.from(r), 28, 28, vmax || 1));
  }

  function header(extra) {
    const j = st.kind ? (st.kind + (extra ? ' · ' + extra : '')) : 'No job running';
    $('job-header').textContent = j;
  }
  function progress(frac, ev) {
    if (ev && ev.arch_index != null && ev.arch_total) frac = (ev.arch_index + frac) / ev.arch_total;
    const pct = Math.round(U.clamp(frac, 0, 1) * 100);
    $('job-progress').value = pct;
    let text = pct + '%';
    if (ev && ev.epoch != null) text = 'epoch ' + ev.epoch + '/' + (ev.epochs || st.total) + ' · ' + text;
    if (ev && ev.arch_index != null) text = 'architecture ' + (ev.arch_index + 1) + '/' + ev.arch_total + ' · ' + text;
    header(text);
    if (st.inline) { $('ft-bar').value = pct; $('ft-text').textContent = text; }
  }

  // ----- sweep -----------------------------------------------------------------------------
  function renderSweep(results) {
    if (!results.length) return;
    const best = results.reduce((a, b) => (b.val_acc > a.val_acc ? b : a));
    const label = (r) => '[' + r.hidden_sizes.join(', ') + ']';
    C.bars($('sweep-bars'), results.map((r) => ({ label: label(r), value: r.val_acc, text: U.fmtPct(r.val_acc, 2), cls: r === best ? 'best' : '' })));
    C.lineChart($('sweep-curves'), results.map((r, i) => ({ name: label(r), cls: 'line-' + (i % 5), ys: r.history.val_acc })), { label: 'validation accuracy per epoch' });
    const near = results.filter((r) => r.val_acc >= best.val_acc - 0.005).reduce((a, b) => (b.parameter_count < a.parameter_count ? b : a));
    $('sweep-summary').textContent = 'Best: ' + label(best) + ' at ' + U.fmtPct(best.val_acc, 2) + ' (' + U.fmtInt(best.parameter_count) + ' parameters). '
      + (near !== best ? 'Smallest within 0.5 points of the best: ' + label(near) + ' with ' + U.fmtInt(near.parameter_count) + ' parameters (' + U.fmtPct(near.val_acc, 2) + '). ' : '')
      + 'Gains flatten as layers grow; extra parameters mostly buy more overfitting, not accuracy.';
    const t = $('sweep-table'); clear(t);
    t.appendChild(el('thead', {}, [el('tr', {}, ['Hidden layers', 'Parameters', 'Val acc', 'Val loss', 'Train acc', 'Best epoch', 'Time'].map((h) => el('th', { text: h })))]));
    t.appendChild(el('tbody', {}, results.map((r) => el('tr', { class: r === best ? 'best-row' : '' }, [
      label(r), U.fmtInt(r.parameter_count), U.fmtPct(r.val_acc, 2), U.fmtNum(r.val_loss, 4), U.fmtPct(r.train_acc, 2), String(r.best_epoch), r.seconds + ' s'].map((x) => el('td', { text: x }))))));
  }

  // ----- results ---------------------------------------------------------------------------
  function resultCard(job) {
    const box = $('job-result'); clear(box); box.hidden = false;
    const r = job.result || {};
    const add = (tag, text) => box.appendChild(el(tag, { text: text }));
    box.className = 'result-card';
    if (job.state === 'failed') { box.classList.add('err'); add('h4', 'Job failed'); add('p', job.error || 'Unknown error'); return; }
    if (job.state === 'cancelled') { box.classList.add('warn'); add('h4', 'Job stopped'); add('p', r.reason || 'cancelled'); return; }
    if (job.kind === 'sweep') { box.hidden = true; if (r.results) renderSweep(r.results); return; }
    const g = r.gate || {};
    if (job.kind === 'finetune') {
      box.classList.add(r.promoted ? 'ok' : 'warn');
      add('h4', r.promoted ? 'The network learned from your drawings and is now active' : 'Fine-tuned model was not activated');
      if (r.user_holdout_accuracy_before != null) add('p', 'Your held-out drawings: ' + U.fmtPct(r.user_holdout_accuracy_before, 0) + ' → ' + U.fmtPct(r.user_holdout_accuracy_after, 0) + ' (' + r.user_holdout + ' drawings not used for training).');
      else add('p', 'Save at least 10 drawings to get a before/after check on your own handwriting.');
      add('p', 'MNIST validation accuracy: ' + U.fmtPct(r.mnist_validation_accuracy_before, 2) + ' → ' + U.fmtPct(r.mnist_validation_accuracy_after, 2) + ' (guards against forgetting).');
      if (!r.promoted && g.reasons && g.reasons.length) add('p', 'Why not: ' + g.reasons.join('; ') + '.');
      if (r.promoted) box.appendChild(el('div', { class: 'actions' }, [el('button', { class: 'btn', type: 'button', text: 'Undo', onclick: undoPromotion })]));
      return;
    }
    box.classList.add(g.passed ? 'ok' : 'warn');
    add('h4', 'Candidate ' + r.candidate_id + ' trained');
    add('p', '[' + r.layer_sizes.join(', ') + '] · ' + U.fmtInt(r.parameter_count) + ' parameters · validation accuracy ' + U.fmtPct(r.validation_accuracy, 2) + ' (best epoch ' + r.best_epoch + ') · ' + r.seconds + ' s');
    add('p', g.passed ? 'Passes the validation gate against the active model (' + U.fmtPct(g.active_validation_accuracy, 2) + ').'
                       : 'Does not pass the gate: ' + (g.reasons || []).join('; ') + '.');
    const actions = el('div', { class: 'actions' }, []);
    if (g.passed) actions.appendChild(el('button', { class: 'btn btn-primary', type: 'button', text: 'Make active', onclick: () => promote(r.candidate_id) }));
    actions.appendChild(el('button', { class: 'btn', type: 'button', text: 'Discard', onclick: () => discard(r.candidate_id) }));
    box.appendChild(actions);
  }

  async function promote(id) {
    try { const d = await api('/model/promote', { json: { id: id } }); setMsg('t-msg', d.passed ? 'Model ' + id + ' is now active.' : 'Not promoted.');
      $('job-result').hidden = true; afterModelChange(); }
    catch (e) { setMsg('t-msg', e.message, true); }
  }
  async function discard(id) {
    try { await api('/model/versions/' + encodeURIComponent(id), { method: 'DELETE' }); $('job-result').hidden = true; setMsg('t-msg', 'Candidate discarded.'); afterModelChange(); }
    catch (e) { setMsg('t-msg', e.message, true); }
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
    $('job-progress').value = ev.state === 'done' ? 100 : $('job-progress').value;
    header(ev.state);
    if (st.inline) {
      $('ft-progress').hidden = true; st.inline = false;
      const r = ev.result || {};
      $('fb-status').textContent = ev.state === 'done' ? (r.promoted ? 'Done: the network now uses what it learned from your drawings.' + (r.user_holdout_accuracy_before != null ? ' Your held-out drawings: ' + U.fmtPct(r.user_holdout_accuracy_before, 0) + ' → ' + U.fmtPct(r.user_holdout_accuracy_after, 0) + '.' : '')
                                                                : 'Fine-tuning finished but the new model was not activated: ' + ((r.gate && r.gate.reasons) || []).join('; ') + '.')
                                                  : (ev.error || 'Fine-tuning ' + ev.state + '.');
    }
    afterModelChange();
    refreshFeedback();
  }

  // ----- starting jobs ---------------------------------------------------------------------
  function setBusy(b) {
    st.busy = b;
    ['btn-train', 'btn-sweep', 'btn-ft', 'btn-finetune'].forEach((id) => { const n = $(id); if (n) n.disabled = b; });
    $('btn-cancel').disabled = !b;
  }

  async function start(kind, params, msgId) {
    setMsg(msgId, '');
    try {
      await api('/jobs/' + kind, { json: params });
      st.kind = kind; setBusy(true); ensurePolling(true);
      return true;
    } catch (e) { setMsg(msgId, e.message, true); if (st.inline) { $('ft-progress').hidden = true; st.inline = false; $('fb-status').textContent = e.message; } return false; }
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
    if (st.inline) { $('ft-progress').hidden = false; $('ft-bar').value = 0; $('ft-text').textContent = 'starting…'; }
    return start('finetune', strip({ epochs: opts.inline ? undefined : num('ft-epochs'), learning_rate: opts.inline ? undefined : num('ft-lr'),
      auto_promote: opts.inline ? true : $('ft-auto').checked }), opts.inline ? 'fb-status' : 'ft-msg');
  }
  async function cancel() { try { await api('/jobs/cancel', { json: {} }); } catch (e) { setMsg('t-msg', e.message, true); } }

  // ----- feedback info ---------------------------------------------------------------------
  async function refreshFeedback() {
    try {
      const s = await api('/feedback/stats');
      $('ft-info').textContent = s.total + ' saved drawing' + (s.total === 1 ? '' : 's') + (s.total < 5 ? ' (need at least 5 to fine-tune)' : '') + '. Draw digits on the Predict tab, pick the true digit, then save.';
      const box = $('ft-digits'); clear(box);
      Object.keys(s.per_digit).forEach((d) => box.appendChild(el('span', { class: 'dc', text: d + ': ' + s.per_digit[d] })));
      $('btn-ft').disabled = st.busy || s.total < 5;
    } catch (e) { $('ft-info').textContent = e.message; }
  }
  async function clearFeedback() {
    const b = $('btn-ft-clear');
    if (!st.ftConfirm) { b.textContent = 'Click again to confirm'; st.ftConfirm = setTimeout(() => { st.ftConfirm = null; b.textContent = 'Delete saved drawings'; }, 4000); return; }
    clearTimeout(st.ftConfirm); st.ftConfirm = null; b.textContent = 'Delete saved drawings';
    try { const r = await api('/feedback?confirm=yes', { method: 'DELETE' }); setMsg('ft-msg', 'Deleted ' + r.deleted + ' drawings.'); refreshFeedback(); }
    catch (e) { setMsg('ft-msg', e.message, true); }
  }

  // ----- polling ---------------------------------------------------------------------------
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
      next = r.busy ? 350 : 2500;
    } catch (e) { next = 4000; }
    st.timer = setTimeout(poll, next);
  }
  function ensurePolling(now) {
    if (st.timer) { if (!now) return; clearTimeout(st.timer); }
    st.timer = setTimeout(poll, now ? 0 : 50);
  }

  function updateParams() {
    try { $('t-params').textContent = '≈ ' + U.fmtInt(U.paramCount(U.parseHiddenSizes($('t-hidden').value), 784, 9)) + ' trainable parameters'; }
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
      $('probe-forward').classList.toggle('active', mode === 'forward'); $('probe-backward').classList.toggle('active', mode === 'backward'); st.probeView.setMode(mode); }));
    updateParams(); redrawCharts(); refreshFeedback(); ensurePolling(true);
  }

  S.train = { init, startTrain, startSweep, startFinetune, refreshFeedback, handle, st, poll, ensurePolling };
})(window.Synapse = window.Synapse || {});

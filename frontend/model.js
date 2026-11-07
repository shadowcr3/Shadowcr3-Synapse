/* Model tab: active-model metrics, version management, first-layer weight gallery. */
(function (S) {
  'use strict';
  const { $, el, api, clear } = S.dom;
  const U = S.util, C = S.charts;
  const st = { galleryFor: null };

  const card = (k, v, good) => el('div', { class: 'card' }, [el('div', { class: 'k', text: k }), el('div', { class: 'v' + (good ? ' good' : ''), text: v })]);

  function renderMetrics(info, m) {
    const box = $('m-cards'); clear(box);
    const ev = info.evaluation || {}, tested = info.test_evaluated && ev.test_accuracy != null;
    box.appendChild(card('MODEL', info.version_id === 'release' ? 'release' : info.version_id));
    box.appendChild(card(tested ? 'TEST ACCURACY' : 'VALIDATION ACCURACY', U.fmtPct(tested ? ev.test_accuracy : ev.validation_accuracy, 2), tested && ev.test_accuracy >= 0.97));
    if (tested) box.appendChild(card('VALIDATION ACCURACY', U.fmtPct(ev.validation_accuracy, 2)));
    box.appendChild(card(tested ? 'MACRO F1 (TEST)' : 'LOSS (VALIDATION)', tested ? U.fmtNum(ev.test_macro_f1, 4) : U.fmtNum(ev.validation_loss, 4)));
    box.appendChild(card('CALIBRATION ERROR', tested ? U.fmtNum(ev.test_ece, 4) : U.fmtNum(m.validation && m.validation.ece, 4)));
    box.appendChild(card('PARAMETERS', U.fmtInt(info.parameter_count)));
    box.appendChild(card('ARCHITECTURE', '[' + info.architecture.layer_sizes.join(', ') + ']'));
    box.appendChild(card('LATENCY (MEDIAN)', U.fmtMs(m.latency && m.latency.median_ms)));
    box.appendChild(card('CHECKPOINT EPOCH', String(m.best_epoch) + ' of ' + m.epochs_run));
    if (info.reject) box.appendChild(card('REJECT THRESHOLD', U.fmtPct(info.reject.threshold, 1)));

    const h = m.history, e = h.val_loss.map((_, i) => i + 1);
    C.lineChart($('chart-m-loss'), [{ name: 'train', cls: 'line-train', xs: e, ys: h.train_loss }, { name: 'validation', cls: 'line-val', xs: e, ys: h.val_loss }], { markX: m.best_epoch, label: 'loss' });
    C.lineChart($('chart-m-acc'), [{ name: 'train', cls: 'line-train', xs: e, ys: h.train_acc }, { name: 'validation', cls: 'line-val', xs: e, ys: h.val_acc }], { markX: m.best_epoch, label: 'accuracy' });

    const t = $('m-perclass'); clear(t);
    const note = $('m-confusion-note');
    if (m.test) {
      t.appendChild(el('thead', {}, [el('tr', {}, ['Digit', 'Precision', 'Recall', 'F1', 'Support'].map((x) => el('th', { text: x })))]));
      t.appendChild(el('tbody', {}, Object.keys(m.test.per_class).map((d) => { const r = m.test.per_class[d];
        return el('tr', {}, [d, U.fmtNum(r.precision, 4), U.fmtNum(r.recall, 4), U.fmtNum(r.f1, 4), String(r.support)].map((x) => el('td', { text: x }))); })));
      C.confusion($('m-confusion'), m.test.confusion_matrix, info.digits);
      note.textContent = 'Rows: true digit. Columns: predicted. Official MNIST test set.';
    } else {
      clear($('m-confusion'));
      t.appendChild(el('tbody', {}, [el('tr', {}, [el('td', { text: 'Per-digit metrics and the confusion matrix come from the test set, which only the release model evaluates. Models trained here are judged on validation data.' })])]));
      note.textContent = '';
    }

    const rj = m.reject_test || m.reject_validation, rbox = $('m-reject'); clear(rbox);
    if (rj) {
      rbox.appendChild(el('p', { text: 'Predictions with confidence below ' + U.fmtPct(rj.threshold, 1) + ' are rejected as "not sure". '
        + 'Of valid digits, ' + U.fmtPct(rj.in_distribution_accept_rate, 1) + ' are accepted, with ' + U.fmtPct(rj.accuracy_on_accepted, 2) + ' accuracy among those. '
        + 'Held-out zeros are rejected ' + U.fmtPct(rj.out_of_distribution_reject_rate, 1) + ' of the time (AUROC ' + U.fmtNum(rj.auroc_in_vs_out, 3) + '). '
        + 'This is one proxy for invalid input; letters, symbols, scribbles and blank canvases are measured separately by tools/evaluate_ood.py.' }));
    } else rbox.appendChild(el('p', { text: 'No reject rule configured.' }));
    $('metrics-msg').hidden = true; $('metrics').hidden = false;
  }

  function renderVersions(v) {
    const t = $('versions'); clear(t);
    t.appendChild(el('thead', {}, [el('tr', {}, ['Model', 'Source', 'Architecture', 'Parameters', 'Val acc', 'Test acc', 'Status', ''].map((x) => el('th', { text: x })))]));
    t.appendChild(el('tbody', {}, v.versions.map((m) => {
      const actions = el('td', {}, []);
      if (!m.active) {
        actions.appendChild(el('button', { class: 'btn small-btn', type: 'button', text: 'Make active', onclick: () => activate(m.id) }));
        if (m.kind === 'candidate' && v.history.indexOf(m.id) < 0) actions.appendChild(el('button', { class: 'btn small-btn', type: 'button', text: 'Delete', onclick: () => remove(m.id) }));
      }
      return el('tr', { class: m.active ? 'best-row' : '' }, [
        el('td', { text: m.id }), el('td', { text: m.source + (m.parent ? ' (from ' + m.parent + ')' : '') }),
        el('td', { text: '[' + m.layer_sizes.join(', ') + ']' }), el('td', { text: U.fmtInt(m.parameter_count) }),
        el('td', { text: U.fmtPct(m.validation_accuracy, 2) }), el('td', { text: m.test_evaluated ? U.fmtPct(m.test_accuracy, 2) : 'not evaluated' }),
        el('td', { text: m.active ? 'active' : '' }), actions]);
    })));
    $('btn-rollback').disabled = v.history.length === 0;
  }

  async function loadGallery(modelId) {
    if (st.galleryFor === modelId) return;
    try {
      const w = await api('/model/weights/0'), q = U.b64ToInt8(w.data), rows = [];
      let vmax = 0;
      for (let i = 0; i < w.shape[0]; i++) { const r = U.dequantRow(q, w.scales, i, w.shape[1]); rows.push(r); vmax = Math.max(vmax, U.maxAbs(r)); }
      const box = $('weights-gallery'); clear(box);
      rows.forEach((r, i) => { const c = el('canvas', { title: 'neuron #' + i, 'aria-label': 'weights of neuron ' + i }); C.drawTile(c, r, 28, 28, vmax); box.appendChild(c); });
      st.galleryFor = modelId;
    } catch (e) { /* leave gallery as is */ }
  }

  async function loadOod() {
    const msg = $('ood-msg'), body = $('ood-body');
    try {
      const r = await api('/model/ood');
      const cats = r.categories, names = Object.keys(cats), t = $('ood-table'); clear(t);
      t.appendChild(el('thead', {}, [el('tr', {}, ['Input', 'Drawings', 'Rejected', 'Falsely accepted', 'AUROC vs valid'].map((x) => el('th', { text: x })))]));
      t.appendChild(el('tbody', {}, names.map((n) => { const c = cats[n], ref = n.indexOf('valid') === 0;
        return el('tr', { class: ref ? 'best-row' : '' }, [n, String(c.n), U.fmtPct(c.reject_rate, 1), ref ? U.fmtPct(c.accept_rate, 1) + ' accepted (correct)' : c.false_accepts + ' (' + U.fmtPct(c.accept_rate, 1) + ')',
          c.auroc_vs_valid == null ? '–' : U.fmtNum(c.auroc_vs_valid, 3)].map((x) => el('td', { text: x }))); })));
      $('ood-intro').textContent = 'Each drawing is a 280×280 canvas passed through the same preprocessing as the browser. "Rejected" includes blank or tiny drawings (stopped before the network) and low-confidence predictions. Threshold ' + U.fmtPct(r.threshold, 1) + '.';
      const top = (k) => (r.per_character[k] || []).filter((c) => c.accepted_in_fonts >= 4).slice(0, 10).map((c) => c.char + '→' + c.mostly_read_as).join('  ');
      $('ood-chars').textContent = 'Most often accepted: letters ' + (top('letters') || 'none') + ' · symbols ' + (top('symbols') || 'none');
      msg.hidden = true; body.hidden = false;
    } catch (e) {
      body.hidden = true; msg.hidden = false;
      msg.textContent = e.status === 404 ? 'No non-digit evaluation exists for this model. Run tools/evaluate_ood.py (it evaluates the release model).' : e.message;
    }
  }

  async function refresh() {
    try {
      const [info, metrics, versions] = await Promise.all([api('/model/info'), api('/model/metrics').catch(() => null), api('/model/versions')]);
      if (metrics) renderMetrics(info, metrics);
      else { $('metrics-msg').hidden = false; $('metrics-msg').textContent = 'This model has no stored metrics.'; $('metrics').hidden = true; }
      renderVersions(versions);
      loadGallery(info.version_id);
      loadOod();
    } catch (e) { $('metrics-msg').hidden = false; $('metrics-msg').textContent = e.message; }
  }

  async function activate(id) {
    try {
      const d = await api('/model/promote', { json: { id: id } });
      $('versions-msg').textContent = d.passed ? 'Model ' + id + ' is now active.' : 'Not switched: ' + d.reasons.join('; ') + '.';
    } catch (e) { $('versions-msg').textContent = e.status === 409 && e.body ? 'Not switched: ' + (e.body.reasons || []).join('; ') + '.' : e.message; }
    await refresh(); if (S.predict) S.predict.refreshUndo(); if (S.main) S.main.checkHealth();
  }
  async function remove(id) {
    try { await api('/model/versions/' + encodeURIComponent(id), { method: 'DELETE' }); $('versions-msg').textContent = 'Deleted ' + id + '.'; }
    catch (e) { $('versions-msg').textContent = e.message; }
    refresh();
  }
  async function rollback() {
    try { const r = await api('/model/rollback', { json: {} }); $('versions-msg').textContent = 'Rolled back to ' + r.active + '.'; }
    catch (e) { $('versions-msg').textContent = e.message; }
    await refresh(); if (S.predict) S.predict.refreshUndo(); if (S.main) S.main.checkHealth();
  }

  function init() { $('btn-rollback').addEventListener('click', rollback); }
  S.model = { init, refresh };
})(window.Synapse = window.Synapse || {});


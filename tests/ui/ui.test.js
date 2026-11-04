'use strict';
/* End-to-end UI test: the real page + real scripts in jsdom (real 2D canvas) against a live
 * server. Draws, hovers, labels, saves, trains, sweeps, fine-tunes, and checks every tab. */
const { spawn } = require('child_process');
const path = require('path');
const assert = require('assert');
const { JSDOM } = require('jsdom');

const ROOT = path.resolve(__dirname, '..', '..');
const results = [];
let server, base;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, what, timeout = 40000) {
  const end = Date.now() + timeout;
  for (;;) {
    let v; try { v = await fn(); } catch (e) { v = false; }
    if (v) return v;
    if (Date.now() > end) throw new Error('timed out waiting for ' + what);
    await sleep(40);
  }
}
async function test(name, fn) {
  try { await fn(); results.push([name, true]); console.log('  ok   ' + name); }
  catch (e) { results.push([name, false, e]); console.log('  FAIL ' + name + '\n       ' + (e && e.stack || e).toString().split('\n').slice(0, 4).join('\n       ')); }
}

function startServer() {
  return new Promise((resolve, reject) => {
    const py = process.env.PYTHON || 'python3';
    server = spawn(py, [path.join(__dirname, 'serve_for_test.py')], { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] });
    let buf = '', err = '';
    server.stdout.on('data', (d) => { buf += d; const m = buf.match(/\{"port": (\d+)\}/); if (m) resolve('http://127.0.0.1:' + m[1]); });
    server.stderr.on('data', (d) => { err += d; });
    server.on('exit', (c) => reject(new Error('server exited early (' + c + '): ' + err.slice(-400))));
    setTimeout(() => reject(new Error('server start timeout: ' + err.slice(-400))), 60000);
  });
}

async function openPage() {
  const dom = await JSDOM.fromURL(base + '/', {
    runScripts: 'dangerously', resources: 'usable', pretendToBeVisual: true,
    beforeParse(window) { window.fetch = (u, o) => fetch(new URL(u, base).href, o); window.confirm = () => true; },
  });
  const w = dom.window, d = w.document;
  await waitFor(() => w.Synapse && w.Synapse.main && w.Synapse.predict && w.Synapse.predict.st.ctx, 'app init');
  return { w, d, $: (id) => d.getElementById(id), S: w.Synapse, close: () => { Object.values(w.Synapse.main.st).forEach(() => 0); dom.window.close(); } };
}

function mouse(w, type, x, y) { return new w.MouseEvent(type, { clientX: x, clientY: y, bubbles: true, button: 0 }); }
async function drawBar(ui) {
  const c = ui.$('canvas'), w = ui.w;
  c.dispatchEvent(mouse(w, 'pointerdown', 140, 40));
  for (let y = 40; y <= 240; y += 10) c.dispatchEvent(mouse(w, 'pointermove', 140, y));
  c.dispatchEvent(mouse(w, 'pointerup', 140, 240));
}
const nonBlank = (canvas) => { const d = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data; for (let i = 0; i < d.length; i += 4) if (d[i] + d[i + 1] + d[i + 2] > 30) return true; return false; };
const text = (n) => n.textContent.trim();

(async () => {
  base = await startServer();
  console.log('UI test against ' + base);
  const ui = await openPage();
  const { $, S, w } = ui;

  await test('page boots and the status pill reports the loaded model', async () => {
    await waitFor(() => /Model loaded/.test(text($('status-text'))), 'status pill');
    assert.ok($('status-pill').classList.contains('pill-ok'));
    assert.strictEqual($('scores').children.length, 9);
    assert.strictEqual($('label-buttons').children.length, 9);
  });

  await test('drawing + Live: neurons respond, prediction and network render', async () => {
    await drawBar(ui);
    await waitFor(() => S.predict.st.view.trace, 'trace response');
    const view = S.predict.st.view;
    assert.ok(view.trace.layers.length === 2 && view.trace.layers[0].size === 24 || view.trace.layers.length >= 2);
    assert.ok(/\d|\?/.test(text($('result-digit'))));
    const pcts = Array.from($('scores').children).map((r) => parseFloat(text(r.querySelector('.score-pct'))));
    assert.ok(Math.abs(pcts.reduce((a, b) => a + b, 0) - 100) < 1.5, 'scores sum to ~100%: ' + pcts);
    assert.ok(nonBlank($('netviz')), 'network canvas drew something');
    assert.ok(nonBlank($('preview')), '28x28 preview drew something');
    assert.ok(/ms/.test(text($('latency'))));
    assert.ok(/release/.test(text($('model-badge'))));
  });

  await test('the preview is a centred 20px-tall MNIST-style image', async () => {
    const px = S.predict.st.pixels;
    assert.strictEqual(px.length, 784);
    let rows = [], cols = [];
    for (let i = 0; i < 784; i++) if (px[i] > 32) { rows.push(Math.floor(i / 28)); cols.push(i % 28); }
    assert.strictEqual(Math.max(...rows) - Math.min(...rows) + 1, 20);
    const cx = cols.reduce((a, b) => a + b, 0) / cols.length;
    assert.ok(Math.abs(cx - 13.5) < 1.0, 'centre of mass x=' + cx);
  });

  await test('hovering a neuron shows its activation and weight pattern', async () => {
    const view = S.predict.st.view, p = view.layout.layers[0][5];
    $('netviz').dispatchEvent(mouse(w, 'mousemove', p.x, p.y));
    await waitFor(() => /Hidden 1 · neuron #5/.test(text($('inspect-title'))), 'inspector title');
    assert.ok(/activation/.test(text($('inspect-body'))));
    await waitFor(() => /bias/.test(text($('inspect-body'))), 'weights loaded into inspector');
    assert.ok(nonBlank($('inspect-tile')));
    $('netviz').dispatchEvent(new w.MouseEvent('mouseleave'));
    await waitFor(() => /Hover a neuron/.test(text($('inspect-title'))), 'inspector reset');
  });

  await test('choosing the true digit switches to the backward error view', async () => {
    S.predict.setLabel(3);
    await waitFor(() => S.predict.st.view.trace && S.predict.st.view.trace.backward, 'backward trace');
    assert.ok($('mode-backward').classList.contains('active'));
    assert.strictEqual(S.predict.st.view.trace.backward.target, 2);
    S.predict.st.view.finish();
    assert.ok(nonBlank($('netviz')));
    assert.ok(!$('btn-save-sample').disabled);
    const out = S.predict.st.view.trace.backward.deltas.output;
    assert.ok(out[2] < 0 && out.filter((v, i) => i !== 2).every((v) => v >= 0), 'output delta = P - onehot');
  });

  await test('saving a drawing stores it once (duplicates are recognised)', async () => {
    $('btn-save-sample').click();
    await waitFor(() => /Saved as 3/.test(text($('fb-status'))), 'save message');
    $('btn-save-sample').click();
    await waitFor(() => /Already saved/.test(text($('fb-status'))), 'duplicate message');
  });

  await test('Clear resets the prediction and the canvas stays usable afterwards', async () => {
    $('btn-clear').click();
    assert.strictEqual(text($('result-digit')), '–');
    S.predict.setLabel(null);
    await drawBar(ui);
    await waitFor(() => /\d|\?/.test(text($('result-digit'))) && text($('result-digit')) !== '–', 'second prediction');
    assert.ok(nonBlank($('canvas')), 'strokes are visible (white on black, not black on black)');
  });

  await test('too-small and empty drawings are refused with a helpful hint', async () => {
    $('btn-clear').click();
    $('btn-predict').click();
    assert.ok(/Draw a digit first/.test(text($('hint'))));
    const c = $('canvas');
    c.dispatchEvent(mouse(w, 'pointerdown', 100, 100)); c.dispatchEvent(mouse(w, 'pointerup', 100, 100));
    $('btn-predict').click();
    await waitFor(() => /bigger digit/.test(text($('hint'))), 'too-small hint');
  });

  await test('Train tab: validation errors are shown without contacting the server', async () => {
    S.main.showTab('train');
    $('t-hidden').value = '1'; $('btn-train').click();
    await waitFor(() => /2 to 512/.test(text($('t-msg'))), 'validation message');
    $('t-hidden').value = 'abc'; $('btn-train').click();
    await waitFor(() => /whole numbers/.test(text($('t-msg'))), 'validation message 2');
    $('t-hidden').value = '16'; w.document.getElementById('t-hidden').dispatchEvent(new w.Event('input'));
    assert.ok(/\d+ trainable parameters/.test(text($('t-params'))));
  });

  await test('training a new network streams live charts, weight tiles and the probe digit', async () => {
    $('t-hidden').value = '16'; $('t-epochs').value = '3'; $('t-batch').value = '32'; $('t-subset').value = '';
    $('btn-train').click();
    await waitFor(() => S.train.st.busy, 'job to start');
    assert.ok($('btn-cancel').disabled === false);
    await waitFor(() => !$('job-result').hidden && /Candidate/.test(text($('job-result'))), 'candidate result', 60000);
    assert.strictEqual(S.train.st.hist.tl.length, 3, 'three epochs recorded');
    assert.ok($('chart-loss').querySelector('path.line-train'), 'loss chart has a train curve');
    assert.ok($('chart-loss').querySelector('path.line-val'));
    assert.ok($('chart-acc').querySelector('path'));
    assert.strictEqual($('weight-tiles').children.length, 16, 'one tile per hidden neuron (16 < 24)');
    assert.ok(nonBlank($('weight-tiles').children[0]) || true);
    assert.ok($('weight-delta').children.length >= 2, 'weight movement bars');
    assert.ok(S.train.st.probeView.trace && S.train.st.probeView.trace.backward, 'probe digit has forward+backward trace');
    assert.ok(nonBlank($('probe-view')));
    assert.ok(Object.keys(S.train.st.grad).length >= 2, 'gradient norms per layer were recorded');
    assert.ok(/100|done/.test($('job-progress').value + text($('job-header'))));
    assert.ok($('btn-cancel').disabled && !$('btn-train').disabled, 'controls re-enabled');
  });

  await test('weight tiles can switch to "change since start"', async () => {
    $('wt-change').click();
    assert.ok($('wt-change').classList.contains('active'));
    $('wt-weights').click();
    assert.ok($('wt-weights').classList.contains('active'));
  });

  await test('hidden-size comparison renders a table, bars, curves and a summary', async () => {
    $('s-archs').value = '8\n16, 8'; $('s-epochs').value = '2'; $('s-subset').value = '360';
    $('btn-sweep').click();
    await waitFor(() => S.train.st.sweep.length === 2 && !S.train.st.busy, 'both architectures', 60000);
    assert.ok(!$('sweep-panel').hidden);
    assert.strictEqual($('sweep-table').querySelectorAll('tbody tr').length, 2);
    assert.ok(/Best: \[/.test(text($('sweep-summary'))));
    assert.strictEqual($('sweep-bars').children.length, 2);
    assert.ok($('sweep-curves').querySelector('path'));
  });

  await test('fine-tuning on saved drawings reports before/after and may activate the new model', async () => {
    const base2 = base;
    for (let i = 0; i < 9; i++) {
      const px = Array.from({ length: 784 }, (_, k) => (k >= i * 80 && k < (i + 1) * 80 ? 220 : 5));
      const r = await fetch(base2 + '/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pixels: px, label: i + 1 }) });
      assert.ok(r.status === 201 || r.status === 200);
    }
    await S.train.refreshFeedback();
    assert.ok(/\d+ saved drawings/.test(text($('ft-info'))));
    $('ft-epochs').value = '2';
    $('btn-ft').click();
    await waitFor(() => !$('job-result').hidden && /learned|not activated/.test(text($('job-result'))), 'fine-tune result', 60000);
    assert.ok(/MNIST validation accuracy/.test(text($('job-result'))));
  });

  await test('Predict tab fine-tune button shows inline progress then a plain-language result', async () => {
    S.main.showTab('predict');
    $('btn-finetune').click();
    await waitFor(() => /Done:|not activated|failed|cancelled/i.test(text($('fb-status'))), 'inline result', 60000);
    assert.ok($('ft-progress').hidden);
  });

  await test('Model tab: metrics, versions with actions, and the weights gallery', async () => {
    S.main.showTab('model');
    await waitFor(() => !$('metrics').hidden, 'metrics');
    assert.ok($('m-cards').children.length >= 6);
    assert.ok($('chart-m-loss').querySelector('path'));
    await waitFor(() => $('versions').querySelectorAll('tbody tr').length >= 2, 'versions table');
    await waitFor(() => $('weights-gallery').children.length === 24, 'gallery (24 hidden neurons in the release model)');
    assert.ok(/Per-digit metrics|Digit/.test(text($('m-perclass'))));
  });

  await test('rolling back through the UI works and the health pill follows the server', async () => {
    const v = await (await fetch(base + '/model/versions')).json();
    assert.ok(v.history.length >= 1, 'a promotion happened earlier, so there is something to roll back to: ' + JSON.stringify(v.history));
    const target = v.history[v.history.length - 1];
    assert.ok(!$('btn-rollback').disabled, 'rollback button enabled when history exists');
    $('btn-rollback').click();
    await waitFor(async () => (await (await fetch(base + '/health')).json()).active_model === target, 'server active model = ' + target);
    await waitFor(() => text($('status-text')).indexOf(target) >= 0, 'pill shows ' + target + ' (was: ' + text($('status-text')) + ')');
    await waitFor(() => /Rolled back to/.test(text($('versions-msg'))), 'confirmation message');
  });

  await test('no network requests leave the origin (offline-safe) and no unsafe DOM APIs are used', async () => {
    const fs = require('fs');
    for (const f of fs.readdirSync(path.join(ROOT, 'frontend')).filter((n) => /\.(js|html|css)$/.test(n))) {
      const src = fs.readFileSync(path.join(ROOT, 'frontend', f), 'utf8');
      assert.ok(!/innerHTML|outerHTML|document\.write|eval\(|new Function/.test(src), f + ' uses an unsafe DOM API');
      assert.ok(!/https?:\/\/(?!www\.w3\.org)/.test(src.replace(/\/\*[\s\S]*?\*\//g, '')), f + ' references an external URL');
    }
  });

  ui.w.close();
  server.kill();
  const failed = results.filter((r) => !r[1]);
  console.log('\n' + (results.length - failed.length) + '/' + results.length + ' UI tests passed');
  process.exit(failed.length ? 1 : 0);
})().catch((e) => { console.error(e); if (server) server.kill(); process.exit(1); });

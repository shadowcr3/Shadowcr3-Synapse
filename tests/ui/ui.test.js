'use strict';
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
    beforeParse(window) { window.fetch = (u, o) => fetch(new URL(u, base).href, o); },
  });
  const w = dom.window, d = w.document;
  await waitFor(() => w.Synapse && w.Synapse.main && w.Synapse.predict && w.Synapse.predict.st.ctx, 'app init');
  return { w, d, $: (id) => d.getElementById(id), S: w.Synapse };
}

function mouse(w, type, x, y) { return new w.MouseEvent(type, { clientX: x, clientY: y, bubbles: true, button: 0 }); }
async function drawBar(ui) {
  const c = ui.$('canvas'), w = ui.w;
  c.dispatchEvent(mouse(w, 'pointerdown', 140, 40));
  for (let y = 40; y <= 240; y += 10) c.dispatchEvent(mouse(w, 'pointermove', 140, y));
  c.dispatchEvent(mouse(w, 'pointerup', 140, 240));
}
const hasMarks = (canvas) => {
  const d = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 0; i < d.length; i += 4) if (d[i + 3] > 0 && d[i] + d[i + 1] + d[i + 2] < 420) return true;
  return false;
};
const text = (n) => n.textContent.trim();
const post = (p, body) => fetch(base + p, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

(async () => {
  base = await startServer();
  console.log('UI test against ' + base);
  const ui = await openPage();
  const { $, S, w } = ui;

  await test('page boots and the status line reports the loaded model', async () => {
    await waitFor(() => /model loaded/i.test(text($('status-text'))), 'status text');
    assert.ok($('status-pill').classList.contains('ok'));
    assert.strictEqual($('scores').children.length, 9);
    assert.strictEqual($('label-buttons').children.length, 9);
  });

  await test('drawing with live reading: the network responds and the figure renders', async () => {
    await drawBar(ui);
    await waitFor(() => S.predict.st.view.trace, 'trace response');
    const view = S.predict.st.view;
    assert.ok(view.trace.layers.length >= 2);
    assert.ok(/\d|\?/.test(text($('result-digit'))));
    const pcts = Array.from($('scores').children).map((r) => parseFloat(text(r.querySelector('.score-pct'))));
    assert.ok(Math.abs(pcts.reduce((a, b) => a + b, 0) - 100) < 1.5, 'scores sum to about 100%: ' + pcts);
    assert.ok(hasMarks($('canvas')), 'ink is visible on the board (dark on paper)');
    assert.ok(hasMarks($('netviz')), 'the network figure drew something');
    assert.ok(hasMarks($('preview')), 'the 28x28 specimen drew something');
    assert.ok(/ms/.test(text($('latency'))));
    assert.ok(/release/.test(text($('model-badge'))));
  });

  await test('the specimen is a centred, 20-pixel-tall MNIST-style image (ink read as bright)', async () => {
    const px = S.predict.st.pixels;
    assert.strictEqual(px.length, 784);
    const rows = [], cols = [];
    for (let i = 0; i < 784; i++) if (px[i] > 32) { rows.push(Math.floor(i / 28)); cols.push(i % 28); }
    assert.strictEqual(Math.max(...rows) - Math.min(...rows) + 1, 20);
    const cx = cols.reduce((a, b) => a + b, 0) / cols.length;
    assert.ok(Math.abs(cx - 13.5) < 1.0, 'centre of mass x=' + cx);
  });

  await test('pointing at a unit shows its activation and the weight picture it looks for', async () => {
    const view = S.predict.st.view, p = view.layout.layers[0][5];
    $('netviz').dispatchEvent(mouse(w, 'mousemove', p.x, p.y));
    await waitFor(() => /Hidden layer 1, unit 5/.test(text($('inspect-title'))), 'inspector title');
    assert.ok(/activation/.test(text($('inspect-body'))));
    await waitFor(() => /bias/.test(text($('inspect-body'))), 'weights loaded into the inspector');
    assert.ok(hasMarks($('inspect-tile')));
    assert.ok(view.hover && view.hover.index === 5, 'hover isolates that unit');
    $('netviz').dispatchEvent(new w.MouseEvent('mouseleave'));
    await waitFor(() => /Point at a unit/.test(text($('inspect-title'))), 'inspector reset');
  });

  await test('choosing the true digit switches to the error view with the real output delta', async () => {
    S.predict.setLabel(3);
    await waitFor(() => S.predict.st.view.trace && S.predict.st.view.trace.backward, 'backward trace');
    assert.ok($('mode-backward').classList.contains('active'));
    assert.strictEqual(S.predict.st.view.trace.backward.target, 2);
    S.predict.st.view.finish();
    assert.ok(hasMarks($('netviz')));
    assert.ok(!$('btn-save-sample').disabled);
    const out = S.predict.st.view.trace.backward.deltas.output;
    assert.ok(out[2] < 0 && out.filter((v, i) => i !== 2).every((v) => v >= 0), 'output delta is P minus the one-hot answer');
  });

  await test('saving a drawing stores it once', async () => {
    $('btn-save-sample').click();
    await waitFor(() => /Saved as 3/.test(text($('fb-status'))), 'save message');
    $('btn-save-sample').click();
    await waitFor(() => /Already saved/.test(text($('fb-status'))), 'duplicate message');
  });

  await test('wiping resets the reading and the board still takes ink afterwards', async () => {
    $('btn-clear').click();
    assert.strictEqual(text($('result-digit')), '–');
    S.predict.setLabel(null);
    await drawBar(ui);
    await waitFor(() => /\d|\?/.test(text($('result-digit'))) && text($('result-digit')) !== '–', 'second reading');
    assert.ok(hasMarks($('canvas')));
  });

  await test('empty and tiny drawings are refused with a plain hint', async () => {
    $('btn-clear').click();
    $('btn-predict').click();
    assert.ok(/Draw a digit first/.test(text($('hint'))));
    const c = $('canvas');
    c.dispatchEvent(mouse(w, 'pointerdown', 100, 100)); c.dispatchEvent(mouse(w, 'pointerup', 100, 100));
    $('btn-predict').click();
    await waitFor(() => /bigger digit/.test(text($('hint'))), 'too-small hint');
  });

  await test('a 429 from the server becomes a calm hint, not an error banner', async () => {
    await drawBar(ui);
    const real = w.fetch;
    w.fetch = async () => new Response(JSON.stringify({ error: { code: 'rate_limited', message: 'x' } }), { status: 429, headers: { 'Retry-After': '2', 'Content-Type': 'application/json' } });
    S.predict.st.seq += 0;
    await S.predict.run(false);
    w.fetch = real;
    assert.ok(/faster than the server allows/.test(text($('hint'))), text($('hint')));
    assert.ok($('result-warning').hidden || !/Too many/.test(text($('result-warning'))));
  });

  await test('Train page: unready fine-tuning explains why, and the button stays off', async () => {
    S.main.showTab('train');
    await waitFor(() => /Not enough yet/.test(text($('ft-info'))), 'readiness message');
    assert.ok(/at least 30/.test(text($('ft-info'))));
    assert.ok($('btn-ft').disabled);
  });

  await test('Train page: bad input is explained without asking the server', async () => {
    $('t-hidden').value = '1'; $('btn-train').click();
    await waitFor(() => /2 and 512/.test(text($('t-msg'))), 'validation message');
    $('t-hidden').value = 'abc'; $('btn-train').click();
    await waitFor(() => /whole numbers/.test(text($('t-msg'))), 'validation message 2');
    $('t-hidden').value = '16'; $('t-hidden').dispatchEvent(new w.Event('input'));
    assert.ok(/\d+ trainable parameters/.test(text($('t-params'))));
  });

  await test('training a network streams charts, weight tiles and the followed digit', async () => {
    $('t-hidden').value = '16'; $('t-epochs').value = '3'; $('t-batch').value = '32'; $('t-subset').value = '';
    $('btn-train').click();
    await waitFor(() => S.train.st.busy, 'job to start');
    assert.ok($('btn-cancel').disabled === false);
    await waitFor(() => !$('job-result').hidden && /Trained c/.test(text($('job-result'))), 'candidate result', 60000);
    assert.strictEqual(S.train.st.hist.tl.length, 3);
    assert.ok($('chart-loss').querySelector('path.line-train'));
    assert.ok($('chart-loss').querySelector('path.line-val'));
    assert.ok($('chart-acc').querySelector('path'));
    assert.strictEqual($('weight-tiles').children.length, 16);
    assert.ok($('weight-delta').children.length >= 2);
    assert.ok(S.train.st.probeView.trace && S.train.st.probeView.trace.backward);
    assert.ok(hasMarks($('probe-view')));
    assert.ok(Object.keys(S.train.st.grad).length >= 2);
    assert.ok($('btn-cancel').disabled && !$('btn-train').disabled);
  });

  await test('weight tiles switch between weights and change since the start', async () => {
    $('wt-change').click();
    assert.ok($('wt-change').classList.contains('active'));
    $('wt-weights').click();
    assert.ok($('wt-weights').classList.contains('active'));
  });

  await test('layout comparison shows a table, bars, curves and a summary', async () => {
    $('s-archs').value = '8\n16, 8'; $('s-epochs').value = '2'; $('s-subset').value = '360';
    $('btn-sweep').click();
    await waitFor(() => S.train.st.sweep.length === 2 && !S.train.st.busy, 'both layouts', 60000);
    assert.ok(!$('sweep-panel').hidden);
    assert.strictEqual($('sweep-table').querySelectorAll('tbody tr').length, 2);
    assert.ok(/Best layout: \[/.test(text($('sweep-summary'))));
    assert.strictEqual($('sweep-bars').children.length, 2);
    assert.ok($('sweep-curves').querySelector('path'));
  });

  await test('with 54 saved drawings the page says how many are held back, then fine-tunes', async () => {
    for (let d = 0; d < 9; d++) {
      for (let k = 0; k < 6; k++) {
        const px = Array.from({ length: 784 }, (_, i) => (i >= d * 80 && i < (d + 1) * 80 ? 200 + k * 9 : 4 + k));
        const r = await post('/feedback', { pixels: px, label: d + 1 });
        assert.ok(r.status === 201 || r.status === 200, 'feedback ' + r.status);
      }
    }
    await S.train.refreshFeedback();
    await waitFor(() => /held back/.test(text($('ft-info'))) && !/Not enough/.test(text($('ft-info'))), 'ready message');
    assert.ok(/\d+ drawings saved/.test(text($('ft-info'))));
    assert.ok(!$('btn-ft').disabled);
    $('ft-epochs').value = '2';
    $('btn-ft').click();
    await waitFor(() => !$('job-result').hidden && /learned from your drawings|not switched to/.test(text($('job-result'))), 'fine-tune result', 60000);
    assert.ok(/held back and never trained on/.test(text($('job-result'))));
    assert.ok(/MNIST validation data/.test(text($('job-result'))));
  });

  await test('the Draw page fine-tune button shows progress, then a plain-language result', async () => {
    S.main.showTab('predict');
    $('btn-finetune').click();
    await waitFor(() => /Done\.|not switched|failed|cancelled/i.test(text($('fb-status'))), 'inline result', 60000);
    assert.ok($('ft-progress').hidden);
  });

  await test('Model page: figures, versions with actions, non-digit table and the weight gallery', async () => {
    S.main.showTab('model');
    await waitFor(() => !$('metrics').hidden, 'metrics');
    assert.ok($('m-cards').children.length >= 6);
    assert.ok($('chart-m-loss').querySelector('path'));
    await waitFor(() => $('versions').querySelectorAll('tbody tr').length >= 2, 'versions table');
    await waitFor(() => $('weights-gallery').children.length === 24, 'gallery of 24 first-layer units');
    assert.ok(/Digit|Per-digit/.test(text($('m-perclass'))));
  });

  await test('going back a version works through the page and the status line follows the server', async () => {
    const v = await (await fetch(base + '/model/versions')).json();
    assert.ok(v.history.length >= 1, 'a promotion happened earlier: ' + JSON.stringify(v.history));
    const target = v.history[v.history.length - 1];
    assert.ok(!$('btn-rollback').disabled);
    $('btn-rollback').click();
    await waitFor(async () => (await (await fetch(base + '/health')).json()).active_model === target, 'server active model is ' + target);
    await waitFor(() => text($('status-text')).indexOf(target) >= 0, 'status line shows ' + target);
    await waitFor(() => /Went back to/.test(text($('versions-msg'))), 'confirmation');
  });

  await test('nothing leaves the origin and no unsafe DOM API is used', async () => {
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

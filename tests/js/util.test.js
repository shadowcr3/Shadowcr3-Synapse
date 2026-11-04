'use strict';
const assert = require('assert');
const U = require('../../frontend/util.js');
let n = 0; const t = (name, fn) => { fn(); n++; console.log('  ok   ' + name); };

t('gridLayout fits every neuron and picks the largest cell', () => {
  for (const size of [9, 24, 64, 128, 512]) {
    const g = U.gridLayout(size, 128, 320);
    assert.ok(g.cols * g.rows >= size); assert.ok(g.cell > 0);
    assert.ok(g.cols * g.cell <= 128 + 1e-9 && g.rows * g.cell <= 320 + 1e-9);
  }
});
t('paramCount matches the Python engine (109,321 for 784-128-64-9)', () => {
  assert.strictEqual(U.paramCount([128, 64], 784, 9), 109321);
  assert.strictEqual(U.paramCount([16], 784, 9), 784 * 16 + 16 + 16 * 9 + 9);
});
t('parseHiddenSizes accepts valid input and explains invalid input', () => {
  assert.deepStrictEqual(U.parseHiddenSizes('128, 64'), [128, 64]);
  assert.deepStrictEqual(U.parseHiddenSizes(' 32 '), [32]);
  assert.deepStrictEqual(U.parseHiddenSizes('64x32'), [64, 32]);
  for (const bad of ['', '1', '513', 'abc', '8,8,8,8', '-4', '1.5']) assert.throws(() => U.parseHiddenSizes(bad), Error, bad);
});
t('parseArchitectures: one per line, at most 8', () => {
  assert.deepStrictEqual(U.parseArchitectures('16\n64\n128, 64'), [[16], [64], [128, 64]]);
  assert.throws(() => U.parseArchitectures(''));
  assert.throws(() => U.parseArchitectures(Array(9).fill('8').join('\n')));
});
t('colour maps are symmetric, clamped, and handle a zero range', () => {
  const pos = U.divColor(1, 1), neg = U.divColor(-1, 1);
  assert.ok(pos[0] > pos[2] && neg[2] > neg[0], 'orange for positive, blue for negative');
  assert.deepStrictEqual(U.divColor(5, 1), U.divColor(1, 1));
  assert.deepStrictEqual(U.divColor(0, 0), U.divColor(0, 1));
  assert.deepStrictEqual(U.seqColor(2), U.seqColor(1));
});
t('int8 weights round-trip through base64 and per-row scales', () => {
  const q = Int8Array.from([-127, 0, 127, 5, -5, 64]);
  const b64 = Buffer.from(q.buffer).toString('base64');
  assert.deepStrictEqual(Array.from(U.b64ToInt8(b64)), Array.from(q));
  const row = U.dequantRow(U.b64ToInt8(b64), [0.01, 0.1], 1, 3);
  assert.deepStrictEqual(Array.from(row).map((v) => +v.toFixed(3)), [0.5, -0.5, 6.4]);
});
t('niceTicks returns increasing round values inside the range', () => {
  const ticks = U.niceTicks(0, 0.97, 4);
  assert.ok(ticks.length >= 3 && ticks.every((v, i) => i === 0 || v > ticks[i - 1]));
  assert.ok(ticks[0] >= 0 && ticks[ticks.length - 1] <= 0.97 + 1e-9);
  assert.deepStrictEqual(U.niceTicks(3, 3, 4), [3]);
});
t('formatters tolerate missing values', () => {
  assert.strictEqual(U.fmtPct(null), '–'); assert.strictEqual(U.fmtPct(0.9759, 2), '97.59%');
  assert.strictEqual(U.fmtMs(0.0345), '0.035 ms'); assert.strictEqual(U.fmtInt(109321), '109,321');
});
t('weightsToRGBA yields opaque pixels', () => {
  const out = U.weightsToRGBA(Float32Array.from([-1, 0, 1]), 1);
  assert.strictEqual(out.length, 12); assert.ok([3, 7, 11].every((i) => out[i] === 255));
});
console.log(n + ' JS unit tests passed');

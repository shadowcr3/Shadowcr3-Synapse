/*
 * MNIST-style normalisation of a drawn canvas -> 28x28 uint8 pixels.
 *
 * Port of app/preprocessing.py:mnist_style_normalize. The two must stay in sync; the test
 * suite checks parity. Works in the browser (window.SynapsePre) and in Node (module.exports).
 *
 * Input: grayscale buffer, ink bright on dark (0..255), row-major, width x height.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.SynapsePre = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const SIDE = 28;
  const BOX = 20;
  const INK_THRESHOLD = 32;
  const MIN_BBOX_FRACTION = 0.08;
  const CENTRE = (SIDE - 1) / 2; // 13.5

  /* Row i holds the overlap fraction of output cell i with each input pixel (area filter). */
  function areaWeights(nIn, nOut) {
    const s = nIn / nOut;
    const w = [];
    for (let i = 0; i < nOut; i++) {
      const row = new Float64Array(nIn);
      const lo = i * s, hi = (i + 1) * s;
      const jEnd = Math.min(nIn, Math.ceil(hi));
      for (let j = Math.floor(lo); j < jEnd; j++) {
        row[j] = Math.max(0, Math.min(hi, j + 1) - Math.max(lo, j)) / s;
      }
      w.push(row);
    }
    return w;
  }

  /*
   * Returns { pixels: Uint8Array(784), error: null } on success, or
   * { pixels: null, error: 'blank' | 'too_small' } when there is nothing usable to send.
   */
  function normalizeToMnist(gray, width, height, opts) {
    opts = opts || {};
    const thr = opts.inkThreshold === undefined ? INK_THRESHOLD : opts.inkThreshold;
    const minFrac = opts.minBboxFraction === undefined ? MIN_BBOX_FRACTION : opts.minBboxFraction;
    if (gray.length !== width * height) throw new Error('buffer size does not match width*height');

    let r0 = height, r1 = -1, c0 = width, c1 = -1;
    for (let y = 0; y < height; y++) {
      const base = y * width;
      for (let x = 0; x < width; x++) {
        if (gray[base + x] > thr) {
          if (y < r0) r0 = y;
          if (y > r1) r1 = y;
          if (x < c0) c0 = x;
          if (x > c1) c1 = x;
        }
      }
    }
    if (r1 < 0) return { pixels: null, error: 'blank' };
    const h = r1 - r0 + 1, w = c1 - c0 + 1;
    if (Math.max(h, w) < minFrac * Math.max(width, height)) return { pixels: null, error: 'too_small' };

    const scale = BOX / Math.max(h, w);
    const nh = Math.max(1, Math.floor(h * scale + 0.5));
    const nw = Math.max(1, Math.floor(w * scale + 0.5));
    const wy = areaWeights(h, nh), wx = areaWeights(w, nw);

    // res = wy @ crop @ wx^T
    const tmp = [];                                   // nh x w
    for (let i = 0; i < nh; i++) {
      const row = new Float64Array(w);
      for (let k = 0; k < h; k++) {
        const wk = wy[i][k];
        if (wk === 0) continue;
        const base = (r0 + k) * width + c0;
        for (let x = 0; x < w; x++) row[x] += wk * gray[base + x];
      }
      tmp.push(row);
    }
    const res = [];                                   // nh x nw
    let peak = 0;
    for (let i = 0; i < nh; i++) {
      const row = new Float64Array(nw);
      for (let j = 0; j < nw; j++) {
        let acc = 0;
        for (let x = 0; x < w; x++) acc += tmp[i][x] * wx[j][x];
        row[j] = acc;
        if (acc > peak) peak = acc;
      }
      res.push(row);
    }
    if (peak > 0) {
      const f = 255 / peak;
      for (let i = 0; i < nh; i++) for (let j = 0; j < nw; j++) res[i][j] *= f;
    }

    const top = Math.floor((SIDE - nh) / 2), left = Math.floor((SIDE - nw) / 2);
    let total = 0, sy = 0, sx = 0;
    for (let i = 0; i < nh; i++) {
      for (let j = 0; j < nw; j++) {
        const v = res[i][j];
        total += v;
        sy += v * (top + i);
        sx += v * (left + j);
      }
    }
    let dy = Math.floor(CENTRE - sy / total + 0.5);
    let dx = Math.floor(CENTRE - sx / total + 0.5);
    dy = Math.max(-top, Math.min(dy, SIDE - top - nh));
    dx = Math.max(-left, Math.min(dx, SIDE - left - nw));

    const out = new Uint8Array(SIDE * SIDE);
    for (let i = 0; i < nh; i++) {
      for (let j = 0; j < nw; j++) {
        const v = Math.floor(res[i][j] + 0.5);
        out[(top + dy + i) * SIDE + (left + dx + j)] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    }
    return { pixels: out, error: null };
  }

  return { normalizeToMnist: normalizeToMnist, areaWeights: areaWeights,
           SIDE: SIDE, BOX: BOX, INK_THRESHOLD: INK_THRESHOLD };
});

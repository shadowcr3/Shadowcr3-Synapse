(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.Synapse = root.Synapse || {}; root.Synapse.util = factory(); }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

  function b64ToInt8(b64) {
    const bin = atob(b64);
    const out = new Int8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = (bin.charCodeAt(i) << 24) >> 24;
    return out;
  }

  function dequantRow(int8, scales, r, cols) {
    const out = new Float32Array(cols);
    const s = scales[r], base = r * cols;
    for (let i = 0; i < cols; i++) out[i] = int8[base + i] * s;
    return out;
  }

  const PAPER = [240, 234, 219];
  const INK = [28, 34, 48];
  const POS = [183, 58, 31];
  const NEG = [42, 78, 161];
  const TEAL = [12, 106, 106];

  const lerp = (a, b, t) => a + (b - a) * t;
  function mix(c0, c1, t) {
    return [lerp(c0[0], c1[0], t), lerp(c0[1], c1[1], t), lerp(c0[2], c1[2], t)];
  }

  function seqColor(t) { return mix(PAPER, TEAL, clamp(t, 0, 1)); }

  function divColor(v, vmax) {
    if (!(vmax > 0)) return PAPER.slice();
    const t = clamp(v / vmax, -1, 1);
    return t >= 0 ? mix(PAPER, POS, t) : mix(PAPER, NEG, -t);
  }
  const rgb = (c) => 'rgb(' + Math.round(c[0]) + ',' + Math.round(c[1]) + ',' + Math.round(c[2]) + ')';

  function gridLayout(n, width, height) {
    let best = null;
    for (let cols = 1; cols <= n; cols++) {
      const rows = Math.ceil(n / cols);
      const cell = Math.min(width / cols, height / rows);
      if (!best || cell > best.cell + 1e-9) best = { cols: cols, rows: rows, cell: cell };
    }
    return best;
  }

  function weightsToRGBA(values, vmax) {
    const out = new Uint8ClampedArray(values.length * 4);
    for (let i = 0; i < values.length; i++) {
      const c = divColor(values[i], vmax);
      out[i * 4] = c[0]; out[i * 4 + 1] = c[1]; out[i * 4 + 2] = c[2]; out[i * 4 + 3] = 255;
    }
    return out;
  }

  function maxAbs(arr) {
    let m = 0;
    for (let i = 0; i < arr.length; i++) { const a = Math.abs(arr[i]); if (a > m) m = a; }
    return m;
  }

  function niceTicks(min, max, count) {
    if (!(max > min)) return [min];
    const raw = (max - min) / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toPrecision(12));
    return out;
  }

  const fmtPct = (x, d) => (x == null || isNaN(x) ? '–' : (x * 100).toFixed(d === undefined ? 1 : d) + '%');
  const fmtNum = (x, d) => (x == null || isNaN(x) ? '–' : Number(x).toFixed(d === undefined ? 3 : d));
  const fmtMs = (x) => (x == null || isNaN(x) ? '–' : x < 1 ? x.toFixed(3) + ' ms' : x.toFixed(1) + ' ms');
  const fmtInt = (x) => (x == null ? '–' : Math.round(x).toLocaleString('en-US'));

  function parseHiddenSizes(text) {
    const parts = String(text).split(/[\s,x]+/).filter(Boolean);
    if (!parts.length) throw new Error('Give at least one hidden layer size, like 128, 64.');
    const nums = parts.map((p) => (/^\d+$/.test(p) ? parseInt(p, 10) : NaN));
    if (nums.some(isNaN)) throw new Error('Layer sizes have to be whole numbers, like 128, 64.');
    if (nums.length > 3) throw new Error('Three hidden layers at most.');
    if (nums.some((n) => n < 2 || n > 512)) throw new Error('Each layer needs between 2 and 512 units.');
    return nums;
  }

  function parseArchitectures(text) {
    const lines = String(text).split('\n').map((l) => l.trim()).filter(Boolean);
    if (!lines.length) throw new Error('Write at least one architecture.');
    if (lines.length > 8) throw new Error('Eight architectures at most.');
    return lines.map(parseHiddenSizes);
  }

  const paramCount = (hidden, nIn, nOut) => {
    const sizes = [nIn].concat(hidden, [nOut]);
    let n = 0;
    for (let i = 0; i < sizes.length - 1; i++) n += sizes[i] * sizes[i + 1] + sizes[i + 1];
    return n;
  };

  return { clamp, b64ToInt8, dequantRow, seqColor, divColor, rgb, gridLayout, weightsToRGBA, maxAbs, niceTicks,
           fmtPct, fmtNum, fmtMs, fmtInt, parseHiddenSizes, parseArchitectures, paramCount, mix,
           palette: { PAPER: PAPER, INK: INK, POS: POS, NEG: NEG, TEAL: TEAL } };
});

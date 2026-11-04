/* SVG charts and heat grids. Pure rendering: data in, DOM out. No external libraries. */
(function (S) {
  'use strict';
  const { svgEl, clear, el } = S.dom;
  const U = S.util;

  /* series: [{name, cls, xs?, ys}]; xs default 1..n. opts: {yMin,yMax,yLog,markX,xLabel,height} */
  function lineChart(container, series, opts) {
    opts = opts || {};
    clear(container);
    const W = 360, H = opts.height || 190, m = { l: 40, r: 10, t: 12, b: 26 };
    const svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'chart', role: 'img', 'aria-label': opts.label || 'chart' });
    const pts = series.map((s) => ({ s: s, xs: s.xs || s.ys.map((_, i) => i + 1) }));
    const allY = [], allX = [];
    pts.forEach((p) => p.s.ys.forEach((y, i) => { if (y != null && isFinite(y) && (!opts.yLog || y > 0)) { allY.push(opts.yLog ? Math.log10(y) : y); allX.push(p.xs[i]); } }));
    if (!allY.length) { container.appendChild(el('p', { class: 'muted small', text: 'No data yet' })); return; }
    let y0 = opts.yMin !== undefined ? opts.yMin : Math.min.apply(null, allY);
    let y1 = opts.yMax !== undefined ? opts.yMax : Math.max.apply(null, allY);
    if (opts.yLog) { y0 = Math.floor(y0); y1 = Math.ceil(y1); }
    if (y1 <= y0) { y1 = y0 + 1; }
    const x0 = Math.min.apply(null, allX), x1 = Math.max(Math.max.apply(null, allX), x0 + 1);
    const sx = (x) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r);
    const sy = (y) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);

    U.niceTicks(y0, y1, 4).forEach((t) => {
      svg.appendChild(svgEl('line', { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), class: 'grid' }));
      const label = opts.yLog ? '1e' + t : (Math.abs(t) >= 100 ? t.toFixed(0) : +t.toPrecision(3));
      const txt = svgEl('text', { x: m.l - 5, y: sy(t) + 3, class: 'tick', 'text-anchor': 'end' }); txt.textContent = label; svg.appendChild(txt);
    });
    U.niceTicks(x0, x1, 5).forEach((t) => {
      const txt = svgEl('text', { x: sx(t), y: H - 8, class: 'tick', 'text-anchor': 'middle' }); txt.textContent = t; svg.appendChild(txt);
    });
    if (opts.markX != null && opts.markX >= x0 && opts.markX <= x1) {
      svg.appendChild(svgEl('line', { x1: sx(opts.markX), x2: sx(opts.markX), y1: m.t, y2: H - m.b, class: 'sel' }));
    }
    pts.forEach((p, idx) => {
      const d = [];
      p.s.ys.forEach((y, i) => {
        if (y == null || !isFinite(y) || (opts.yLog && y <= 0)) return;
        d.push((d.length ? 'L' : 'M') + sx(p.xs[i]).toFixed(1) + ' ' + sy(opts.yLog ? Math.log10(y) : y).toFixed(1));
      });
      svg.appendChild(svgEl('path', { d: d.join(' '), class: 'line ' + (p.s.cls || 'line-' + (idx % 5)) }));
      const lg = svgEl('text', { x: W - m.r - 2, y: m.t + 9 + idx * 11, class: 'legend ' + (p.s.cls || 'line-' + (idx % 5)), 'text-anchor': 'end' });
      lg.textContent = p.s.name; svg.appendChild(lg);
    });
    container.appendChild(svg);
  }

  /* Draw one 28x28 (or w x h) weight tile onto a canvas element. */
  function drawTile(canvas, values, w, h, vmax) {
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(w, h);
    img.data.set(U.weightsToRGBA(values, vmax));
    ctx.putImageData(img, 0, 0);
  }

  /* Confusion matrix as a CSS grid of coloured cells. */
  function confusion(container, matrix, labels) {
    clear(container);
    const n = matrix.length;
    container.style.gridTemplateColumns = '22px repeat(' + n + ', 1fr)';
    container.appendChild(el('div', { class: 'h' }));
    labels.forEach((l) => container.appendChild(el('div', { class: 'h', text: String(l) })));
    const rowMax = matrix.map((r) => Math.max.apply(null, r));
    matrix.forEach((row, i) => {
      container.appendChild(el('div', { class: 'h', text: String(labels[i]) }));
      row.forEach((v, j) => {
        const c = el('div', { class: 'c', text: v ? String(v) : '', title: 'true ' + labels[i] + ' → predicted ' + labels[j] + ': ' + v });
        const t = v / Math.max(1, rowMax[i]);
        c.style.background = i === j ? 'rgba(52,211,153,' + (0.12 + 0.6 * t).toFixed(2) + ')' : (v ? 'rgba(248,113,113,' + Math.min(1, 0.25 + t).toFixed(2) + ')' : '');
        container.appendChild(c);
      });
    });
  }

  /* Horizontal bars: rows [{label, value, text, cls}] scaled to max(value). */
  function bars(container, rows) {
    clear(container);
    const max = Math.max.apply(null, rows.map((r) => r.value).concat([1e-12]));
    rows.forEach((r) => {
      const fill = el('div', { class: 'bar-fill ' + (r.cls || '') });
      fill.style.width = (100 * r.value / max).toFixed(1) + '%';
      container.appendChild(el('div', { class: 'bar-row' }, [
        el('span', { class: 'bar-label', text: r.label }), el('div', { class: 'bar-track' }, [fill]),
        el('span', { class: 'bar-val mono', text: r.text })]));
    });
  }

  S.charts = { lineChart, drawTile, confusion, bars };
})(window.Synapse = window.Synapse || {});

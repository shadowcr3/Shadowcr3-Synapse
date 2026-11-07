(function (S) {
  'use strict';
  const { svgEl, clear, el } = S.dom;
  const U = S.util;

  function lineChart(container, series, opts) {
    opts = opts || {};
    clear(container);
    const W = 380, H = opts.height || 200, m = { l: 44, r: 12, t: 14, b: 28 };
    const svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'chart', role: 'img', 'aria-label': opts.label || 'chart' });
    const pts = series.map((s) => ({ s: s, xs: s.xs || s.ys.map((_, i) => i + 1) }));
    const allY = [], allX = [];
    pts.forEach((p) => p.s.ys.forEach((y, i) => {
      if (y != null && isFinite(y) && (!opts.yLog || y > 0)) { allY.push(opts.yLog ? Math.log10(y) : y); allX.push(p.xs[i]); }
    }));
    if (!allY.length) { container.appendChild(el('p', { class: 'note', text: 'Nothing plotted yet.' })); return; }
    let y0 = opts.yMin !== undefined ? opts.yMin : Math.min.apply(null, allY);
    let y1 = opts.yMax !== undefined ? opts.yMax : Math.max.apply(null, allY);
    if (opts.yLog) { y0 = Math.floor(y0); y1 = Math.ceil(y1); }
    if (y1 <= y0) y1 = y0 + 1;
    const x0 = Math.min.apply(null, allX), x1 = Math.max(Math.max.apply(null, allX), x0 + 1);
    const sx = (x) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r);
    const sy = (y) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);

    svg.appendChild(svgEl('line', { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, class: 'axis' }));
    svg.appendChild(svgEl('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: 'axis' }));
    U.niceTicks(y0, y1, 4).forEach((t) => {
      svg.appendChild(svgEl('line', { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), class: 'grid' }));
      const label = opts.yLog ? '1e' + t : (Math.abs(t) >= 100 ? t.toFixed(0) : +t.toPrecision(3));
      const txt = svgEl('text', { x: m.l - 6, y: sy(t) + 3, class: 'tick', 'text-anchor': 'end' });
      txt.textContent = label; svg.appendChild(txt);
    });
    U.niceTicks(x0, x1, 5).forEach((t) => {
      const txt = svgEl('text', { x: sx(t), y: H - 10, class: 'tick', 'text-anchor': 'middle' });
      txt.textContent = t; svg.appendChild(txt);
    });
    if (opts.markX != null && opts.markX >= x0 && opts.markX <= x1) {
      svg.appendChild(svgEl('line', { x1: sx(opts.markX), x2: sx(opts.markX), y1: m.t, y2: H - m.b, class: 'mark' }));
      const lab = svgEl('text', { x: sx(opts.markX) + 4, y: m.t + 8, class: 'tick' });
      lab.textContent = 'kept'; svg.appendChild(lab);
    }
    pts.forEach((p, idx) => {
      const d = [];
      p.s.ys.forEach((y, i) => {
        if (y == null || !isFinite(y) || (opts.yLog && y <= 0)) return;
        d.push((d.length ? 'L' : 'M') + sx(p.xs[i]).toFixed(1) + ' ' + sy(opts.yLog ? Math.log10(y) : y).toFixed(1));
      });
      const cls = p.s.cls || 'line-' + (idx % 5);
      svg.appendChild(svgEl('path', { d: d.join(' '), class: 'line ' + cls }));
      const lg = svgEl('text', { x: W - m.r - 2, y: m.t + 10 + idx * 12, class: 'legend ' + cls, 'text-anchor': 'end' });
      lg.textContent = p.s.name; svg.appendChild(lg);
    });
    container.appendChild(svg);
  }

  function drawTile(canvas, values, w, h, vmax) {
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(w, h);
    img.data.set(U.weightsToRGBA(values, vmax));
    ctx.putImageData(img, 0, 0);
  }

  function confusion(container, matrix, labels) {
    clear(container);
    const n = matrix.length;
    container.style.gridTemplateColumns = '1.4rem repeat(' + n + ', 1fr)';
    container.appendChild(el('div', { class: 'h' }));
    labels.forEach((l) => container.appendChild(el('div', { class: 'h', text: String(l) })));
    const rowMax = matrix.map((r) => Math.max.apply(null, r));
    matrix.forEach((row, i) => {
      container.appendChild(el('div', { class: 'h', text: String(labels[i]) }));
      row.forEach((v, j) => {
        const c = el('div', { class: 'c' + (i === j ? ' diag' : ''), text: v ? String(v) : '',
                              title: 'drawn as ' + labels[i] + ', read as ' + labels[j] + ': ' + v });
        const t = v / Math.max(1, rowMax[i]);
        if (v) c.style.background = i === j ? 'rgba(12,106,106,' + (0.1 + 0.5 * t).toFixed(2) + ')' : 'rgba(183,58,31,' + Math.min(0.85, 0.2 + t).toFixed(2) + ')';
        container.appendChild(c);
      });
    });
  }

  function bars(container, rows) {
    clear(container);
    const max = Math.max.apply(null, rows.map((r) => r.value).concat([1e-12]));
    rows.forEach((r) => {
      const fill = el('div', { class: 'bar-fill ' + (r.cls || '') });
      fill.style.width = (100 * r.value / max).toFixed(1) + '%';
      container.appendChild(el('div', { class: 'bar-row' }, [
        el('span', { class: 'bar-label', text: r.label }), el('div', { class: 'bar-track' }, [fill]),
        el('span', { class: 'bar-val', text: r.text })]));
    });
  }

  S.charts = { lineChart, drawTile, confusion, bars };
})(window.Synapse = window.Synapse || {});

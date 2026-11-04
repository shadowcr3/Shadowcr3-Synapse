/* Live network visualiser (canvas). Draws every neuron of every layer, the strongest
 * connections between layers, the forward signal (activations) and the backward error signal
 * (delta = dLoss/dz per neuron), all taken from the server's /trace response.
 */
(function (S) {
  'use strict';
  const U = S.util;
  const W = 900, H = 420;
  const COL = { pos: '#22d3ee', neg: '#f472b6', bpos: '#f87171', bneg: '#60a5fa', text: '#8d94a3',
                ring: '#22d3ee', target: '#34d399', grid: '#262a33' };

  class NetworkView {
    constructor(canvas, opts) {
      opts = opts || {};
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      const dpr = Math.max(1, Math.min(2, (window.devicePixelRatio || 1)));
      canvas.width = W * dpr; canvas.height = H * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.mode = 'forward';
      this.t = 0; this.anim = null;
      this.pixels = null; this.trace = null; this.layout = null; this.hover = null;
      this.onHover = opts.onHover || null;
      this.reduced = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
      canvas.addEventListener('mousemove', (e) => this._move(e));
      canvas.addEventListener('mouseleave', () => this._leave());
      this.renderEmpty();
    }

    // ----- data ---------------------------------------------------------------------------
    update(pixels, trace, opts) {
      this.pixels = pixels; this.trace = trace; this._layout();
      if (opts && opts.animate && !this.reduced) this.play(); else this.finish();
    }
    setMode(mode) {
      this.mode = mode;
      if (this.trace && !this.reduced) this.play(); else this.finish();
    }
    get tMax() { return this.trace ? this.trace.layers.length : 1; }
    finish() { this._stop(); this.t = this.tMax; this.render(); }
    _stop() { if (this.anim) { cancelAnimationFrame(this.anim); this.anim = null; } }
    play() {
      if (!this.trace) return;
      this._stop();
      const total = 380 * this.tMax + 250, start = performance.now();
      this.t = 0;
      const step = (now) => {
        const p = U.clamp((now - start) / total, 0, 1);
        this.t = p * this.tMax;
        this.render();
        this.anim = p < 1 ? requestAnimationFrame(step) : null;
      };
      this.anim = requestAnimationFrame(step);
    }
    get hasBackward() { return !!(this.trace && this.trace.backward); }

    // ----- layout -------------------------------------------------------------------------
    _layout() {
      const layers = this.trace.layers, n = layers.length;
      const xs = [];
      for (let i = 0; i <= n; i++) xs.push(70 + (W - 140) * i / n);
      const scale = 4, size = 28 * scale;
      this.layout = { input: { x: xs[0] - size / 2, y: (H - size) / 2 + 14, size: size, scale: scale }, layers: [], xs: xs };
      layers.forEach((ly, j) => {
        const x = xs[j + 1], isOut = j === n - 1, pos = [];
        if (isOut) {
          const gap = Math.min(40, (H - 90) / ly.size), top = 56 + (H - 90 - gap * (ly.size - 1)) / 2;
          for (let i = 0; i < ly.size; i++) pos.push({ x: x, y: top + i * gap, r: Math.min(14, gap * 0.42) });
        } else {
          const g = U.gridLayout(ly.size, 128, H - 100);
          const w = g.cols * g.cell, h = g.rows * g.cell, y0 = 52 + (H - 100 - h) / 2;
          for (let i = 0; i < ly.size; i++) {
            pos.push({ x: x - w / 2 + (i % g.cols + 0.5) * g.cell, y: y0 + (Math.floor(i / g.cols) + 0.5) * g.cell, r: g.cell * 0.4 });
          }
        }
        this.layout.layers.push(pos);
      });
    }
    _pt(layerIdx, i) {   // layerIdx: -1 = input pixels, 0.. = trace layers
      if (layerIdx < 0) {
        const L = this.layout.input;
        return { x: L.x + (i % 28 + 0.5) * L.scale, y: L.y + (Math.floor(i / 28) + 0.5) * L.scale };
      }
      return this.layout.layers[layerIdx][i];
    }
    _layerName(j) { return j < 0 ? 'input' : this.trace.layers[j].name; }

    // ----- drawing ------------------------------------------------------------------------
    renderEmpty() {
      const c = this.ctx;
      c.clearRect(0, 0, W, H);
      c.fillStyle = COL.text; c.font = '14px Inter, system-ui, sans-serif'; c.textAlign = 'center';
      c.fillText('Draw a digit to watch the network respond', W / 2, H / 2);
    }

    _reveal(p, back) {   // p = 1..L layer position; returns 0..1
      const L = this.tMax;
      return U.clamp(back ? this.t - (L - p) : this.t - (p - 1), 0, 1);
    }

    render() {
      if (!this.trace || !this.layout) return this.renderEmpty();
      const c = this.ctx, tr = this.trace, back = this.mode === 'backward' && this.hasBackward;
      c.clearRect(0, 0, W, H);
      c.textAlign = 'center';

      // edges (under the neurons)
      const edgeGroups = back ? tr.backward.edges : tr.edges;
      tr.layers.forEach((ly, j) => {
        const a = this._reveal(j + 1, back);
        if (a <= 0) return;
        const prev = j - 1;
        const key = back ? (ly.name + '_' + this._layerName(prev)) : (this._layerName(prev) + '_' + ly.name);
        const list = edgeGroups[key] || [];
        let vmax = 1e-9;
        list.forEach((e) => { vmax = Math.max(vmax, Math.abs(e[2])); });
        list.forEach((e) => {
          const from = this._pt(prev, e[0]), to = this._pt(j, e[1]);
          const s = Math.pow(Math.abs(e[2]) / vmax, 0.6);
          c.globalAlpha = a * (0.12 + 0.75 * s);
          c.strokeStyle = back ? (e[2] >= 0 ? COL.bpos : COL.bneg) : (e[2] >= 0 ? COL.pos : COL.neg);
          c.lineWidth = 0.6 + 1.4 * s;
          c.beginPath(); c.moveTo(back ? to.x : from.x, back ? to.y : from.y); c.lineTo(back ? from.x : to.x, back ? from.y : to.y); c.stroke();
        });
      });
      c.globalAlpha = 1;

      // input image
      const L = this.layout.input;
      if (this.pixels) {
        c.fillStyle = '#000'; c.fillRect(L.x, L.y, L.size, L.size);
        for (let i = 0; i < 784; i++) {
          const v = this.pixels[i];
          if (v > 0) { c.fillStyle = 'rgb(' + v + ',' + v + ',' + v + ')'; c.fillRect(L.x + (i % 28) * L.scale, L.y + Math.floor(i / 28) * L.scale, L.scale, L.scale); }
        }
      }
      c.strokeStyle = COL.grid; c.lineWidth = 1; c.strokeRect(L.x - 0.5, L.y - 0.5, L.size + 1, L.size + 1);
      this._title(this.layout.xs[0], 'Input', '28×28 = 784 pixels');

      // neurons
      tr.layers.forEach((ly, j) => {
        const a = this._reveal(j + 1, back), pos = this.layout.layers[j];
        const isOut = j === tr.layers.length - 1;
        const values = back ? tr.backward.deltas[ly.name] : (isOut ? ly.probs : ly.activation);
        const vmax = Math.max(1e-9, U.maxAbs(values));
        const pre = Math.max(0.25, a);
        pos.forEach((p, i) => {
          const v = values[i];
          const col = back ? U.divColor(v, vmax) : U.seqColor(isOut ? v : v / vmax);
          c.globalAlpha = pre;
          c.fillStyle = a > 0 ? U.rgb(col) : '#1b1e25';
          c.beginPath();
          if (isOut) c.arc(p.x, p.y, p.r, 0, Math.PI * 2);
          else c.rect(p.x - p.r, p.y - p.r, p.r * 2, p.r * 2);
          c.fill();
        });
        c.globalAlpha = 1;
        if (isOut) this._outputDecor(pos, ly, back, a);
        const info = back ? '‖δ‖ = ' + U.fmtNum(tr.backward.delta_norms[ly.name], 3)
                          : (isOut ? 'softmax' : ly.activation.filter((v) => v > 0).length + ' of ' + ly.size + ' active');
        this._title(this.layout.xs[j + 1], (isOut ? 'Output' : 'Hidden ' + (j + 1)) + ' · ' + ly.size, info);
      });

      if (this.hover) {
        const p = this.layout.layers[this.hover.layer][this.hover.index];
        c.strokeStyle = '#fff'; c.lineWidth = 1.5; c.beginPath(); c.arc(p.x, p.y, p.r + 3, 0, Math.PI * 2); c.stroke();
      }
      if (this.mode === 'backward' && !this.hasBackward) {
        c.fillStyle = COL.text; c.font = '12px Inter, system-ui, sans-serif';
        c.fillText('Choose the true digit to see the error flow backwards', W / 2, H - 10);
      }
    }

    _title(x, line1, line2) {
      const c = this.ctx;
      c.fillStyle = '#e8eaf0'; c.font = '600 12px Inter, system-ui, sans-serif'; c.textAlign = 'center';
      c.fillText(line1, x, 22);
      c.fillStyle = COL.text; c.font = '11px Inter, system-ui, sans-serif';
      c.fillText(line2, x, 38);
    }

    _outputDecor(pos, ly, back, a) {
      const c = this.ctx, tr = this.trace, digits = tr.digits || [];
      c.font = '12px Inter, system-ui, sans-serif';
      pos.forEach((p, i) => {
        c.fillStyle = '#e8eaf0'; c.textAlign = 'left';
        c.fillText(String(digits[i] != null ? digits[i] : i) + '  ' + U.fmtPct(ly.probs[i], 0), p.x + p.r + 8, p.y + 4);
      });
      const pr = tr.predicted;
      if (a > 0.99 && pos[pr]) { c.strokeStyle = COL.ring; c.lineWidth = 2; c.beginPath(); c.arc(pos[pr].x, pos[pr].y, pos[pr].r + 4, 0, Math.PI * 2); c.stroke(); }
      if (tr.backward && pos[tr.backward.target]) {
        const t = pos[tr.backward.target];
        c.strokeStyle = COL.target; c.lineWidth = 2; c.setLineDash([3, 3]); c.beginPath(); c.arc(t.x, t.y, t.r + 7, 0, Math.PI * 2); c.stroke(); c.setLineDash([]);
      }
    }

    // ----- hover --------------------------------------------------------------------------
    _move(e) {
      if (!this.trace || !this.layout) return;
      const r = this.canvas.getBoundingClientRect();
      const x = (e.clientX - r.left) * (W / (r.width || W)), y = (e.clientY - r.top) * (H / (r.height || H));
      let best = null, bd = 1e9;
      this.layout.layers.forEach((pos, j) => pos.forEach((p, i) => {
        const d = Math.hypot(p.x - x, p.y - y);
        if (d < Math.max(p.r + 4, 9) && d < bd) { bd = d; best = { layer: j, index: i }; }
      }));
      const same = (!best && !this.hover) || (best && this.hover && best.layer === this.hover.layer && best.index === this.hover.index);
      if (same) return;
      this.hover = best;
      this.render();
      if (this.onHover) this.onHover(best ? this.describe(best.layer, best.index) : null);
    }
    _leave() { if (this.hover) { this.hover = null; this.render(); if (this.onHover) this.onHover(null); } }

    describe(layer, index) {
      const tr = this.trace, ly = tr.layers[layer], isOut = layer === tr.layers.length - 1;
      return { layer: layer, index: index, name: ly.name, size: ly.size, isOutput: isOut,
               activation: isOut ? ly.probs[index] : ly.activation[index],
               logit: isOut ? ly.logits[index] : null,
               delta: tr.backward ? tr.backward.deltas[ly.name][index] : null,
               digit: isOut && tr.digits ? tr.digits[index] : null };
    }
  }

  S.NetworkView = NetworkView;
  S.NETVIZ_SIZE = { W: W, H: H };
})(window.Synapse = window.Synapse || {});

(function (S) {
  'use strict';
  const U = S.util;
  const W = 900, H = 460;
  const C = { ink: '#1c2230', muted: '#6a7081', rule: '#cdc5b0', plate: '#fbf8f0',
              teal: '#0c6a6a', brown: '#9a4b2c', verm: '#b73a1f', ultra: '#2a4ea1' };
  const SERIF = 'Georgia, "Iowan Old Style", Palatino, serif';
  const MONO = 'ui-monospace, Menlo, Consolas, monospace';
  const TAU = Math.PI * 2;

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
      this.pixels = null; this.trace = null; this.layout = null; this.hover = null; this.focus = null;
      this.onHover = opts.onHover || null;
      this.reduced = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
      canvas.addEventListener('mousemove', (e) => this._move(e));
      canvas.addEventListener('mouseleave', () => this._leave());
      this.renderEmpty();
    }

    update(pixels, trace, opts) {
      this.pixels = pixels; this.trace = trace; this._layout();
      if (opts && opts.animate && !this.reduced) this.play(); else this.finish();
    }
    setMode(mode) {
      this.mode = mode;
      if (this.trace && !this.reduced) this.play(); else this.finish();
    }
    setFocus(edges) { this.focus = edges && edges.length ? edges : null; this.render(); }
    get tMax() { return this.trace ? this.trace.layers.length : 1; }
    get hasBackward() { return !!(this.trace && this.trace.backward); }
    finish() { this._stop(); this.t = this.tMax; this.render(); }
    _stop() { if (this.anim) { cancelAnimationFrame(this.anim); this.anim = null; } }
    play() {
      if (!this.trace) return;
      this._stop();
      const total = 420 * this.tMax + 300, start = performance.now();
      this.t = 0;
      const step = (now) => {
        const p = U.clamp((now - start) / total, 0, 1);
        this.t = p * this.tMax;
        this.render();
        this.anim = p < 1 ? requestAnimationFrame(step) : null;
      };
      this.anim = requestAnimationFrame(step);
    }

    _layout() {
      const layers = this.trace.layers, n = layers.length;
      const left = 84, right = W - 196, xs = [];
      for (let i = 0; i <= n; i++) xs.push(left + (right - left) * i / n);
      const scale = 4, size = 28 * scale;
      this.layout = { input: { x: xs[0] - size / 2, y: (H - size) / 2 + 10, size: size, scale: scale }, layers: [], xs: xs };
      layers.forEach((ly, j) => {
        const x = xs[j + 1], pos = [];
        if (j === n - 1) {
          const gap = Math.min(42, (H - 120) / ly.size), top = 70 + (H - 120 - gap * (ly.size - 1)) / 2;
          for (let i = 0; i < ly.size; i++) pos.push({ x: x, y: top + i * gap, r: Math.min(15, gap * 0.42) });
        } else {
          const g = U.gridLayout(ly.size, 132, H - 130);
          const w = g.cols * g.cell, h = g.rows * g.cell, y0 = 66 + (H - 130 - h) / 2;
          for (let i = 0; i < ly.size; i++) {
            pos.push({ x: x - w / 2 + (i % g.cols + 0.5) * g.cell, y: y0 + (Math.floor(i / g.cols) + 0.5) * g.cell, r: g.cell * 0.46 });
          }
        }
        this.layout.layers.push(pos);
      });
    }

    _pt(layerIdx, i) {
      if (layerIdx < 0) {
        const L = this.layout.input;
        return { x: L.x + (i % 28 + 0.5) * L.scale, y: L.y + (Math.floor(i / 28) + 0.5) * L.scale };
      }
      return this.layout.layers[layerIdx][i];
    }
    _name(j) { return j < 0 ? 'input' : this.trace.layers[j].name; }
    _reveal(p, back) {
      const L = this.tMax;
      return U.clamp(back ? this.t - (L - p) : this.t - (p - 1), 0, 1);
    }

    renderEmpty() {
      const c = this.ctx;
      c.clearRect(0, 0, W, H);
      c.fillStyle = C.muted; c.font = 'italic 17px ' + SERIF; c.textAlign = 'center';
      c.fillText('Draw a digit on the board and the network shows up here.', W / 2, H / 2);
    }

    _edge(from, to, color, alpha, width, progress) {
      const c = this.ctx, dx = to.x - from.x;
      c.beginPath();
      c.moveTo(from.x, from.y);
      c.bezierCurveTo(from.x + dx * 0.5, from.y, from.x + dx * 0.5, to.y, to.x, to.y);
      if (progress < 1) { const len = Math.hypot(dx, to.y - from.y) * 1.2; c.setLineDash([len * progress, len]); }
      c.globalAlpha = alpha; c.strokeStyle = color; c.lineWidth = width; c.stroke();
      c.setLineDash([]); c.globalAlpha = 1;
    }

    _unit(p, value, vmax, color, grow) {
      const c = this.ctx;
      c.beginPath(); c.arc(p.x, p.y, p.r, 0, TAU);
      c.fillStyle = C.plate; c.fill();
      c.lineWidth = 1; c.strokeStyle = 'rgba(28,34,48,0.28)'; c.stroke();
      const a = vmax > 0 ? Math.abs(value) / vmax : 0;
      if (grow <= 0) return;
      if (a > 0.015) {
        c.beginPath(); c.arc(p.x, p.y, Math.max(1.4, p.r * Math.sqrt(a) * 0.94 * grow), 0, TAU);
        c.fillStyle = color; c.fill();
      } else {
        c.beginPath(); c.arc(p.x, p.y, 1.1, 0, TAU); c.fillStyle = C.muted; c.fill();
      }
    }

    render() {
      if (!this.trace || !this.layout) return this.renderEmpty();
      const c = this.ctx, tr = this.trace, back = this.mode === 'backward' && this.hasBackward;
      const hv = this.hover;
      c.clearRect(0, 0, W, H);
      c.textAlign = 'center';

      const groups = back ? tr.backward.edges : tr.edges;
      tr.layers.forEach((ly, j) => {
        const a = this._reveal(j + 1, back);
        if (a <= 0) return;
        const prev = j - 1;
        const key = back ? (ly.name + '_' + this._name(prev)) : (this._name(prev) + '_' + ly.name);
        const list = groups[key] || [];
        let vmax = 1e-9;
        list.forEach((e) => { vmax = Math.max(vmax, Math.abs(e[2])); });
        list.forEach((e) => {
          const left = this._pt(prev, e[0]), right = this._pt(j, e[1]);
          const s = Math.pow(Math.abs(e[2]) / vmax, 0.65);
          const touches = hv && ((hv.layer === j && hv.index === e[1]) || (hv.layer === prev && hv.index === e[0]));
          const focus = hv ? (touches ? 1 : (this.focus ? 0.1 : 0.16)) : 1;
          const color = back ? (e[2] >= 0 ? C.verm : C.ultra) : (e[2] >= 0 ? C.teal : C.brown);
          const width = (0.6 + 1.5 * s) * (touches ? 1.5 : 1);
          if (back) this._edge(right, left, color, (0.14 + 0.62 * s) * focus, width, a);
          else this._edge(left, right, color, (0.14 + 0.62 * s) * focus, width, a);
        });
      });

      if (this.focus) {
        this.focus.forEach((e) => {
          const a = this._pt(e.fromLayer, e.fromIdx), b = this._pt(e.toLayer, e.toIdx);
          const color = back ? (e.v >= 0 ? C.verm : C.ultra) : (e.v >= 0 ? C.teal : C.brown);
          this._edge(a, b, color, 0.35 + 0.6 * e.s, 0.9 + 1.7 * e.s, 1);
        });
      }

      const L = this.layout.input;
      c.fillStyle = C.plate; c.fillRect(L.x, L.y, L.size, L.size);
      if (this.pixels) {
        for (let i = 0; i < 784; i++) {
          const v = this.pixels[i];
          if (v > 6) {
            c.fillStyle = 'rgba(28,34,48,' + (v / 255).toFixed(3) + ')';
            c.fillRect(L.x + (i % 28) * L.scale, L.y + Math.floor(i / 28) * L.scale, L.scale, L.scale);
          }
        }
      }
      c.strokeStyle = C.ink; c.lineWidth = 1; c.strokeRect(L.x - 0.5, L.y - 0.5, L.size + 1, L.size + 1);
      this._heading(this.layout.xs[0], 'input', '28 × 28 pixels');

      tr.layers.forEach((ly, j) => {
        const a = this._reveal(j + 1, back), pos = this.layout.layers[j], isOut = j === tr.layers.length - 1;
        const values = back ? tr.backward.deltas[ly.name] : (isOut ? ly.probs : ly.activation);
        const vmax = back ? Math.max(1e-9, U.maxAbs(values)) : Math.max(1e-9, isOut ? 1 : U.maxAbs(values));
        pos.forEach((p, i) => {
          const v = values[i];
          const color = back ? (v >= 0 ? C.verm : C.ultra) : C.teal;
          this._unit(p, v, vmax, color, a);
        });
        if (isOut) this._outputLabels(pos, ly, a);
        const note = back ? 'error size ' + U.fmtNum(tr.backward.delta_norms[ly.name], 3)
                          : (isOut ? 'one unit per digit' : ly.activation.filter((v) => v > 0).length + ' of ' + ly.size + ' firing');
        this._heading(this.layout.xs[j + 1], isOut ? 'output' : 'hidden ' + (j + 1), isOut ? note : ly.size + ' units · ' + note);
      });

      this._arrow(back);

      if (hv && this.layout.layers[hv.layer] && this.layout.layers[hv.layer][hv.index]) {
        const p = this.layout.layers[hv.layer][hv.index];
        c.strokeStyle = C.ink; c.lineWidth = 1.6; c.beginPath(); c.arc(p.x, p.y, p.r + 3, 0, TAU); c.stroke();
      }
      if (this.mode === 'backward' && !this.hasBackward) {
        c.fillStyle = C.muted; c.font = 'italic 14px ' + SERIF; c.textAlign = 'center';
        c.fillText('Pick the digit you drew (under the figure) to see the error travel back.', W / 2, H - 38);
      }
    }

    _heading(x, title, note) {
      const c = this.ctx;
      c.textAlign = 'center';
      c.fillStyle = C.ink; c.font = 'italic 16px ' + SERIF; c.fillText(title, x, 26);
      c.fillStyle = C.muted; c.font = '11px ' + MONO; c.fillText(note, x, 44);
    }

    _outputLabels(pos, ly, reveal) {
      const c = this.ctx, tr = this.trace, digits = tr.digits || [];
      pos.forEach((p, i) => {
        const isTop = i === tr.predicted, isTarget = tr.backward && i === tr.backward.target;
        c.textAlign = 'left';
        c.fillStyle = C.ink; c.font = (isTop ? 'bold ' : '') + '20px ' + SERIF;
        c.fillText(String(digits[i] != null ? digits[i] : i), p.x + p.r + 10, p.y + 7);
        const x0 = p.x + p.r + 36, full = 84, len = full * ly.probs[i] * reveal;
        c.fillStyle = 'rgba(28,34,48,0.08)'; c.fillRect(x0, p.y - 3, full, 6);
        c.fillStyle = isTop ? C.teal : C.muted; c.fillRect(x0, p.y - 3, Math.max(len, 0.6), 6);
        c.fillStyle = C.muted; c.font = '11px ' + MONO;
        c.fillText(U.fmtPct(ly.probs[i], 0), x0 + full + 6, p.y + 4);
        if (isTarget) {
          c.font = 'italic 13px ' + SERIF; c.textAlign = 'right';
          const label = 'you drew', w = c.measureText(label).width + 10;
          c.clearRect(p.x - p.r - 14 - w, p.y - 9, w + 6, 18);
          c.fillStyle = C.verm; c.fillText(label, p.x - p.r - 10, p.y + 4);
        }
      });
    }

    _arrow(back) {
      const c = this.ctx, x0 = this.layout.xs[0], x1 = this.layout.xs[this.layout.xs.length - 1], y = H - 16;
      c.strokeStyle = back ? C.verm : C.teal; c.fillStyle = c.strokeStyle; c.lineWidth = 1.2;
      c.beginPath(); c.moveTo(x0, y); c.lineTo(x1, y); c.stroke();
      const dir = back ? -1 : 1, tip = back ? x0 : x1;
      c.beginPath(); c.moveTo(tip, y); c.lineTo(tip - 8 * dir, y - 4); c.lineTo(tip - 8 * dir, y + 4); c.closePath(); c.fill();
      c.font = 'italic 13px ' + SERIF; c.textAlign = 'center';
      const mid = (x0 + x1) / 2, text = back ? 'error, traced back from the answer' : 'signal, forward through the layers';
      const w = c.measureText(text).width + 16;
      c.clearRect(mid - w / 2, y - 9, w, 18);
      c.fillText(text, mid, y + 4);
    }

    _move(e) {
      if (!this.trace || !this.layout) return;
      const r = this.canvas.getBoundingClientRect();
      const x = (e.clientX - r.left) * (W / (r.width || W)), y = (e.clientY - r.top) * (H / (r.height || H));
      let best = null, bd = 1e9;
      this.layout.layers.forEach((pos, j) => pos.forEach((p, i) => {
        const d = Math.hypot(p.x - x, p.y - y);
        if (d < Math.max(p.r + 3, 8) && d < bd) { bd = d; best = { layer: j, index: i }; }
      }));
      const same = (!best && !this.hover) || (best && this.hover && best.layer === this.hover.layer && best.index === this.hover.index);
      if (same) return;
      this.hover = best;
      this.render();
      if (this.onHover) this.onHover(best ? this.describe(best.layer, best.index) : null);
    }
    _leave() {
      if (this.hover) { this.hover = null; this.focus = null; this.render(); if (this.onHover) this.onHover(null); }
    }

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

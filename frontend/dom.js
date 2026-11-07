(function (S) {
  'use strict';

  function el(tag, props, children) {
    const node = document.createElement(tag);
    Object.keys(props || {}).forEach((k) => {
      const v = props[k];
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
      else if (v === true) node.setAttribute(k, '');
      else if (v !== false && v != null) node.setAttribute(k, v);
    });
    (children || []).forEach((c) => { if (c != null) node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return node;
  }
  const svgEl = (tag, attrs) => {
    const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.keys(attrs || {}).forEach((k) => n.setAttribute(k, attrs[k]));
    return n;
  };
  const $ = (id) => document.getElementById(id);
  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

  class ApiError extends Error {
    constructor(status, code, message, body, retryAfter) {
      super(message);
      this.status = status; this.code = code; this.body = body; this.retryAfter = retryAfter || 0;
    }
  }

  async function api(path, options) {
    const opts = Object.assign({ method: 'GET' }, options || {});
    if (opts.json !== undefined) {
      opts.method = opts.method === 'GET' ? 'POST' : opts.method;
      opts.headers = Object.assign({ 'Content-Type': 'application/json' }, opts.headers);
      opts.body = JSON.stringify(opts.json);
      delete opts.json;
    }
    let res;
    try { res = await fetch(path, opts); }
    catch (e) { throw new ApiError(0, 'network', 'The server is not answering. Is it still running?'); }
    let body = null;
    try { body = await res.json(); } catch (e) { body = null; }
    if (!res.ok) {
      const err = (body && body.error) || {};
      const wait = Number(res.headers && res.headers.get && res.headers.get('Retry-After')) || 0;
      const message = res.status === 429
        ? 'Too many requests. Try again in ' + (wait || 1) + ' s.'
        : (err.message || ('Request failed (' + res.status + ')'));
      throw new ApiError(res.status, err.code || 'http_error', message, body, wait);
    }
    return body;
  }

  S.dom = { el, svgEl, $, clear, api, ApiError };
})(window.Synapse = window.Synapse || {});

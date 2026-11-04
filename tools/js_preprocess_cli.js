#!/usr/bin/env node
/* Test/tool helper: runs frontend/preprocess.js on images read from stdin.
 * stdin JSON: {"width": W, "height": H, "images_b64": ["<base64 of W*H uint8>", ...]}
 * stdout JSON: [{"pixels": [784 ints] | null, "error": null | "blank" | "too_small"}, ...]
 */
'use strict';
const path = require('path');
const { normalizeToMnist } = require(path.join(__dirname, '..', 'frontend', 'preprocess.js'));

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (c) => (input += c));
process.stdin.on('end', () => {
  const req = JSON.parse(input);
  const out = req.images_b64.map((b64) => {
    const gray = Float64Array.from(Buffer.from(b64, 'base64'));
    const r = normalizeToMnist(gray, req.width, req.height);
    return { pixels: r.pixels ? Array.from(r.pixels) : null, error: r.error };
  });
  process.stdout.write(JSON.stringify(out));
});

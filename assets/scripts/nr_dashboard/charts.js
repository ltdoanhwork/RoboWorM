/* Minimal SVG chart helpers. No chart library: the dashboard must work offline from a
 * file:// URL, and these four primitives cover every panel. */
'use strict';

const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  for (const k in (attrs || {})) if (attrs[k] !== null && attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}
function svgRoot(w, h) {
  const s = el('svg', { width: '100%', viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: 'xMinYMin meet' });
  return s;
}
function txt(parent, x, y, s, attrs) {
  const t = el('text', { x, y, ...(attrs || {}) }, parent);
  t.textContent = s;
  return t;
}

/* Tooltip, one instance shared by every chart. */
const TIP = (() => {
  const d = document.createElement('div');
  d.className = 'tip';
  document.body.appendChild(d);
  return d;
})();
function tipOn(html) { TIP.innerHTML = html; TIP.classList.add('on'); }
function tipMove(e) {
  const pad = 14, w = TIP.offsetWidth, h = TIP.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > innerWidth) x = e.clientX - w - pad;
  if (y + h > innerHeight) y = e.clientY - h - pad;
  TIP.style.left = x + 'px'; TIP.style.top = y + 'px';
}
function tipOff() { TIP.classList.remove('on'); }
function hover(node, html) {
  node.addEventListener('mouseenter', () => tipOn(html));
  node.addEventListener('mousemove', tipMove);
  node.addEventListener('mouseleave', tipOff);
}

function niceTicks(lo, hi, want) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / (want || 5);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || mag * 10;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) out.push(+t.toFixed(10));
  return out;
}

/* ---- horizontal forest: rows of median + CI, with the claim vocabulary of F5 ---- */
function forest(rows, opts) {
  // rows: {label, n, med, lo, hi, p, resolved, artifact, color, tip}
  const o = Object.assign({ w: 560, rowH: 19, padL: 118, padR: 128, padT: 16, padB: 30, center: 0, unit: '' }, opts);
  const h = o.padT + rows.length * o.rowH + o.padB;
  const s = svgRoot(o.w, h);
  const fin = v => v !== null && v !== undefined && isFinite(v);
  let lo = o.center, hi = o.center;
  rows.forEach(r => [r.lo, r.hi, r.med].forEach(v => { if (fin(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
  if (!(hi > lo)) { hi = lo + 1; }
  const pad = (hi - lo) * 0.08 || 1;
  lo -= pad; hi += pad;
  const X = v => o.padL + (v - lo) / (hi - lo) * (o.w - o.padL - o.padR);
  const g = el('g', {}, s);
  niceTicks(lo, hi, 5).forEach(t => {
    el('line', { x1: X(t), x2: X(t), y1: o.padT - 6, y2: h - o.padB, class: 'gridline' }, g);
    txt(g, X(t), h - o.padB + 12, String(t), { 'text-anchor': 'middle' });
  });
  el('line', { x1: X(o.center), x2: X(o.center), y1: o.padT - 6, y2: h - o.padB, class: 'zero' }, g);
  txt(g, o.w / 2, h - o.padB + 24, o.unit, { 'text-anchor': 'middle' });
  rows.forEach((r, i) => {
    const y = o.padT + i * o.rowH + o.rowH / 2;
    if (r.n !== null && r.n < MIN_N) {
      el('rect', { x: o.padL, y: y - o.rowH / 2 + 1, width: o.w - o.padL - o.padR, height: o.rowH - 2, fill: 'var(--panel-2)' }, g);
    }
    txt(g, o.padL - 7, y + 3.2, r.label, { 'text-anchor': 'end', class: 'lbl' });
    if (!fin(r.med)) { txt(g, o.padL + 4, y + 3.2, 'no comparable pairs', { 'fill': 'var(--text-dim)' }); return; }
    const c = r.color || 'var(--accent)';
    if (fin(r.lo) && fin(r.hi)) {
      el('line', {
        x1: X(r.lo), x2: X(r.hi), y1: y, y2: y, stroke: c,
        'stroke-width': r.resolved ? 3 : 1.4, 'stroke-opacity': r.resolved ? 0.95 : 0.4,
      }, g);
    }
    let mk;
    if (r.artifact) {
      mk = el('rect', { x: X(r.med) - 4, y: y - 4, width: 8, height: 8, fill: 'none', stroke: '#c04747', 'stroke-width': 1.8 }, g);
    } else {
      mk = el('circle', { cx: X(r.med), cy: y, r: 4, fill: r.resolved ? c : 'var(--panel)', stroke: c, 'stroke-width': 1.5 }, g);
    }
    mk.style.cursor = 'help';
    hover(mk, r.tip || r.label);
    const cls = r.artifact ? 'artifact' : (r.resolved ? 'resolved' : '');
    const t = txt(g, o.w - o.padR + 6, y + 3.2, `n=${r.n}` + (r.p !== null && r.p !== undefined ? `  p=${fmtP(r.p)}` : '')
      + (r.artifact ? '  artifact+' : (r.resolved ? '  *' : '')), { 'text-anchor': 'start' });
    if (cls === 'artifact') t.setAttribute('fill', '#c04747');
    else if (cls === 'resolved') { t.setAttribute('fill', 'var(--text)'); t.setAttribute('font-weight', '700'); }
  });
  return s;
}

/* ---- distribution: box + whisker per group, for absolute levels ---- */
function boxes(groups, opts) {
  // groups: {label, values, color}
  const o = Object.assign({ w: 560, h: 210, padL: 52, padR: 14, padT: 14, padB: 44 }, opts);
  const s = svgRoot(o.w, o.h);
  const g = el('g', {}, s);
  const all = groups.flatMap(x => x.values);
  if (!all.length) { txt(g, o.w / 2, o.h / 2, 'no valid observations', { 'text-anchor': 'middle' }); return s; }
  const q = (v, p) => { const a = v.slice().sort((x, y) => x - y); const i = (a.length - 1) * p; const l = Math.floor(i); return a[l] + (a[Math.ceil(i)] - a[l]) * (i - l); };
  let lo = Math.min(...groups.map(x => x.values.length ? q(x.values, 0.02) : Infinity));
  let hi = Math.max(...groups.map(x => x.values.length ? q(x.values, 0.98) : -Infinity));
  if (!(hi > lo)) hi = lo + 1;
  const pad = (hi - lo) * 0.1; lo -= pad; hi += pad;
  const Y = v => o.padT + (1 - (v - lo) / (hi - lo)) * (o.h - o.padT - o.padB);
  niceTicks(lo, hi, 4).forEach(t => {
    el('line', { x1: o.padL, x2: o.w - o.padR, y1: Y(t), y2: Y(t), class: 'gridline' }, g);
    txt(g, o.padL - 6, Y(t) + 3, String(t), { 'text-anchor': 'end' });
  });
  const bw = (o.w - o.padL - o.padR) / groups.length;
  groups.forEach((grp, i) => {
    const cx = o.padL + bw * (i + 0.5);
    txt(g, cx, o.h - o.padB + 13, grp.label, { 'text-anchor': 'middle', class: 'lbl' });
    txt(g, cx, o.h - o.padB + 24, `n=${grp.values.length}`, { 'text-anchor': 'middle' });
    if (grp.values.length < 3) return;
    const v = grp.values, c = grp.color || 'var(--accent)';
    const q1 = q(v, 0.25), q3 = q(v, 0.75), md = median(v), w1 = q(v, 0.05), w9 = q(v, 0.95);
    el('line', { x1: cx, x2: cx, y1: Y(w1), y2: Y(w9), stroke: c, 'stroke-opacity': .55 }, g);
    const r = el('rect', { x: cx - bw * 0.22, y: Y(q3), width: bw * 0.44, height: Math.max(1, Y(q1) - Y(q3)), fill: c, 'fill-opacity': .30, stroke: c }, g);
    el('line', { x1: cx - bw * 0.22, x2: cx + bw * 0.22, y1: Y(md), y2: Y(md), stroke: 'var(--text)', 'stroke-width': 2 }, g);
    r.style.cursor = 'help';
    hover(r, `<b>${esc(grp.label)}</b><br>median ${md.toFixed(4)}<br>IQR ${q1.toFixed(4)} \u2013 ${q3.toFixed(4)}<br>n=${v.length}`);
  });
  return s;
}

/* ---- multi-series line chart, for the per-frame traces ---- */
function lines(series, opts) {
  // series: {label, values, color, dash}
  const o = Object.assign({ w: 700, h: 170, padL: 46, padR: 10, padT: 16, padB: 26, hline: null, hlabel: '', logy: false, ylabel: '' }, opts);
  const s = svgRoot(o.w, o.h);
  const g = el('g', {}, s);
  const flat = series.flatMap(x => x.values || []).filter(v => v !== null && v !== undefined && isFinite(v) && (!o.logy || v > 0));
  if (!flat.length) { txt(g, o.w / 2, o.h / 2, 'no data', { 'text-anchor': 'middle' }); return s; }
  const n = Math.max(...series.map(x => (x.values || []).length));
  let lo = Math.min(...flat), hi = Math.max(...flat);
  if (o.hline !== null && o.hline !== undefined && (!o.logy || o.hline > 0)) { lo = Math.min(lo, o.hline); hi = Math.max(hi, o.hline); }
  if (!(hi > lo)) hi = lo + 1;
  const tr = v => o.logy ? Math.log10(Math.max(v, 1e-9)) : v;
  const tlo = tr(lo), thi = tr(hi), spanPad = (thi - tlo) * 0.08 || 1;
  const Y = v => o.padT + (1 - (tr(v) - tlo + spanPad) / (thi - tlo + 2 * spanPad)) * (o.h - o.padT - o.padB);
  const X = i => o.padL + (n <= 1 ? 0 : i / (n - 1) * (o.w - o.padL - o.padR));
  (o.logy ? [lo, Math.sqrt(lo * hi), hi] : niceTicks(lo, hi, 4)).forEach(t => {
    el('line', { x1: o.padL, x2: o.w - o.padR, y1: Y(t), y2: Y(t), class: 'gridline' }, g);
    txt(g, o.padL - 5, Y(t) + 3, o.logy ? t.toPrecision(2) : String(+t.toFixed(4)), { 'text-anchor': 'end' });
  });
  if (o.hline !== null && o.hline !== undefined) {
    el('line', { x1: o.padL, x2: o.w - o.padR, y1: Y(o.hline), y2: Y(o.hline), stroke: '#c8922a', 'stroke-width': 1.2, 'stroke-dasharray': '4 3' }, g);
    txt(g, o.padL + 4, Y(o.hline) - 3, o.hlabel, { fill: '#c8922a' });
  }
  series.forEach(ser => {
    let d = '', pen = false;
    (ser.values || []).forEach((v, i) => {
      if (v === null || v === undefined || !isFinite(v) || (o.logy && v <= 0)) { pen = false; return; }
      d += (pen ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1) + ' ';
      pen = true;
    });
    el('path', { d, fill: 'none', stroke: ser.color || 'var(--accent)', 'stroke-width': 1.2, 'stroke-dasharray': ser.dash || null, 'stroke-opacity': .92 }, g);
  });
  if (o.ylabel) txt(g, 3, o.padT - 5, o.ylabel, { class: 'lbl' });
  txt(g, o.w - o.padR, o.h - 4, 'index within clip', { 'text-anchor': 'end' });
  return s;
}

/* ---- heatmap grid, for the model x method matrix ---- */
function heat(cells, rowsLbl, colsLbl, opts) {
  // cells[r][c] = {v, n, text, tip, blocked}
  const o = Object.assign({ cw: 62, ch: 19, padL: 150, padT: 34, ramp: ['#1f3b57', '#2f7d4f', '#d9c04a'] }, opts);
  const w = o.padL + colsLbl.length * o.cw + 10, h = o.padT + rowsLbl.length * o.ch + 10;
  const s = svgRoot(w, h);
  const g = el('g', {}, s);
  const nums = cells.flat().map(c => c && c.v).filter(v => v !== null && v !== undefined && isFinite(v));
  const lo = nums.length ? Math.min(...nums) : 0, hi = nums.length ? Math.max(...nums) : 1;
  const ramp = t => {
    const a = o.ramp, x = Math.max(0, Math.min(1, t)) * (a.length - 1), i = Math.min(a.length - 2, Math.floor(x)), f = x - i;
    const hx = c => [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
    const A = hx(a[i]), B = hx(a[i + 1]);
    return `rgb(${A.map((v, k) => Math.round(v + (B[k] - v) * f)).join(',')})`;
  };
  colsLbl.forEach((cl, c) => {
    const t = txt(g, o.padL + c * o.cw + o.cw / 2, o.padT - 8, cl, { 'text-anchor': 'start', class: 'lbl' });
    t.setAttribute('transform', `rotate(-38 ${o.padL + c * o.cw + o.cw / 2} ${o.padT - 8})`);
  });
  rowsLbl.forEach((rl, r) => {
    txt(g, o.padL - 7, o.padT + r * o.ch + o.ch / 2 + 3.2, rl, { 'text-anchor': 'end', class: 'lbl' });
    colsLbl.forEach((_, c) => {
      const cell = (cells[r] || [])[c];
      const x = o.padL + c * o.cw, y = o.padT + r * o.ch;
      if (!cell) {
        el('rect', { x, y, width: o.cw - 1, height: o.ch - 1, fill: 'none', stroke: 'var(--border)' }, g);
        return;
      }
      const has = cell.v !== null && cell.v !== undefined && isFinite(cell.v);
      const rect = el('rect', {
        x, y, width: o.cw - 1, height: o.ch - 1,
        fill: has && !cell.blocked ? ramp(hi > lo ? (cell.v - lo) / (hi - lo) : 0.5) : 'var(--panel-2)',
        stroke: 'var(--panel)', 'fill-opacity': cell.blocked ? 0.5 : 1,
      }, g);
      const t = txt(g, x + o.cw / 2 - 0.5, y + o.ch / 2 + 3.2, cell.text || '\u2014',
        { 'text-anchor': 'middle', 'font-size': 9 });
      if (has && !cell.blocked) t.setAttribute('fill', '#f2f2f2');
      if (cell.tip) { rect.style.cursor = 'help'; hover(rect, cell.tip); }
    });
  });
  return s;
}

/* ---- scatter, for the confound / Pareto panels ---- */
function scatter(points, opts) {
  // points: {x, y, r, color, tip, filled}
  const o = Object.assign({ w: 330, h: 240, padL: 46, padR: 12, padT: 12, padB: 34, xlabel: '', ylabel: '', trend: null, vline: null, hline: null }, opts);
  const s = svgRoot(o.w, o.h);
  const g = el('g', {}, s);
  if (!points.length) { txt(g, o.w / 2, o.h / 2, 'too few clips', { 'text-anchor': 'middle' }); return s; }
  const q = (a, p) => { const v = a.slice().sort((x, y) => x - y); return v[Math.min(v.length - 1, Math.floor(p * v.length))]; };
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  let x0 = q(xs, 0.005), x1 = q(xs, 0.99), y0 = q(ys, 0.01), y1 = q(ys, 0.99);
  if (o.vline !== null && o.vline !== undefined) { x0 = Math.min(x0, o.vline); x1 = Math.max(x1, o.vline); }
  if (o.hline !== null && o.hline !== undefined) { y0 = Math.min(y0, o.hline); y1 = Math.max(y1, o.hline); }
  if (!(x1 > x0)) x1 = x0 + 1;
  if (!(y1 > y0)) y1 = y0 + 1;
  const px = (x1 - x0) * .06, py = (y1 - y0) * .08;
  x0 -= px; x1 += px; y0 -= py; y1 += py;
  const X = v => o.padL + (v - x0) / (x1 - x0) * (o.w - o.padL - o.padR);
  const Y = v => o.padT + (1 - (v - y0) / (y1 - y0)) * (o.h - o.padT - o.padB);
  niceTicks(y0, y1, 4).forEach(t => {
    el('line', { x1: o.padL, x2: o.w - o.padR, y1: Y(t), y2: Y(t), class: 'gridline' }, g);
    txt(g, o.padL - 5, Y(t) + 3, String(+t.toFixed(3)), { 'text-anchor': 'end' });
  });
  niceTicks(x0, x1, 4).forEach(t => txt(g, X(t), o.h - o.padB + 12, String(+t.toFixed(2)), { 'text-anchor': 'middle' }));
  if (o.hline !== null && o.hline !== undefined) el('line', { x1: o.padL, x2: o.w - o.padR, y1: Y(o.hline), y2: Y(o.hline), class: 'zero' }, g);
  if (o.vline !== null && o.vline !== undefined) el('line', { x1: X(o.vline), x2: X(o.vline), y1: o.padT, y2: o.h - o.padB, class: 'zero', 'stroke-dasharray': '3 3' }, g);
  points.forEach(p => {
    const c = el('circle', {
      cx: X(p.x), cy: Y(p.y), r: p.r || 3, fill: p.filled === false ? 'var(--panel)' : (p.color || 'var(--accent)'),
      stroke: p.color || 'var(--accent)', 'stroke-width': 1.1, 'fill-opacity': p.opacity === undefined ? .7 : p.opacity,
    }, g);
    if (p.tip) { c.style.cursor = 'help'; hover(c, p.tip); }
  });
  if (o.trend && o.trend.length > 1) {
    let d = '';
    o.trend.forEach((t, i) => { d += (i ? 'L' : 'M') + X(t[0]).toFixed(1) + ' ' + Y(t[1]).toFixed(1) + ' '; });
    el('path', { d, fill: 'none', stroke: 'var(--text)', 'stroke-width': 1.6, 'stroke-opacity': .85 }, g);
  }
  txt(g, (o.padL + o.w) / 2, o.h - 3, o.xlabel, { 'text-anchor': 'middle' });
  if (o.ylabel) txt(g, 2, o.padT - 3, o.ylabel, { class: 'lbl' });
  return s;
}

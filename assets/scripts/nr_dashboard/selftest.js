/* Smoke test for the NR dashboard: does the client-side re-medianing agree with the
 * Python pipeline, and does every tab renderer run without throwing?
 *
 * There is no headless browser on this box, so the DOM is stubbed just deeply enough for
 * the renderers to execute. That is enough to catch the two failure modes that matter:
 *   1. the client's median / delta pairing disagreeing with `table_dense_nr` (a wrong
 *      number is worse than a broken page);
 *   2. a renderer throwing on real data.
 *
 * Usage: node assets/scripts/nr_dashboard/selftest.js <built .html> <cache agg.json>
 */
'use strict';
const fs = require('fs');
const path = require('path');

const htmlPath = process.argv[2];
const aggPath = process.argv[3];
if (!htmlPath || !aggPath) {
  console.error('usage: node selftest.js <dashboard.html> <agg.json>');
  process.exit(2);
}
const html = fs.readFileSync(htmlPath, 'utf8');
const agg = JSON.parse(fs.readFileSync(aggPath, 'utf8'));

/* ---- minimal DOM ---- */
function makeNode(tag) {
  const n = {
    tagName: tag, children: [], attributes: {}, style: {}, dataset: {},
    textContent: '', _html: '',
    get innerHTML() { return this._html; },
    set innerHTML(v) { this._html = String(v); },
    setAttribute(k, v) { this.attributes[k] = String(v); },
    getAttribute(k) { return this.attributes[k] === undefined ? null : this.attributes[k]; },
    appendChild(c) { this.children.push(c); return c; },
    insertAdjacentHTML(_, h) { this._html += h; },
    addEventListener() { },
    querySelector() { return makeNode('div'); },
    querySelectorAll() { return []; },
    classList: { add() { }, remove() { }, toggle() { } },
    get offsetWidth() { return 100; }, get offsetHeight() { return 40; },
  };
  return n;
}
const registry = new Map();
global.document = {
  createElement: makeNode,
  createElementNS: (_, tag) => makeNode(tag),
  addEventListener(ev, fn) { if (ev === 'DOMContentLoaded') global.__ready = fn; },
  documentElement: makeNode('html'),
  body: makeNode('body'),
  querySelector(sel) {
    if (!registry.has(sel)) registry.set(sel, makeNode('div'));
    return registry.get(sel);
  },
};
global.window = global;
global.innerWidth = 1600; global.innerHeight = 900;
global.performance = { now: () => Date.now() };
global.matchMedia = () => ({ matches: false });
global.localStorage = { getItem: () => null, setItem: () => { } };

/* ---- load the four scripts out of the built HTML (proves what shipped works) ---- */
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
if (scripts.length < 5) { console.error(`expected 5 inline scripts, found ${scripts.length}`); process.exit(1); }
const src = scripts.join('\n;\n').replace(/<\\\//g, '</');
const run = new Function(`${src}\n; return {CLIPS, COL, KEYS, median, bootCI, wilcoxonP, deltas, vals, state, refilter, apply, resolved, TABS, DATA, FIELDS};`);
let api;
try { api = run(); } catch (e) { console.error('script load FAILED:', e.message); process.exit(1); }
console.log(`loaded: ${api.CLIPS.length.toLocaleString()} clips, ${api.KEYS.length} metrics`);

let fails = 0;
const near = (a, b, tol, what) => {
  if (a === null || b === null || a === undefined || b === undefined) {
    console.log(`  SKIP ${what} (a=${a} b=${b})`); return;
  }
  const d = Math.abs(a - b);
  const scale = Math.max(1e-9, Math.abs(b));
  const ok = d <= tol || d / scale <= tol;
  if (!ok) { console.error(`  FAIL ${what}: js=${a} py=${b} (diff ${d})`); fails++; }
  else console.log(`  ok   ${what}: ${a.toFixed(4)} vs ${b.toFixed(4)}`);
};

/* ---- 1. absolute medians must match the Python descriptive table ---- */
console.log('\n[1] absolute medians vs t1_descriptive');
api.refilter();
let checked1 = 0;
for (const row of agg.t1_descriptive) {
  if (checked1 >= 8) break;
  // t1 rows are keyed (model, embodiment, camera_mount) and cover BOTH gt and pred
  // clips of that stratum; `method` is not a column there.
  const sub = api.CLIPS.filter(c => c.model === row.model
    && c.embodiment === row.embodiment && c.camera_mount === row.camera_mount);
  if (!sub.length) { console.error(`  FAIL no clips for ${row.model}/${row.embodiment}/${row.camera_mount}`); fails++; continue; }
  for (const key of api.KEYS) {
    const py = row[`${key}_median`];
    if (py === null || py === undefined) continue;
    const i = api.COL[key].idx;
    near(api.median(sub.map(c => c.v[i])), py, 2e-3,
      `${row.model}/${row.embodiment}/${row.camera_mount} ${key}`);
    checked1++;
    break;
  }
}
if (!checked1) { console.error('  FAIL nothing was compared; the t1 schema changed'); fails++; }

/* ---- 2. paired accel deltas must match t3b_accel_pooled ---- */
console.log('\n[2] pooled accel deltas vs t3b_accel_pooled');
for (const row of agg.t3b_accel_pooled) {
  for (const key of api.KEYS) {
    const pyMed = row[`${key}_delta_median`], pyN = row[`${key}_n_pairs`];
    if (pyMed === null || pyMed === undefined || !pyN) continue;
    const d = api.deltas(key, row.accel_method);
    if (d.length !== pyN) {
      console.error(`  FAIL ${row.accel_method} ${key}: n js=${d.length} py=${pyN}`);
      fails++;
    } else {
      near(api.median(d), pyMed, 2e-3, `${row.accel_method} ${key} (n=${pyN})`);
    }
    break;
  }
}

/* ---- 3. Wilcoxon p should land in the same decade as Python's ---- */
console.log('\n[3] Wilcoxon p order of magnitude');
for (const row of agg.t3b_accel_pooled.slice(0, 4)) {
  const key = api.KEYS.find(k => row[`${k}_p`] !== null && row[`${k}_p`] !== undefined);
  if (!key) continue;
  const p = api.wilcoxonP(api.deltas(key, row.accel_method));
  const py = row[`${key}_p`];
  if (p === null) continue;
  // Python's scipy-free implementation and this one are both normal approximations, so
  // they agree to within a decade; below 1e-12 both are "astronomically significant"
  // and the exact exponent is meaningless (JS underflows to 0).
  const okDecade = py < 1e-12 ? p < 1e-8
    : Math.abs(Math.log10(Math.max(p, 1e-300)) - Math.log10(Math.max(py, 1e-300))) < 1.2;
  if (!okDecade) { console.error(`  FAIL ${row.accel_method} ${key}: p js=${p} py=${py}`); fails++; }
  else console.log(`  ok   ${row.accel_method} ${key}: p js=${p.toExponential(1)} py=${py.toExponential(1)}`);
}

/* ---- 4. bootstrap CI should bracket the median and roughly match Python's ---- */
console.log('\n[4] bootstrap CI sanity');
for (const row of agg.t3b_accel_pooled.slice(0, 3)) {
  const key = api.KEYS.find(k => row[`${k}_ci_lo`] !== null && row[`${k}_ci_lo`] !== undefined);
  if (!key) continue;
  const d = api.deltas(key, row.accel_method);
  const [lo, hi] = api.bootCI(d, 11);
  const md = api.median(d);
  if (!(lo <= md && md <= hi)) { console.error(`  FAIL CI does not bracket median: ${lo} ${md} ${hi}`); fails++; continue; }
  const pyW = row[`${key}_ci_hi`] - row[`${key}_ci_lo`], jsW = hi - lo;
  const ratio = jsW / (pyW || 1e-9);
  if (ratio < 0.5 || ratio > 2.0) { console.error(`  FAIL CI width ${row.accel_method} ${key}: js=${jsW} py=${pyW}`); fails++; }
  else console.log(`  ok   ${row.accel_method} ${key}: width js=${jsW.toFixed(4)} py=${pyW.toFixed(4)}`);
}

/* ---- 5. every tab renderer must run, under no filter and under a narrow one ---- */
console.log('\n[5] tab renderers');
// One metric per STATUS class, not all six: the statuses are what change the code path
// (a DROPPED column takes the refusal branch, a VALID one the full render), and the full
// 6x6x3 matrix is minutes of bootstrapping for no extra coverage.
const byStatus = new Map();
api.KEYS.forEach(k => { if (!byStatus.has(api.COL[k].status)) byStatus.set(api.COL[k].status, k); });
const probeKeys = [...byStatus.values()];
console.log(`  probing metrics: ${probeKeys.map(k => `${k} [${api.COL[k].status}]`).join(', ')}`);
const scenarios = [['no filter', () => { }],
['wrist only', () => api.state.sel.camera_mount.add('wrist')],
['one method', () => api.state.sel.method.add('itm')]];
for (const [name, setup] of scenarios) {
  api.FIELDS.forEach(f => api.state.sel[f].clear());
  setup();
  api.refilter();
  const t0 = Date.now();
  for (const [id, , fn] of api.TABS) {
    for (const key of probeKeys) {
      api.state.metric = key;
      try { fn(); } catch (e) {
        console.error(`  FAIL [${name}] tab ${id} metric ${key}: ${e.message}`);
        fails++;
      }
    }
  }
  console.log(`  ok   [${name}] ${api.TABS.length} tabs x ${probeKeys.length} statuses in ${Date.now() - t0} ms`);
}

/* ---- 6. the refusals must actually refuse ---- */
console.log('\n[6] policy refusals');
const dropped = api.KEYS.filter(k => api.COL[k].status === 'DROPPED');
const ceiling = api.KEYS.filter(k => api.COL[k].status === 'CEILING');
const pairedOnly = api.KEYS.filter(k => api.COL[k].status === 'PAIRED_ONLY');
if (!dropped.length) { console.error('  FAIL no DROPPED column found; the status wiring is broken'); fails++; }
else console.log(`  ok   DROPPED: ${dropped.join(', ')}`);
if (!ceiling.length) { console.error('  FAIL no CEILING column found'); fails++; }
else console.log(`  ok   CEILING: ${ceiling.join(', ')}`);
if (!pairedOnly.length) { console.error('  FAIL no PAIRED_ONLY column found'); fails++; }
else console.log(`  ok   PAIRED_ONLY: ${pairedOnly.join(', ')}`);
for (const k of [...dropped, ...ceiling]) {
  api.state.metric = k;
  api.refilter();
  const node = document.querySelector('#models-body');
  node._html = '';
  api.TABS.find(t => t[0] === 'models')[2]();
  if (!/not shown for/.test(node.innerHTML)) {
    console.error(`  FAIL models tab did not refuse to rank on ${k} (status ${api.COL[k].status})`);
    fails++;
  } else console.log(`  ok   models tab refuses to rank on ${k}`);
}

console.log(fails ? `\n${fails} CHECK(S) FAILED` : '\nall checks passed');
process.exit(fails ? 1 : 0);

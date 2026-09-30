/* NR metric dashboard. DATA is injected above this script as a global.
 *
 * Core invariant: every number on screen is a median of per-clip GATED values --
 * recomputed from CLIPS under the current filter, never an average of
 * precomputed medians. That is why filtering is trustworthy: narrowing to
 * "wrist only" gives the same number the paper pipeline would give for wrist only.
 */
'use strict';

const KEYS = DATA.columns.map(c => c.key);
const COL = {}; DATA.columns.forEach((c, i) => { COL[c.key] = { ...c, idx: i }; });
const POLICY = DATA.policy;
const MIN_N = POLICY.min_pairs_for_claim;
const SS = DATA.status_style;
const L = DATA.labels;

const lbl = (kind, v) => (L[kind] && L[kind][v]) || v || '?';
const famColor = m => (m === 'dense' || m === 'gt') ? L.family_color[m]
  : (L.family_color[L.family[m]] || '#888');
const ordered = (vals, order) => vals.slice().sort(
  (a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99) || String(a).localeCompare(b));

/* ---------------- rehydrate the interned clip rows -------------------------------- */
/* The payload interns every repeated string (strata values, gate reasons, captions) to
 * keep the single-file HTML about a third of its naive size. Reversing it once here means
 * the rest of the code sees plain objects and never has to know. */
const CLIPS = (() => {
  const fields = DATA.clip_fields;          // position 0 is 'id'
  const tabs = DATA.tables, reasons = DATA.reasons;
  const nk = DATA.columns.length;
  const ZERO = new Array(nk).fill(-1);
  return DATA.clips.map(row => {
    const o = { id: row[0] };
    for (let i = 1; i < fields.length; i++) {
      const f = fields[i], ix = row[i];
      o[f] = ix < 0 ? null : tabs[f][ix];
    }
    o.v = row[fields.length];
    const gr = row[fields.length + 1];
    o.gr = gr === 0 ? ZERO : gr;            // 0 = nothing gated out on this clip
    o.reasonOf = k => {                     // index -> the verbatim sentence
      const ix = o.gr[k];
      return ix < 0 ? null : reasons[ix];
    };
    return o;
  });
})();

/* ---------------- statistics (must match assets/scripts/table_dense_nr.py) --------- */

function median(xs) {
  const v = xs.filter(x => x !== null && x !== undefined).sort((a, b) => a - b);
  if (!v.length) return null;
  const n = v.length;
  return n % 2 ? v[(n - 1) / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* In-place Hoare selection: the k-th smallest of a typed array, O(n) and allocation-free.
 * The bootstrap below calls this 600 times per CI on arrays up to ~30k long; doing it
 * with sort() instead cost ~100 ms per CI and made the Overview tab take a minute. */
function selectKth(a, n, k) {
  let lo = 0, hi = n - 1;
  while (lo < hi) {
    const p = a[(lo + hi) >> 1];
    let i = lo, j = hi;
    while (i <= j) {
      while (a[i] < p) i++;
      while (a[j] > p) j--;
      if (i <= j) { const t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
    }
    if (k <= j) hi = j; else if (k >= i) lo = i; else return a[k];
  }
  return a[lo];
}

/* Percentile bootstrap CI for the median. B is deliberately lower than the paper's
 * 10000: this runs on every filter change, and 600 resamples already settle the 2.5/97.5
 * percentiles to well within the printed precision. The number is shown in the UI so
 * nobody mistakes a dashboard CI for the published one. */
const BOOT_B = 600;
/* Above this n the percentile bootstrap is replaced by the exact distribution-free
 * order-statistic CI for the median (a binomial interval, normal-approximated for the
 * index). For n in the thousands the two agree far beyond the printed precision, but the
 * bootstrap costs 600 resamples of n while this costs one sort -- the difference between
 * a 4-second Overview tab and a 90-millisecond one. Panels state which was used. */
const BOOT_EXACT_ABOVE = 2000;
function bootCI(xs, seed) {
  let n = 0;
  const v = new Float64Array(xs.length);
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i];
    if (x !== null && x !== undefined) v[n++] = x;
  }
  if (n < 3) return [null, null];
  if (n > BOOT_EXACT_ABOVE) return orderStatCI(v, n);
  let s = (seed || 12345) >>> 0;
  const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  const meds = new Float64Array(BOOT_B);
  const buf = new Float64Array(n);
  const even = n % 2 === 0, k = even ? n / 2 : (n - 1) / 2;
  for (let b = 0; b < BOOT_B; b++) {
    for (let i = 0; i < n; i++) buf[i] = v[(rnd() * n) | 0];
    // Same median convention as median(): mean of the two middles when n is even.
    const hi = selectKth(buf, n, k);
    meds[b] = even ? 0.5 * (hi + selectKth(buf, n, k - 1)) : hi;
  }
  meds.sort();
  return [meds[Math.floor(0.025 * BOOT_B)], meds[Math.min(BOOT_B - 1, Math.floor(0.975 * BOOT_B))]];
}

/* 95% CI for the median from the order statistics: the rank window is
 * n/2 +- 1.96*sqrt(n)/2, which is the normal approximation to the exact binomial
 * interval. Assumes nothing about the distribution -- appropriate here, where the
 * per-clip metric distributions are skewed and bounded. */
function orderStatCI(v, n) {
  const arr = v.length === n ? v : v.subarray(0, n);
  const half = 0.5 * n, w = 0.98 * Math.sqrt(n);   // 1.96 * sqrt(n) / 2
  const lo = Math.max(0, Math.floor(half - w));
  const hi = Math.min(n - 1, Math.ceil(half + w));
  return [selectKth(arr, n, lo), selectKth(arr, n, hi)];
}
/* Which method a panel's CI came from, so the caption can say so. */
const ciKind = n => n > BOOT_EXACT_ABOVE
  ? 'exact order-statistic interval'
  : `${BOOT_B}-resample percentile bootstrap`;

/* Two-sided Wilcoxon signed-rank, normal approximation with tie correction --
 * the same test and the same zero-handling as `table_dense_nr.wilcoxon_p`. */
function wilcoxonP(diffs) {
  const d = diffs.filter(x => x !== null && x !== undefined && x !== 0);
  const n = d.length;
  if (n < 6) return null;
  const abs = d.map(Math.abs);
  const idx = abs.map((_, i) => i).sort((a, b) => abs[a] - abs[b]);
  const rank = new Array(n);
  let i = 0, tieSum = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && abs[idx[j + 1]] === abs[idx[i]]) j++;
    const avg = (i + j) / 2 + 1, t = j - i + 1;
    for (let k = i; k <= j; k++) rank[idx[k]] = avg;
    if (t > 1) tieSum += t * t * t - t;
    i = j + 1;
  }
  let wPlus = 0;
  for (let k = 0; k < n; k++) if (d[k] > 0) wPlus += rank[k];
  const mean = n * (n + 1) / 4;
  const sd = Math.sqrt((n * (n + 1) * (2 * n + 1) - tieSum / 2) / 24);
  if (!sd) return null;
  const z = (wPlus - mean) / sd;
  return 2 * (1 - normCdf(Math.abs(z)));
}
function normCdf(z) {
  const t = 1 / (1 + 0.2316419 * z);
  const p = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return 1 - 0.3989422804014327 * Math.exp(-z * z / 2) * p;
}

/* Does a delta support a claim? Mirrors `resolved_flag`: the CI must exclude zero, the
 * sample must reach MIN_N, Wilcoxon must agree, and the metric must not be one whose
 * positive direction is a known artifact. */
function resolved(key, n, lo, hi, p, deltaMed) {
  if (n === null || n < MIN_N || lo === null || hi === null) return false;
  if (!(lo > 0 || hi < 0)) return false;
  if (p !== null && p !== undefined && p >= POLICY.alpha) return false;
  if (POLICY.no_improvement_vs_dense && POLICY.no_improvement_vs_dense[key] && deltaMed > 0) return false;
  return true;
}
function artifactPositive(key, n, lo, p) {
  return !!(POLICY.no_improvement_vs_dense && POLICY.no_improvement_vs_dense[key]
    && n >= MIN_N && lo !== null && lo > 0
    && (p === null || p === undefined || p < POLICY.alpha));
}

/* ---------------- what each status permits ---------------------------------------- */

const CAN_RANK_MODELS = st => st === 'VALID' || st === 'DESCRIPTIVE';
const CAN_SHOW_ABSOLUTE = st => st !== 'PAIRED_ONLY';
const IS_RESULT = st => st !== 'DROPPED' && st !== 'DIAGNOSTIC';

function chip(key) {
  const st = COL[key].status, s = SS[st];
  return `<span class="chip" style="background:${s.color}" title="${esc(s.caption)}">${s.short}</span>`;
}
function esc(s) {
  return String(s === null || s === undefined ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function fmt(key, v, extraDigits) {
  if (v === null || v === undefined) return '\u2014';
  return v.toFixed(COL[key].digits + (extraDigits || 0));
}
function fmtSigned(key, v) {
  if (v === null || v === undefined) return '\u2014';
  return (v > 0 ? '+' : '') + v.toFixed(COL[key].digits);
}
function fmtP(p) {
  if (p === null || p === undefined) return '\u2014';
  return p < 0.001 ? p.toExponential(0) : p.toFixed(3);
}

/* ---------------- global filter --------------------------------------------------- */

const FIELDS = ['embodiment', 'model', 'split', 'camera_mount', 'method'];
const state = {
  tab: 'overview',
  metric: KEYS.find(k => COL[k].status === 'VALID') || KEYS[0],
  sel: {},                      // field -> Set of allowed values (empty = all)
  domain: {},                   // field -> every value present in the cache
};
FIELDS.forEach(f => {
  state.domain[f] = ordered([...new Set(CLIPS.map(c => c[f]).filter(v => v))],
    L[f + '_order'] || L[({ camera_mount: 'mount', embodiment: 'emb' })[f] + '_order'] || []);
  state.sel[f] = new Set();
});

function passes(c) {
  for (const f of FIELDS) {
    const s = state.sel[f];
    if (s.size && !s.has(c[f])) return false;
  }
  return true;
}
let FILTERED = CLIPS;
function refilter() { FILTERED = CLIPS.filter(passes); }

/* Memo keyed by the filter signature. Re-selecting the same tab, or toggling the theme,
 * must not recompute a 2-second correlation grid; changing any filter must. */
function filterSig() {
  return FIELDS.map(f => [...state.sel[f]].sort().join(',')).join('|');
}
const MEMO = new Map();
function memo(tag, fn) {
  const k = tag + '#' + filterSig();
  if (MEMO.has(k)) return MEMO.get(k);
  const v = fn();
  if (MEMO.size > 60) MEMO.clear();     // bounded: the payload is already 5 MB of heap
  MEMO.set(k, v);
  return v;
}

/* Values of `key` among the filtered clips, optionally narrowed further. */
function vals(key, extra) {
  const i = COL[key].idx, out = [];
  for (const c of FILTERED) {
    if (extra && !extra(c)) continue;
    const v = c.v[i];
    if (v !== null && v !== undefined) out.push(v);
  }
  return out;
}

/* Paired accel-minus-dense deltas, recomputed from the FILTERED clips.
 *
 * `comparablePair` is a direct port of `table_dense_nr.comparable_pair` and is NOT
 * optional: the dense and accel trees of dreamgen/single_arm were rendered with different
 * rollout configs (277 frames @16fps vs 93 @10fps), so pairing on episode+view alone
 * invents a +10 AMC improvement for every method in that cell. The gate rejects ~12% of
 * candidate pairs, concentrated there.
 *
 * The dense partner is looked up in the UNFILTERED index on purpose: filtering to
 * "method = itm" must not destroy the dense clip that itm is measured against. */
const DENSE_INDEX = (() => {
  const m = new Map();
  for (const c of CLIPS) if (c.method === 'dense') m.set(c.pair_key + '\u0000' + c.model + '\u0000' + c.view, c);
  return m;
})();
function comparablePair(a, b) {
  if (a.source_fps !== b.source_fps) return false;
  if (a.effective_fps !== b.effective_fps) return false;
  const na = a.n_sampled || 0, nb = b.n_sampled || 0;
  if (!na || !nb) return false;
  return Math.abs(na - nb) / Math.max(na, nb) <= POLICY.gates.max_length_mismatch;
}
function denseOf(c) {
  const base = DENSE_INDEX.get(c.pair_key + '\u0000' + c.model + '\u0000' + c.view);
  return (base && comparablePair(c, base)) ? base : null;
}
function deltas(key, method, extra) {
  const i = COL[key].idx, out = [];
  for (const c of FILTERED) {
    if (c.method !== method) continue;
    if (extra && !extra(c)) continue;
    const base = denseOf(c);
    if (!base) continue;
    const a = base.v[i], b = c.v[i];
    if (a === null || a === undefined || b === null || b === undefined) continue;
    out.push(b - a);
  }
  return out;
}

/* One pass over FILTERED that buckets deltas by an arbitrary key.
 *
 * The per-cell views need ~150 cells x 9 methods of deltas; calling `deltas()` for each
 * would rescan all 39k clips 1350 times (~50M iterations, several seconds). Bucketing in
 * a single pass makes the same view ~40 ms. */
function deltasBy(key, keyOf, tag) {
  if (tag) return memo(`dby:${key}:${tag}`, () => deltasByRaw(key, keyOf));
  return deltasByRaw(key, keyOf);
}
function deltasByRaw(key, keyOf) {
  const i = COL[key].idx, out = new Map();
  for (const c of FILTERED) {
    const m = c.method;
    if (!m || m === 'gt' || m === 'dense') continue;
    const base = denseOf(c);
    if (!base) continue;
    const a = base.v[i], b = c.v[i];
    if (a === null || a === undefined || b === null || b === undefined) continue;
    const k = keyOf(c);
    if (k === null || k === undefined) continue;
    let arr = out.get(k);
    if (!arr) { arr = []; out.set(k, arr); }
    arr.push(b - a);
  }
  return out;
}

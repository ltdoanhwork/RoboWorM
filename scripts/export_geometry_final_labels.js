#!/usr/bin/env node
/**
 * Export the FINAL geometry human scores (post conflict-resolution) to JSON.
 *
 * The resolution logic is not reimplemented here: this loads the deployed
 * webpage/labeling/vgp.js and calls VGP.resolveHuman(), so the export can never
 * drift from what the site itself shows.
 *
 * Resolution order for a geometry clip (see resolveUnit in vgp.js):
 *   raters   - both regular raters gave the same score          -> that score
 *   arbiter  - they disagreed and thanh_check scored the clip   -> arbiter score
 *   checkers - they disagreed, two checkers agree               -> checker score
 *   checker  - they disagreed, a single checker scored it       -> checker score
 *   (null)   - still unresolved (fewer than 2 raters, or a conflict nobody settled)
 *
 * Usage:
 *   node scripts/export_geometry_final_labels.js [-o OUT.json] [--include-unresolved]
 */
const fs = require('fs');
const path = require('path');
const https = require('https');

const REPO = path.resolve(__dirname, '..');
// The labeling site lives on `main`, which this repo checks out as the
// `.main-edit` worktree; the feature branch still carries an old 8KB stub of
// vgp.js without the resolver. So pick whichever copy actually exports
// resolveHuman rather than trusting a fixed path. Override with VGP_SITE=<dir>.
const CANDIDATES = [
  process.env.VGP_SITE,
  path.join(REPO, '.main-edit', 'webpage', 'labeling'),
  path.join(REPO, 'webpage', 'labeling'),
].filter(Boolean);
const ROOT = CANDIDATES.find((dir) => {
  const f = path.join(dir, 'vgp.js');
  return fs.existsSync(f) && /resolveHuman\s*,/.test(fs.readFileSync(f, 'utf8'));
});
if (!ROOT) {
  console.error('FAILED: no vgp.js exporting resolveHuman found in:\n  ' + CANDIDATES.join('\n  '));
  process.exit(1);
}

const argv = process.argv.slice(2);
const outArg = argv.indexOf('-o') >= 0 ? argv[argv.indexOf('-o') + 1]
             : argv.indexOf('--out') >= 0 ? argv[argv.indexOf('--out') + 1] : null;
const includeUnresolved = argv.includes('--include-unresolved');
const OUT = outArg || path.join(REPO, 'sampling_dataset', 'data_for_web', 'geometry_final_labels.json');

// --- browser-ish globals so vgp.js loads outside a page -----------------------
global.window = global;
global.localStorage = {
  _d: {}, getItem(k) { return this._d[k] ?? null; },
  setItem(k, v) { this._d[k] = String(v); }, removeItem(k) { delete this._d[k]; },
};
global.document = { addEventListener() {}, querySelectorAll() { return []; } };
global.location = { search: '', pathname: '/', href: 'http://localhost/' };
// Node 16 has no global fetch.
if (typeof global.fetch !== 'function') {
  global.fetch = (url, opts = {}) => new Promise((resolve, reject) => {
    const req = https.request(url, { method: opts.method || 'GET', headers: opts.headers || {} }, (res) => {
      let body = '';
      res.on('data', (c) => { body += c; });
      res.on('end', () => resolve({
        ok: res.statusCode >= 200 && res.statusCode < 300,
        status: res.statusCode,
        headers: { get: (k) => res.headers[String(k).toLowerCase()] || null },
        json: async () => JSON.parse(body),
        text: async () => body,
      }));
    });
    req.on('error', reject);
    if (opts.body) req.write(opts.body);
    req.end();
  });
}

// vgp.js declares `const VGP` at top level, so require() would not expose it.
const VGP = new Function(
  fs.readFileSync(path.join(ROOT, 'vgp.js'), 'utf8') + '\nreturn VGP;'
)();

const CFG = new Function(
  fs.readFileSync(path.join(ROOT, 'config.js'), 'utf8') + '\nreturn CONFIG;'
)();

/** PostgREST caps a response at max-rows (1000); page until content-range is exhausted. */
async function selectAll(table, cols) {
  const out = [];
  let from = 0;
  for (;;) {
    const res = await fetch(`${CFG.SUPABASE_URL}/rest/v1/${table}?select=${cols}`, {
      headers: {
        apikey: CFG.SUPABASE_PUBLISHABLE_KEY,
        Authorization: `Bearer ${CFG.SUPABASE_PUBLISHABLE_KEY}`,
        Range: `${from}-${from + 999}`,
        Prefer: 'count=exact',
      },
    });
    if (!res.ok) throw new Error(`${table}: HTTP ${res.status} ${await res.text()}`);
    const rows = await res.json();
    out.push(...rows);
    const total = Number((res.headers.get('content-range') || '').split('/')[1] || 0);
    if (!rows.length || out.length >= total) break;
    from += rows.length;
  }
  return out;
}

(async () => {
  const rows = await selectAll('labels', 'item_id,rater_name,score,updated_at');

  const catalog = JSON.parse(fs.readFileSync(path.join(ROOT, 'catalog.json'), 'utf8'));
  const items = Array.isArray(catalog) ? catalog : catalog.items;
  const metrics = JSON.parse(fs.readFileSync(path.join(ROOT, 'metrics.multiview.json'), 'utf8'));
  const byId = new Map((metrics.items || []).map((m) => [m.id, m]));

  const resolved = VGP.resolveHuman(rows, items);

  const clips = [];
  const tally = { raters: 0, arbiter: 0, checkers: 0, checker: 0, unresolved: 0 };
  for (const it of items) {
    const r = resolved[it.id] || {};
    const m = byId.get(it.id) || {};
    const key = r.human_score == null ? 'unresolved' : (r.consensus || 'unresolved');
    tally[key] = (tally[key] || 0) + 1;
    if (r.human_score == null && !includeUnresolved) continue;
    clips.push({
      id: it.id,
      video_path: m.video_path ?? null,
      embodiment: it.embodiment ?? m.embodiment ?? null,
      model: it.model ?? null,
      view: it.view ?? null,
      split: it.split ?? null,
      task: it.task ?? null,
      // final human answer after conflict resolution
      human_score: r.human_score ?? null,          // 1 bad / 2 medium / 3 good
      human_rating: r.human_rating ?? null,
      consensus: r.consensus ?? null,              // raters | arbiter | checkers | checker
      conflict: !!r.conflict,
      // audit trail
      n_raters: r.n_raters ?? 0,
      rater_scores: r.rater_scores ?? {},
      checker_scores: r.checker_scores ?? {},
      arbiter_score: r.arbiter_score ?? null,
      // machine side, for correlation work
      machine_score: m.score ?? null,
      combined_score: m.combined_score ?? null,
      geometric_consistency: m.geometric_consistency ?? null,
      pose_stability: m.pose_stability ?? null,
      disjoint_frame_ratio: m.disjoint_frame_ratio ?? null,
    });
  }

  const scored = clips.filter((c) => c.human_score != null);
  const dist = scored.reduce((acc, c) => { acc[c.human_score] = (acc[c.human_score] || 0) + 1; return acc; }, {});
  const payload = {
    generated_at: new Date().toISOString(),
    source: {
      labels: `${CFG.SUPABASE_URL}/rest/v1/labels`,
      catalog: 'webpage/labeling/catalog.json',
      metrics: metrics.source || 'webpage/labeling/metrics.multiview.json',
      resolver: `webpage/labeling/vgp.js (BATCH_VERSION ${VGP.BATCH_VERSION})`,
    },
    check: 'geometry',
    score_labels: { 1: 'bad', 2: 'medium', 3: 'good' },
    geo_arbiter: 'thanh_check',
    counts: {
      catalog_clips: items.length,
      exported: clips.length,
      resolved: scored.length,
      unresolved: tally.unresolved,
      by_consensus: {
        raters: tally.raters, arbiter: tally.arbiter,
        checkers: tally.checkers, checker: tally.checker,
      },
      human_score_distribution: dist,
    },
    clips,
  };

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(payload, null, 2) + '\n');

  console.log(`labels fetched      : ${rows.length}`);
  console.log(`catalog clips       : ${items.length}`);
  console.log(`resolved            : ${scored.length}`);
  console.log(`  by raters agree   : ${tally.raters}`);
  console.log(`  by geo arbiter    : ${tally.arbiter}`);
  console.log(`  by checkers (2)   : ${tally.checkers}`);
  console.log(`  by checker (1)    : ${tally.checker}`);
  console.log(`unresolved          : ${tally.unresolved}`);
  console.log(`score distribution  : ${JSON.stringify(dist)}`);
  console.log(`\nwrote ${OUT}`);
})().catch((e) => { console.error('FAILED:', e.message); process.exit(1); });

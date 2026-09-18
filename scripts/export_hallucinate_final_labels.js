#!/usr/bin/env node
/**
 * Export the FINAL hallucinate labels (post conflict-resolution) to JSON.
 *
 * Nothing is reimplemented here: this loads the deployed
 * webpage/labeling/vgp.js and calls VGP.resolveSegments() / VGP.resolveHuman(),
 * so the export cannot drift from what the labeling site itself resolves.
 *
 * Per SEGMENT (the unit that was actually labeled):
 *   raters   - both regular raters gave the same score -> that score
 *   checker  - they differed, so a checker adjudicated  -> the checker's score
 * Per VIDEO: the mean of the 64 segment flags, rounded back to 1-3.
 *
 * Score scale: humans rate QUALITY 1 Bad / 2 Medium / 3 Good. The hallucinate
 * bucket rates the AMOUNT of hallucination, so the mapping to bucket levels is
 * inverted: 3 -> low, 2 -> medium, 1 -> high. Both are emitted.
 *
 * Usage:
 *   node scripts/export_hallucinate_final_labels.js [-o OUT.json]
 */
const fs = require('fs');
const path = require('path');
const https = require('https');

const REPO = path.resolve(__dirname, '..');
// The labeling site lives on `main`, checked out as the `.main-edit` clone; the
// feature branch carries an old stub of vgp.js. Pick the copy that actually has
// the resolver. Override with VGP_SITE=<dir>.
const CANDIDATES = [
  process.env.VGP_SITE,
  path.join(REPO, '.main-edit', 'webpage', 'labeling'),
  path.join(REPO, 'webpage', 'labeling'),
].filter(Boolean);
const ROOT = CANDIDATES.find((dir) => {
  const f = path.join(dir, 'vgp.js');
  return fs.existsSync(f) && /resolveSegments\s*,/.test(fs.readFileSync(f, 'utf8'));
});
if (!ROOT) {
  console.error('FAILED: no vgp.js exporting resolveSegments found in:\n  ' + CANDIDATES.join('\n  '));
  process.exit(1);
}

const argv = process.argv.slice(2);
const outArg = argv.indexOf('-o') >= 0 ? argv[argv.indexOf('-o') + 1]
             : argv.indexOf('--out') >= 0 ? argv[argv.indexOf('--out') + 1] : null;
const OUT = outArg || path.join(REPO, 'sampling_dataset', 'data_for_web', 'hallucinate_final_labels.json');

global.window = global;
global.localStorage = {
  _d: {}, getItem(k) { return this._d[k] ?? null; },
  setItem(k, v) { this._d[k] = String(v); }, removeItem(k) { delete this._d[k]; },
};
global.document = { addEventListener() {}, querySelectorAll() { return []; } };
global.location = { search: '', pathname: '/', href: 'http://localhost/' };
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

const VGP = new Function(fs.readFileSync(path.join(ROOT, 'vgp.js'), 'utf8') + '\nreturn VGP;')();
const CFG = new Function(fs.readFileSync(path.join(ROOT, 'config.js'), 'utf8') + '\nreturn CONFIG;')();

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

const HUMAN_TO_LEVEL = { 3: 'low', 2: 'medium', 1: 'high' };

(async () => {
  const rows = await selectAll('labels', 'item_id,rater_name,score,updated_at');
  const cat = JSON.parse(fs.readFileSync(path.join(ROOT, 'catalog.segment.json'), 'utf8'));
  const items = Array.isArray(cat) ? cat : cat.items;

  const perSeg = VGP.resolveSegments(rows, items);
  const perVid = VGP.resolveHuman(rows, items);

  const videos = [];
  const segTally = { raters: 0, checker: 0, checkers: 0, unresolved: 0 };
  const vidTally = { raters: 0, checker: 0, checkers: 0, unresolved: 0 };
  const segDist = {};
  const vidDist = {};
  let nSeg = 0;

  for (const it of items) {
    const v = perVid[it.id] || {};
    const vKey = v.human_score == null ? 'unresolved' : (v.consensus || 'unresolved');
    vidTally[vKey] = (vidTally[vKey] || 0) + 1;
    if (v.human_score != null) vidDist[v.human_score] = (vidDist[v.human_score] || 0) + 1;

    const segments = [];
    for (const idx of VGP.scoreableSegmentIndices(it) || []) {
      const uid = VGP.segmentId(it.id, idx);
      const s = perSeg[uid] || {};
      const meta = (it.segments || []).find((x) => x.index === idx) || {};
      const key = s.human_score == null ? 'unresolved' : (s.consensus || 'unresolved');
      segTally[key] = (segTally[key] || 0) + 1;
      if (s.human_score != null) segDist[s.human_score] = (segDist[s.human_score] || 0) + 1;
      nSeg += 1;
      segments.push({
        index: idx,
        item_id: uid,
        start_frame: meta.start_frame ?? null,
        end_frame: meta.end_frame ?? null,
        n_frames: meta.n_frames ?? null,
        human_score: s.human_score ?? null,           // 1 bad / 2 medium / 3 good
        human_rating: s.human_rating ?? null,
        human_level: s.human_score != null ? HUMAN_TO_LEVEL[s.human_score] : null,
        consensus: s.consensus ?? null,               // raters | checker | checkers
        rater_scores: s.rater_scores ?? {},
        checker_scores: s.checker_scores ?? {},
      });
    }

    videos.push({
      id: it.id,
      clip_id: it.clip_id ?? null,
      video_path: it.video_path ?? null,
      embodiment: it.embodiment ?? null,
      split: it.split ?? null,
      task: it.task ?? null,
      fps: it.fps ?? null,
      n_frames: it.n_frames ?? null,
      segment_len_frames: it.segment_len_frames ?? null,
      n_segments: segments.length,
      // video-level answer = mean of the segment flags, rounded to 1-3
      human_score: v.human_score ?? null,
      human_rating: v.human_rating ?? null,
      consensus: v.consensus ?? null,
      conflict: !!v.conflict,
      n_raters: v.n_raters ?? 0,
      n_checkers: v.n_checkers ?? 0,
      rater_scores: v.rater_scores ?? {},
      checker_scores: v.checker_scores ?? {},
      segments,
    });
  }

  const payload = {
    generated_at: new Date().toISOString(),
    source: {
      labels: `${CFG.SUPABASE_URL}/rest/v1/labels`,
      catalog: 'webpage/labeling/catalog.segment.json',
      resolver: `webpage/labeling/vgp.js (BATCH_VERSION ${VGP.BATCH_VERSION})`,
      n_label_rows: rows.length,
    },
    check: 'hallucinate',
    score_labels: { 1: 'bad', 2: 'medium', 3: 'good' },
    level_mapping: { comment: 'humans rate quality; the bucket rates hallucination amount', 3: 'low', 2: 'medium', 1: 'high' },
    checkers: VGP.REVIEWERS,
    counts: {
      videos: videos.length,
      segments: nSeg,
      videos_resolved: videos.length - vidTally.unresolved,
      videos_unresolved: vidTally.unresolved,
      segments_resolved: nSeg - segTally.unresolved,
      segments_unresolved: segTally.unresolved,
      video_consensus: { raters: vidTally.raters, checker: vidTally.checker, checkers: vidTally.checkers },
      segment_consensus: { raters: segTally.raters, checker: segTally.checker, checkers: segTally.checkers },
      video_score_distribution: vidDist,
      segment_score_distribution: segDist,
    },
    videos,
  };

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(payload, null, 2) + '\n');

  console.log(`labels fetched        : ${rows.length}`);
  console.log(`videos                : ${videos.length}  (resolved ${videos.length - vidTally.unresolved})`);
  console.log(`segments              : ${nSeg}  (resolved ${nSeg - segTally.unresolved})`);
  console.log(`segment consensus     : raters ${segTally.raters}, checker ${segTally.checker}, checkers ${segTally.checkers}`);
  console.log(`segment distribution  : ${JSON.stringify(segDist)}`);
  console.log(`video distribution    : ${JSON.stringify(vidDist)}`);
  console.log(`\nwrote ${OUT}`);
})().catch((e) => { console.error('FAILED:', e.message); process.exit(1); });

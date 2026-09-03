// Shared logic for the labeling site — no build step, plain <script> include.
// Uses direct Supabase REST calls (fetch) instead of the JS SDK to avoid a
// CDN dependency; this is a pure static site (Vercel Hobby, $0).

const VGP = (() => {
  const NAME_KEY = "vgp_rater_name";
  const PASS_KEY = "vgp_pass_ok";
  const ADMIN_KEY = "vgp_admin_ok";
  const BATCH_PREFIX = "vgp_batch::";
  const BATCH_VERSION = 2; // bump when catalog ids change (e.g. rate900)
  const HALF = 25;
  const ADMIN_PAGES = /(^|\/)(analytics|admin)\.html?$/i;

  function fnv1a(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }

  function mulberry32(seed) {
    let s = seed >>> 0;
    return function () {
      s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function getRater() {
    return (localStorage.getItem(NAME_KEY) || "").trim();
  }
  function setRater(name) {
    localStorage.setItem(NAME_KEY, name.trim());
  }
  function passOk() {
    return sessionStorage.getItem(PASS_KEY) === "1";
  }
  function setPassOk() {
    sessionStorage.setItem(PASS_KEY, "1");
  }
  function adminOk() {
    return sessionStorage.getItem(ADMIN_KEY) === "1";
  }
  function setAdminOk() {
    sessionStorage.setItem(ADMIN_KEY, "1");
  }
  function clearSession() {
    sessionStorage.removeItem(PASS_KEY);
    sessionStorage.removeItem(ADMIN_KEY);
  }
  function isAdminPath(path) {
    return ADMIN_PAGES.test((path || "").split("?")[0]);
  }

  function requireGate() {
    if (!passOk() || !getRater()) {
      const next = encodeURIComponent(location.pathname + location.search);
      location.replace(`gate.html?next=${next}`);
      return false;
    }
    return true;
  }

  function requireAdmin() {
    if (!adminOk()) {
      const next = encodeURIComponent(location.pathname + location.search);
      location.replace(`gate.html?next=${next}&admin=1`);
      return false;
    }
    return true;
  }

  function segmentId(itemId, s) {
    return `${itemId}_s${String(s).padStart(3, "0")}`;
  }

  function parentVideoId(itemId) {
    const m = String(itemId).match(/^(.*)_s\d{3}$/);
    return m ? m[1] : itemId;
  }

  function scoreableSegmentIndices(item) {
    if (!item || !item.segments) return null;
    if (Number.isInteger(item.score_segment_index)) return [item.score_segment_index];
    const rated = item.segments
      .map((seg, i) => (seg && seg.rateable === false ? null : i))
      .filter((i) => i != null);
    return rated.length ? rated : item.segments.map((_, i) => i);
  }

  function videoComplete(item, labels) {
    if (!item) return false;
    if (!item.segments) return labels[item.id] != null;
    // rate900: one score on video id OR on the middle segment id
    if (item.rate_target === "middle" || Number.isInteger(item.score_segment_index)) {
      const s = item.score_segment_index ?? 1;
      return labels[item.id] != null || labels[segmentId(item.id, s)] != null;
    }
    const idxs = scoreableSegmentIndices(item) || [];
    return idxs.every((s) => labels[segmentId(item.id, s)] != null);
  }

  function videoTouched(item, labels) {
    if (!item) return false;
    if (!item.segments) return labels[item.id] != null;
    if (item.rate_target === "middle" || Number.isInteger(item.score_segment_index)) {
      const s = item.score_segment_index ?? 1;
      return labels[item.id] != null || labels[segmentId(item.id, s)] != null;
    }
    return item.segments.some((_, s) => labels[segmentId(item.id, s)] != null);
  }

  function raterStats(labels, mvItems, svItems) {
    const mvLabeled = (mvItems || []).filter((it) => labels[it.id] != null).length;
    const svDone = (svItems || []).filter((it) => videoComplete(it, labels)).length;
    const svTouched = (svItems || []).filter((it) => videoTouched(it, labels)).length;
    let svSegments = 0;
    for (const id of Object.keys(labels)) {
      if (/_s\d{3}$/.test(id) || /^rate900_/.test(id)) svSegments += 1;
    }
    return {
      mvLabeled,
      svDone,
      svTouched,
      svSegments,
      totalVideos: mvLabeled + svDone,
    };
  }

  function coverageCounts(rows, items) {
    const known = new Set((items || []).map((it) => it.id));
    const raters = new Map();
    for (const row of rows || []) {
      const vid = known.has(row.item_id) ? row.item_id : parentVideoId(row.item_id);
      if (!known.has(vid)) continue;
      if (!raters.has(vid)) raters.set(vid, new Set());
      raters.get(vid).add(row.rater_name);
    }
    const counts = {};
    for (const it of items || []) counts[it.id] = (raters.get(it.id) || new Set()).size;
    return counts;
  }

  // Prefer videos this rater has not finished; among those, least-labeled
  // first, random among ties. Cross-rater overlap is intentional.
  function pickLeastLabeled(items, coverage, doneIds, n) {
    const fresh = [];
    for (const it of items || []) {
      if (!doneIds.has(it.id)) fresh.push(it);
    }
    fresh.sort((a, b) => {
      const d = (coverage[a.id] || 0) - (coverage[b.id] || 0);
      if (d !== 0) return d;
      return Math.random() - 0.5;
    });
    return fresh.slice(0, n);
  }

  function batchStorageKey() {
    return BATCH_PREFIX + getRater().toLowerCase();
  }

  function getBatch() {
    try {
      return JSON.parse(localStorage.getItem(batchStorageKey()) || "null");
    } catch {
      return null;
    }
  }

  function saveBatch(batch) {
    localStorage.setItem(batchStorageKey(), JSON.stringify(batch));
    return batch;
  }

  function submitBatch() {
    const batch = getBatch();
    if (!batch) return null;
    batch.submittedAt = new Date().toISOString();
    return saveBatch(batch);
  }

  function clearBatch() {
    localStorage.removeItem(batchStorageKey());
  }

  async function getOrCreateBatch(mvItems, svItems) {
    const existing = getValidBatch(mvItems, svItems);
    if (existing) return existing;
    return createBatch(mvItems, svItems);
  }

  async function createBatch(mvItems, svItems) {
    const [mine, allRows] = await Promise.all([
      sbLoadLabels(getRater()),
      sbLoadAllLabels(),
    ]);
    const mvDone = new Set((mvItems || []).filter((it) => mine[it.id] != null).map((it) => it.id));
    const svDone = new Set((svItems || []).filter((it) => videoComplete(it, mine)).map((it) => it.id));
    const mvPick = pickLeastLabeled(mvItems, coverageCounts(allRows, mvItems), mvDone, HALF);
    const svPick = pickLeastLabeled(svItems, coverageCounts(allRows, svItems), svDone, HALF);
    return saveBatch({
      id: `${Date.now().toString(36)}-${fnv1a(getRater() + Math.random())}`,
      version: BATCH_VERSION,
      createdAt: new Date().toISOString(),
      submittedAt: null,
      mvIds: mvPick.map((it) => it.id),
      svIds: svPick.map((it) => it.id),
    });
  }

  function itemsFromIds(allItems, ids) {
    const map = new Map((allItems || []).map((it) => [it.id, it]));
    return (ids || []).map((id) => map.get(id)).filter(Boolean);
  }

  /** Open batch whose ids no longer exist in current catalogs. */
  function isStaleOpenBatch(batch, mvItems, svItems) {
    if (!batch || batch.submittedAt) return false;
    if (batch.version != null && batch.version !== BATCH_VERSION) return true;
    const mvIds = batch.mvIds || [];
    const svIds = batch.svIds || [];
    if (!mvIds.length && !svIds.length) return true;
    if (mvIds.length && itemsFromIds(mvItems, mvIds).length === 0) return true;
    if (svIds.length && itemsFromIds(svItems, svIds).length === 0) return true;
    return false;
  }

  function getValidBatch(mvItems, svItems) {
    const batch = getBatch();
    if (!batch) return null;
    if (isStaleOpenBatch(batch, mvItems, svItems)) {
      clearBatch();
      return null;
    }
    return batch;
  }

  function sbHeaders() {
    return {
      apikey: CONFIG.SUPABASE_PUBLISHABLE_KEY,
      Authorization: `Bearer ${CONFIG.SUPABASE_PUBLISHABLE_KEY}`,
      "Content-Type": "application/json",
    };
  }

  async function sbUpsertLabel(raterName, itemId, score) {
    const url = `${CONFIG.SUPABASE_URL}/rest/v1/labels`;
    const res = await fetch(url, {
      method: "POST",
      headers: { ...sbHeaders(), Prefer: "resolution=merge-duplicates,return=minimal" },
      body: JSON.stringify([{ rater_name: raterName, item_id: itemId, score }]),
    });
    if (!res.ok) throw new Error(`Supabase upsert failed: ${res.status} ${await res.text()}`);
  }

  async function sbLoadLabels(raterName) {
    const url = `${CONFIG.SUPABASE_URL}/rest/v1/labels?rater_name=eq.${encodeURIComponent(raterName)}&select=item_id,score`;
    const res = await fetch(url, { headers: sbHeaders() });
    if (!res.ok) throw new Error(`Supabase load failed: ${res.status} ${await res.text()}`);
    const rows = await res.json();
    const map = {};
    for (const r of rows) map[r.item_id] = r.score;
    return map;
  }

  async function sbLoadAllLabels() {
    const page = 1000;
    const rows = [];
    let from = 0;
    while (true) {
      const url = `${CONFIG.SUPABASE_URL}/rest/v1/labels?select=rater_name,item_id,score,updated_at&order=updated_at.desc`;
      const res = await fetch(url, {
        headers: { ...sbHeaders(), Range: `${from}-${from + page - 1}`, Prefer: "count=exact" },
      });
      if (!res.ok) throw new Error(`Supabase load-all failed: ${res.status} ${await res.text()}`);
      const chunk = await res.json();
      rows.push(...chunk);
      if (chunk.length < page) break;
      from += page;
      if (from > 20000) break;
    }
    return rows;
  }

  return {
    HALF,
    getRater, setRater, passOk, setPassOk, adminOk, setAdminOk,
    clearSession, isAdminPath, requireGate, requireAdmin,
    segmentId, parentVideoId, scoreableSegmentIndices, videoComplete, videoTouched, raterStats,
    coverageCounts, pickLeastLabeled,
    getBatch, getValidBatch, isStaleOpenBatch, saveBatch, submitBatch, clearBatch, createBatch, getOrCreateBatch,
    itemsFromIds,
    sbUpsertLabel, sbLoadLabels, sbLoadAllLabels,
    fnv1a, mulberry32,
    BATCH_VERSION,
  };
})();

// Shared logic for the labeling site — no build step, plain <script> include.
// Uses direct Supabase REST calls (fetch) instead of the JS SDK to avoid a
// CDN dependency; this is a pure static site (Vercel Hobby, $0).

const VGP = (() => {
  const NAME_KEY = "vgp_rater_name";
  const PASS_KEY = "vgp_pass_ok";
  const ADMIN_KEY = "vgp_admin_ok";
  const BATCH_PREFIX = "vgp_batch::";
  const BATCH_VERSION = 4; // checkers get conflict-only batches
  const GEO_N = 25;
  const HALLU_N = 75;
  const HALF = GEO_N; // back-compat alias
  const TARGET_COVER = 2;
  const REVIEWERS = ["quang_check", "thanh_check"];
  const ADMIN_PAGES = /(^|\/)(analytics|admin)\.html?$/i;
  const CHECK = { geo: "geometry", hallu: "hallucinate" };
  const SCORE_LABEL = { 1: "Bad", 2: "Medium", 3: "Good" };
  const EMB_LABEL = { bimanual: "Bimanual", humanoid: "Humanoid", single_arm: "Single-arm" };
  const BAND_TO_SCORE = { high: 1, middle: 2, medium: 2, low: 3, clean: 3, control: 3 };

  function normalizeEmbodiment(emb) {
    const x = String(emb || "unknown").toLowerCase();
    if (x === "single_arm_a1x" || x === "single-arm" || x === "singlearm") return "single_arm";
    return x;
  }

  function embLabel(emb) {
    const key = normalizeEmbodiment(emb);
    return EMB_LABEL[key] || String(emb || "unknown").replace(/_/g, " ");
  }

  function scoreLabel(score) {
    return SCORE_LABEL[score] || String(score ?? "—");
  }

  function checkLabel(kind) {
    return kind === "hallu" || kind === CHECK.hallu ? "Hallucinate" : "Geometry";
  }

  function machineScore(item) {
    if (!item) return null;
    if (item.machine_score != null) return Number(item.machine_score);
    if (item.score != null) return Number(item.score);
    const mapped = BAND_TO_SCORE[item.band];
    return mapped != null ? mapped : null;
  }

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

  function defaultLang() {
    try {
      if (localStorage.getItem("vgp_lang_v2") === "vi") return "vi";
    } catch (_) {}
    return "en";
  }

  function applyLang(next) {
    const lang = next === "vi" ? "vi" : "en";
    document.documentElement.setAttribute("data-lang", lang);
    document.documentElement.lang = lang;
    try { localStorage.setItem("vgp_lang_v2", lang); } catch (_) {}
    return lang;
  }

  function normalizeName(name) {
    return String(name || "").trim().toLowerCase();
  }

  function isReviewer(name) {
    return REVIEWERS.includes(normalizeName(name));
  }

  function reviewerPassOk(pass) {
    return String(pass || "") === String((typeof CONFIG !== "undefined" && CONFIG.REVIEWER_PASSCODE) || "1");
  }

  function raterPassOk(name, pass) {
    if (isReviewer(name)) return reviewerPassOk(pass);
    return String(pass || "") === String((typeof CONFIG !== "undefined" && CONFIG.ACCESS_PASSCODE) || "");
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
      halluDone: svDone,
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
      if (isReviewer(row.rater_name)) continue;
      const vid = known.has(row.item_id) ? row.item_id : parentVideoId(row.item_id);
      if (!known.has(vid)) continue;
      if (!raters.has(vid)) raters.set(vid, new Set());
      raters.get(vid).add(normalizeName(row.rater_name));
    }
    const counts = {};
    for (const it of items || []) counts[it.id] = (raters.get(it.id) || new Set()).size;
    return counts;
  }

  // Each video is assigned to at most TARGET_COVER distinct raters.
  // Prefer uncovered, then 1-label videos. Conflict checkers only
  // receive disagreement clips and may exceed the cap.
  function pickLeastLabeled(items, coverage, doneIds, n, opts) {
    const options = opts || {};
    const maxCover = options.maxCover != null ? options.maxCover : TARGET_COVER;
    const prefer = new Set(options.preferIds || []);
    const fresh = [];
    for (const it of items || []) {
      if (doneIds.has(it.id)) continue;
      const c = coverage[it.id] || 0;
      if (c >= maxCover && !prefer.has(it.id)) continue;
      fresh.push(it);
    }
    fresh.sort((a, b) => {
      const ap = prefer.has(a.id) ? 0 : 1;
      const bp = prefer.has(b.id) ? 0 : 1;
      if (ap !== bp) return ap - bp;
      const d = (coverage[a.id] || 0) - (coverage[b.id] || 0);
      if (d !== 0) return d;
      return Math.random() - 0.5;
    });
    return fresh.slice(0, n);
  }

  function labelsByVideo(rows, items) {
    const known = new Set((items || []).map((it) => it.id));
    const byVid = new Map();
    for (const row of rows || []) {
      const vid = known.has(row.item_id) ? row.item_id : parentVideoId(row.item_id);
      if (!known.has(vid)) continue;
      if (!byVid.has(vid)) byVid.set(vid, new Map());
      byVid.get(vid).set(normalizeName(row.rater_name), row);
    }
    return byVid;
  }

  function conflictIds(rows, items) {
    const byVid = labelsByVideo(rows, items);
    const out = [];
    for (const [vid, scores] of byVid) {
      const regular = [...scores.values()].filter((row) => !isReviewer(row.rater_name));
      const uniq = new Set(regular.map((row) => row.score));
      if (regular.length >= 2 && uniq.size >= 2) out.push(vid);
    }
    return out;
  }

  function conflictRows(rows, items) {
    const ids = new Set(conflictIds(rows, items));
    const known = new Set((items || []).map((it) => it.id));
    const grouped = {};
    for (const row of rows || []) {
      const vid = known.has(row.item_id) ? row.item_id : parentVideoId(row.item_id);
      if (!ids.has(vid)) continue;
      if (!grouped[vid]) grouped[vid] = [];
      grouped[vid].push(row);
    }
    return Object.entries(grouped).map(([id, labels]) => ({ id, labels }));
  }

  function pickConflictItems(items, rows, doneIds, n) {
    const want = new Set(conflictIds(rows, items));
    const fresh = (items || []).filter((it) => want.has(it.id) && !doneIds.has(it.id));
    fresh.sort(() => Math.random() - 0.5);
    return n != null ? fresh.slice(0, n) : fresh;
  }

  function conflictCheckProgress(rows, items) {
    const ids = conflictIds(rows, items);
    const byVid = labelsByVideo(rows, items);
    const per = {};
    for (const name of REVIEWERS) per[name] = 0;
    let open = 0, one = 0, both = 0;
    for (const id of ids) {
      const labels = byVid.get(id) || new Map();
      const checkers = REVIEWERS.filter((name) => labels.has(name));
      checkers.forEach((name) => { per[name] += 1; });
      if (checkers.length === 0) open += 1;
      else if (checkers.length === 1) one += 1;
      else both += 1;
    }
    return { total: ids.length, open, one, both, per };
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
    const reviewer = isReviewer(getRater());
    const mvPick = reviewer
      ? pickConflictItems(mvItems, allRows, mvDone)
      : pickLeastLabeled(mvItems, coverageCounts(allRows, mvItems), mvDone, GEO_N);
    const svPick = reviewer
      ? pickConflictItems(svItems, allRows, svDone)
      : pickLeastLabeled(svItems, coverageCounts(allRows, svItems), svDone, HALLU_N);
    if (reviewer && !mvPick.length && !svPick.length) {
      clearBatch();
      return { id: null, version: BATCH_VERSION, reviewer: true, mvIds: [], svIds: [], empty: true };
    }
    return saveBatch({
      id: `${Date.now().toString(36)}-${fnv1a(getRater() + Math.random())}`,
      version: BATCH_VERSION,
      createdAt: new Date().toISOString(),
      submittedAt: null,
      reviewer: reviewer || undefined,
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

  async function sbDeleteLabel(raterName, itemId) {
    const url = `${CONFIG.SUPABASE_URL}/rest/v1/labels?rater_name=eq.${encodeURIComponent(raterName)}&item_id=eq.${encodeURIComponent(itemId)}`;
    const res = await fetch(url, { method: "DELETE", headers: sbHeaders() });
    if (!res.ok) throw new Error(`Supabase delete failed: ${res.status} ${await res.text()}`);
  }

  // --- local autosave mirror + undo stack ---------------------------------
  const LABEL_CACHE_PREFIX = "vgp_labels_cache::";
  const PENDING_PREFIX = "vgp_labels_pending::";
  const UNDO_MAX = 40;
  let undoStack = [];

  function labelCacheKey(rater) {
    return LABEL_CACHE_PREFIX + String(rater || "").toLowerCase();
  }
  function pendingKey(rater) {
    return PENDING_PREFIX + String(rater || "").toLowerCase();
  }

  function readLabelCache(rater) {
    try {
      return JSON.parse(localStorage.getItem(labelCacheKey(rater)) || "{}") || {};
    } catch {
      return {};
    }
  }

  function writeLabelCache(rater, map) {
    localStorage.setItem(labelCacheKey(rater), JSON.stringify(map || {}));
  }

  function mergeLabels(remote, rater) {
    const local = readLabelCache(rater);
    return { ...local, ...(remote || {}) };
  }

  function mirrorLabel(rater, itemId, score) {
    const map = readLabelCache(rater);
    if (score == null) delete map[itemId];
    else map[itemId] = score;
    writeLabelCache(rater, map);
  }

  function readPending(rater) {
    try {
      return JSON.parse(localStorage.getItem(pendingKey(rater)) || "[]") || [];
    } catch {
      return [];
    }
  }

  function writePending(rater, list) {
    localStorage.setItem(pendingKey(rater), JSON.stringify(list || []));
  }

  function enqueuePending(rater, itemId, score) {
    const list = readPending(rater).filter((p) => p.itemId !== itemId);
    list.push({ itemId, score, at: Date.now() });
    writePending(rater, list);
  }

  function dequeuePending(rater, itemId) {
    writePending(rater, readPending(rater).filter((p) => p.itemId !== itemId));
  }

  /** Upsert/delete + local mirror. Queues on network failure for flush later. */
  async function autosaveLabel(raterName, itemId, score) {
    mirrorLabel(raterName, itemId, score);
    try {
      if (score == null) await sbDeleteLabel(raterName, itemId);
      else await sbUpsertLabel(raterName, itemId, score);
      dequeuePending(raterName, itemId);
      return true;
    } catch (err) {
      enqueuePending(raterName, itemId, score);
      throw err;
    }
  }

  async function flushPending(raterName) {
    const list = readPending(raterName);
    for (const entry of list) {
      try {
        if (entry.score == null) await sbDeleteLabel(raterName, entry.itemId);
        else await sbUpsertLabel(raterName, entry.itemId, entry.score);
        dequeuePending(raterName, entry.itemId);
      } catch (_) {
        /* keep queued */
      }
    }
    return readPending(raterName).length;
  }

  function pushUndo(entry) {
    undoStack.push(entry);
    if (undoStack.length > UNDO_MAX) undoStack.shift();
  }

  function canUndo() {
    return undoStack.length > 0;
  }

  function clearUndo() {
    undoStack = [];
  }

  /**
   * Pop last score action and restore server + local map.
   * entry: { itemId, prevScore, nextScore, idx?, cursor? }
   * Returns restored entry or null.
   */
  async function undoLast(raterName, labelsMap) {
    const entry = undoStack.pop();
    if (!entry) return null;
    const restore = entry.prevScore == null ? null : entry.prevScore;
    if (restore == null) delete labelsMap[entry.itemId];
    else labelsMap[entry.itemId] = restore;
    await autosaveLabel(raterName, entry.itemId, restore);
    return entry;
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

  function renderStepStrip(host, items, current, isDone) {
    if (!host) return;
    host.replaceChildren();
    (items || []).forEach((it, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "step" + (i === current ? " current" : "") + (isDone(it) ? " done" : "");
      btn.textContent = String(i + 1);
      btn.dataset.idx = String(i);
      host.append(btn);
    });
  }

  const exampleCache = new Map();
  function preloadVideos(urls) {
    (urls || []).forEach((url) => {
      if (!url || exampleCache.has(url)) return;
      const v = document.createElement("video");
      v.preload = "auto";
      v.muted = true;
      v.playsInline = true;
      v.src = url;
      v.load();
      exampleCache.set(url, v);
    });
  }

  return {
    HALF, GEO_N, HALLU_N, TARGET_COVER, REVIEWERS, BATCH_VERSION,
    CHECK, SCORE_LABEL, BAND_TO_SCORE, normalizeEmbodiment, embLabel, scoreLabel, checkLabel, machineScore,
    getRater, setRater, passOk, setPassOk, adminOk, setAdminOk,
    clearSession, isAdminPath, requireGate, requireAdmin,
    defaultLang, applyLang, isReviewer, raterPassOk, conflictCheckProgress,
    segmentId, parentVideoId, scoreableSegmentIndices, videoComplete, videoTouched, raterStats,
    coverageCounts, pickLeastLabeled, conflictIds, conflictRows,
    getBatch, getValidBatch, isStaleOpenBatch, saveBatch, submitBatch, clearBatch, createBatch, getOrCreateBatch,
    itemsFromIds, renderStepStrip, preloadVideos,
    sbUpsertLabel, sbDeleteLabel, sbLoadLabels, sbLoadAllLabels,
    mergeLabels, mirrorLabel, autosaveLabel, flushPending, readPending,
    pushUndo, canUndo, clearUndo, undoLast,
    fnv1a, mulberry32,
  };
})();

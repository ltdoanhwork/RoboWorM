// Shared logic for the labeling site — no build step, plain <script> include.
// Uses direct Supabase REST calls (fetch) instead of the JS SDK to avoid a
// CDN dependency; this is a pure static site (Vercel Hobby, $0).

const VGP = (() => {
  const NAME_KEY = "vgp_rater_name";
  const PASS_KEY = "vgp_pass_ok";
  const ADMIN_KEY = "vgp_admin_ok";
  const BATCH_PREFIX = "vgp_batch::";
  const BATCH_VERSION = 7; // batch is 25 geometry videos + 25 hallucinate videos
  const GEO_N = 25;
  const HALLU_N = 25; // whole clips; each one is then scored segment by segment
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

  /** Segment indices a rater must score. `only` restricts it (conflict review). */
  function scoreableSegmentIndices(item, only) {
    if (!item || !item.segments) return null;
    const all = item.segments
      .map((seg, i) => (seg && seg.rateable === false ? null : i))
      .filter((i) => i != null);
    const idxs = all.length ? all : item.segments.map((_, i) => i);
    if (!only || !only.length) return idxs;
    const want = new Set(only.map(Number));
    const kept = idxs.filter((i) => want.has(i));
    return kept.length ? kept : idxs;
  }

  /** Label ids one item produces: one per segment, or the bare id if unsegmented. */
  function itemLabelIds(item, only) {
    if (!item) return [];
    if (!item.segments) return [item.id];
    return (scoreableSegmentIndices(item, only) || []).map((s) => segmentId(item.id, s));
  }

  function segmentCount(item) {
    if (!item) return 0;
    return item.segments ? item.segments.length : 1;
  }

  function videoComplete(item, labels, only) {
    if (!item) return false;
    if (!item.segments) return labels[item.id] != null;
    return itemLabelIds(item, only).every((id) => labels[id] != null);
  }

  function videoTouched(item, labels, only) {
    if (!item) return false;
    if (!item.segments) return labels[item.id] != null;
    return itemLabelIds(item, only).some((id) => labels[id] != null);
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

  function raterStats(labels, mvItems, svItems, svSegs) {
    const only = (it) => (svSegs ? svSegs[it.id] : null);
    const mvLabeled = (mvItems || []).filter((it) => labels[it.id] != null).length;
    const svDone = (svItems || []).filter((it) => videoComplete(it, labels, only(it))).length;
    const svTouched = (svItems || []).filter((it) => videoTouched(it, labels, only(it))).length;
    let svSegments = 0;
    for (const id of Object.keys(labels)) {
      if (/_s\d{3}$/.test(id)) svSegments += 1;
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

  /**
   * rows grouped by scoring unit (segment id, or item id when unsegmented).
   * Map unitId -> { itemId, segIndex, raters: Map(name -> row) }.
   */
  function labelsByUnit(rows, items) {
    const units = new Map();
    for (const it of items || []) {
      for (const uid of itemLabelIds(it)) {
        const m = String(uid).match(/_s(\d{3})$/);
        units.set(uid, { itemId: it.id, segIndex: m ? Number(m[1]) : null, raters: new Map() });
      }
    }
    for (const row of rows || []) {
      const unit = units.get(row.item_id);
      if (!unit) continue;
      unit.raters.set(normalizeName(row.rater_name), row);
    }
    return units;
  }

  function unitsByItem(units) {
    const byItem = new Map();
    for (const [uid, unit] of units) {
      if (!byItem.has(unit.itemId)) byItem.set(unit.itemId, []);
      byItem.get(unit.itemId).push([uid, unit]);
    }
    for (const list of byItem.values()) list.sort((a, b) => (a[1].segIndex ?? 0) - (b[1].segIndex ?? 0));
    return byItem;
  }

  function regularOf(unit) {
    return [...unit.raters.values()].filter((row) => !isReviewer(row.rater_name));
  }
  function checkersOf(unit) {
    return [...unit.raters.values()].filter((row) => isReviewer(row.rater_name));
  }

  /** Distinct regular raters per scoring unit. */
  function segmentCoverage(rows, items) {
    const units = labelsByUnit(rows, items);
    const counts = {};
    for (const [uid, unit] of units) {
      counts[uid] = new Set(regularOf(unit).map((row) => normalizeName(row.rater_name))).size;
    }
    return counts;
  }

  /** Item coverage = the weakest of its segments, so partial work keeps it open. */
  function coverageCounts(rows, items) {
    const units = labelsByUnit(rows, items);
    const byItem = unitsByItem(units);
    const counts = {};
    for (const it of items || []) {
      const list = byItem.get(it.id) || [];
      let min = Infinity;
      for (const [, unit] of list) {
        min = Math.min(min, new Set(regularOf(unit).map((row) => normalizeName(row.rater_name))).size);
      }
      counts[it.id] = Number.isFinite(min) ? min : 0;
    }
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
    const rank = (a, b) => {
      const ap = prefer.has(a.id) ? 0 : 1;
      const bp = prefer.has(b.id) ? 0 : 1;
      if (ap !== bp) return ap - bp;
      const d = (coverage[a.id] || 0) - (coverage[b.id] || 0);
      if (d !== 0) return d;
      return Math.random() - 0.5;
    };
    fresh.sort(rank);
    if (options.stratify === false || fresh.length <= n) return fresh.slice(0, n);
    const groups = new Map();
    for (const it of fresh) {
      const emb = normalizeEmbodiment(it.embodiment);
      if (!groups.has(emb)) groups.set(emb, []);
      groups.get(emb).push(it);
    }
    const embs = [...groups.keys()].sort();
    if (embs.length <= 1) return fresh.slice(0, n);
    embs.forEach((emb) => groups.get(emb).sort(rank));
    const picked = [];
    const cursor = Object.fromEntries(embs.map((emb) => [emb, 0]));
    let guard = 0;
    while (picked.length < n && guard < n * embs.length + embs.length) {
      const emb = embs[guard % embs.length];
      const list = groups.get(emb);
      const i = cursor[emb];
      if (i < list.length) {
        picked.push(list[i]);
        cursor[emb] = i + 1;
      }
      guard += 1;
      if (embs.every((e) => cursor[e] >= groups.get(e).length)) break;
    }
    return picked;
  }

  /** Resolve one scoring unit: 2 agreeing raters, else a checker decides. */
  function resolveUnit(unit) {
    const regular = regularOf(unit);
    const checkers = checkersOf(unit);
    const raterScores = {};
    const checkerScores = {};
    regular.forEach((row) => { raterScores[row.rater_name] = row.score; });
    checkers.forEach((row) => { checkerScores[row.rater_name] = row.score; });
    const uniq = new Set(regular.map((row) => row.score));
    const conflict = regular.length >= 2 && uniq.size >= 2;
    let humanScore = null;
    let consensus = null;
    if (regular.length >= 2 && uniq.size === 1) {
      humanScore = regular[0].score;
      consensus = "raters";
    } else if (conflict) {
      const cuniq = new Set(checkers.map((row) => row.score));
      if (checkers.length >= 2 && cuniq.size === 1) {
        humanScore = checkers[0].score;
        consensus = "checkers";
      } else if (checkers.length === 1) {
        humanScore = checkers[0].score;
        consensus = "checker";
      }
    }
    return {
      human_score: humanScore,
      human_rating: humanScore != null ? (SCORE_LABEL[humanScore] || "").toLowerCase() : null,
      n_raters: new Set(regular.map((row) => normalizeName(row.rater_name))).size,
      n_checkers: checkers.length,
      rater_scores: raterScores,
      checker_scores: checkerScores,
      conflict,
      consensus,
    };
  }

  /** Per-segment resolution keyed by label id. */
  function resolveSegments(rows, items) {
    const units = labelsByUnit(rows, items);
    const out = {};
    for (const [uid, unit] of units) out[uid] = resolveUnit(unit);
    return out;
  }

  // key.json scale: good=0, medium=1, bad=2. A video score is the mean of its
  // segment flags, rounded back onto the 1 / 2 / 3 rater scale.
  function scoreToFlagValue(score) {
    return 3 - score;
  }
  function flagValueToScore(value) {
    return 3 - Math.round(value);
  }

  /** Per-item resolution; segmented items aggregate their segment scores. */
  function resolveHuman(rows, items) {
    const units = labelsByUnit(rows, items);
    const byItem = unitsByItem(units);
    const out = {};
    for (const it of items || []) {
      const list = byItem.get(it.id) || [];
      if (!it.segments) {
        const unit = list.length ? list[0][1] : { raters: new Map() };
        out[it.id] = { ...resolveUnit(unit), n_segments: 1, n_resolved: 0 };
        out[it.id].n_resolved = out[it.id].human_score != null ? 1 : 0;
        continue;
      }
      const resolved = [];
      const raterScores = {};
      const checkerScores = {};
      let conflicts = 0;
      let raterSet = new Set();
      for (const [uid, unit] of list) {
        const r = resolveUnit(unit);
        if (r.conflict) conflicts += 1;
        if (r.human_score != null) resolved.push(r.human_score);
        Object.keys(r.rater_scores).forEach((name) => raterSet.add(normalizeName(name)));
        raterScores[uid] = r.rater_scores;
        checkerScores[uid] = r.checker_scores;
      }
      const complete = list.length > 0 && resolved.length === list.length;
      const meanFlag = resolved.length
        ? resolved.reduce((a, s) => a + scoreToFlagValue(s), 0) / resolved.length
        : null;
      const humanScore = complete ? flagValueToScore(meanFlag) : null;
      out[it.id] = {
        human_score: humanScore,
        human_rating: humanScore != null ? (SCORE_LABEL[humanScore] || "").toLowerCase() : null,
        mean_flag: meanFlag != null ? Number(meanFlag.toFixed(4)) : null,
        n_segments: list.length,
        n_resolved: resolved.length,
        n_raters: raterSet.size,
        rater_scores: raterScores,
        checker_scores: checkerScores,
        conflict: conflicts > 0,
        n_conflicts: conflicts,
        consensus: complete ? "segments" : null,
      };
    }
    return out;
  }

  /** Label ids where the two raters disagree. */
  function conflictUnitIds(rows, items) {
    const units = labelsByUnit(rows, items);
    const out = [];
    for (const [uid, unit] of units) {
      if (resolveUnit(unit).conflict) out.push(uid);
    }
    return out;
  }

  /** Item ids holding at least one conflicting unit. */
  function conflictIds(rows, items) {
    return [...new Set(conflictUnitIds(rows, items).map((uid) => parentVideoId(uid)))];
  }

  /** { itemId: [segIndex, …] } so a checker only re-scores the disputed parts. */
  function conflictSegmentsByItem(rows, items) {
    const out = {};
    const units = labelsByUnit(rows, items);
    for (const uid of conflictUnitIds(rows, items)) {
      const unit = units.get(uid);
      const itemId = unit ? unit.itemId : parentVideoId(uid);
      if (!out[itemId]) out[itemId] = [];
      if (unit && unit.segIndex != null) out[itemId].push(unit.segIndex);
    }
    for (const list of Object.values(out)) list.sort((a, b) => a - b);
    return out;
  }

  function conflictRows(rows, items) {
    const units = labelsByUnit(rows, items);
    const out = [];
    for (const uid of conflictUnitIds(rows, items)) {
      const unit = units.get(uid);
      out.push({ id: uid, itemId: unit ? unit.itemId : parentVideoId(uid), labels: [...unit.raters.values()] });
    }
    return out;
  }

  function pickConflictItems(items, rows, doneIds, n) {
    const bySeg = conflictSegmentsByItem(rows, items);
    const fresh = (items || []).filter((it) => bySeg[it.id] && !doneIds.has(it.id));
    fresh.sort(() => Math.random() - 0.5);
    const picked = n != null ? fresh.slice(0, n) : fresh;
    return { items: picked, segments: bySeg };
  }

  function conflictCheckProgress(rows, items) {
    const units = labelsByUnit(rows, items);
    const per = {};
    for (const name of REVIEWERS) per[name] = 0;
    let total = 0, open = 0, one = 0, both = 0;
    for (const uid of conflictUnitIds(rows, items)) {
      total += 1;
      const unit = units.get(uid);
      const checkers = REVIEWERS.filter((name) => unit.raters.has(name));
      checkers.forEach((name) => { per[name] += 1; });
      if (checkers.length === 0) open += 1;
      else if (checkers.length === 1) one += 1;
      else both += 1;
    }
    return { total, open, one, both, per };
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
    const reviewer = isReviewer(getRater());
    const svConflicts = reviewer ? conflictSegmentsByItem(allRows, svItems) : null;
    const mvDone = new Set((mvItems || []).filter((it) => mine[it.id] != null).map((it) => it.id));
    const svDone = new Set(
      (svItems || [])
        .filter((it) => videoComplete(it, mine, svConflicts ? svConflicts[it.id] : null))
        .map((it) => it.id)
    );
    let mvPick;
    let svPick;
    let svSegs;
    if (reviewer) {
      mvPick = pickConflictItems(mvItems, allRows, mvDone).items;
      const sv = pickConflictItems(svItems, allRows, svDone);
      svPick = sv.items;
      svSegs = {};
      svPick.forEach((it) => { svSegs[it.id] = sv.segments[it.id] || []; });
    } else {
      mvPick = pickLeastLabeled(mvItems, coverageCounts(allRows, mvItems), mvDone, GEO_N);
      svPick = pickLeastLabeled(svItems, coverageCounts(allRows, svItems), svDone, HALLU_N);
    }
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
      svSegs: svSegs,
      svSegments: svPick.reduce((n, it) => n + (svSegs ? (svSegs[it.id] || []).length : segmentCount(it)), 0),
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
  const LABEL_CACHE_PREFIX = "vgp_labels_cache::v" + BATCH_VERSION + "::";
  const PENDING_PREFIX = "vgp_labels_pending::v" + BATCH_VERSION + "::";
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

  function anyDialogOpen() {
    return !!document.querySelector("dialog[open]");
  }

  function bindExampleRemind(check, opts) {
    const dialog = document.getElementById("exampleRemind");
    if (!dialog) return;
    const key = "vgp_ex_remind::" + check;
    const close = () => {
      sessionStorage.setItem(key, "1");
      if (dialog.open) dialog.close();
    };
    const showExamples = () => {
      close();
      const panel = document.getElementById("examplesPanel");
      if (panel) {
        panel.classList.add("nudge");
        panel.scrollIntoView({ behavior: "smooth", block: "start" });
        setTimeout(() => panel.classList.remove("nudge"), 2400);
      }
      if (opts && typeof opts.onShow === "function") opts.onShow();
    };
    const closeBtn = document.getElementById("closeExRemind");
    const showBtn = document.getElementById("showExRemind");
    const skipBtn = document.getElementById("skipExRemind");
    if (closeBtn) closeBtn.onclick = close;
    if (skipBtn) skipBtn.onclick = close;
    if (showBtn) showBtn.onclick = showExamples;
    dialog.addEventListener("click", (e) => { if (e.target === dialog) close(); });
    dialog.addEventListener("cancel", () => { sessionStorage.setItem(key, "1"); });
    if (!sessionStorage.getItem(key)) {
      requestAnimationFrame(() => {
        if (!sessionStorage.getItem(key) && !dialog.open) dialog.showModal();
      });
    }
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

  function videoKey(item) {
    if (!item) return null;
    return item.clip_id || item.id;
  }

  function itemSplit(item) {
    const s = String((item && item.split) || "").toLowerCase();
    if (s === "makovian") return "makovian";
    if (s === "non_makovian" || s === "non-makovian") return "non_makovian";
    return "unknown";
  }

  function raterBucket(name) {
    const n = normalizeName(name).normalize("NFD").replace(/[\u0300-\u036f]/g, "");
    if (n === "quang" || (n.startsWith("quang") && !n.includes("check"))) return "quang";
    if (n === "thanh" || n === "thanhnv" || (n.startsWith("thanh") && !n.includes("check"))) return "thanh";
    return n;
  }

  function renderMarkdown(value) {
    function escapeHtml(s) {
      return String(s).replace(/[&<>'"]/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
      }[c]));
    }
    function inlineMd(s) {
      return escapeHtml(s)
        .replace(/&lt;br\s*\/?&gt;/gi, "<br>")
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    }
    const lines = String(value || "").replace(/\r/g, "").split("\n");
    const out = [];
    let list = false;
    let table = [];
    const flushList = () => { if (list) { out.push("</ul>"); list = false; } };
    const flushTable = () => {
      if (!table.length) return;
      const rows = table.filter((x) => !/^[\s|:-]+$/.test(x));
      if (rows.length) {
        out.push("<table>");
        rows.forEach((row, i) => {
          const cells = row.split("|").slice(1, -1);
          out.push("<tr>" + cells.map((c) => `<${i ? "td" : "th"}>${inlineMd(c.trim())}</${i ? "td" : "th"}>`).join("") + "</tr>");
        });
        out.push("</table>");
      }
      table = [];
    };
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (line.startsWith("|")) { flushList(); table.push(line); continue; }
      flushTable();
      if (!line) { flushList(); continue; }
      const h = line.match(/^(#{1,3})\s+(.+)/);
      if (h) { flushList(); out.push(`<h${h[1].length}>${inlineMd(h[2])}</h${h[1].length}>`); continue; }
      const li = line.match(/^[-*]\s+(.+)/);
      if (li) {
        if (!list) { out.push("<ul>"); list = true; }
        out.push(`<li>${inlineMd(li[1])}</li>`);
        continue;
      }
      flushList();
      out.push(`<p>${inlineMd(line)}</p>`);
    }
    flushTable();
    flushList();
    return out.join("\n");
  }

  return {
    HALF, GEO_N, HALLU_N, TARGET_COVER, REVIEWERS, BATCH_VERSION,
    CHECK, SCORE_LABEL, BAND_TO_SCORE, normalizeEmbodiment, embLabel, scoreLabel, checkLabel, machineScore,
    getRater, setRater, passOk, setPassOk, adminOk, setAdminOk,
    clearSession, isAdminPath, requireGate, requireAdmin,
    defaultLang, applyLang, isReviewer, raterPassOk, conflictCheckProgress,
    segmentId, parentVideoId, scoreableSegmentIndices, itemLabelIds, segmentCount,
    videoComplete, videoTouched, raterStats,
    coverageCounts, segmentCoverage, pickLeastLabeled, conflictIds, conflictUnitIds,
    conflictSegmentsByItem, conflictRows, resolveHuman, resolveSegments,
    scoreToFlagValue, flagValueToScore,
    getBatch, getValidBatch, isStaleOpenBatch, saveBatch, submitBatch, clearBatch, createBatch, getOrCreateBatch,
    itemsFromIds, renderStepStrip, preloadVideos, anyDialogOpen, bindExampleRemind,
    videoKey, itemSplit, raterBucket, renderMarkdown,
    sbUpsertLabel, sbDeleteLabel, sbLoadLabels, sbLoadAllLabels,
    mergeLabels, mirrorLabel, autosaveLabel, flushPending, readPending,
    pushUndo, canUndo, clearUndo, undoLast,
    fnv1a, mulberry32,
  };
})();

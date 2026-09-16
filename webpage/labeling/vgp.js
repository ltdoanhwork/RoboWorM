// Shared logic for the labeling site — no build step, plain <script> include.
// Uses direct Supabase REST calls (fetch) instead of the JS SDK to avoid a
// CDN dependency; this is a pure static site (Vercel Hobby, $0).

const VGP = (() => {
  const NAME_KEY = "vgp_rater_name";
  const PASS_KEY = "vgp_pass_ok";
  const ADMIN_KEY = "vgp_admin_ok";
  const BATCH_PREFIX = "vgp_batch::";
  const BATCH_VERSION = 10; // fixed 12-rater assignment; conflicts remain indexed by whole video
  const GEO_N = 50;
  const HALLU_N = 49; // maximum; 291 clips give six raters 49 and six raters 48
  const HALF = GEO_N; // back-compat alias
  const TARGET_COVER = 2;
  const REVIEWERS = ["quang_check", "thanh_check"];
  const RATER_ACCOUNTS = Object.freeze({
    nhi_pham: "Nhi Phạm",
    hoang_tran: "Hoàng Trần",
    minh_loi: "Minh Lợi",
    luu_minh_tung: "Lưu Minh Tùng",
    christopher_wewer: "Christopher Wewer",
    minh_duc: "Minh Đức",
    bao_ngoc: "Bảo Ngọc",
    gia_binh: "Gia Bình",
    duc_duy: "Đức Duy",
    kien_bui: "Kiên Bùi",
    doanh_1: "Doanh 1",
    doanh_2: "Doanh 2",
  });
  const RATER_PASSCODE = "1";
  const RATER_IDS = Object.freeze(Object.keys(RATER_ACCOUNTS));
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
    localStorage.setItem(NAME_KEY, canonicalRaterName(name));
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
    if (!passOk() || !isKnownRater(getRater())) {
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

  function accountToken(name) {
    return normalizeName(name)
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/đ/g, "d")
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "");
  }

  /** Accept account id, accented display name, or unaccented display name. */
  function canonicalRaterName(name) {
    const normalized = normalizeName(name);
    if (REVIEWERS.includes(normalized) || Object.hasOwn(RATER_ACCOUNTS, normalized)) return normalized;
    const token = accountToken(normalized);
    return RATER_IDS.find((id) => accountToken(id) === token || accountToken(RATER_ACCOUNTS[id]) === token) || normalized;
  }

  function isReviewer(name) {
    return REVIEWERS.includes(normalizeName(name));
  }

  function isRegularRater(name) {
    return Object.hasOwn(RATER_ACCOUNTS, canonicalRaterName(name));
  }

  function isKnownRater(name) {
    return isReviewer(name) || isRegularRater(name);
  }

  function reviewerPassOk(pass) {
    return String(pass || "") === String((typeof CONFIG !== "undefined" && CONFIG.REVIEWER_PASSCODE) || "1");
  }

  function raterPassOk(name, pass) {
    if (isReviewer(name)) return reviewerPassOk(pass);
    return isRegularRater(name) && String(pass || "") === RATER_PASSCODE;
  }

  /** Eleven perfect-match rounds for twelve raters; no pair contains same rater twice. */
  function roundRobinPairings() {
    const ring = [...RATER_IDS];
    const rounds = [];
    for (let round = 0; round < ring.length - 1; round++) {
      const pairs = [];
      for (let i = 0; i < ring.length / 2; i++) {
        pairs.push([ring[i], ring[ring.length - 1 - i]]);
      }
      rounds.push(pairs);
      ring.splice(1, 0, ring.pop());
    }
    return rounds;
  }

  const FIXED_PAIRINGS = roundRobinPairings();

  /**
   * Assign each catalog item to exactly two raters using six pairs per round.
   * Geometry: 300 items -> 50/rater. Hallucinate: 291 -> 48 or 49/rater.
   * Hallucinate starts at a different pairing round to vary co-raters.
   */
  function fixedAssignedItems(items, raterName, kind) {
    const rater = canonicalRaterName(raterName);
    if (!isRegularRater(rater)) return [];
    const ordered = [...(items || [])].sort((a, b) => String(a.id).localeCompare(String(b.id)));
    const catalogSize = kind === CHECK.hallu ? 291 : 300;
    if (ordered.length !== catalogSize) {
      throw new Error(`Fixed assignment requires ${catalogSize} ${kind} items; got ${ordered.length}.`);
    }
    const roundOffset = kind === CHECK.hallu ? 5 : 0;
    const picked = ordered.filter((_, itemIndex) => {
      const round = Math.floor(itemIndex / 6);
      const pairIndex = itemIndex % 6;
      const pairs = FIXED_PAIRINGS[(round + roundOffset) % FIXED_PAIRINGS.length];
      return pairs[pairIndex].includes(rater);
    });
    const validCount = kind === CHECK.hallu
      ? picked.length === HALLU_N || picked.length === HALLU_N - 1
      : picked.length === GEO_N;
    if (!validCount) {
      throw new Error(`Invalid fixed assignment for ${rater}: got ${picked.length} ${kind} items.`);
    }
    return picked;
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
    // Legacy/free-form raters stay in raw data, but current campaign consensus
    // and conflicts use only the twelve preassigned accounts.
    return [...unit.raters.values()].filter((row) => isRegularRater(row.rater_name));
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

  /**
   * Item coverage. Geometry has one unit. Hallucinate counts only regular raters
   * who finished every rateable segment, because assignment/conflict is per video.
   */
  function coverageCounts(rows, items) {
    const units = labelsByUnit(rows, items);
    const byItem = unitsByItem(units);
    const counts = {};
    for (const it of items || []) {
      const list = byItem.get(it.id) || [];
      if (it.segments) {
        counts[it.id] = Object.keys(videoRaterScores(rows, it, false)).length;
        continue;
      }
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

  /**
   * Per-segment resolution keyed by label id. Segment disagreements never open
   * a checker task; hallucinate conflict is decided only by whole-video overall.
   */
  function resolveSegments(rows, items) {
    const units = labelsByUnit(rows, items);
    const out = {};
    for (const it of items || []) {
      if (!it.segments) {
        const unit = units.get(it.id);
        if (unit) out[it.id] = resolveUnit(unit);
        continue;
      }
      const video = resolveHalluVideo(rows, it);
      const regular = videoRaterScores(rows, it, false);
      const checkers = videoRaterScores(rows, it, true);
      const adjudicated = video.conflict && video.human_score != null && Object.keys(checkers).length;
      const chosen = adjudicated ? checkers : regular;
      for (const uid of itemLabelIds(it)) {
        const raterScores = Object.fromEntries(
          Object.entries(regular).map(([name, p]) => [name, p.segment_scores[uid]])
        );
        const checkerScores = Object.fromEntries(
          Object.entries(checkers).map(([name, p]) => [name, p.segment_scores[uid]])
        );
        const scores = Object.values(chosen).map((p) => p.segment_scores[uid]);
        const uniq = new Set(scores);
        const humanScore = scores.length && uniq.size === 1 ? scores[0] : null;
        out[uid] = {
          human_score: humanScore,
          human_rating: humanScore != null ? (SCORE_LABEL[humanScore] || "").toLowerCase() : null,
          n_raters: Object.keys(regular).length,
          n_checkers: Object.keys(checkers).length,
          rater_scores: raterScores,
          checker_scores: checkerScores,
          conflict: false,
          video_conflict: video.conflict,
          consensus: humanScore != null ? (adjudicated ? video.consensus : "raters") : null,
        };
      }
    }
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

  /**
   * Complete per-rater hallucinate profiles for one video.
   * Result: name -> { overall_score, mean_flag, segment_scores }.
   */
  function videoRaterScores(rows, item, reviewers) {
    if (!item || !item.segments) return {};
    const ids = itemLabelIds(item);
    const wanted = new Set(ids);
    const grouped = new Map();
    for (const row of rows || []) {
      if (!wanted.has(row.item_id)) continue;
      const reviewer = isReviewer(row.rater_name);
      if (reviewers === true && !reviewer) continue;
      if (reviewers === false && !isRegularRater(row.rater_name)) continue;
      const name = normalizeName(row.rater_name);
      if (!grouped.has(name)) grouped.set(name, new Map());
      grouped.get(name).set(row.item_id, Number(row.score));
    }
    const out = {};
    for (const [name, scores] of grouped) {
      if (!ids.every((id) => scores.has(id))) continue;
      const ordered = ids.map((id) => scores.get(id));
      const meanFlag = ordered.reduce((sum, score) => sum + scoreToFlagValue(score), 0) / ordered.length;
      out[name] = {
        overall_score: flagValueToScore(meanFlag),
        mean_flag: Number(meanFlag.toFixed(4)),
        segment_scores: Object.fromEntries(ids.map((id, i) => [id, ordered[i]])),
      };
    }
    return out;
  }

  /** Whole-video hallucinate consensus/conflict. */
  function resolveHalluVideo(rows, item) {
    const regular = videoRaterScores(rows, item, false);
    const checkers = videoRaterScores(rows, item, true);
    const regularEntries = Object.entries(regular);
    const checkerEntries = Object.entries(checkers);
    const regularScores = new Set(regularEntries.map(([, p]) => p.overall_score));
    const conflict = regularEntries.length >= TARGET_COVER && regularScores.size >= 2;
    let humanScore = null;
    let consensus = null;
    if (regularEntries.length >= TARGET_COVER && regularScores.size === 1) {
      humanScore = regularEntries[0][1].overall_score;
      consensus = "raters";
    } else if (conflict && checkerEntries.length) {
      const checkerScores = new Set(checkerEntries.map(([, p]) => p.overall_score));
      if (checkerEntries.length === 1 || checkerScores.size === 1) {
        humanScore = checkerEntries[0][1].overall_score;
        consensus = checkerEntries.length === 1 ? "checker" : "checkers";
      }
    }
    return {
      human_score: humanScore,
      human_rating: humanScore != null ? (SCORE_LABEL[humanScore] || "").toLowerCase() : null,
      mean_flag: humanScore != null ? scoreToFlagValue(humanScore) : null,
      n_segments: itemLabelIds(item).length,
      n_resolved: humanScore != null ? itemLabelIds(item).length : 0,
      n_raters: regularEntries.length,
      n_checkers: checkerEntries.length,
      rater_scores: Object.fromEntries(regularEntries.map(([name, p]) => [name, p.overall_score])),
      checker_scores: Object.fromEntries(checkerEntries.map(([name, p]) => [name, p.overall_score])),
      rater_segment_scores: Object.fromEntries(regularEntries.map(([name, p]) => [name, p.segment_scores])),
      checker_segment_scores: Object.fromEntries(checkerEntries.map(([name, p]) => [name, p.segment_scores])),
      conflict,
      n_conflicts: conflict ? 1 : 0,
      consensus,
    };
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
      out[it.id] = resolveHalluVideo(rows, it);
    }
    return out;
  }

  /** Conflict ids: geometry unit ids; hallucinate whole-video ids. */
  function conflictUnitIds(rows, items) {
    const units = labelsByUnit(rows, items);
    const out = [];
    for (const it of items || []) {
      if (it.segments) {
        if (resolveHalluVideo(rows, it).conflict) out.push(it.id);
      } else {
        const unit = units.get(it.id);
        if (unit && resolveUnit(unit).conflict) out.push(it.id);
      }
    }
    return out;
  }

  /** Item ids holding at least one conflicting unit. */
  function conflictIds(rows, items) {
    return [...new Set(conflictUnitIds(rows, items).map((uid) => parentVideoId(uid)))];
  }

  /** Hallucinate conflicts always assign every rateable segment in the video. */
  function conflictSegmentsByItem(rows, items) {
    const out = {};
    const conflicts = new Set(conflictIds(rows, items));
    for (const it of items || []) {
      if (!conflicts.has(it.id)) continue;
      out[it.id] = it.segments ? scoreableSegmentIndices(it) : [];
    }
    return out;
  }

  function conflictRows(rows, items) {
    const units = labelsByUnit(rows, items);
    const out = [];
    for (const it of items || []) {
      if (it.segments) {
        const video = resolveHalluVideo(rows, it);
        if (!video.conflict) continue;
        const regular = videoRaterScores(rows, it, false);
        const checkers = videoRaterScores(rows, it, true);
        const labels = [
          ...Object.entries(regular).map(([name, p]) => ({ rater_name: name, score: p.overall_score, segment_scores: p.segment_scores })),
          ...Object.entries(checkers).map(([name, p]) => ({ rater_name: name, score: p.overall_score, segment_scores: p.segment_scores })),
        ];
        out.push({ id: it.id, itemId: it.id, labels, overall: true });
      } else {
        const unit = units.get(it.id);
        if (unit && resolveUnit(unit).conflict) {
          out.push({ id: it.id, itemId: it.id, labels: [...unit.raters.values()], overall: false });
        }
      }
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
      const item = (items || []).find((it) => it.id === parentVideoId(uid));
      const checkers = item && item.segments
        ? REVIEWERS.filter((name) => videoRaterScores(rows, item, true)[name])
        : REVIEWERS.filter((name) => units.get(uid)?.raters.has(name));
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

  async function submitBatch() {
    const batch = getBatch();
    if (!batch) return null;
    batch.submittedAt = new Date().toISOString();
    saveBatch(batch);
    await sbUpsertBatch(batch);
    return batch;
  }

  /**
   * Re-open a submitted batch. Regular assignments are fixed and deterministic,
   * so a rater who submitted early (or whose browser only saw a truncated label
   * set) must be able to walk straight back into the leftover clips instead of
   * landing on a page with every button hidden.
   */
  async function reopenBatch() {
    const batch = getBatch();
    if (!batch) return null;
    batch.submittedAt = null;
    saveBatch(batch);
    await sbUpsertBatch(batch);
    return batch;
  }

  /** Unscored work left in a batch: geometry clips + incomplete hallu videos. */
  function batchRemaining(batch, mvItems, svItems, labels) {
    if (!batch) return { mv: 0, sv: 0, total: 0 };
    const svSegs = batch.svSegs || null;
    const mv = itemsFromIds(mvItems, batch.mvIds).filter((it) => labels[it.id] == null).length;
    const sv = itemsFromIds(svItems, batch.svIds).filter(
      (it) => !videoComplete(it, labels, svSegs ? svSegs[it.id] : null)
    ).length;
    return { mv, sv, total: mv + sv };
  }

  function clearBatch() {
    localStorage.removeItem(batchStorageKey());
  }

  async function getOrCreateBatch(mvItems, svItems) {
    const existing = getValidBatch(mvItems, svItems);
    if (existing) {
      // Backfill batches created before server-side audit storage existed.
      await sbUpsertBatch(existing);
      return existing;
    }
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
      mvPick = fixedAssignedItems(mvItems, getRater(), CHECK.geo);
      svPick = fixedAssignedItems(svItems, getRater(), CHECK.hallu);
      // Hallucinate assignment is atomic by video. Every regular rater scores
      // every rateable segment so their per-video overall scores are comparable.
      svSegs = {};
      svPick.forEach((it) => {
        svSegs[it.id] = scoreableSegmentIndices(it) || [];
      });
    }
    if (reviewer && !mvPick.length && !svPick.length) {
      clearBatch();
      return { id: null, version: BATCH_VERSION, reviewer: true, mvIds: [], svIds: [], empty: true };
    }
    const batch = {
      id: reviewer
        ? `${Date.now().toString(36)}-${fnv1a(getRater() + Math.random())}`
        : `assigned-v${BATCH_VERSION}-${normalizeName(getRater())}`,
      version: BATCH_VERSION,
      createdAt: new Date().toISOString(),
      submittedAt: null,
      reviewer: reviewer || undefined,
      mvIds: mvPick.map((it) => it.id),
      svIds: svPick.map((it) => it.id),
      svSegs: svSegs,
      svSegments: svPick.reduce((n, it) => n + (svSegs ? (svSegs[it.id] || []).length : segmentCount(it)), 0),
    };
    saveBatch(batch);
    await sbUpsertBatch(batch);
    return batch;
  }

  function itemsFromIds(allItems, ids) {
    const map = new Map((allItems || []).map((it) => [it.id, it]));
    return (ids || []).map((id) => map.get(id)).filter(Boolean);
  }

  /** Batch from another campaign version, or open batch with missing catalog ids. */
  function isStaleOpenBatch(batch, mvItems, svItems) {
    if (!batch) return false;
    if (batch.version != null && batch.version !== BATCH_VERSION) return true;
    if (batch.submittedAt) return false;
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

  function batchDbRow(batch) {
    return {
      id: batch.id,
      rater_name: getRater(),
      version: Number(batch.version || BATCH_VERSION),
      reviewer: !!batch.reviewer,
      mv_ids: batch.mvIds || [],
      sv_ids: batch.svIds || [],
      sv_segments: batch.svSegs || {},
      created_at: batch.createdAt || new Date().toISOString(),
      submitted_at: batch.submittedAt || null,
    };
  }

  function missingBatchTable(status, body) {
    return status === 404 || /rating_batches|PGRST205|schema cache/i.test(body || "");
  }

  async function sbUpsertBatch(batch) {
    if (!batch || !batch.id) return false;
    const url = `${CONFIG.SUPABASE_URL}/rest/v1/rating_batches`;
    const res = await fetch(url, {
      method: "POST",
      headers: { ...sbHeaders(), Prefer: "resolution=merge-duplicates,return=minimal" },
      body: JSON.stringify([batchDbRow(batch)]),
    });
    if (res.ok) return true;
    const body = await res.text();
    if (missingBatchTable(res.status, body)) {
      console.warn("Batch audit table missing; run supabase/batch_audit.sql.");
      return false;
    }
    throw new Error(`Supabase batch upsert failed: ${res.status} ${body}`);
  }

  async function sbLoadAllBatches() {
    const fixedRows = async () => {
      const [mvRes, svRes] = await Promise.all([fetch("catalog.json"), fetch("catalog.segment.json")]);
      if (!mvRes.ok || !svRes.ok) throw new Error("Could not load catalogs for assignment audit.");
      const [mvCatalog, svCatalog] = await Promise.all([mvRes.json(), svRes.json()]);
      return RATER_IDS.map((rater) => ({
        id: `assigned-v${BATCH_VERSION}-${rater}`,
        rater_name: rater,
        version: BATCH_VERSION,
        reviewer: false,
        mv_ids: fixedAssignedItems(mvCatalog.items || [], rater, CHECK.geo).map((it) => it.id),
        sv_ids: fixedAssignedItems(svCatalog.items || [], rater, CHECK.hallu).map((it) => it.id),
        sv_segments: {},
        created_at: "2026-09-12T00:00:00.000Z",
        submitted_at: null,
        assignment_source: "fixed-v10",
      }));
    };
    const url = `${CONFIG.SUPABASE_URL}/rest/v1/rating_batches?select=*&order=created_at.desc`;
    const res = await fetch(url, { headers: sbHeaders() });
    if (res.ok) {
      const stored = await res.json();
      const merged = new Map((await fixedRows()).map((row) => [row.id, row]));
      stored.forEach((row) => merged.set(row.id, row));
      return [...merged.values()];
    }
    const body = await res.text();
    if (missingBatchTable(res.status, body)) return fixedRows();
    throw new Error(`Supabase batch load failed: ${res.status} ${body}`);
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
    const res = await fetch(url, {
      method: "DELETE",
      headers: { ...sbHeaders(), Prefer: "return=representation" },
    });
    if (!res.ok) throw new Error(`Supabase delete failed: ${res.status} ${await res.text()}`);
    const deleted = await res.json();
    if (!Array.isArray(deleted) || deleted.length !== 1) {
      throw new Error("Supabase delete changed no rows.");
    }
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
    // Production intentionally grants no anonymous DELETE policy. A first
    // score can still be changed by revisiting the item; only score changes
    // enter undo history, so undo always restores through a safe upsert.
    if (entry.prevScore == null) return;
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

  /**
   * PostgREST caps every response at its `max-rows` setting (1000 here), so any
   * unpaged select silently truncates. A rater with 50 geometry + ~1400
   * hallucinate segments lost ~470 labels this way and the UI reported work as
   * unsaved. Always page through with Range until the server stops sending.
   */
  async function sbSelectAll(query, label) {
    const page = 1000;
    const rows = [];
    let from = 0;
    while (true) {
      const res = await fetch(`${CONFIG.SUPABASE_URL}/rest/v1/${query}`, {
        headers: { ...sbHeaders(), Range: `${from}-${from + page - 1}` },
      });
      if (!res.ok) throw new Error(`Supabase ${label} failed: ${res.status} ${await res.text()}`);
      const chunk = await res.json();
      if (!Array.isArray(chunk)) throw new Error(`Supabase ${label} returned no rows array.`);
      rows.push(...chunk);
      // Trust what the server granted, not our requested page size: a smaller
      // max-rows would otherwise stop the loop after one short chunk.
      if (!chunk.length) break;
      const range = res.headers.get("content-range") || "";
      const total = Number((range.split("/")[1] || "").trim());
      if (Number.isFinite(total) && total > 0 && rows.length >= total) break;
      from += chunk.length;
      if (from > 200000) break;
    }
    return rows;
  }

  async function sbLoadLabels(raterName) {
    const rows = await sbSelectAll(
      `labels?rater_name=eq.${encodeURIComponent(raterName)}&select=item_id,score&order=item_id.asc`,
      "load"
    );
    const map = {};
    for (const r of rows) map[r.item_id] = r.score;
    return map;
  }

  async function sbLoadAllLabels() {
    return sbSelectAll(
      "labels?select=rater_name,item_id,score,updated_at&order=item_id.asc,rater_name.asc",
      "load-all"
    );
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
    HALF, GEO_N, HALLU_N, TARGET_COVER, REVIEWERS, BATCH_VERSION, RATER_IDS, RATER_ACCOUNTS,
    CHECK, SCORE_LABEL, BAND_TO_SCORE, normalizeEmbodiment, embLabel, scoreLabel, checkLabel, machineScore,
    getRater, setRater, passOk, setPassOk, adminOk, setAdminOk,
    clearSession, isAdminPath, requireGate, requireAdmin,
    defaultLang, applyLang, canonicalRaterName, isReviewer, isRegularRater, isKnownRater, raterPassOk, conflictCheckProgress,
    segmentId, parentVideoId, scoreableSegmentIndices, itemLabelIds, segmentCount,
    videoComplete, videoTouched, raterStats,
    coverageCounts, segmentCoverage, pickLeastLabeled, fixedAssignedItems, conflictIds, conflictUnitIds,
    conflictSegmentsByItem, conflictRows, resolveHuman, resolveSegments,
    videoRaterScores, resolveHalluVideo,
    scoreToFlagValue, flagValueToScore,
    getBatch, getValidBatch, isStaleOpenBatch, saveBatch, submitBatch, reopenBatch, batchRemaining, clearBatch, createBatch, getOrCreateBatch,
    itemsFromIds, renderStepStrip, preloadVideos, anyDialogOpen, bindExampleRemind,
    videoKey, itemSplit, raterBucket, renderMarkdown,
    sbUpsertLabel, sbDeleteLabel, sbLoadLabels, sbLoadAllLabels, sbUpsertBatch, sbLoadAllBatches,
    mergeLabels, mirrorLabel, autosaveLabel, flushPending, readPending,
    pushUndo, canUndo, clearUndo, undoLast,
    fnv1a, mulberry32,
  };
})();

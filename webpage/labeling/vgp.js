// Shared logic for the labeling site — no build step, plain <script> include.
// Uses direct Supabase REST calls (fetch) instead of the JS SDK to avoid a
// CDN dependency; this is a pure static site (Vercel Hobby, $0).

const VGP = (() => {
  const NAME_KEY = "vgp_rater_name";
  const PASS_KEY = "vgp_pass_ok";

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

  // Redirect to gate.html unless the passcode was entered this session and a
  // rater name is stored. Call at the top of every protected page.
  function requireGate() {
    if (!passOk() || !getRater()) {
      const next = encodeURIComponent(location.pathname + location.search);
      location.replace(`gate.html?next=${next}`);
      return false;
    }
    return true;
  }

  // Deterministic per-rater queue: a shared CORE subset (for inter-annotator
  // agreement) that every rater sees, plus a personal subset assigned by a
  // seeded hash of (rater name, item id) so coverage spreads out as more
  // raters join, with no backend / assignment table needed.
  function buildQueue(allItems, raterName, opts) {
    const coreFrac = opts?.coreFraction ?? 0.10;
    const targetTotal = opts?.targetTotal ?? 100;
    const sorted = [...allItems].sort((a, b) => a.id.localeCompare(b.id));
    const coreCount = Math.max(1, Math.round(sorted.length * coreFrac));
    const core = sorted.slice(0, coreCount);
    const rest = sorted.slice(coreCount);

    const personalTarget = Math.max(0, targetTotal - core.length);
    const coverage = rest.length ? Math.min(1, personalTarget / rest.length) : 0;

    const personal = rest.filter((it) => {
      const h = fnv1a(`${raterName}::${it.id}`) / 4294967296;
      return h < coverage;
    });

    // Stable per-rater shuffle so the viewing order differs between raters
    // (reduces order effects) but is reproducible across sessions.
    const rng = mulberry32(fnv1a(`order::${raterName}`));
    const queue = [...core, ...personal];
    for (let i = queue.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [queue[i], queue[j]] = [queue[j], queue[i]];
    }
    return queue;
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
    const url = `${CONFIG.SUPABASE_URL}/rest/v1/labels?select=rater_name,item_id,score,updated_at&order=updated_at.desc`;
    const res = await fetch(url, { headers: sbHeaders() });
    if (!res.ok) throw new Error(`Supabase load-all failed: ${res.status} ${await res.text()}`);
    return res.json();
  }

  return {
    getRater, setRater, passOk, setPassOk, requireGate,
    buildQueue, sbUpsertLabel, sbLoadLabels, sbLoadAllLabels,
  };
})();

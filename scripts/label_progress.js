#!/usr/bin/env node
/**
 * Per-rater labeling progress, straight from the labels table.
 *
 * Uses the real vgp.js so the numbers match what each rater sees on home.html:
 * the same deterministic fixed assignment, the same videoComplete() rule, and
 * paginated reads (PostgREST caps a single response at 1000 rows, which is what
 * once made finished raters look ~20 clips short).
 *
 * Usage: node scripts/label_progress.js
 */
const fs = require("fs");
const vm = require("vm");
const path = require("path");
const https = require("https");

const LABELING = path.join(__dirname, "..", "webpage", "labeling");

function request(url, { headers = {} } = {}) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const req = https.request(
      { hostname: u.hostname, path: u.pathname + u.search, method: "GET", headers },
      (res) => {
        let text = "";
        res.on("data", (d) => (text += d));
        res.on("end", () =>
          resolve({
            ok: res.statusCode >= 200 && res.statusCode < 300,
            status: res.statusCode,
            headers: { get: (k) => res.headers[String(k).toLowerCase()] || null },
            json: async () => JSON.parse(text),
            text: async () => text,
          })
        );
      }
    );
    req.on("error", reject);
    req.end();
  });
}

function loadVgp() {
  const noStore = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
  const ctx = {
    console, Math, Date, JSON, RegExp, Object, Array, String, Number, Boolean, Error, Set, Map, Promise, URL,
    fetch: request, encodeURIComponent, decodeURIComponent, isNaN, parseInt, parseFloat,
    localStorage: noStore, sessionStorage: noStore,
    location: { pathname: "/home.html", search: "", replace: () => {} },
    document: { documentElement: { dataset: {} } },
    navigator: { onLine: true },
  };
  vm.createContext(ctx);
  const run = (file, tail) =>
    new vm.Script(fs.readFileSync(path.join(LABELING, file), "utf8") + tail).runInContext(ctx);
  run("config.js", ";globalThis.CONFIG=CONFIG;");
  run("vgp.js", ";globalThis.__V=VGP;");
  return { V: ctx.__V, CONFIG: ctx.CONFIG };
}

const pct = (n, d) => (d ? `${Math.round((100 * n) / d)}%` : "—");
const ago = (iso) => {
  if (!iso) return "never";
  const h = (Date.now() - new Date(iso).getTime()) / 3.6e6;
  if (h < 1) return `${Math.round(h * 60)}m ago`;
  if (h < 48) return `${Math.round(h)}h ago`;
  return `${Math.round(h / 24)}d ago`;
};

async function main() {
  const { V, CONFIG } = loadVgp();
  const headers = {
    apikey: CONFIG.SUPABASE_PUBLISHABLE_KEY,
    Authorization: `Bearer ${CONFIG.SUPABASE_PUBLISHABLE_KEY}`,
  };
  const rows = [];
  for (let from = 0; ; from += 1000) {
    const res = await request(
      `${CONFIG.SUPABASE_URL}/rest/v1/labels?select=rater_name,item_id,score,updated_at`,
      { headers: { ...headers, "Range-Unit": "items", Range: `${from}-${from + 999}` } }
    );
    if (!res.ok) throw new Error(`labels fetch failed: ${res.status} ${await res.text()}`);
    const chunk = await res.json();
    rows.push(...chunk);
    if (chunk.length < 1000) break;
  }

  const mvItems = JSON.parse(fs.readFileSync(path.join(LABELING, "catalog.json"), "utf8")).items;
  const svItems = JSON.parse(fs.readFileSync(path.join(LABELING, "catalog.segment.json"), "utf8")).items;

  const byRater = new Map();
  for (const r of rows) {
    if (!byRater.has(r.rater_name)) byRater.set(r.rater_name, { scores: {}, last: null });
    const e = byRater.get(r.rater_name);
    e.scores[r.item_id] = r.score;
    if (!e.last || r.updated_at > e.last) e.last = r.updated_at;
  }

  console.log(`labels on server: ${rows.length}\n`);
  console.log("rater                geo      hallu vids   hallu segs   total    left  last seen");
  console.log("-".repeat(84));
  const done = [];
  const active = [];
  const idle = [];
  for (const rater of V.RATER_IDS) {
    const e = byRater.get(rater) || { scores: {}, last: null };
    const geo = V.fixedAssignedItems(mvItems, rater, V.CHECK.geo);
    const hal = V.fixedAssignedItems(svItems, rater, V.CHECK.hallu);
    const geoDone = geo.filter((it) => e.scores[it.id] != null).length;
    const halDone = hal.filter((it) => V.videoComplete(it, e.scores, null)).length;
    const segIds = hal.flatMap((it) => V.itemLabelIds(it));
    const segDone = segIds.filter((id) => e.scores[id] != null).length;
    const total = geo.length + hal.length;
    const doneV = geoDone + halDone;
    const left = total - doneV;
    console.log(
      `${rater.padEnd(20)} ${String(geoDone).padStart(2)}/${geo.length}` +
      `    ${String(halDone).padStart(2)}/${hal.length}` +
      `      ${String(segDone).padStart(4)}/${segIds.length}` +
      `   ${String(doneV).padStart(2)}/${total} ${pct(doneV, total).padStart(5)}` +
      `  ${String(left).padStart(3)}  ${ago(e.last)}`
    );
    (left === 0 ? done : doneV === 0 ? idle : active).push(rater);
  }
  console.log("-".repeat(84));
  console.log(`finished (${done.length}): ${done.join(", ") || "-"}`);
  console.log(`in progress (${active.length}): ${active.join(", ") || "-"}`);
  console.log(`not started (${idle.length}): ${idle.join(", ") || "-"}`);

  // Catalog-level coverage and conflict state.
  for (const [name, items, kind] of [["GEOMETRY", mvItems, V.CHECK.geo], ["HALLUCINATE", svItems, V.CHECK.hallu]]) {
    const resolved = V.resolveHuman(rows, items);
    const list = Object.values(resolved);
    const byC = {};
    let full = 0, conflicts = 0, unresolved = 0;
    for (const r of list) {
      if (r.n_raters >= 2) full += 1;
      if (r.conflict) conflicts += 1;
      if (r.human_score != null) byC[r.consensus] = (byC[r.consensus] || 0) + 1;
      else if (r.conflict) unresolved += 1;
    }
    const prog = V.conflictCheckProgress(rows, items);
    console.log(`\n${name} (${items.length} items)`);
    console.log(`  both raters in: ${full}/${items.length} (${pct(full, items.length)})`);
    console.log(`  resolved: ${Object.values(byC).reduce((a, b) => a + b, 0)}  ${JSON.stringify(byC)}`);
    console.log(`  conflicts: ${conflicts}  unresolved: ${unresolved}  (checker queue open: ${prog.open})`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });

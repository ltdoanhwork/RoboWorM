#!/usr/bin/env node
/**
 * Sync the rating_batches audit table with ground truth derived from the labels
 * table + the deterministic fixed assignment in vgp.js.
 *
 * Batch/submit state normally lives only in one browser's localStorage, so a
 * rater who clears cache or switches device loses it, and the admin view cannot
 * see who actually finished. This rebuilds the server-side audit rows:
 *   - mv_ids / sv_ids  : the rater's fixed assignment (deterministic)
 *   - created_at       : their earliest label timestamp (when they really started)
 *   - submitted_at     : their latest label timestamp, ONLY if the assignment is
 *                        100% complete. Never fabricated; null while work remains.
 *
 * Audit-only: the labeling pages read their batch from localStorage, and only
 * analytics.html reads this table, so running this cannot disturb a live rater.
 *
 * Usage: node scripts/sync_rating_batches.js [--dry-run]
 */
const fs = require("fs");
const vm = require("vm");
const path = require("path");
const https = require("https");

const LABELING = path.join(__dirname, "..", "webpage", "labeling");
const DRY = process.argv.includes("--dry-run");

function request(url, { method = "GET", headers = {}, body } = {}) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const req = https.request(
      { hostname: u.hostname, path: u.pathname + u.search, method, headers },
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
    if (body) req.write(body);
    req.end();
  });
}

function loadVgp() {
  const noStore = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
  const ctx = {
    console, Math, Date, JSON, Object, Array, String, Number, Boolean, Error, Set, Map, Promise, URL,
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

async function main() {
  const { V, CONFIG } = loadVgp();
  const headers = {
    apikey: CONFIG.SUPABASE_PUBLISHABLE_KEY,
    Authorization: `Bearer ${CONFIG.SUPABASE_PUBLISHABLE_KEY}`,
    "Content-Type": "application/json",
  };
  const mvItems = JSON.parse(fs.readFileSync(path.join(LABELING, "catalog.json"), "utf8")).items;
  const svItems = JSON.parse(fs.readFileSync(path.join(LABELING, "catalog.segment.json"), "utf8")).items;

  // Every label row, paginated (PostgREST caps a single response at max-rows).
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
  console.log(`labels: ${rows.length} rows`);

  const raterIds = V.RATER_IDS;
  const payload = [];
  console.log("\nrater                geo     hallu   complete  submitted_at");
  for (const rater of raterIds) {
    const mine = {};
    const stamps = [];
    for (const r of rows) {
      if (r.rater_name !== rater) continue;
      mine[r.item_id] = r.score;
      if (r.updated_at) stamps.push(r.updated_at);
    }
    const geo = V.fixedAssignedItems(mvItems, rater, V.CHECK.geo);
    const hal = V.fixedAssignedItems(svItems, rater, V.CHECK.hallu);
    const geoDone = geo.filter((it) => mine[it.id] != null).length;
    const halDone = hal.filter((it) => V.videoComplete(it, mine, null)).length;
    const complete = geoDone === geo.length && halDone === hal.length;
    stamps.sort();
    const submittedAt = complete && stamps.length ? stamps[stamps.length - 1] : null;
    payload.push({
      id: `assigned-v${V.BATCH_VERSION ?? 10}-${rater}`,
      rater_name: rater,
      version: V.BATCH_VERSION ?? 10,
      reviewer: false,
      mv_ids: geo.map((it) => it.id),
      sv_ids: hal.map((it) => it.id),
      sv_segments: {},
      created_at: stamps.length ? stamps[0] : new Date().toISOString(),
      submitted_at: submittedAt,
    });
    console.log(
      `${rater.padEnd(20)} ${String(geoDone).padStart(2)}/${geo.length}` +
      `   ${String(halDone).padStart(2)}/${hal.length}` +
      `   ${complete ? "YES     " : "no      "}  ${submittedAt || "-"}`
    );
  }

  if (DRY) {
    console.log("\n--dry-run: nothing written.");
    return;
  }
  const res = await request(`${CONFIG.SUPABASE_URL}/rest/v1/rating_batches`, {
    method: "POST",
    headers: { ...headers, Prefer: "resolution=merge-duplicates,return=minimal" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`rating_batches upsert failed: ${res.status} ${await res.text()}`);
  console.log(`\nupserted ${payload.length} rating_batches rows.`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

# Labeling site — architecture & setup

How the Video World human-labeling web app is built, hosted and operated, and
how to stand up an independent copy from scratch.

Scope: the static site under `webpage/labeling/` plus its Supabase database and
the data pipeline that feeds it. For the rating rules raters follow, see
`webpage/labeling/RUBRIC.md`; for the account roster and the conflict contract,
`webpage/labeling/ACCOUNTS.md`; for terse deploy notes, `webpage/labeling/README.md`.

---

## 1. What this system is

A two-track human study over generated robot videos:

| Track | Unit of judgement | Catalog | Scores per rater |
|---|---|---|---|
| **Geometry** | one whole clip | `catalog.json` — 300 clips | 1 score |
| **Hallucinate** | one 16-frame segment | `catalog.segment.json` — 291 videos / 8,093 segments | 1 score per rateable segment |

Every score is `1 = Bad`, `2 = Medium`, `3 = Good`. Each catalog item is rated by
**exactly two** raters; disagreement is resolved by a third signal (§6).

The whole thing is deliberately **serverless and free**: a static HTML/JS folder
on Vercel, a Postgres table on Supabase reached directly from the browser, and
videos served from HuggingFace. There is no backend to run and no build step.

---

## 2. Architecture

```
                     ┌──────────────────────────────┐
   rater's browser   │  Vercel (static, free tier)  │
  ┌───────────────┐  │  root dir = webpage/labeling │
  │ gate.html     │◄─┤                              │
  │ home.html     │  │  *.html  app.css  vgp.js     │
  │ index.html    │  │  config.js (committed)       │
  │ segment.html  │  │  catalog*.json examples*.json│
  │ analytics.html│  │  metrics*.json rubric*.md    │
  └──┬─────────┬──┘  └──────────────────────────────┘
     │         │
     │ mp4     │ fetch() REST, publishable key
     ▼         ▼
┌──────────┐  ┌────────────────────────────────────┐
│HuggingFace│  │ Supabase Postgres (free tier)      │
│ dataset   │  │  labels(rater_name, item_id,score) │
│  CDN      │  │  rating_batches(...) — audit only  │
└──────────┘  └────────────────────────────────────┘
```

Consequences of this shape, all intentional:

- **No server code.** The browser talks to PostgREST directly. Any logic that
  must be consistent between rater UI and admin analytics lives in one file,
  `vgp.js`, and is imported by every page.
- **No Supabase Auth.** Its free-tier email OTP is capped ~2 emails/hour, which
  stalls when several raters onboard at once. Access is a client-side account
  allowlist + passcode instead (§4). This is a *soft gate*, not security.
- **No video hosting on Vercel.** Videos come from HuggingFace, so Vercel
  bandwidth stays negligible and the 100 GB/month Hobby cap is irrelevant.
- **RLS is open.** Any holder of the publishable key can read/write any row of
  `labels`. Acceptable for a small trusted study; never put anything sensitive
  in this database.

### Files and their roles

| File | Role |
|---|---|
| `gate.html` | login: account name + passcode, or admin passcode |
| `home.html` | hub: shows the rater's fixed assignment and progress, creates/submits the batch |
| `index.html` | Geometry labeling (raw + optical-flow overlay, 3 buttons) |
| `segment.html` | Hallucinate labeling (player pauses at each segment boundary) |
| `analytics.html` | admin only: coverage, conflicts, agreement, box plot, exports |
| `admin.html` | redirect → `analytics.html` |
| `vgp.js` | the shared brain: assignment, completeness, conflict resolution, Supabase REST, batch state |
| `config.js` | Supabase URL + publishable key + passcodes (**committed on purpose**, see §4) |
| `catalog.json` / `catalog.segment.json` | blind rater-facing catalogs (no method, no machine band) |
| `examples*.json`, `rubric*.md` | reference clips and bilingual rubrics shown while labeling |
| `metrics.multiview.json` / `metrics.segment.json` | admin-only machine metrics; **never** loaded by rater pages |
| `vercel.json` | rewrites + cache headers |
| `build-config.js` | no-op; only exists because a stale Vercel dashboard "Build Command" still points at it |

The split between **blind rater catalogs** and **admin-only metrics** is a
correctness requirement, not tidiness: if a band, bundle, ratio or method label
reached `catalog*.json`, raters could infer the answer.

---

## 3. Infra setup from zero

### 3.1 Supabase

1. Create a free project at <https://supabase.com> (no credit card).
2. SQL Editor → New query → paste **`supabase/schema.sql`** → Run. This creates
   both `labels` and `rating_batches`, their open RLS policies, and the
   `updated_at` triggers.
3. Settings → API → copy the **Project URL** and the **publishable** (anon) key.

> If you inherited a deployment where `rating_batches` is missing, apply
> `supabase/batch_audit.sql` — it is the same table as an idempotent migration.
> Symptom of it missing: batch upserts silently no-op and the admin view cannot
> tell who has submitted.

Free tier auto-pauses after 7 days idle (one click to resume, no data loss). For
a long study, a free GitHub Actions cron pinging the REST URL weekly avoids it.

### 3.2 Vercel

1. Import the repo, set **Root Directory = `webpage/labeling`**.
2. No env vars, no build command needed (`config.js` is committed).
3. Deploys on every push to `main`.

Share `https://<project>.vercel.app/` plus each account name + passcode. Do
**not** hand out `/analytics.html`: it needs `ADMIN_PASSCODE`, and rater
passcodes are rejected there.

`vercel.json` sets `Cache-Control: no-cache, must-revalidate` for `vgp.js`,
`config.js` and all `*.html` / `*.json`, while `*.mp4` stays immutable for a
year. Keep it that way — see the cache incident in §8.

### 3.3 Video hosting (HuggingFace)

Geometry clips:

```
https://huggingface.co/datasets/<user>/<dataset>/resolve/main/assessment/multiview_static2/clips/{id}/{raw,flow}.mp4
```

Hallucinate videos are referenced by absolute URL inside `catalog.segment.json`
(HF `hallucinate-bench`, revision `web`, path `300vid/{robot}/{split}/{band}/{n}.mp4`).

Upload with:

```bash
HF_TOKEN=... python scripts/upload_content_to_hf.py <local_dir> <user>/<dataset> \
  --repo-type dataset --prefix assessment/multiview_static2/clips
```

### 3.4 Local run (no deploy)

```bash
cp webpage/labeling/config.example.js webpage/labeling/config.js
# put a real Supabase URL/key + any passcodes in it
cd webpage/labeling && python3 -m http.server 8899
# http://localhost:8899/gate.html
```

---

## 4. Access model

Four independent secrets, all in `config.js` except the rater passcode:

| Who | Secret | Where defined | Opens |
|---|---|---|---|
| Rater | `RATER_PASSCODE` (currently `1`) | `vgp.js` | `home/index/segment` for their own account |
| Conflict checker | `REVIEWER_PASSCODE` | `config.js` | same pages, but the batch is conflict clips |
| Admin | `ADMIN_PASSCODE` | `config.js` | `analytics.html` only |

`ACCESS_PASSCODE` also sits in `config.js` but is **dead** — no code reads it and
`config.example.js` does not declare it. Safe to drop when you copy the config.

Account names are normalized (`canonicalRaterName`), so `"Christopher Wewer"`,
`"christopher wewer"` and `"CHRISTOPHER_WEWER"` all resolve to the same account.

**Why `config.js` is committed** (unusual, but correct here): Supabase's
publishable key is designed to be public, and any passcode in a static site
ships inside the JS bundle anyway. Putting them in a gitignored env var would
add a manual dashboard step without adding security. Never put a
`sb_secret_...` key in this folder.

---

## 5. Assignment: deterministic, not stored

Assignment is **computed**, never allocated from a database. `vgp.js` builds a
round-robin over the account roster:

```
roundRobinPairings()      11 rounds × 6 pairs over the 12 accounts
fixedAssignedItems(items, rater, kind)
    item i  ->  round = i / 6, pairIndex = i % 6
    the rater is assigned iff they appear in that round's pair
```

Result: 50 geometry + 48–49 hallucinate videos per account, and every one of the
300 geometry and 291 hallucinate items lands on exactly **two distinct** raters.
Hallucinate is atomic per video — an assigned rater scores *every* rateable
segment, so the two raters' per-video overall scores are directly comparable.

Two properties worth knowing before touching the roster:

- Pairing depends only on **position** in `RATER_ACCOUNTS`. Renaming a key in
  place hands that person's exact items to the new name and leaves everyone else
  byte-identical. **Adding a 13th account reshuffles everybody.**
- The batch in `localStorage` is a convenience cache. Ground truth is
  `labels` + the deterministic assignment, which is why a rater can move
  browsers and why `scripts/label_progress.js` can rebuild anyone's state.

`rating_batches` is an **audit mirror** only; no rater flow reads it.

---

## 6. Conflict resolution

Two assigned raters, three possible outcomes:

| Track | Raters agree | Raters disagree |
|---|---|---|
| Geometry | that score (`consensus: raters`) | **arbiter score** (`consensus: arbiter`) |
| Hallucinate | that score (`consensus: raters`) | a checker re-scores the video (`consensus: checker`/`checkers`) |

The **geometry arbiter** is the account `thanh_check`, who scored all 300
geometry clips offline. That pass is the `score` column of
`sampling_dataset/data_for_web/data_for_web.csv`, imported into `labels` by
`scripts/import_geo_arbiter_labels.py`. Because it covers the whole catalog
rather than only the conflicts a checker happened to reach, it settles every
geometry conflict the moment one appears, and geometry conflicts are therefore
never queued as checker work.

Hallucinate has **no** arbiter: `quang_check` / `thanh_check` adjudicate those
conflicts in the UI, and a single checker's verdict resolves the video (two
disagreeing checkers leave it unresolved on purpose).

Implementation is centralized in `resolveUnit()` (geometry, unsegmented items)
and `resolveHalluVideo()` (per-video overall). Per-segment disagreement alone
never creates a conflict.

---

## 7. Data pipeline

Rebuild order, internal → blind public bundle:

```bash
# Geometry (Mode A): render clips + flow overlays, sync blind catalog
models/dreamgen/.venv/bin/python scripts/build_assessment_multiview.py --sync-web
python scripts/build_multiview_metrics_json.py     # admin metrics, id-aligned

# Hallucinate (Mode B): blind segment catalog from the HF bucket
python scripts/build_hallucination_segment_catalog.py --sync-web
python scripts/sync_labeling_admin_metrics.py      # validates catalog/metrics pair
```

Two invariants these builders enforce, both of which have broken before:

1. **Row order == catalog id.** Row *i* of `data_for_web.csv` is catalog id
   `{i:05d}`. `build_multiview_metrics_json.py` refuses to write unless each
   rendered clip's `meta.json` `source_path` equals that row's `video_path`; a
   shifted mapping would show one video while scoring another.
   `import_geo_arbiter_labels.py` re-checks the same thing before importing.
2. **Frame counts come from ffprobe, not the source index.** `key.json` lists
   161 phantom segments past end-of-file on 7 single_arm clips, and 16 clips have
   a trailing frame no segment covers. Counts are cached in `frames.json`; pass
   `--reprobe` after the bucket changes.

---

## 8. Operations

```bash
node scripts/label_progress.js              # per-rater + catalog progress
node scripts/sync_rating_batches.js         # rebuild the audit table (--dry-run)
python3 scripts/import_geo_arbiter_labels.py --dry-run   # verify arbiter mapping
python3 scripts/_sb_fetch_all.py labels /tmp/labels.json # dump a table
```

`label_progress.js` and `sync_rating_batches.js` load the real `vgp.js`, so the
numbers they print are exactly what a rater sees on `home.html` — do not
reimplement the assignment or completeness rules in a reporting script.

### Incidents worth not repeating

- **PostgREST caps a single response at 1000 rows.** Raters above 1000 labels
  looked ~20 clips short of finished, and `home.html` re-opened their submitted
  batch. Every read now goes through `sbSelectAll()`, which pages with `Range`
  headers and trusts the granted `content-range`. Any new query must do the same.
- **Stale JS cache.** The pagination fix shipped, but raters kept seeing the old
  number because the bundle was cached. Hence the `no-cache` headers in
  `vercel.json` for `vgp.js` / `config.js` / html / json.
- **`rating_batches` was never created**, so `sbUpsertBatch()` no-oped for weeks
  and submit state existed only in one browser's `localStorage`.
- **A submitted-but-incomplete batch used to be a dead end** — every button
  hidden. `reopenBatch()` + `batchRemaining()` now keep the phase cards visible
  and offer "continue remaining clips".

---

## 9. Replicating this for a different study

Minimum edits to reuse the stack with new content:

1. `supabase/schema.sql` → new project. Copy URL + publishable key into `config.js`.
2. Replace `catalog.json` / `catalog.segment.json` with your own blind catalogs
   (keep the `id` / `segments[]` shape; segment ids are `{videoId}_s{index}`).
3. Edit `RATER_ACCOUNTS` in `vgp.js`. Keep the count **even** — the round-robin
   needs an even roster to pair everyone, and 12 accounts is what yields exactly
   two raters per item at these catalog sizes. Bump `BATCH_VERSION` so old
   `localStorage` batches invalidate while stored labels survive.
4. Rewrite the rubrics (`rubric.*.md`) and example sets.
5. Decide the conflict rule: reuse the checker flow, or supply a full-catalog
   arbiter pass like §6 (much cheaper — it resolves conflicts before they queue).

Sanity check before inviting anyone: run `node scripts/label_progress.js` and
confirm the coverage line reads `both raters in: 0/N`, and that per-rater
assignment sizes sum to `2 × N`.

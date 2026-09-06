# Video World — free-tier labeling site

Static site + Supabase Postgres for storing scores. $0 on Vercel Hobby +
Supabase Free, as long as this stays a non-commercial research deployment
(see cost notes below).

```
gate.html     rater passcode + name  |  admin passcode (analytics only)
home.html     hub: 100-video batch (25 geometry + 75 hallucinate), lifetime stats
index.html    Geometry — 25 clips / batch, raw + flow, Bad/Medium/Good
segment.html  Hallucinate — 75 clips / batch, score the middle 16 frames
analytics.html admin-only charts (coverage, scores, raters). Not linked from rater UI.
admin.html    redirect → analytics.html
app.css       shared UI
vgp.js        least-labeled batch assignment + Supabase REST
catalog.json  Mode A public catalog (HF URLs)
examples.json 9 references: 3 ratings × 3 robot embodiments
catalog.segment.json  Mode B blind public catalog (300 videos / explicit frame ranges)
examples.segment.json 45 references: 3 ratings × 3 robot embodiments × 5 clips
rubric.segment.{vi,en}.md Mode B bilingual rubrics shown during labeling
```

Mode A videos live at
`https://huggingface.co/datasets/doanh25032004/video_gen_physics/resolve/main/assessment/multiview_static2/clips/{id}/{raw,flow}.mp4`.
Rebuild:

```bash
models/dreamgen/.venv/bin/python scripts/build_assessment_multiview.py --sync-web
HF_TOKEN=... python scripts/upload_content_to_hf.py \
  sampling_dataset/data_for_web/assessment/multiview_static2/clips \
  doanh25032004/video_gen_physics --repo-type dataset \
  --prefix assessment/multiview_static2/clips
HF_TOKEN=... python scripts/upload_content_to_hf.py \
  sampling_dataset/data_for_web/assessment/multiview_static2/examples \
  doanh25032004/video_gen_physics --repo-type dataset \
  --prefix assessment/multiview_static2/examples
```

`assets/all_videos.md` fixes source order for all 300 items.
`assets/seed_points.json` provides CoTracker initialization points; entries
without points fall back to optical flow. `assets/examples.md` controls the
nine references shown in the sticky example panel.

Mode B source data lives under
`sampling_dataset/data_for_web/hallucinate_bench`. Rebuild
its blind catalog and sync public metadata/rubrics after downloading:

```bash
python scripts/build_hallucination_segment_catalog.py --sync-web
```

Builder strips source bundle, quality stratum, detector outputs, and violation
counts. Example clips remain reference-only and never enter rating queue.

## 1. Supabase (free Postgres)

1. Create a free project at https://supabase.com (no credit card).
2. Project -> SQL Editor -> New query -> paste `supabase/schema.sql` -> Run.
3. Project -> Settings -> API -> copy **Project URL** and **anon public key**.
4. If the project is idle 7 days it auto-pauses (free tier only) — one click
   to resume in the dashboard, no data loss. A cheap way to avoid this if you
   run a long study: a free GitHub Actions cron hitting the REST URL weekly.

No Supabase Auth is used (its free-tier email OTP is capped at 2 emails/hour,
which stalls if several raters sign up at once). Instead raters type a shared
passcode (see below) + their own display name; RLS on the `labels` table is
open read/write. This is **not** a real security boundary — do not put
anything sensitive in this table, and rotate the passcode if it leaks.

## 2. Vercel (free static hosting)

Already done for you: a Vercel project `vgp-labeling` is linked to this repo's
`main` branch (root directory `webpage/labeling`), auto-deploying on every
push. `config.js` is committed directly (see design note below) so there is
no build step and nothing to configure in the Vercel dashboard.

To change the passcode or point at a different Supabase project: edit
`webpage/labeling/config.js` and push to `main` — Vercel redeploys
automatically.

Share `https://<project>.vercel.app/` (or `/gate.html`) + the **rater**
passcode. Do **not** give raters `/analytics.html` — that URL uses a
separate `ADMIN_PASSCODE` in `config.js`. Rater passcodes cannot open it.

Cost note: Vercel Hobby is free but restricted to **non-commercial personal
use** per its Fair Use Guidelines — fine for an internal research study, not
for a monetized product. Bandwidth cap is ~100GB/month fast data transfer,
which this static HTML/JS page (videos are served from HuggingFace, not
Vercel) will not come close to.

## 3. Local testing (no deploy)

```bash
cp webpage/labeling/config.example.js webpage/labeling/config.js
# edit config.js with a real Supabase URL/key (or a scratch project) + any passcode
cd webpage/labeling && python3 -m http.server 8899
# open http://localhost:8899/gate.html
```

## Rater assignment

Each rater gets **100 videos at a time** (25 geometry, then 75 hallucinate).
The batch is stored in `localStorage` so refresh does not re-roll. **Submit
works before the batch is finished.** After submit they can request another 100.

Assignment reads current Supabase labels and picks videos this rater has not
finished, **least-labeled first**, random among ties. That evens coverage
while keeping intentional overlap across raters. Same display name restores
lifetime counts (multiview videos + completed single-view videos + segments).

## Hallucinate segments (Mode B)

Do not cut 8k tiny MP4s. `scripts/build_hallucination_segment_catalog.py`
copies `start_frame` / `end_frame` / `n_frames` from
`sampling_dataset/data_for_web/hallucinate_bench/300-prune/{robot}/segments.json`
(including the last short tail — 297 of 8370 segments are &lt;16 frames).
The browser streams the full HF video and seeks:

```
start_time = start_frame / fps
end_time   = (end_frame + 1) / fps
```

Rebuild after HF metadata changes:

```bash
python scripts/build_hallucination_segment_catalog.py --sync-web
```

`--sync-web` also writes `metrics.segment.json` (admin-only): per-segment
rigidity / jerk / teleport / joint_limit / self_collision from each
`300-prune/{robot}/segments.json`. Analytics compares those machine flags
to human 1–2–3. The rater catalog stays blind.

## Design tradeoffs (why it's free)

- No Supabase Auth -> shared passcode instead (avoids the 2 email/hour cap).
- No video hosting on Vercel -> videos on the HuggingFace dataset CDN.
- No server -> plain `fetch()` to Supabase's REST API (PostgREST) from the
  browser using the publishable key; RLS policies are the only guard, and
  they are intentionally open (any passcode holder can write any rater's rows).
- `config.js` is committed to git (unusual for a "config" file, but correct
  here): Supabase's publishable key is explicitly designed to be public
  (equivalent to the old anon key), and the passcode is a soft gate that
  ships in the JS bundle on any static deploy anyway — putting it in a
  gitignored env var would not add real security, only an extra manual step
  in the Vercel dashboard. Do NOT put the `sb_secret_...` key here or
  anywhere client-side.
- `catalog.json` in this folder is a **different, blind** file from
  `sampling_dataset/data_for_web/catalog.json` (the internal one, which does
  not leak the method either but lives next to the full manifest with method
  labels — keep this deployed copy separate and regenerate it with
  `build_public_labeling_bundle.py`, never hand-edit the source manifest).

# Video World — free-tier labeling site

Static site + Supabase Postgres for storing scores. $0 on Vercel Hobby +
Supabase Free, as long as this stays a non-commercial research deployment
(see cost notes below).

```
gate.html     rater passcode + name  |  admin passcode (analytics only)
home.html     hub: batch = 25 geometry videos + ~75 hallucinate segments, lifetime stats
index.html    Geometry — 25 clips / batch, raw + flow, Bad/Medium/Good
segment.html  Hallucinate — the clip pauses at every 16-frame segment for one score
analytics.html admin-only charts (coverage, confusion, raters). Not linked from rater UI.
admin.html    redirect → analytics.html
app.css       shared UI
vgp.js        least-labeled batch assignment + Supabase REST
catalog.json  Mode A public catalog (HF URLs)
examples.json 9 references: 3 ratings × 3 robot embodiments
catalog.segment.json  Mode B blind public catalog (291 videos, explicit frame ranges)
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

Each batch is **25 geometry videos + hallucinate clips worth ~75 segments**
(whole clips only — a clip is never split across raters, so the actual count
lands between 75 and 94). The batch is stored in `localStorage` so refresh does
not re-roll. **Submit works before the batch is finished.**

Assignment reads current Supabase labels and picks items this rater has not
finished, **least-labeled first**, random among ties, round-robin across
embodiments. A hallucinate clip counts as covered only when *every* one of its
segments has that many raters, so partial work leaves it open. Same display
name restores lifetime counts.

Conflict checkers (`quang_check`, `thanh_check`) get only the clips holding a
disagreement, and inside them only the disputed segment indices (`svSegs` on
the batch).

## Hallucinate segments (Mode B, 300vid pool)

Source: the HF bucket `twanghcmut/hallucinate-bench`, revision `web`, path
`300vid/{robot}/key.json` — 291 videos (bimanual 100, humanoid 100,
single_arm 91) cut into 8,093 segments of 16 frames at 16 fps, each carrying a
markov / non-markov split and a machine band.

One catalog item = one video. Labels are per segment (`{videoId}_s{index}`);
the player streams the clip and pauses at each boundary:

```
end_time = (end_frame + 1) / fps
```

Rebuild:

```bash
python scripts/build_hallucination_segment_catalog.py --sync-web
python scripts/sync_labeling_admin_metrics.py   # validates the pair
```

The builder **ffprobes every video** (HTTP range requests, ~40s for all 291)
and caches the frame counts in
`sampling_dataset/data_for_web/hallucinate_bench/300vid/frames.json`. This is
not optional: `key.json` lists 161 phantom segments past the end of file on 7
single_arm clips, and 16 clips have one trailing frame no segment covers. Pass
`--reprobe` after the bucket changes.

`metrics.segment.json` (admin-only) holds `items` = one row per segment
(`ratio`, `violated`, frame range) and `videos` = the per-video rollup
(`band`, `bundle`, `score_mean`, …). The rater catalog stays blind — no band,
bundle, ratio or violation ever reaches `catalog.segment.json`.

Analytics exports three files: `vgp_labels.json`, `vgp_labels.csv`, and
`hallucinate_key_labeled.json`, the last one shaped exactly like the machine
`key.json` (`{robot: {videos, segments}}`) with the `flag` fields filled in
from resolved human scores.

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

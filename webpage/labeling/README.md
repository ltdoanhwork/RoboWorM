# Video World — free-tier labeling site

Static site + Supabase Postgres for storing scores. $0 on Vercel Hobby +
Supabase Free, as long as this stays a non-commercial research deployment
(see cost notes below).

> For the full picture — architecture diagram, access model, deterministic
> assignment math, conflict resolution, operational runbook and how to
> replicate the stack for another study — see
> [`docs/labeling-site-architecture.md`](../../docs/labeling-site-architecture.md).
> This file stays the terse deploy reference.

```
gate.html     fixed account login  |  admin passcode (analytics only)
home.html     hub: fixed assignment = 50 geometry + 48–49 hallucinate videos
index.html    Geometry — 50 clips / assignment, raw + flow, Bad/Medium/Good
segment.html  Hallucinate — the clip pauses at every 16-frame segment for one score
analytics.html admin-only charts (coverage, confusion, raters). Not linked from rater UI.
admin.html    redirect → analytics.html
app.css       shared UI
vgp.js        fixed two-rater assignment + conflict logic + Supabase REST
ACCOUNTS.md   rater names, account IDs, assignment contract
catalog.json  Mode A public catalog (HF URLs)
examples.json 27 references: 3 videos × 3 ratings × 3 robot embodiments
catalog.segment.json  Mode B blind public catalog (291 videos, explicit frame ranges)
examples.segment.json 34 references: whole clips, 3-6 per rating × embodiment
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
which stalls if several raters sign up at once). Instead raters use a fixed
client-side account allowlist and assigned passcode; RLS on the `labels` table
is open read/write. This is **not** a real security boundary — do not put
anything sensitive in this table.

## 2. Vercel (free static hosting)

Already done for you: a Vercel project `vgp-labeling` is linked to this repo's
`main` branch (root directory `webpage/labeling`), auto-deploying on every
push. `config.js` is committed directly (see design note below) so there is
no build step and nothing to configure in the Vercel dashboard.

To change the account allowlist/passcode, edit `webpage/labeling/vgp.js`. To
point at a different Supabase project or change admin/checker passcodes, edit
`webpage/labeling/config.js`. Push to `main`; Vercel redeploys automatically.

Share `https://<project>.vercel.app/` (or `/gate.html`) + each assigned account.
Do **not** give raters `/analytics.html` — that URL uses a
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

Each regular account has one fixed assignment: **50 geometry videos + 48 or 49
hallucinate videos**. With twelve accounts, all 300 Geometry and all 291
Hallucinate videos receive exactly two distinct raters. Geometry is one score
per clip; a Hallucinate clip is scored segment by segment. The batch is stored
in `localStorage` and mirrored to Supabase `rating_batches` with exact
Geometry/Hallucinate order for audit. Version changes invalidate old local
batches while preserving submitted labels.
**Submit works before the batch is finished.**

Admin reconstructs all twelve deterministic assignments from the catalogs, so
batch indices remain available even before a rater starts or when the optional
`rating_batches` migration has not been applied. If that table exists, stored
submission timestamps override the reconstructed audit rows.

Assignment is deterministic from account ID and sorted catalog item ID.
Hallucinate coverage is indexed by video: one rater counts only after scoring
every rateable segment. Each assigned rater receives the whole video, so two
per-rater overall scores can be compared directly. Legacy free-form labels stay
in raw storage but do not participate in current campaign consensus/conflicts.

Conflict checkers (`quang_check`, `thanh_check`) get only the clips holding a
whole-video overall disagreement. A checker sees both original raters' overall
and per-segment labels, then re-scores every rateable segment in that video.
Per-segment disagreement alone does not create a conflict.

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

Reference examples come from `300vid/example/key.json` — 34 whole clips (3-6
per embodiment × rating) from the same generators as the pool but held out of
it, so showing them leaks no answer. Their `band` is a per-video verdict, not
per-segment, which is why they supplement the rubric rather than replace it.
The bucket names the levels Low / Medium / High; `relabel_scale()` rewrites the
rubric to the Good / Medium / Bad the buttons show, and fails loudly if a
future bucket edit changes that wording.

`metrics.segment.json` (admin-only) holds `items` = one row per segment
(`ratio`, `violated`, frame range) and `videos` = the per-video rollup
(`band`, `bundle`, `score_mean`, …). The rater catalog stays blind — no band,
bundle, ratio or violation ever reaches `catalog.segment.json`.

Analytics exports three files: `vgp_labels.json`, `vgp_labels.csv`, and
`hallucinate_key_labeled.json`, the last one shaped exactly like the machine
`key.json` (`{robot: {videos, segments}}`) with the `flag` fields filled in
from resolved human scores.

## Design tradeoffs (why it's free)

- No Supabase Auth -> client-side account allowlist instead (avoids the 2 email/hour cap).
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

# Video World — free-tier labeling site

Static site + Supabase Postgres for storing scores. $0 on Vercel Hobby +
Supabase Free, as long as this stays a non-commercial research deployment
(see cost notes below).

```
gate.html    passcode + display name (sets localStorage/sessionStorage)
index.html   labeling UI (1-5 score), personal queue, syncs to Supabase
admin.html   cross-rater progress + CSV/JSON export
vgp.js       shared logic: queue assignment, Supabase REST calls
catalog.json blind public catalog (400 items; no method/role/path fields)
build-config.js  generates config.js from Vercel env vars at build time
config.example.js  template — copy to config.js for LOCAL testing only
```

Video files are **not** hosted here — `catalog.json`'s `video` field points at
the public HuggingFace dataset CDN
(`https://huggingface.co/datasets/doanh25032004/video_gen_physics/resolve/main/data_for_web/clips/{id}.mp4`).
Rebuild the blind bundle + catalog with:

```bash
python scripts/build_public_labeling_bundle.py   # writes clips/ + catalog.public.json
cp sampling_dataset/data_for_web/catalog.public.json webpage/labeling/catalog.json
HF_TOKEN=... python scripts/upload_content_to_hf.py \
  sampling_dataset/data_for_web/clips doanh25032004/video_gen_physics \
  --repo-type dataset --prefix data_for_web/clips
```

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

1. Push this repo (or just `webpage/labeling/`) to a Git remote, import it in
   Vercel as a new project, **Root Directory = `webpage/labeling`**,
   Framework Preset = **Other**.
2. Project -> Settings -> Environment Variables, add:
   - `SUPABASE_URL` = your Project URL
   - `SUPABASE_PUBLISHABLE_KEY` = your anon public key
   - `ACCESS_PASSCODE` = whatever passcode you'll share with raters
3. Deploy. `vercel.json`'s `buildCommand` (`node build-config.js`) generates
   `config.js` from those env vars at build time — the real secrets never go
   into git (`config.js` is gitignored; only `config.example.js` is tracked).
4. Share `https://<your-project>.vercel.app/gate.html` + the passcode with
   raters. Admin dashboard: `/admin.html` (same passcode gate).

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

`vgp.buildQueue()` gives every rater a personal ~100-item queue:
10% is a shared "core" subset (same clips for everyone, for inter-annotator
agreement), the rest is a deterministic hash-based sample so coverage spreads
out as more raters join — no backend assignment table needed.

## Design tradeoffs (why it's free)

- No Supabase Auth -> shared passcode instead (avoids the 2 email/hour cap).
- No video hosting on Vercel -> videos on the HuggingFace dataset CDN.
- No server -> plain `fetch()` to Supabase's REST API (PostgREST) from the
  browser using the anon key; RLS policies are the only guard, and they are
  intentionally open (any passcode holder can write any rater's rows).
- `catalog.json` in this folder is a **different, blind** file from
  `sampling_dataset/data_for_web/catalog.json` (the internal one, which does
  not leak the method either but lives next to the full manifest with method
  labels — keep this deployed copy separate and regenerate it with
  `build_public_labeling_bundle.py`, never hand-edit the source manifest).

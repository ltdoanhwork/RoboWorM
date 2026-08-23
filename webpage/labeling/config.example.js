// Copy this file to config.js (gitignored) and fill in real values before
// deploying. config.js is NOT committed — Vercel needs it uploaded as a
// project file or injected at build time (see webpage/labeling/README.md).
const CONFIG = {
  // Shared passcode raters must type once per browser session. This is NOT
  // real security (it ships in the JS bundle) — it only avoids Supabase's
  // free-tier email-OTP rate limit (2 emails/hour) by skipping email auth
  // entirely. Anyone who has the passcode can write labels under any name.
  ACCESS_PASSCODE: "change-me",

  // Supabase project REST endpoint + anon (public) key. Anon key is safe to
  // ship client-side; RLS policies in supabase/schema.sql scope what it can
  // do (here: open read/write on the `labels` table only).
  SUPABASE_URL: "https://YOUR-PROJECT-REF.supabase.co",
  SUPABASE_PUBLISHABLE_KEY: "YOUR-ANON-KEY",
};

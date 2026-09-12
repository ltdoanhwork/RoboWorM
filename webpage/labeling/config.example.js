// Copy this file to config.js (gitignored) and fill in real values before
// deploying. config.js is NOT committed — Vercel needs it uploaded as a
// project file or injected at build time (see webpage/labeling/README.md).
const CONFIG = {
  // Separate gate for /analytics.html. Rater passcode cannot open admin.
  ADMIN_PASSCODE: "change-me-admin",
  // Passcode for reserved conflict checkers (quang_check / thanh_check).
  REVIEWER_PASSCODE: "1",

  // Supabase project REST endpoint + anon (public) key. Anon key is safe to
  // ship client-side; RLS policies in supabase/schema.sql scope what it can
  // do (here: open read/write on the `labels` table only).
  SUPABASE_URL: "https://YOUR-PROJECT-REF.supabase.co",
  SUPABASE_PUBLISHABLE_KEY: "YOUR-ANON-KEY",
};

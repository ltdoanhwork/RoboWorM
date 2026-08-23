// Committed on purpose — see webpage/labeling/README.md "Design tradeoffs".
// SUPABASE_URL/PUBLISHABLE_KEY are meant to be public (RLS is the real
// guard); ACCESS_PASSCODE is a soft anti-bot gate, not real security.
// To rotate: edit the values below and push to main (Vercel auto-redeploys).
const CONFIG = {
  ACCESS_PASSCODE: "vgp-label-2026",

  SUPABASE_URL: "https://usuctukltxniyhyjwhln.supabase.co",
  SUPABASE_PUBLISHABLE_KEY: "sb_publishable_eMHf2-CYSUB8Ot_GWqOBvA_J8NcWEa_",
};

#!/usr/bin/env node
// Vercel build step (Node runtime, no framework) — generates config.js from
// Environment Variables set in the Vercel dashboard, so secrets never touch
// git. Runs automatically via vercel.json's "buildCommand".
const fs = require("fs");
const path = require("path");

const required = ["SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "ACCESS_PASSCODE"];
const missing = required.filter((k) => !process.env[k]);
if (missing.length) {
  console.error(
    `[build-config] Missing env vars: ${missing.join(", ")}. ` +
      "Set them in Vercel -> Project -> Settings -> Environment Variables."
  );
  process.exit(1);
}

const out = `// AUTO-GENERATED at build time from Vercel Environment Variables. Do not edit.
const CONFIG = {
  ACCESS_PASSCODE: ${JSON.stringify(process.env.ACCESS_PASSCODE)},
  SUPABASE_URL: ${JSON.stringify(process.env.SUPABASE_URL)},
  SUPABASE_PUBLISHABLE_KEY: ${JSON.stringify(process.env.SUPABASE_PUBLISHABLE_KEY)},
};
`;

fs.writeFileSync(path.join(__dirname, "config.js"), out);
console.log("[build-config] wrote config.js from environment variables");

#!/usr/bin/env node
// No-op build step kept ONLY because Vercel's cached Project Settings
// "Build Command" override still points at this file (set on first deploy,
// before the labeling site switched to a committed config.js — there is no
// MCP/CLI tool available here to clear that dashboard override). It simply
// verifies config.js exists; it does not generate or overwrite it.
const fs = require("fs");
const path = require("path");

const configPath = path.join(__dirname, "config.js");
if (!fs.existsSync(configPath)) {
  console.error("[build] config.js is missing — commit a real config.js (see config.example.js).");
  process.exit(1);
}
console.log("[build] config.js present, nothing to do.");

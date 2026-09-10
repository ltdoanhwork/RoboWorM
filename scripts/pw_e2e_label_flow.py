#!/usr/bin/env python3
"""Playwright smoke: gate → assign batch → score 1 multiview → score 1 rate900."""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

BASE = "https://vgp-labeling-doanhles-projects.vercel.app"
PASS = "vgp-label-2026"
RATER = f"pw_e2e_{int(time.time())}"
OUT = Path("/tmp/vgp_pw_e2e")
OUT.mkdir(parents=True, exist_ok=True)


def sb_has_label(rater: str, item_id: str) -> bool:
    cfg = urllib.request.urlopen(f"{BASE}/config.js").read().decode()
    url = re.search(r'SUPABASE_URL:\s*"([^"]+)"', cfg).group(1)
    key = re.search(r'SUPABASE_PUBLISHABLE_KEY:\s*"([^"]+)"', cfg).group(1)
    req = urllib.request.Request(
        f"{url}/rest/v1/labels?rater_name=eq.{urllib.parse.quote(rater)}&item_id=eq.{urllib.parse.quote(item_id)}&select=item_id,score",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        rows = json.loads(resp.read().decode())
    return bool(rows)


def main() -> int:
    steps = []
    ok_mv = ok_sv = False
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()
        page.set_default_timeout(60000)

        # 1) Gate
        page.goto(f"{BASE}/gate.html?next=home.html", wait_until="networkidle")
        page.fill("#name", RATER)
        page.fill("#pass", PASS)
        with page.expect_navigation(timeout=30000):
            page.click("#go")
        assert "/home" in page.url, page.url
        steps.append("gate_ok")
        page.screenshot(path=str(OUT / "01_home.png"))

        # 2) Clear stale batch + assign
        page.evaluate("""() => {
          const keys = Object.keys(localStorage).filter(k => k.startsWith('vgp_batch::'));
          keys.forEach(k => localStorage.removeItem(k));
        }""")
        page.reload(wait_until="networkidle")
        btn = page.locator("#btnAssign")
        expect(btn).to_be_visible(timeout=15000)
        btn.click()
        page.wait_for_selector("#phaseGrid:not([hidden])", timeout=90000)
        steps.append("batch_assigned")
        page.screenshot(path=str(OUT / "02_batch.png"))

        # 3) Multiview: score one video (key 3 = Good)
        page.goto(f"{BASE}/index.html", wait_until="networkidle")
        page.wait_for_selector("#vidRaw", timeout=30000)
        page.wait_for_function(
            """() => {
              const s = document.getElementById('status');
              return s && !/Loading|Failed|Không tải|err/i.test(s.textContent || '')
                && document.getElementById('vidRaw') && document.getElementById('vidRaw').src;
            }""",
            timeout=60000,
        )
        before = page.locator("#nLabeled").inner_text()
        upsert = {"item_id": None, "score": None}

        def on_req(request):
            if "rest/v1/labels" in request.url and request.method == "POST":
                try:
                    body = request.post_data_json
                    if isinstance(body, list) and body:
                        upsert["item_id"] = body[0].get("item_id")
                        upsert["score"] = body[0].get("score")
                except Exception:
                    pass

        page.on("request", on_req)
        page.locator('.score[data-s="3"]').click()
        page.wait_for_timeout(3000)
        if not upsert["item_id"]:
            page.wait_for_function(
                f"() => document.getElementById('nLabeled').textContent !== '{before}'",
                timeout=20000,
            )
        steps.append(f"mv_scored:{upsert['item_id']}:{upsert['score']}")
        page.screenshot(path=str(OUT / "03_mv_scored.png"))
        mv_id = upsert["item_id"]

        # 4) Single-view rate900: score one
        upsert["item_id"] = None
        upsert["score"] = None
        page.goto(f"{BASE}/segment.html", wait_until="networkidle")
        page.wait_for_selector("#stage:not([hidden])", timeout=60000)
        page.wait_for_selector("#player", timeout=15000)
        page.wait_for_function(
            """() => {
              const s = document.getElementById('status');
              const p = document.getElementById('player');
              return s && !/Load error|Không tải|Failed/i.test(s.textContent || '')
                && p && p.src;
            }""",
            timeout=60000,
        )
        page.locator('.score[data-s="2"]').click()
        page.wait_for_timeout(3000)
        steps.append(f"sv_scored:{upsert['item_id']}:{upsert['score']}")
        page.screenshot(path=str(OUT / "04_sv_scored.png"))
        sv_id = upsert["item_id"]

        # 5) Verify Supabase
        ok_mv = bool(mv_id) and sb_has_label(RATER, mv_id)
        ok_sv = bool(sv_id) and sb_has_label(RATER, sv_id)
        steps.append(f"sb_mv={ok_mv}")
        steps.append(f"sb_sv={ok_sv}")

        # 6) Home progress
        page.goto(f"{BASE}/home.html", wait_until="networkidle")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "05_home_done.png"))
        life = page.locator("#nLife").inner_text()
        steps.append(f"life={life}")

        browser.close()

    report = {"rater": RATER, "steps": steps, "pass": ok_mv and ok_sv}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

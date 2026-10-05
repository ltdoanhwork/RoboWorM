#!/usr/bin/env python3
"""Screenshot the Mode A (multiview consistency) labeling page with real video frames.

Serves webpage/labeling over localhost, fakes the gate + batch in browser storage,
loads the HF-hosted clip, seeks to a mid frame, and captures a paper-ready PNG.

Usage:
    /tmp/pwvenv/bin/python scripts/shot_labeling_multiview.py [--item 00000] [--lang en]
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import socket
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "webpage" / "labeling"
OUT_DIR = REPO / "assets" / "paper" / "figures"

# This box has no system chromium deps (libatk/libatspi/libcups/glib are absent
# from /usr/lib64), so we run the ms-playwright chromium build with a shim dir of
# symlinks into the workspace miniforge pkgs. See .chromium-libs/.
CHROME = Path(
    os.environ.get(
        "VGP_CHROME",
        "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.cache/ms-playwright/"
        "chromium-1234/chrome-linux64/chrome",
    )
)
LIB_SHIM = Path(os.environ.get("VGP_CHROME_LIBS", str(REPO / ".chromium-libs")))


def ensure_lib_shim() -> None:
    """Re-exec with LD_LIBRARY_PATH pointing at the shim (must be set pre-exec)."""
    if not LIB_SHIM.is_dir():
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if str(LIB_SHIM) in current.split(":"):
        return
    os.environ["LD_LIBRARY_PATH"] = f"{LIB_SHIM}:{current}".rstrip(":")
    os.execv(sys.executable, [sys.executable, *sys.argv])


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):  # keep stdout readable
        pass


def serve(directory: Path) -> tuple[str, http.server.ThreadingHTTPServer]:
    handler = functools.partial(_QuietHandler, directory=str(directory))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}", httpd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", default=None, help="catalog item id to show (default: first in batch)")
    ap.add_argument("--lang", default="en", choices=["en", "vi"])
    ap.add_argument("--n-batch", type=int, default=25)
    ap.add_argument("--seek", type=float, default=0.45, help="fraction of duration to freeze at")
    ap.add_argument("--guide", default="closed", choices=["closed", "open"],
                    help="leave the 'How to rate' rubric accordion open or collapsed")
    ap.add_argument("--pick", type=int, default=0, choices=[0, 1, 2, 3],
                    help="highlight a score button (0 = none), visual only, no DB write")
    ap.add_argument("--clip", default="content", choices=["content", "viewport"],
                    help="content = crop to the workspace card, viewport = full window")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=1680)
    ap.add_argument("--height", type=int, default=1150)
    args = ap.parse_args()
    ensure_lib_shim()

    catalog = json.loads((SITE / "catalog.json").read_text())
    items = catalog["items"]
    ids = [it["id"] for it in items]
    if args.item:
        if args.item not in ids:
            raise SystemExit(f"item {args.item} not in catalog")
        start = ids.index(args.item)
    else:
        start = 0
    mv_ids = (ids[start : start + args.n_batch] + ids[:start])[: args.n_batch]

    base, httpd = serve(SITE)
    out = Path(args.out) if args.out else OUT_DIR / f"fig_labeling_multiview_{args.lang}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=str(CHROME) if CHROME.exists() else None,
            args=["--autoplay-policy=no-user-gesture-required", "--no-sandbox"],
        )
        ctx = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=2,
        )
        page = ctx.new_page()
        errors: list[str] = []
        page.on("console", lambda m: errors.append(f"[{m.type}] {m.text}") if m.type == "error" else None)
        page.on("requestfailed", lambda r: errors.append(f"FAILED {r.url} {r.failure}"))
        page.on(
            "response",
            lambda r: errors.append(f"HTTP {r.status} {r.url}") if r.status >= 400 else None,
        )

        page.goto(f"{base}/gate.html", wait_until="domcontentloaded")
        page.evaluate(
            """([rater, lang, batch]) => {
                sessionStorage.setItem('vgp_pass_ok', '1');
                localStorage.setItem('vgp_rater_name', rater);
                localStorage.setItem('vgp_lang', lang);
                localStorage.setItem('vgp_batch::' + rater.toLowerCase(), JSON.stringify(batch));
            }""",
            [
                "rater_demo",
                args.lang,
                {
                    "id": "figure-shot",
                    "createdAt": "2026-09-26T00:00:00.000Z",
                    "submittedAt": None,
                    "mvIds": mv_ids,
                    "svIds": [],
                },
            ],
        )
        page.goto(f"{base}/index.html", wait_until="load")
        page.wait_for_timeout(1500)

        # wait until both videos have decoded frames
        page.wait_for_function(
            """() => {
                const a = document.getElementById('vidRaw');
                const b = document.getElementById('vidOverlay');
                return a && b && a.readyState >= 2 && b.readyState >= 2 && a.videoWidth > 0 && b.videoWidth > 0;
            }""",
            timeout=120_000,
        )
        info = page.evaluate(
            """() => {
                const a = document.getElementById('vidRaw'), b = document.getElementById('vidOverlay');
                return {raw: a.currentSrc, overlay: b.currentSrc, dur: a.duration,
                        wr: a.videoWidth + 'x' + a.videoHeight, wo: b.videoWidth + 'x' + b.videoHeight,
                        status: document.getElementById('status').textContent};
            }"""
        )
        print("raw     :", info["raw"])
        print("overlay :", info["overlay"])
        print("sizes   :", info["wr"], "/", info["wo"], "dur", info["dur"])
        print("status  :", info["status"])

        page.evaluate(
            """(frac) => new Promise(res => {
                const a = document.getElementById('vidRaw'), b = document.getElementById('vidOverlay');
                const t = (a.duration || 1) * frac;
                let n = 0;
                const done = () => { if (++n >= 2) res(true); };
                a.addEventListener('seeked', done, {once: true});
                b.addEventListener('seeked', done, {once: true});
                a.pause(); b.pause();
                a.currentTime = t; b.currentTime = t;
                setTimeout(() => res(false), 15000);
            })""",
            args.seek,
        )
        page.wait_for_timeout(1200)
        if args.guide == "closed":
            page.evaluate("document.querySelector('details.guide')?.removeAttribute('open')")
        else:
            page.evaluate("document.querySelector('details.guide')?.setAttribute('open','')")
        if args.pick:
            page.evaluate(
                """(s) => document.querySelectorAll('.score').forEach(b =>
                       b.classList.toggle('active', Number(b.dataset.s) === s))""",
                args.pick,
            )
        page.wait_for_timeout(500)
        if args.clip == "content":
            box = page.evaluate(
                """() => {
                    const r = document.querySelector('main .workspace').getBoundingClientRect();
                    const n = document.querySelector('nav').getBoundingClientRect();
                    const pad = 18;
                    return {x: Math.max(0, r.x - pad), y: Math.max(0, n.y),
                            width: r.width + pad * 2,
                            height: (r.bottom - n.top) + pad};
                }"""
            )
            page.screenshot(path=str(out), clip=box)
        else:
            page.screenshot(path=str(out))
        print("saved", out)

        noise = ("favicon", "ERR_ABORTED", "net::ERR_CACHE")
        bad = [e for e in errors if not any(n in e for n in noise)]
        if bad:
            print("--- browser issues ---")
            for e in bad[:20]:
                print(" ", e)
        else:
            print("no browser errors")
        browser.close()
    httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

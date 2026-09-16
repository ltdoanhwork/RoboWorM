#!/usr/bin/env python3
"""Fetch every row of a Supabase table with Range pagination (PostgREST max-rows safe)."""
import json
import re
import sys
import urllib.request
from pathlib import Path

LABELING = Path(__file__).resolve().parent.parent / "webpage" / "labeling"


def supabase_config():
    txt = (LABELING / "config.js").read_text()
    url = re.search(r'SUPABASE_URL:\s*"([^"]+)"', txt).group(1)
    key = re.search(r'SUPABASE_PUBLISHABLE_KEY:\s*"([^"]+)"', txt).group(1)
    return url.rstrip("/"), key


def fetch_all(table, select="*", page=1000, extra=""):
    url, key = supabase_config()
    rows, start = [], 0
    while True:
        req = urllib.request.Request(
            f"{url}/rest/v1/{table}?select={select}{extra}",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Range-Unit": "items",
                "Range": f"{start}-{start + page - 1}",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            chunk = json.load(resp)
        rows.extend(chunk)
        if len(chunk) < page:
            return rows
        start += page


if __name__ == "__main__":
    table = sys.argv[1] if len(sys.argv) > 1 else "labels"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/{table}_fresh.json"
    rows = fetch_all(table)
    Path(out).write_text(json.dumps(rows))
    print(f"{table}: {len(rows)} rows -> {out}")

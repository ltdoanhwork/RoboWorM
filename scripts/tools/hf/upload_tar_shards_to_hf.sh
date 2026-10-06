#!/usr/bin/env bash
# Back up a many-small-file tree to HF as tar shards, ONE SHARD AT A TIME.
#
# Two constraints shaped this script:
#   * HF allows 128 commits/hour/repo, and `upload-large-folder` commits in small
#     batches -- a 100k-file tree needs thousands of commits and stalls on 429.
#     Packing into a few dozen tars fixes that.
#   * This workspace has a disk QUOTA (writing all shards up front died with
#     "Disk quota exceeded" after ~87G). So each shard is packed, uploaded, then
#     deleted before the next one -- peak extra disk = one shard.
#
#   HF_TOKEN=... bash scripts/tools/hf/upload_tar_shards_to_hf.sh sampling_dataset vgp_sampling_dataset_tar dataset
#
# Env: SHARD_BYTES (default 8G), BIG_FILE_BYTES (default = SHARD_BYTES), WORK_DIR
set -uo pipefail

SRC="${1:?usage: $0 <local-dir> <repo-name> [model|dataset]}"
REPO_NAME="${2:?}"
RTYPE="${3:-model}"

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
: "${HF_TOKEN:?HF_TOKEN is required}"
export HF_TOKEN
# The Xet backend fails on this parallel filesystem ("MerkleDB Shard error: File
# I/O error"), so fall back to plain LFS uploads.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_USER="${HF_USER:-doanh25032004}"
REPO_ID="$HF_USER/$REPO_NAME"

SHARD_BYTES="${SHARD_BYTES:-$((8 * 1024 * 1024 * 1024))}"
BIG_FILE_BYTES="${BIG_FILE_BYTES:-$SHARD_BYTES}"
WORK="${WORK_DIR:-$(dirname "$REPO_ROOT")/.hf_tars}/$REPO_NAME"
LOG_DIR="$REPO_ROOT/logs/exp_logs/hf_upload"
mkdir -p "$WORK/lists" "$LOG_DIR"
UPLOADED="$WORK/uploaded.txt"; touch "$UPLOADED"

echo "=== tar-shard backup: $SRC -> $REPO_ID ($RTYPE) ==="
echo "    shard budget: $((SHARD_BYTES / 1024 / 1024 / 1024))G, work dir: $WORK"

# ---- 1. plan: group files into shard-sized batches (batch lists are tiny) ----
if [ ! -f "$WORK/plan.done" ]; then
  echo "[1/3] planning shards"
  SRC="$SRC" WORK="$WORK" SHARD_BYTES="$SHARD_BYTES" BIG="$BIG_FILE_BYTES" python3 - <<'PY'
import os
src, work = os.environ["SRC"], os.environ["WORK"]
limit, big = int(os.environ["SHARD_BYTES"]), int(os.environ["BIG"])
SKIP = {"__pycache__", ".git", ".venv", ".venv_svg", ".cache"}
batches, cur, cur_sz, bigs = [], [], 0, []
# NOTE: do NOT wrap os.walk() in sorted() -- that materialises the whole tree up
# front and makes the `dns[:] = ...` prune below a no-op (it only works while
# os.walk is still a live generator), which silently pulls in .cache/.git/.venv.
for dp, dns, fns in os.walk(src):
    dns[:] = sorted(d for d in dns if d not in SKIP and not d.endswith(".egg-info"))
    for f in sorted(fns):
        if f.endswith((".pyc", ".pyo")) or f == ".env":
            continue
        p = os.path.join(dp, f)
        if os.path.islink(p):
            continue
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        rel = os.path.relpath(p, src)
        if sz >= big:            # too large to bundle: uploaded as a plain file
            bigs.append(rel)
            continue
        if cur and cur_sz + sz > limit:
            batches.append(cur); cur, cur_sz = [], 0
        cur.append(rel); cur_sz += sz
if cur:
    batches.append(cur)
os.makedirs(f"{work}/lists", exist_ok=True)
for i, b in enumerate(batches, 1):
    with open(f"{work}/lists/shard_{i:04d}.list", "w") as fh:
        fh.write("\n".join(b) + "\n")
with open(f"{work}/large_files.txt", "w") as fh:
    fh.write("\n".join(bigs) + ("\n" if bigs else ""))
with open(f"{work}/manifest.txt", "w") as fh:
    for i, b in enumerate(batches, 1):
        fh.write(f"shards/shard_{i:04d}.tar\t{len(b)} files\n")
    for r in bigs:
        fh.write(f"large/{r}\t1 file (plain)\n")
print(f"      {len(batches)} shard(s) + {len(bigs)} oversized file(s)")
PY
  [ $? -eq 0 ] && touch "$WORK/plan.done" || { echo "planning failed"; exit 1; }
fi
NSHARDS=$(ls -1 "$WORK/lists" | wc -l)
NBIG=$(grep -c . "$WORK/large_files.txt" 2>/dev/null | head -1)
NBIG=${NBIG:-0}
echo "    plan: $NSHARDS shard(s), $NBIG oversized file(s)"

hf repo create "$REPO_ID" --repo-type "$RTYPE" --exist-ok 2>&1 | tail -1

# ---- 2. pack -> upload -> delete, one shard at a time (quota safe) ----
echo "[2/3] packing and uploading"
fail=0
for list in "$WORK"/lists/*.list; do
  name="$(basename "$list" .list).tar"
  grep -qxF "shards/$name" "$UPLOADED" && { echo "    [$name] done"; continue; }
  tar -c -C "$SRC" -T "$list" -f "$WORK/$name" 2>"$WORK/$name.err"
  trc=$?
  if [ $trc -ne 0 ] && [ ! -s "$WORK/$name" ]; then
    echo "    [$name] TAR FAILED: $(tail -1 "$WORK/$name.err")"; rm -f "$WORK/$name"; fail=1; continue
  fi
  sz=$(stat -c %s "$WORK/$name")
  if hf upload "$REPO_ID" "$WORK/$name" "shards/$name" --repo-type "$RTYPE" \
        --commit-message "shard $name ($(( sz / 1024 / 1024 ))MB)" >/dev/null 2>&1; then
    echo "    [$name] uploaded $(( sz / 1024 / 1024 ))MB ($(wc -l < "$list") files)"
    echo "shards/$name" >> "$UPLOADED"
  else
    echo "    [$name] UPLOAD FAILED"; fail=1
  fi
  rm -f "$WORK/$name" "$WORK/$name.err"
done

while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  grep -qxF "large/$rel" "$UPLOADED" && continue
  if hf upload "$REPO_ID" "$SRC/$rel" "large/$rel" --repo-type "$RTYPE" \
        --commit-message "oversized file $rel" >/dev/null 2>&1; then
    echo "    [large/$rel] uploaded"; echo "large/$rel" >> "$UPLOADED"
  else
    echo "    [large/$rel] UPLOAD FAILED"; fail=1
  fi
done < "$WORK/large_files.txt"

# ---- 3. manifest + README, then verify ----
echo "[3/3] manifest and verification"
cat > "$WORK/README.md" <<EOF
---
tags: [backup]
---

# $REPO_NAME

Backup of \`$SRC\` from the \`video_gen_physics\` workspace. The tree has too many
small files to commit individually (the Hub allows 128 commits/hour/repo), so it is
stored as **independent tar shards** of ~$((SHARD_BYTES / 1024 / 1024 / 1024))GB under
\`shards/\`. Files too large to bundle are stored verbatim under \`large/\`.
\`manifest.txt\` lists how many files each shard holds.

Paths inside every tar are relative to \`$SRC\`, so extract all shards into one
directory that plays the role of \`$SRC\`.

Excluded: \`__pycache__\`, \`*.pyc\`, \`.git\`, \`.venv*\`, \`.cache/huggingface\`, \`.env\`,
symlinks. Credentials found in text files were redacted before upload.

## Restore

\`\`\`bash
hf download $REPO_ID --repo-type $RTYPE --local-dir ./dl
DEST=./$(basename "$SRC"); mkdir -p "\$DEST"
for t in ./dl/shards/*.tar; do tar -xf "\$t" -C "\$DEST"; done
[ -d ./dl/large ] && cp -a ./dl/large/. "\$DEST"/
\`\`\`
EOF
hf upload "$REPO_ID" "$WORK/README.md" README.md --repo-type "$RTYPE" \
    --commit-message "README" >/dev/null 2>&1 || true
hf upload "$REPO_ID" "$WORK/manifest.txt" manifest.txt --repo-type "$RTYPE" \
    --commit-message "manifest" >/dev/null 2>&1 || true

EXPECTED=$(( NSHARDS + NBIG ))
REMOTE=$(REPO="$REPO_ID" RT="$RTYPE" python3 -c "
import os
from huggingface_hub import HfApi
fs = HfApi().list_repo_files(os.environ['REPO'], repo_type=os.environ['RT'])
print(sum(1 for f in fs if f.startswith(('shards/', 'large/'))))" 2>/dev/null || echo 0)
echo "    expected payload files: $EXPECTED | remote: $REMOTE"

if [ "$REMOTE" -ge "$EXPECTED" ] && [ "$fail" -eq 0 ]; then
  echo "OK: https://huggingface.co/$( [ "$RTYPE" = dataset ] && echo datasets/ )$REPO_ID"
  rm -rf "$WORK"
  exit 0
fi
echo "INCOMPLETE ($REMOTE/$EXPECTED) -- re-run to resume (progress in $UPLOADED)"
exit 1

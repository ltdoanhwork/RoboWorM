#!/usr/bin/env bash
# Upload the SOURCE tree (git-tracked + untracked-not-ignored files, submodule
# contents included, nested .git dirs excluded) to a HuggingFace model repo.
#
#   bash scripts/upload_source_to_hf.sh
#
# Env overrides: HF_TOKEN, HF_REPO, HF_REPO_TYPE, PRIVATE=1, STAGE_DIR
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HF_REPO="${HF_REPO:-doanh25032004/video_gen_physics_backup}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
STAGE_DIR="${STAGE_DIR:-$(dirname "$REPO_ROOT")/.hf_stage_$(basename "$HF_REPO")}"
: "${HF_TOKEN:?HF_TOKEN is required}"
export HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

echo "[1/5] repo root : $REPO_ROOT"
echo "      hf repo   : $HF_REPO ($HF_REPO_TYPE)"
echo "      stage dir : $STAGE_DIR"

echo "[2/5] collecting source file list"
LIST="$(mktemp /tmp/hf_src_list.XXXXXX.z)"
trap 'rm -f "$LIST"' EXIT
{
  git ls-files --recurse-submodules -z
  git ls-files --others --exclude-standard -z
} > "$LIST"
NFILES=$(tr -dc '\0' < "$LIST" | wc -c)
echo "      files: $NFILES"

echo "[3/5] staging via hardlinks (no extra disk usage)"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
STAGE_DIR="$STAGE_DIR" python3 - "$LIST" <<'PY'
import os, sys, shutil
stage = os.environ["STAGE_DIR"]
with open(sys.argv[1], "rb") as fh:
    names = [n.decode() for n in fh.read().split(b"\0") if n]
made = 0
for rel in names:
    if not os.path.isfile(rel) or os.path.islink(rel):
        continue
    dst = os.path.join(stage, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        continue
    try:
        os.link(rel, dst)
    except OSError:
        shutil.copy2(rel, dst)
    made += 1
print(f"      staged files: {made}")
PY

# Public repo: never ship hardcoded API tokens. Break the hardlink, then redact.
echo "      redacting hardcoded tokens in staged copies"
while IFS= read -r f; do
  [ -f "$STAGE_DIR/$f" ] || continue
  cp -f --remove-destination "$f" "$STAGE_DIR/$f"
  sed -i -E 's/hf_[A-Za-z0-9]{30,}/HF_TOKEN_REDACTED/g; s/sk-[A-Za-z0-9]{20,}/API_KEY_REDACTED/g' "$STAGE_DIR/$f"
  echo "        redacted: $f"
done < <(cd "$STAGE_DIR" && grep -rIl -E 'hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}' . 2>/dev/null | sed 's|^\./||')

if (cd "$STAGE_DIR" && grep -rIq -E 'hf_[A-Za-z0-9]{30,}' . 2>/dev/null); then
  echo "ERROR: token still present in staging tree, aborting" >&2
  exit 1
fi
echo "      staged size: $(du -sh "$STAGE_DIR" | cut -f1)"

echo "[4/5] ensuring HF repo exists"
CREATE_FLAGS=(--repo-type "$HF_REPO_TYPE" --exist-ok)
[ "${PRIVATE:-0}" = "1" ] && CREATE_FLAGS+=(--private)
hf repo create "$HF_REPO" "${CREATE_FLAGS[@]}"

echo "[5/5] uploading"
hf upload "$HF_REPO" "$STAGE_DIR" . \
  --repo-type "$HF_REPO_TYPE" \
  --commit-message "Backup source tree of video_gen_physics ($(date -u +%Y-%m-%dT%H:%M:%SZ))"

echo "DONE: https://huggingface.co/$HF_REPO"

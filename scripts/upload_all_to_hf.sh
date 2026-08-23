#!/usr/bin/env bash
# Back up the WHOLE workspace to HuggingFace, split into several repos because a
# single 1.3T repo is not viable (HF recommends < 300GB / < 100k files per repo).
#
#   HF_TOKEN=... bash scripts/upload_all_to_hf.sh            # run every job
#   HF_TOKEN=... JOBS="sampling datasets" bash scripts/...   # run a subset
#
# Uses `hf upload-large-folder`, which is resumable: re-running the script picks
# up where it stopped (state kept in <local_path>/.cache/huggingface).
#
# Env: HF_TOKEN (required), HF_USER, PRIVATE=1, JOBS, NUM_WORKERS, DRY_RUN=1
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${HF_TOKEN:?HF_TOKEN is required}"
export HF_TOKEN
# The Xet backend fails on this parallel filesystem ("MerkleDB Shard error: File
# I/O error"), so fall back to plain LFS uploads.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_USER="${HF_USER:-doanh25032004}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRIVATE="${PRIVATE:-0}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="$REPO_ROOT/exp_logs/hf_upload"
STATE_DIR="$LOG_DIR/state"
STAGE_BASE="$(dirname "$REPO_ROOT")/.hf_stage"
mkdir -p "$LOG_DIR" "$STATE_DIR"

# Never upload: python caches, virtualenvs (185k files, absolute paths, all
# rebuildable), git internals (HF rejects .git* paths), uploader state, secrets.
COMMON_EXCLUDES=(
  "*__pycache__*" "*.pyc" "*.pyo"
  "*.venv/*" "*.venv_svg/*" "*/.venv/*" "*/.venv_svg/*"
  "*/.git/*" ".git/*" "*/.git" "*.gitmodules"
  "*.cache/huggingface/*" "*.env" "*.egg-info/*"
  "*.nfs*" "*.lock"
)

# job_name | local_path | repo_suffix | repo_type
#
# Only trees with FEW files belong here. HF caps commits at 128/hour, and
# `upload-large-folder` commits in small batches, so a 100k-file tree stalls in a
# 429 retry loop -- those go through scripts/upload_tar_shards_to_hf.sh instead
# (see JOBS_TAR below).
JOB_SPECS=(
  "ckpt_dreamdojo|checkpoints/dreamdojo|vgp_checkpoints_dreamdojo|model"
  "ckpt_t5|checkpoints/google-t5|vgp_checkpoints_google_t5|model"
  "ckpt_nvidia|checkpoints/nvidia|vgp_checkpoints_nvidia|model"
  "explogs|exp_logs|vgp_exp_logs|model"
  "ckpt_own|@stage_ckpt_own|vgp_checkpoints_own|model"
)

# Trees packed into tar shards: name|path|repo|type (many small files each).
JOB_SPECS_TAR=(
  "sampling|sampling_dataset|vgp_sampling_dataset_tar|dataset"
  "datasets|datasets|vgp_datasets_tar|dataset"
  "models|@stage_models|vgp_models_code_tar|model"
  "misc|@stage_misc|vgp_misc_tar|model"
)

JOBS="${JOBS:-ckpt_dreamdojo ckpt_t5 ckpt_nvidia explogs ckpt_own sampling datasets models misc}"

# ---------------------------------------------------------------- staging ----
# Hardlink trees (same filesystem => no extra disk usage) for the composite jobs.
link_tree() { # link_tree <src-rel> <stage-root> [<dst-rel>]
  local src="$1" stage="$2" dst="${3:-$1}"
  [ -e "$src" ] || { echo "      skip (missing): $src"; return 0; }
  SRC="$src" DST="$stage/$dst" python3 - <<'PY'
import os, shutil
src, dst = os.environ["SRC"], os.environ["DST"]
SKIP_DIRS = {"__pycache__", ".git", ".venv", ".venv_svg", ".cache", ".egg-info"}
n = 0
if os.path.isfile(src):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        try: os.link(src, dst)
        except OSError: shutil.copy2(src, dst)
    n = 1
else:
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        rel = os.path.relpath(root, src)
        out = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(out, exist_ok=True)
        for f in files:
            if f.endswith((".pyc", ".pyo")) or f == ".env":
                continue
            s, d = os.path.join(root, f), os.path.join(out, f)
            if os.path.islink(s) or os.path.exists(d):
                continue
            try: os.link(s, d)
            except OSError:
                try: shutil.copy2(s, d)
                except OSError: continue
            n += 1
print(f"      linked {n:>7} files  {src} -> {dst}")
PY
}

# Public repos must not carry credentials. Staged files are hardlinks, so break
# the link (cp --remove-destination) before rewriting, leaving originals intact.
SECRET_RES=(
  'hf_[A-Za-z0-9]{34,}' 'ghp_[A-Za-z0-9]{30,}' 'sk-[A-Za-z0-9]{32,}'
  'AKIA[0-9A-Z]{16}'
  '(?i)(wandb|api|secret|access)[_-]?(key|token)["'"'"' :=]+[A-Za-z0-9]{28,}'
)
redact_stage() { # redact_stage <stage-root> <src-root>
  local stage="$1" src="$2" rel n=0
  local rgargs=()
  for p in "${SECRET_RES[@]}"; do rgargs+=(-e "$p"); done
  while IFS= read -r rel; do
    [ -f "$stage/$rel" ] || continue
    grep -Iq . "$stage/$rel" 2>/dev/null || continue   # text files only
    cp -f --remove-destination "$src/$rel" "$stage/$rel"
    sed -i -E \
      -e 's/hf_[A-Za-z0-9]{34,}/HF_TOKEN_REDACTED/g' \
      -e 's/ghp_[A-Za-z0-9]{30,}/GITHUB_TOKEN_REDACTED/g' \
      -e 's/sk-[A-Za-z0-9]{32,}/API_KEY_REDACTED/g' \
      -e 's/AKIA[0-9A-Z]{16}/AWS_KEY_REDACTED/g' \
      -e 's/((wandb|WANDB|api|API|secret|SECRET|access|ACCESS)[_-]?(key|KEY|token|TOKEN)["'"'"' :=]{1,4})[A-Za-z0-9]{28,}/\1SECRET_REDACTED/g' \
      "$stage/$rel"
    echo "      redacted: $rel"
    n=$((n+1))
  done < <(cd "$stage" && rg -l -uu --no-messages --max-filesize 4M "${rgargs[@]}" . 2>/dev/null | sed 's|^\./||')
  echo "      redacted $n file(s)"
  if (cd "$stage" && rg -q -uu --no-messages --max-filesize 4M -e 'hf_[A-Za-z0-9]{34,}' -e 'ghp_[A-Za-z0-9]{30,}' . 2>/dev/null); then
    echo "      !! ERROR: credentials still present in $stage" >&2
    return 1
  fi
}

build_stage_models() {
  local stage="$STAGE_BASE/models_code"
  rm -rf "$stage"; mkdir -p "$stage"
  for d in models/*/; do
    [ "$(basename "$d")" = "__pycache__" ] && continue
    link_tree "${d%/}" "$stage" "$(basename "${d%/}")"
  done
  redact_stage "$stage" "$REPO_ROOT/models" || return 1
  echo "$stage"
}

build_stage_misc() {
  local stage="$STAGE_BASE/misc"
  rm -rf "$stage"; mkdir -p "$stage"
  for p in tmp tmp_2 output inference_times webpage finetuning dist docs tests \
           metrics methods scripts .jitcache .codegraph .agent \
           nohup.out logs_hmv_singleview_dense_makovian_fill.log \
           README.md README_cache.md README_prunning.md CLAUDE.md finetuning.md; do
    link_tree "$p" "$stage"
  done
  redact_stage "$stage" "$REPO_ROOT" || return 1
  echo "$stage"
}

build_stage_ckpt_own() {
  local stage="$STAGE_BASE/ckpt_own"
  rm -rf "$stage"; mkdir -p "$stage"
  # our own / small third-party checkpoints; the giant public mirrors get their own repos
  for p in ctrl_world pi05_droid videophy_2_auto worldscore droid mjcf \
           stabilityai openai meta-llama; do
    link_tree "checkpoints/$p" "$stage" "$p"
  done
  echo "$stage"
}

# ------------------------------------------------------------------- main ----
echo "=============================================================="
echo " HF full backup   user=$HF_USER  private=$PRIVATE  jobs=$JOBS"
echo "=============================================================="

for spec in "${JOB_SPECS[@]}"; do
  IFS='|' read -r name path suffix rtype <<< "$spec"
  case " $JOBS " in *" $name "*) ;; *) continue ;; esac

  repo_id="$HF_USER/$suffix"
  marker="$STATE_DIR/$name.done"
  log="$LOG_DIR/${name}.log"

  if [ -f "$marker" ]; then
    echo ">>> [$name] already done ($(cat "$marker")) -- skipping"
    continue
  fi

  echo ""
  echo ">>> [$name] -> $repo_id ($rtype)   $(date -u +%H:%M:%SZ)"

  local_path="$path"
  case "$path" in
    @stage_models)   echo "    building staging tree"; local_path="$(build_stage_models   | tail -1)" ;;
    @stage_misc)     echo "    building staging tree"; local_path="$(build_stage_misc     | tail -1)" ;;
    @stage_ckpt_own) echo "    building staging tree"; local_path="$(build_stage_ckpt_own | tail -1)" ;;
  esac
  [ -d "$local_path" ] || { echo "    !! missing path: $local_path -- skipping"; continue; }

  size=$(du -sh "$local_path" 2>/dev/null | cut -f1)
  echo "    local: $local_path ($size)"

  if [ "$DRY_RUN" = "1" ]; then echo "    DRY_RUN -> not uploading"; continue; fi

  create_flags=(--repo-type "$rtype" --exist-ok)
  [ "$PRIVATE" = "1" ] && create_flags+=(--private)
  hf repo create "$repo_id" "${create_flags[@]}" 2>&1 | tail -1

  up_flags=(--repo-type "$rtype" --num-workers "$NUM_WORKERS" --no-bars)
  [ "$PRIVATE" = "1" ] && up_flags+=(--private)

  start=$(date +%s)
  hf upload-large-folder "$repo_id" "$local_path" \
      "${up_flags[@]}" --exclude "${COMMON_EXCLUDES[@]}" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  dur=$(( $(date +%s) - start ))

  # `upload-large-folder` can exit 0 while still having skipped files (e.g. after
  # HTTP 429 commit-rate throttling), so compare remote vs local counts instead
  # of trusting the exit code.
  echo "    verifying remote vs local file count"
  n_local=$(REL="$local_path" python3 -c "
import os, sys
root = os.environ['REL']
skip_dirs = {'__pycache__', '.git', '.venv', '.venv_svg', '.cache'}
n = 0
for dp, dns, fns in os.walk(root):
    dns[:] = [d for d in dns if d not in skip_dirs and not d.endswith('.egg-info')]
    n += sum(1 for f in fns if not f.endswith(('.pyc', '.pyo')) and f != '.env')
print(n)")
  n_remote=$(REPO="$repo_id" RT="$rtype" python3 -c "
import os
from huggingface_hub import HfApi
fs = HfApi().list_repo_files(os.environ['REPO'], repo_type=os.environ['RT'])
print(sum(1 for f in fs if not f.startswith('.') and f != 'README.md'))" 2>/dev/null || echo 0)
  echo "    local=$n_local remote=$n_remote (rc=$rc, ${dur}s)"

  if [ "$rc" -eq 0 ] && [ "$n_remote" -ge "$n_local" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ok ${dur}s $size local=$n_local remote=$n_remote" > "$marker"
    echo "    OK [$name] -> https://huggingface.co/$( [ "$rtype" = dataset ] && echo datasets/ )$repo_id"
  else
    echo "    INCOMPLETE [$name] rc=$rc remote=$n_remote/$n_local -- re-run to resume"
  fi
done

# ------------------------------------------------- tar-shard jobs (many files) --
for spec in "${JOB_SPECS_TAR[@]}"; do
  IFS='|' read -r name path suffix rtype <<< "$spec"
  case " $JOBS " in *" $name "*) ;; *) continue ;; esac

  marker="$STATE_DIR/$name.done"
  if [ -f "$marker" ]; then
    echo ">>> [$name] already done ($(cat "$marker")) -- skipping"
    continue
  fi

  echo ""
  echo ">>> [$name] (tar shards) -> $HF_USER/$suffix   $(date -u +%H:%M:%SZ)"

  local_path="$path"
  case "$path" in
    @stage_models)   echo "    building staging tree"; local_path="$(build_stage_models   | tail -1)" ;;
    @stage_misc)     echo "    building staging tree"; local_path="$(build_stage_misc     | tail -1)" ;;
    @stage_ckpt_own) echo "    building staging tree"; local_path="$(build_stage_ckpt_own | tail -1)" ;;
  esac
  [ -d "$local_path" ] || { echo "    !! missing path: $local_path -- skipping"; continue; }

  if [ "$DRY_RUN" = "1" ]; then echo "    DRY_RUN -> not uploading"; continue; fi

  start=$(date +%s)
  bash "$REPO_ROOT/scripts/upload_tar_shards_to_hf.sh" "$local_path" "$suffix" "$rtype"
  rc=$?
  dur=$(( $(date +%s) - start ))
  if [ "$rc" -eq 0 ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ok ${dur}s tar-shards" > "$marker"
    echo "    OK [$name] in ${dur}s"
  else
    echo "    INCOMPLETE [$name] rc=$rc after ${dur}s -- re-run to resume"
  fi
done

echo ""
echo "=============== summary ==============="
for spec in "${JOB_SPECS[@]}" "${JOB_SPECS_TAR[@]}"; do
  IFS='|' read -r name path suffix rtype <<< "$spec"
  if [ -f "$STATE_DIR/$name.done" ]; then
    printf "  %-16s DONE  %s\n" "$name" "$HF_USER/$suffix"
  else
    printf "  %-16s ----  %s\n" "$name" "$HF_USER/$suffix"
  fi
done

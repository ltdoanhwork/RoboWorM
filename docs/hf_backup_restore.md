# Backup & Restore — video_gen_physics on HuggingFace

Full backup of this workspace lives on the Hub under user **`doanh25032004`**
(all repos **public**). This file is the map: what is where, how to pull it back,
and what has to be rebuilt by hand (virtualenvs, secrets).

Scripts that produced it:

| script | purpose |
|---|---|
| `scripts/tools/hf/upload_source_to_hf.sh` | source tree only (git-tracked + untracked-not-ignored) |
| `scripts/tools/hf/upload_all_to_hf.sh` | driver for every large tree; resumable, verifies remote vs local |
| `scripts/tools/hf/upload_tar_shards_to_hf.sh` | packs a many-small-file tree into tar shards, then uploads |

---

## 1. Repo map

`SRC` = path inside the workspace. `layout` = how it is stored on the Hub.

| repo (`doanh25032004/…`) | type | SRC | size | layout |
|---|---|---|---|---|
| `video_gen_physics_backup` | model | source tree (code only) | 0.4G | plain files |
| `vgp_checkpoints_nvidia` | model | `checkpoints/base/nvidia` | 209G | plain files |
| `vgp_checkpoints_dreamdojo` | model | `checkpoints/base/dreamdojo` | 187G | plain files |
| `vgp_checkpoints_google_t5` | model | `checkpoints/base/google-t5` | 84G | plain files |
| `vgp_checkpoints_own` | model | `checkpoints/{ctrl_world,pi05_droid,videophy_2_auto,stabilityai,openai,worldscore,droid,mjcf,meta-llama}` (uploaded before the 2026-10-05 re-layout; now `checkpoints/base/*` + `checkpoints/metrics/worldscore`) | 73G | plain files |
| `vgp_exp_logs` | model | `exp_logs` (uploaded before the re-layout; `*_logs/` now live in `checkpoints/finetuned/`, the rest in `logs/exp_logs/`) | 291G | plain files |
| `vgp_sampling_dataset_tar` | dataset | `sampling_dataset` | 104G | **tar shards** |
| `vgp_datasets_tar` | dataset | `datasets` | 125G | **tar shards** |
| `vgp_models_code_tar` | model | `models` (minus `.venv*`) | 22G | **tar shards** |
| `vgp_misc_tar` | model | `tmp tmp_2 output inference_times webpage finetuning methods metrics scripts docs tests dist .jitcache .codegraph .agent` + root `*.md`/logs | 13G | **tar shards** |

Deliberately **not** backed up:

- `.venv`, `.venv_svg` (~40G, 372k files, absolute paths baked in) → rebuild, see §4.
- `.git` directories, `__pycache__`, `*.pyc`, `.cache/huggingface`, `.env`.

---

## 2. Prerequisites

```bash
pip install -U "huggingface_hub[cli]"

export HF_TOKEN=<your token>            # write token for upload, any token for download
export HF_HUB_DISABLE_XET=1             # REQUIRED on this cluster, see §6
```

---

## 3. Restore

### 3.1 Source code (start here)

```bash
cd /path/to/parent
hf download doanh25032004/video_gen_physics_backup \
    --repo-type model --local-dir video_gen_physics
cd video_gen_physics
```

One file lives at a non-standard path because the repo's own `.gitignore`
(`models/Ctrl-World/dataset_example/*`) makes the Hub API skip it — move it back:

```bash
mv _gitignored_extras/models/Ctrl-World/dataset_example/extract_latent.py \
   models/Ctrl-World/dataset_example/
rm -rf _gitignored_extras
```

### 3.2 Plain-file repos (checkpoints, exp_logs)

Each repo unpacks into the directory named in the table above.

```bash
hf download doanh25032004/vgp_checkpoints_nvidia    --repo-type model   --local-dir checkpoints/base/nvidia
hf download doanh25032004/vgp_checkpoints_dreamdojo --repo-type model   --local-dir checkpoints/base/dreamdojo
hf download doanh25032004/vgp_checkpoints_google_t5 --repo-type model   --local-dir checkpoints/base/google-t5
hf download doanh25032004/vgp_checkpoints_own       --repo-type model   --local-dir checkpoints
hf download doanh25032004/vgp_exp_logs              --repo-type model   --local-dir exp_logs
```

The last two repos were uploaded with the pre-2026-10-05 layout. After downloading
them, run the re-layout snippet in [checkpoints/CHECKPOINTS.md](../checkpoints/CHECKPOINTS.md#restore-from-huggingface)
(moves `exp_logs/*_logs` to `checkpoints/finetuned/`, `checkpoints/worldscore` to
`checkpoints/metrics/`, the rest of `checkpoints/` to `checkpoints/base/`, and leaves
compat symlinks at the old paths).

`vgp_checkpoints_own` already contains the per-model subdirs (`ctrl_world/`,
`pi05_droid/`, …), so it targets `checkpoints` directly.

Only need one model? Pull a subset instead of the whole repo:

```bash
hf download doanh25032004/vgp_checkpoints_own --repo-type model \
    --include "ctrl_world/*" --local-dir checkpoints
```

### 3.3 Tar-shard repos (datasets, sampling_dataset, models, misc)

These are tar shards under `shards/` (plus any oversized file kept verbatim under
`large/`). Each repo carries a `README.md` with the same recipe and a
`manifest.txt` listing how many files every shard holds.

```bash
restore_shards () {   # restore_shards <repo> <repo-type> <dest-dir>
  local repo="$1" rtype="$2" dest="$3"
  local tmp="./_dl_$(basename "$repo")"
  hf download "$repo" --repo-type "$rtype" --local-dir "$tmp"
  mkdir -p "$dest"
  for t in "$tmp"/shards/*.tar; do
    echo "extracting $(basename "$t")"
    tar -xf "$t" -C "$dest"
  done
  [ -d "$tmp/large" ] && cp -a "$tmp/large/." "$dest"/
  rm -rf "$tmp"
}

restore_shards doanh25032004/vgp_datasets_tar         dataset datasets
restore_shards doanh25032004/vgp_sampling_dataset_tar dataset sampling_dataset
restore_shards doanh25032004/vgp_models_code_tar      model   models
restore_shards doanh25032004/vgp_misc_tar             model   .

# these two repos carry regenerable Hub metadata (see gotcha 7)
rm -rf datasets/.cache sampling_dataset/.cache
```

Paths inside every tar are relative to the source dir, so the destination has to
be named explicitly. `vgp_misc_tar` is the exception: it was packed from a staging
tree whose children already are `tmp/`, `output/`, `metrics/`, … so it extracts at
the workspace root.

Needs roughly one shard (~8G) of scratch space plus the extracted result.

---

## 4. Rebuild the virtualenvs (not backed up)

Four envs, never mix them (see `.cursor/rules/model-environments.mdc`). Exact
package lists as of the backup are frozen in `docs/env_freeze/`:

| env | python | freeze file |
|---|---|---|
| `models/dreamgen/.venv` | 3.12.13 | `docs/env_freeze/dreamgen.txt` |
| `models/dreamgen/.venv_svg` | 3.12.13 | `docs/env_freeze/dreamgen__svg.txt` |
| `models/DreamDojo/.venv` | 3.10.19 | `docs/env_freeze/DreamDojo.txt` |
| `models/Ctrl-World/.venv` | 3.10.19 | `docs/env_freeze/Ctrl-World.txt` |

```bash
python3.12 -m venv models/dreamgen/.venv
models/dreamgen/.venv/bin/pip install -r docs/env_freeze/dreamgen.txt
```

Notes:

- Install torch first with the right CUDA wheel index (dreamgen: torch 2.6+cu124,
  DreamDojo: torch 2.7+cu128), then the rest — otherwise pip resolves a CPU build.
- `.venv_svg` exists only because SVG2 needs **flashinfer 0.3.1** for the real
  block-sparse kernel; the default `.venv` has 0.2.5 and falls back to a dense
  mask that OOMs. Keep them separate.
- The freeze files are a snapshot including editable/local installs — lines
  pointing at local paths (`-e /pfss/...`) need re-pointing at the new checkout.

## 5. Secrets (must be re-created)

Credentials were **redacted** in the uploaded copies, so a restored tree has
placeholders instead of keys. Files affected:

| file | placeholder |
|---|---|
| `scripts/tools/hf/upload_sampling_to_hf.sh` | `HF_TOKEN_REDACTED` |
| `models/dreamgen/script_cache.md` | `HF_TOKEN_REDACTED` |
| `models/dreamgen/bash-running-script/Finetuning-bash-dreamgen/{debug,realworld_training,cosmos-predict-training,robocasa_100_training}.sh` | `SECRET_REDACTED` (wandb API key) |

Also absent by design: the workspace `.env`. Recreate it and prefer reading
tokens from the environment over hardcoding them.

## 6. Gotchas learned the hard way

1. **`HF_HUB_DISABLE_XET=1` is mandatory here.** The Xet backend dies on this
   parallel filesystem with `Data processing error: MerkleDB Shard error: File
   I/O error`. Without it, uploads silently stall after a handful of files.
   Cost: throughput drops from ~230MB/s to ~40MB/s, but it completes.
2. **The Hub allows 128 commits/hour/repo.** `hf upload-large-folder` commits in
   small batches, so a 100k-small-file tree needs thousands of commits and gets
   stuck in a 429 retry loop (one run burned 12h and produced a 31MB log of
   retries). That is exactly why `datasets`, `sampling_dataset`, `models` and
   `misc` are stored as ~20G tar shards — a few dozen commits instead.
3. **`hf upload-large-folder` exits 0 even when files were skipped.** Never trust
   the exit code: `upload_all_to_hf.sh` compares remote vs local file counts and
   only then writes `logs/exp_logs/hf_upload/state/<job>.done`. Re-running the script
   resumes and re-verifies.
4. **The Hub API applies the repo's own `.gitignore` on upload.** Files matching
   it (e.g. `models/Ctrl-World/dataset_example/*`) get silently dropped — hence
   the `_gitignored_extras/` workaround in §3.1.
5. Per-file limit is 50GB. The two largest real files here
   (`checkpoints/base/google-t5/t5-11b/{pytorch_model.bin,tf_model.h5}`, 42.1G each)
   fit, but anything bigger must be split.
6. **This workspace has a disk quota** separate from the filesystem's free space
   (`df` showed 339T free while `split` died with "Disk quota exceeded" after
   ~185G of staging). That is why `upload_tar_shards_to_hf.sh` packs, uploads and
   deletes **one shard at a time** — peak extra disk is a single shard (~8G)
   rather than the whole tree.
7. **`sorted(os.walk(...))` breaks directory pruning.** `sorted()` materialises
   the whole tree before iteration, so the usual `dns[:] = [...]` prune becomes a
   no-op. `vgp_datasets_tar` / `vgp_sampling_dataset_tar` were packed with this
   bug and therefore also contain each tree's `.cache/huggingface` (~204MB of
   regenerable Hub metadata, 376k tiny files). Harmless — no real data is missing
   and the byte totals match — but on restore you can delete
   `datasets/.cache` and `sampling_dataset/.cache`. Fixed in the script.

## 7. Re-running the backup

```bash
export HF_TOKEN=<write token>

bash scripts/tools/hf/upload_source_to_hf.sh      # source only
bash scripts/tools/hf/upload_all_to_hf.sh         # everything (skips jobs already done)

JOBS="explogs misc" bash scripts/tools/hf/upload_all_to_hf.sh   # subset
DRY_RUN=1 bash scripts/tools/hf/upload_all_to_hf.sh             # show plan, upload nothing
```

State/logs: `logs/exp_logs/hf_upload/` (`state/<job>.done` markers, `<job>.log`,
`driver_*.log`). Delete a marker to force that job to run again.

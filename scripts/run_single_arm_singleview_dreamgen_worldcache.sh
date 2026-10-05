#!/usr/bin/env bash
# =============================================================================
# run_single_arm_singleview_dreamgen_worldcache.sh
# DreamGen single-arm SINGLEVIEW (RoboChallenge global static view, static16fps
# checkpoint) with WorldCache acceleration.
#
# This mirrors the dense baseline run_single_arm_singleview_dreamgen_static16fps.sh
# EXACTLY in logic/paths (same prepare step, same LVG autoregressive inference,
# same checkpoint / resolution / fps / guidance / aspect / cond frames), and only:
#   1. forces single-GPU (cache backends require context_parallel_size == 1, so
#      no torchrun) and
#   2. adds the WorldCache flags and
#   3. swaps the leading sampling_dataset/dense/ for sampling_dataset/worldcache/.
#
# Output (method = top-level dir, per sampling-dataset-layout rule):
#   sampling_dataset/worldcache/single_arm/output/singleview/dreamgen/{split}/iter_XXXXXX_static16fps/{view}/
#
# Usage (repo root):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_singleview_dreamgen_worldcache.sh
#   CUDA_VISIBLE_DEVICES=0 SPLITS="makovian" NUM_CHUNKS="2 3" bash scripts/run_single_arm_singleview_dreamgen_worldcache.sh
#   WORLDCACHE_REL_L1_THRESH=0.05 bash scripts/run_single_arm_singleview_dreamgen_worldcache.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
# Repo root must precede models/dreamgen so `methods.cache_strategy` is importable
# (config imports it at module load, before runtime sys.path injection).
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

# --- Checkpoint (from single_arm_static_16fps training) ---
CKPT_BASE="${CKPT_BASE:-${ROOT}/exp_logs/dreamgen_logs/posttraining/singleview/single_arm_static_singleview_16fps_v2}"
CKPT_MODEL_DIR="${CKPT_BASE}/checkpoints_40_per/model"

# Default to the latest checkpoint recorded by the trainer if ITER_NAME unset.
if [[ -z "${ITER_NAME:-}" ]]; then
  latest_file="${CKPT_BASE}/checkpoints_40_per/latest_checkpoint.txt"
  if [[ -f "${latest_file}" ]]; then
    ITER_NAME="$(basename "$(cat "${latest_file}")" .pt)"
  else
    ITER_NAME="iter_000090000"
  fi
fi

DIT_PATH="${DIT_PATH:-${CKPT_MODEL_DIR}/${ITER_NAME}.pt}"

if [[ ! -f "${DIT_PATH}" ]]; then
  echo "ERROR: Checkpoint not found at ${DIT_PATH}" >&2
  echo "Available checkpoints:" >&2
  ls "${CKPT_MODEL_DIR}/"*.pt 2>/dev/null >&2 || true
  exit 1
fi

# --- Inference settings (identical to dense baseline) ---
# Cache backends are single-GPU only.
NUM_GPUS=1
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
ASPECT_RATIO="${ASPECT_RATIO:-4:3}"
NUM_CONDITIONAL_FRAMES="${NUM_CONDITIONAL_FRAMES:-1}"

# --- Views to use for inference (must match training: RoboChallenge global view) ---
VIEWS="${VIEWS:-global}"

# --- Dataset paths ---
# Inputs are method-specific because the prepare step embeds the (worldcache)
# output_video path inside each batch JSON. First_frames extraction is cheap.
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT}/datasets/single_arm/singleview}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/worldcache/single_arm/input/singleview/dreamgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/worldcache/single_arm/output/singleview/dreamgen}"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-2 3}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
PROMPT_REFINE="${PROMPT_REFINE:-0}"
PROMPT_PREFIX="${PROMPT_PREFIX:-}"
SOURCE_FPS=15

OUT_ITER="${ITER_NAME}_static16fps"

# --- WorldCache defaults (README_cache.md starting points, 28-block) ---
WORLDCACHE_NUM_STEPS="${WORLDCACHE_NUM_STEPS:-35}"
WORLDCACHE_REL_L1_THRESH="${WORLDCACHE_REL_L1_THRESH:-0.08}"
WORLDCACHE_RET_RATIO="${WORLDCACHE_RET_RATIO:-0.5}"
WORLDCACHE_PROBE_DEPTH="${WORLDCACHE_PROBE_DEPTH:-4}"
WORLDCACHE_MOTION_SENSITIVITY="${WORLDCACHE_MOTION_SENSITIVITY:-4.0}"
WORLDCACHE_DYNAMIC_DECAY="${WORLDCACHE_DYNAMIC_DECAY:-1}"

echo "============================================"
echo "DreamGen Single-arm Singleview — WorldCache (single-GPU)"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Iter tag:   ${OUT_ITER}"
echo "  Views:      ${VIEWS} (from single_arm/singleview dataset)"
echo "  Model:      ${MODEL_SIZE} | Resolution: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Guidance:   ${GUIDANCE}"
echo "  Cond frames: ${NUM_CONDITIONAL_FRAMES} | Aspect: ${ASPECT_RATIO}"
echo "  Prompt refine: ${PROMPT_REFINE} | Prefix: '${PROMPT_PREFIX}'"
echo "  Splits:     ${SPLITS}"
echo "  Chunks:     ${NUM_CHUNKS}"
echo "  WorldCache: thresh=${WORLDCACHE_REL_L1_THRESH} ret=${WORLDCACHE_RET_RATIO} probe=${WORLDCACHE_PROBE_DEPTH} motion=${WORLDCACHE_MOTION_SENSITIVITY} decay=${WORLDCACHE_DYNAMIC_DECAY} steps=${WORLDCACHE_NUM_STEPS}"
echo "  Output:     ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
echo "============================================"

# =============================================================================
# Step 1: Prepare batch_input JSONs from static view (same logic as dense)
# =============================================================================
echo ""
echo "--- Step 1: Preparing batch inputs from views: ${VIEWS} ---"

prepare_batch_inputs() {
  local split="$1"
  local view="$2"
  local view_cam="observation.images.${view}"
  local split_dir="${SOURCE_ROOT}/${split}"
  local input_dir="${INPUT_ROOT}/${view}/${split}"
  local frames_dir="${input_dir}/first_frames"

  if [[ ! -d "${split_dir}" ]]; then
    echo "  [SKIP] ${split}/${view}: source dir not found at ${split_dir}"
    return
  fi

  mkdir -p "${frames_dir}"

  python3 - "${split_dir}" "${input_dir}" "${frames_dir}" "${view_cam}" "${OUTPUT_ROOT}/${split}/${OUT_ITER}/${view}" "${SOURCE_FPS}" <<'PY'
import json, math, re, subprocess, sys
from collections import defaultdict
from pathlib import Path

split_dir = Path(sys.argv[1])
input_dir = Path(sys.argv[2])
frames_dir = Path(sys.argv[3])
view_cam = sys.argv[4]
output_dir = sys.argv[5]
source_fps = float(sys.argv[6])

FRAMES_PER_CHUNK = 93
FRAMES_PER_ADDITIONAL = 92
MODEL_FPS = 16

# Load tasks from meta
def load_tasks(split_dir):
    ep_task = {}
    episodes_file = split_dir / "meta" / "episodes.jsonl"
    tasks_file = split_dir / "meta" / "tasks.jsonl"
    if not episodes_file.exists():
        return ep_task
    with episodes_file.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            ep_idx = rec.get("episode_index", -1)
            tasks = rec.get("tasks", [])
            if ep_idx >= 0 and tasks and tasks[0]:
                ep_task[ep_idx] = tasks[0]
    if ep_task:
        return ep_task
    task_map = {}
    if tasks_file.exists():
        with tasks_file.open() as f:
            for line in f:
                line = line.strip()
                if not line: continue
                rec = json.loads(line)
                if rec.get("task"):
                    task_map[rec["task_index"]] = rec["task"]
    if task_map and episodes_file.exists():
        with episodes_file.open() as f:
            for line in f:
                line = line.strip()
                if not line: continue
                rec = json.loads(line)
                ep_idx = rec.get("episode_index", -1)
                t_idx = rec.get("task_index", -1)
                if ep_idx >= 0 and t_idx in task_map:
                    ep_task[ep_idx] = task_map[t_idx]
    return ep_task

# Find episodes
videos_root = split_dir / "videos"
episodes = []
for mp4 in sorted(videos_root.glob(f"chunk-*/{view_cam}/*.mp4")):
    ep_match = re.search(r"(\d+)$", mp4.stem)
    if ep_match:
        episodes.append((int(ep_match.group(1)), mp4))

if not episodes:
    print(f"  No episodes found for view {view_cam}")
    sys.exit(0)

ep_tasks = load_tasks(split_dir)

PROMPT_PREFIXES = [
    "A robot arm",
    "A single robotic arm",
    "A tabletop robot arm",
    "A robot manipulator",
]
TASK_REPHRASINGS = {
    "place the cup on the coaster": "places the cup on the coaster",
    "turn on the light switch": "turns on the light switch",
    "press the pink, blue, and green buttons in sequence":
        "presses the pink, blue, and green buttons in sequence",
    "pull out a tissue, wipe the stains on the table clean, "
    "and then discard the tissue into the trash bin":
        "pulls out a tissue, wipes the stains on the table clean, "
        "and then discards the tissue into the trash bin",
}
GENERIC_CAPTION = "A robot arm performing a manipulation task."

def build_caption(raw_task, ep_id):
    if not raw_task or raw_task == GENERIC_CAPTION:
        return GENERIC_CAPTION
    prefix = PROMPT_PREFIXES[ep_id % len(PROMPT_PREFIXES)]
    task_phrase = TASK_REPHRASINGS.get(raw_task, raw_task)
    return f"{prefix} {task_phrase}."

# Group by num_chunks
chunks_groups = defaultdict(list)
for ep_id, video_path in episodes:
    frame_png = Path(frames_dir) / f"episode_{ep_id:06d}.png"
    if not frame_png.exists():
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vf", "scale=640:480", "-vframes", "1", "-q:v", "2", str(frame_png)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            print(f"  [WARN] Failed to extract frame: {video_path.name}")
            continue

    # Get frame count
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "stream=nb_frames",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    n_frames = None
    if result.returncode == 0 and result.stdout.strip():
        try:
            n_frames = int(result.stdout.strip().split("\n")[0])
        except ValueError:
            pass

    if n_frames:
        target_frames = n_frames * (MODEL_FPS / source_fps)
        if target_frames <= FRAMES_PER_CHUNK:
            n_chunks = 1
        else:
            n_chunks = math.ceil((target_frames - FRAMES_PER_CHUNK) / FRAMES_PER_ADDITIONAL) + 1
    else:
        n_chunks = 2

    raw_task = ep_tasks.get(ep_id, GENERIC_CAPTION)
    caption = build_caption(raw_task, ep_id)
    out_video = f"{output_dir}/episode_{ep_id:06d}.mp4"

    chunks_groups[n_chunks].append({
        "input_video": str(frame_png),
        "prompt": caption,
        "output_video": out_video,
    })

# Write batch files
total = 0
for nc, items in sorted(chunks_groups.items()):
    out_json = Path(input_dir) / f"batch_input_{nc}chunks.json"
    out_json.write_text(json.dumps(items, indent=2))
    total += len(items)
    print(f"  {out_json.name}: {len(items)} episodes")
print(f"  Total: {total} episodes across {len(chunks_groups)} chunk groups")
PY
}

for split in ${SPLITS}; do
  for view in ${VIEWS}; do
    prepare_batch_inputs "${split}" "${view}"
  done
done

# =============================================================================
# Step 2: Run inference (WorldCache, single-GPU)
# =============================================================================
echo ""
echo "--- Step 2: Running inference ---"

WORLDCACHE_FLAGS=(
  --worldcache-enabled
  --worldcache-num-steps          "${WORLDCACHE_NUM_STEPS}"
  --worldcache-rel-l1-thresh      "${WORLDCACHE_REL_L1_THRESH}"
  --worldcache-ret-ratio          "${WORLDCACHE_RET_RATIO}"
  --worldcache-probe-depth        "${WORLDCACHE_PROBE_DEPTH}"
  --worldcache-motion-sensitivity "${WORLDCACHE_MOTION_SENSITIVITY}"
)
if [[ "${WORLDCACHE_DYNAMIC_DECAY}" == "1" ]]; then
  WORLDCACHE_FLAGS+=(--worldcache-dynamic-decay)
fi

run_inference() {
  local batch_json="$1"
  local label="$2"
  local num_chunks="$3"

  if [[ ! -f "${batch_json}" ]]; then
    echo "  [SKIP] ${label}: batch file not found (${batch_json})"
    return
  fi

  local item_count
  item_count="$(python3 -c "import json; print(len(json.load(open('${batch_json}'))))")"

  local tmp_batch_json=""
  local run_json="${batch_json}"
  tmp_batch_json="$(mktemp --suffix=_dreamgen_worldcache.json)"
  local pending
  pending="$(python3 - "${batch_json}" "${tmp_batch_json}" "${FORCE_REGENERATE}" "${MAX_EPISODES}" <<'PY'
import json, sys
from pathlib import Path
src, dst, force, max_eps = sys.argv[1:5]
items = json.loads(Path(src).read_text())
pending = list(items) if force == "1" else [x for x in items if not Path(x["output_video"]).is_file()]
if max_eps:
    pending = pending[: int(max_eps)]
Path(dst).write_text(json.dumps(pending, indent=2))
print(len(pending))
PY
)"
  if [[ "${pending}" -eq 0 ]]; then
    echo "  [SKIP] ${label}: all ${item_count} episodes done"
    rm -f "${tmp_batch_json}"
    return
  fi
  echo "  === Running: ${label} (${pending}/${item_count} pending, chunks=${num_chunks}, cap=${MAX_EPISODES:-none}) ==="
  run_json="${tmp_batch_json}"

  local pr_flag=""
  if [[ "${PROMPT_REFINE}" == "1" ]]; then
    pr_flag="--enable_prompt_refiner_per_chunk"
  fi

  python -m models.dreamgen.examples.video2world_gr00t_lvg \
    --model_size "${MODEL_SIZE}" \
    --gr00t_variant droid \
    --dit_path "${DIT_PATH}" \
    --batch_input_json "${run_json}" \
    --num_chunks "${num_chunks}" \
    --guidance "${GUIDANCE}" \
    --num_gpus "${NUM_GPUS}" \
    --aspect_ratio "${ASPECT_RATIO}" \
    --num_conditional_frames "${NUM_CONDITIONAL_FRAMES}" \
    --prompt_prefix "${PROMPT_PREFIX}" \
    --disable_guardrail \
    --no_skip_existing \
    ${pr_flag} \
    "${WORLDCACHE_FLAGS[@]}"

  [[ -n "${tmp_batch_json}" ]] && rm -f "${tmp_batch_json}"
  echo "  === Done: ${label} ==="
}

for split in ${SPLITS}; do
  for view in ${VIEWS}; do
    input_dir="${INPUT_ROOT}/${view}/${split}"
    if [[ ! -d "${input_dir}" ]]; then
      echo "  [SKIP] ${view}/${split}: input dir not found at ${input_dir}"
      continue
    fi

    for nc in ${NUM_CHUNKS}; do
      batch_json="${input_dir}/batch_input_${nc}chunks.json"
      run_inference "${batch_json}" "single_arm_worldcache/${view}/${split}/${nc}chunks" "${nc}"
    done
  done
done

echo ""
echo "============================================"
echo "Done! Results: ${OUTPUT_ROOT}/{split}/${OUT_ITER}/{view}/"
echo "============================================"

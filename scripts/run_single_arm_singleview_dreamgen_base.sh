#!/usr/bin/env bash
# =============================================================================
# Test DreamGen single-arm using the ORIGINAL (base) Cosmos-Predict2-2B
# Video2World checkpoint — i.e. the pretrained model BEFORE any RoboChallenge
# posttraining. Use this to compare base-vs-posttrained generation on the same
# global static-view inputs.
#
# Base checkpoint:
#   checkpoints/base/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt
#
# Same pipeline as run_single_arm_singleview_dreamgen_static16fps.sh, but:
#   - DIT_PATH points to the base .pt (no posttraining)
#   - Outputs are tagged "base_2b" so they DON'T collide with iter_* results
#
# Output:
#   sampling_dataset/dense/single_arm/output/singleview/dreamgen/{split}/base_2b/global/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2,3 PROMPT_REFINE=0 bash scripts/run_single_arm_singleview_dreamgen_base.sh
#   NUM_CHUNKS="2 3" SPLITS="makovian" bash scripts/run_single_arm_singleview_dreamgen_base.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/dreamgen:${ROOT}:${PYTHONPATH:-}"

# --- Checkpoint (ORIGINAL base Cosmos-Predict2-2B-Video2World, 480p/16fps) ---
DIT_PATH="${DIT_PATH:-${ROOT}/checkpoints/base/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt}"

if [[ ! -f "${DIT_PATH}" ]]; then
  echo "ERROR: Base checkpoint not found at ${DIT_PATH}" >&2
  exit 1
fi

# --- Inference settings ---
NUM_GPUS="${NUM_GPUS:-2}"
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
MASTER_PORT="${MASTER_PORT:-29560}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"

# --- Views (same global static view as posttraining inference) ---
VIEWS="${VIEWS:-global}"

# --- Dataset paths ---
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT}/datasets/single_arm/singleview}"
INPUT_ROOT="${ROOT}/sampling_dataset/dense/single_arm/input/singleview/dreamgen"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/dense/single_arm/output/singleview/dreamgen}"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-2 3}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
PROMPT_REFINE="${PROMPT_REFINE:-1}"
SOURCE_FPS=15

# Tag outputs so base-model results don't collide with posttrained iter_* ones.
OUT_ITER="${OUT_ITER:-base_2b}"

echo "============================================"
echo "DreamGen Single-arm — ORIGINAL base 2B ckpt (no posttraining)"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Iter tag:   ${OUT_ITER}"
echo "  Views:      ${VIEWS} (from single_arm/singleview dataset)"
echo "  Model:      ${MODEL_SIZE} | Resolution: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Guidance:   ${GUIDANCE}"
echo "  Splits:     ${SPLITS}"
echo "  Chunks:     ${NUM_CHUNKS}"
echo "  Output:     ${OUTPUT_ROOT}/{split}/${OUT_ITER}/{view}/"
echo "============================================"

# =============================================================================
# Step 1: Prepare batch_input JSONs from static view
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

chunks_groups = defaultdict(list)
for ep_id, video_path in episodes:
    frame_png = Path(frames_dir) / f"episode_{ep_id:06d}.png"
    if not frame_png.exists():
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vf", "scale=768:432", "-vframes", "1", "-q:v", "2", str(frame_png)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            print(f"  [WARN] Failed to extract frame: {video_path.name}")
            continue

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

    caption = ep_tasks.get(ep_id, "A robot arm performing a manipulation task.")
    out_video = f"{output_dir}/episode_{ep_id:06d}.mp4"

    chunks_groups[n_chunks].append({
        "input_video": str(frame_png),
        "prompt": caption,
        "output_video": out_video,
    })

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
# Step 2: Run inference
# =============================================================================
echo ""
echo "--- Step 2: Running inference ---"

run_inference() {
  local batch_json="$1"
  local label="$2"
  local num_chunks="$3"
  local sub_path="$4"

  if [[ ! -f "${batch_json}" ]]; then
    echo "  [SKIP] ${label}: batch file not found (${batch_json})"
    return
  fi

  local out_dir="${OUTPUT_ROOT}/${sub_path}"
  mkdir -p "${out_dir}"

  local item_count
  item_count="$(python3 -c "import json; print(len(json.load(open('${batch_json}'))))")"

  if [[ "${FORCE_REGENERATE}" != "1" ]]; then
    local tmp_batch_json
    tmp_batch_json="$(mktemp --suffix=_dreamgen_base.json)"
    local pending
    pending="$(python3 -c "
import json
from pathlib import Path
items = json.loads(Path('${batch_json}').read_text())
pending = [x for x in items if not Path(x['output_video']).is_file()]
Path('${tmp_batch_json}').write_text(json.dumps(pending, indent=2))
print(len(pending))
")"
    if [[ "${pending}" -eq 0 ]]; then
      echo "  [SKIP] ${label}: all ${item_count} episodes done"
      rm -f "${tmp_batch_json}"
      return
    fi
    echo "  === Running: ${label} (${pending}/${item_count} pending, chunks=${num_chunks}) ==="
    local run_json="${tmp_batch_json}"
  else
    echo "  === Running: ${label} (${item_count} episodes, chunks=${num_chunks}) ==="
    local run_json="${batch_json}"
    local tmp_batch_json=""
  fi

  local pr_flag=""
  if [[ "${PROMPT_REFINE}" == "1" ]]; then
    pr_flag="--enable_prompt_refiner_per_chunk"
  fi

  if [[ "${NUM_GPUS}" -gt 1 ]]; then
    torchrun \
      --nproc_per_node="${NUM_GPUS}" \
      --master_port="${MASTER_PORT}" \
      -m models.dreamgen.examples.video2world_gr00t_lvg \
      --model_size "${MODEL_SIZE}" \
      --gr00t_variant droid \
      --dit_path "${DIT_PATH}" \
      --batch_input_json "${run_json}" \
      --num_chunks "${num_chunks}" \
      --guidance "${GUIDANCE}" \
      --num_gpus "${NUM_GPUS}" \
      --aspect_ratio "${ASPECT_RATIO}" \
      --disable_guardrail \
      --no_skip_existing \
      ${pr_flag}
  else
    python -m models.dreamgen.examples.video2world_gr00t_lvg \
      --model_size "${MODEL_SIZE}" \
      --gr00t_variant droid \
      --dit_path "${DIT_PATH}" \
      --batch_input_json "${run_json}" \
      --num_chunks "${num_chunks}" \
      --guidance "${GUIDANCE}" \
      --num_gpus "${NUM_GPUS}" \
      --aspect_ratio "${ASPECT_RATIO}" \
      --disable_guardrail \
      --no_skip_existing \
      ${pr_flag}
  fi

  MASTER_PORT=$((MASTER_PORT + 1))
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
      run_inference "${batch_json}" "single_arm_base/${view}/${split}/${nc}chunks" "${nc}" "${split}/${OUT_ITER}/${view}"
    done
  done
done

echo ""
echo "============================================"
echo "Done! Results: ${OUTPUT_ROOT}/{split}/${OUT_ITER}/{view}/"
echo "============================================"

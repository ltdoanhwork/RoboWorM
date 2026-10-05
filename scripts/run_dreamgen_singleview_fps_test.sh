#!/usr/bin/env bash
# =============================================================================
# Test DreamGen singleview inference with fps=10 (state_t=16) vs default fps=16.
#
# Hypothesis: fps=10 produces "slower" videos since each frame covers more time.
# Training data had native fps (15/20/30) with fps conditioning — testing if
# fps=10 better matches the slow robot execution pace.
#
# Saves to: tmp/2/{embodiment}/{split}/fps_{FPS}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_dreamgen_singleview_fps_test.sh
#   CUDA_VISIBLE_DEVICES=0,1 FPS=16 bash scripts/run_dreamgen_singleview_fps_test.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/dreamgen:${PYTHONPATH:-}"

# --- Checkpoint ---
# BROKEN (2026-10-05 refactor): the combined all_singleview run does not exist on disk (not in
# checkpoints/finetuned/ nor the HF backup). Set CKPT_BASE/DIT_PATH explicitly to run this script.
CKPT_BASE="${ROOT}/checkpoints/finetuned/dreamgen/posttraining/singleview/all_singleview"
ITER_NAME="${ITER_NAME:-iter_000090000}"
DIT_PATH="${DIT_PATH:-${CKPT_BASE}/checkpoints_40_per/model/${ITER_NAME}.pt}"

if [[ ! -f "${DIT_PATH}" ]]; then
  ALT_PATH="${CKPT_BASE}/checkpoints_40_per/backup_60k_70k/model/${ITER_NAME}.pt"
  if [[ -f "${ALT_PATH}" ]]; then
    DIT_PATH="${ALT_PATH}"
  else
    echo "ERROR: Checkpoint not found at ${DIT_PATH}" >&2
    exit 1
  fi
fi

# --- Settings ---
NUM_GPUS="${NUM_GPUS:-2}"
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-10}"
GUIDANCE="${GUIDANCE:-7}"
MASTER_PORT="${MASTER_PORT:-29510}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
MAX_EPISODES="${MAX_EPISODES:-3}"

# --- Output ---
SAVE_ROOT="${ROOT}/tmp/2"

echo "============================================"
echo "DreamGen FPS Test"
echo "  Checkpoint: ${DIT_PATH}"
echo "  FPS:        ${FPS}"
echo "  Resolution: ${RESOLUTION}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Guidance:   ${GUIDANCE}"
echo "  Max eps:    ${MAX_EPISODES}"
echo "  Save root:  ${SAVE_ROOT}"
echo "============================================"

# --- Embodiment configs ---
# source_fps is used for num_chunks calculation
declare -A SOURCE_FPS=( [bimanual]=30 [single_arm]=15 [humanoid]=20 )
declare -A CAMERAS=( [bimanual]="observation.images.cam_high" [single_arm]="observation.images.wrist_left" [humanoid]="observation.images.ego_view_freq20" )

# --- Prepare inputs and run per embodiment ---
for embodiment in bimanual single_arm humanoid; do
  source_fps="${SOURCE_FPS[$embodiment]}"
  camera="${CAMERAS[$embodiment]}"

  for split in makovian non_makovian; do
    dataset_dir="${ROOT}/datasets/${embodiment}/singleview/${split}"
    if [[ ! -d "${dataset_dir}" ]]; then
      echo "[SKIP] ${embodiment}/${split}: dataset not found"
      continue
    fi

    output_dir="${SAVE_ROOT}/${embodiment}/${split}/fps_${FPS}"
    input_dir="${SAVE_ROOT}/${embodiment}/${split}/input"
    mkdir -p "${output_dir}" "${input_dir}/first_frames"

    echo ""
    echo "=== ${embodiment}/${split} (source_fps=${source_fps}, inference_fps=${FPS}) ==="

    # Prepare batch_input.json: extract first frames + compute chunks
    batch_json="${input_dir}/batch_input_fps${FPS}.json"
    python3 - "${dataset_dir}" "${input_dir}" "${output_dir}" "${camera}" \
              "${source_fps}" "${FPS}" "${MAX_EPISODES}" "${batch_json}" "${embodiment}" <<'PYTHON'
import json, math, os, re, subprocess, sys
from pathlib import Path

dataset_dir = Path(sys.argv[1])
input_dir = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
camera = sys.argv[4]
source_fps = float(sys.argv[5])
target_fps = int(sys.argv[6])
max_episodes = int(sys.argv[7])
batch_json = Path(sys.argv[8])
embodiment = sys.argv[9]

FRAMES_PER_CHUNK = 93
FRAMES_PER_ADDITIONAL = 92

# For fps=10: state_t=16, fewer latent frames → 61 output frames per chunk
# For fps=16: state_t=24, normal → 93 output frames per chunk
if target_fps == 10:
    FRAMES_PER_CHUNK = 61
    FRAMES_PER_ADDITIONAL = 60

def compute_num_chunks(n_frames, src_fps):
    target_frames = n_frames * (target_fps / src_fps)
    if target_frames <= FRAMES_PER_CHUNK:
        return 1
    return math.ceil((target_frames - FRAMES_PER_CHUNK) / FRAMES_PER_ADDITIONAL) + 1

def get_frame_count(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "stream=nb_frames",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        try:
            return int(result.stdout.strip().split("\n")[0])
        except ValueError:
            pass
    return None

def extract_first_frame(video_path, out_png):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vf", "scale=768:432",
         "-vframes", "1", "-q:v", "2", str(out_png)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_png.exists()

# Find episodes
videos_root = dataset_dir / "videos"
if embodiment == "bimanual":
    # bimanual has task subdirs
    all_episodes = []
    for task_dir in sorted(dataset_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name in ("meta", "captions"):
            continue
        task_name = task_dir.name
        task_videos = task_dir / "videos"
        if not task_videos.exists():
            continue
        for mp4 in sorted(task_videos.glob(f"chunk-*/{camera}/*.mp4")):
            ep_match = re.search(r"(\d+)$", mp4.stem)
            if ep_match:
                all_episodes.append((int(ep_match.group(1)), task_name, mp4))
    all_episodes = all_episodes[:max_episodes]
    
    items = []
    for ep_id, task_name, video_path in all_episodes:
        frame_png = input_dir / "first_frames" / f"{task_name}__episode_{ep_id:06d}.png"
        if not frame_png.exists():
            if not extract_first_frame(video_path, frame_png):
                continue
        n_frames = get_frame_count(video_path)
        n_chunks = compute_num_chunks(n_frames, source_fps) if n_frames else 5
        name = task_name.replace("_", " ")
        name = name[0].upper() + name[1:]
        caption = f"A bimanual robot performing the task: {name}."
        items.append({
            "input_video": str(frame_png),
            "prompt": caption,
            "output_video": str(output_dir / f"{task_name}__episode_{ep_id:06d}.mp4"),
            "num_chunks": n_chunks,
        })
else:
    mp4_files = sorted(videos_root.glob(f"chunk-*/{camera}/*.mp4"))[:max_episodes]
    
    # Load captions/tasks
    captions = {}
    if embodiment == "humanoid":
        cap_dir = dataset_dir / "captions"
        if cap_dir.exists():
            for txt in cap_dir.glob("*.txt"):
                m = re.search(r"(\d+)$", txt.stem)
                if m:
                    captions[int(m.group(1))] = txt.read_text().strip()
    elif embodiment == "single_arm":
        ep_file = dataset_dir / "meta" / "episodes.jsonl"
        if ep_file.exists():
            with ep_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    ep_idx = rec.get("episode_index", -1)
                    tasks = rec.get("tasks", [])
                    if ep_idx >= 0 and tasks and tasks[0]:
                        captions[ep_idx] = tasks[0]
    
    generic = "A GR-1 humanoid robot performing a manipulation task." if embodiment == "humanoid" else "A robot arm performing a manipulation task."
    
    items = []
    for mp4 in mp4_files:
        ep_match = re.search(r"(\d+)$", mp4.stem)
        if not ep_match:
            continue
        ep_id = int(ep_match.group(1))
        frame_png = input_dir / "first_frames" / f"episode_{ep_id:06d}.png"
        if not frame_png.exists():
            if not extract_first_frame(mp4, frame_png):
                continue
        n_frames = get_frame_count(mp4)
        n_chunks = compute_num_chunks(n_frames, source_fps) if n_frames else 3
        caption = captions.get(ep_id, generic)
        items.append({
            "input_video": str(frame_png),
            "prompt": caption,
            "output_video": str(output_dir / f"episode_{ep_id:06d}.mp4"),
            "num_chunks": n_chunks,
        })

# Group by num_chunks and write separate batch files
from collections import defaultdict
groups = defaultdict(list)
for item in items:
    nc = item.pop("num_chunks")
    groups[nc].append(item)

for nc, group_items in sorted(groups.items()):
    out_json = input_dir / f"batch_input_fps{target_fps}_{nc}chunks.json"
    with open(out_json, "w") as f:
        json.dump(group_items, f, indent=2, ensure_ascii=False)
    print(f"  {len(group_items)} episodes → {nc} chunks → {out_json.name}")

PYTHON

    # Run inference for each chunk group
    for batch_file in $(ls "${input_dir}"/batch_input_fps${FPS}_*chunks.json 2>/dev/null | sort -t_ -k4 -n); do
      fname="$(basename "${batch_file}")"
      # Extract num_chunks from filename: batch_input_fps10_5chunks.json → 5
      num_chunks="${fname##*_}"
      num_chunks="${num_chunks%chunks.json}"

      item_count="$(python3 -c "import json; print(len(json.load(open('${batch_file}'))))")"
      if [[ "${item_count}" -eq 0 ]]; then
        continue
      fi

      # Skip already done
      pending="$(python3 -c "
import json
from pathlib import Path
items = json.load(open('${batch_file}'))
pending = [x for x in items if not Path(x['output_video']).is_file()]
print(len(pending))
")"
      if [[ "${pending}" -eq 0 ]]; then
        echo "  [SKIP] ${num_chunks} chunks: all done"
        continue
      fi

      echo "  Running ${pending} episodes (${num_chunks} chunks, fps=${FPS})..."
      if [[ "${NUM_GPUS}" -gt 1 ]]; then
        torchrun \
          --nproc_per_node="${NUM_GPUS}" \
          --master_port="${MASTER_PORT}" \
          -m models.dreamgen.examples.video2world_lvg \
          --model_size "${MODEL_SIZE}" \
          --resolution "${RESOLUTION}" \
          --fps "${FPS}" \
          --dit_path "${DIT_PATH}" \
          --batch_input_json "${batch_file}" \
          --guidance "${GUIDANCE}" \
          --num_gpus "${NUM_GPUS}" \
          --num_chunks "${num_chunks}" \
          --num_conditional_frames 1 \
          --aspect_ratio "${ASPECT_RATIO}" \
          --disable_guardrail \
          --disable_prompt_refiner
      else
        python -m models.dreamgen.examples.video2world_lvg \
          --model_size "${MODEL_SIZE}" \
          --resolution "${RESOLUTION}" \
          --fps "${FPS}" \
          --dit_path "${DIT_PATH}" \
          --batch_input_json "${batch_file}" \
          --guidance "${GUIDANCE}" \
          --num_gpus "${NUM_GPUS}" \
          --num_chunks "${num_chunks}" \
          --num_conditional_frames 1 \
          --aspect_ratio "${ASPECT_RATIO}" \
          --disable_guardrail \
          --disable_prompt_refiner
      fi
      MASTER_PORT=$((MASTER_PORT + 1))
    done
  done
done

echo ""
echo "Done! Results at: ${SAVE_ROOT}/{embodiment}/{split}/fps_${FPS}/"
echo "Compare with fps=16: FPS=16 bash scripts/run_dreamgen_singleview_fps_test.sh"

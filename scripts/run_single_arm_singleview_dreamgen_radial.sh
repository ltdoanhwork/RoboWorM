#!/usr/bin/env bash
# =============================================================================
# run_single_arm_singleview_dreamgen_pisa.sh
# DreamGen single-arm SINGLEVIEW (RoboChallenge global static view, static16fps
# checkpoint) with PISA sparse attention.
#
# Exact mirror of run_single_arm_singleview_dreamgen_dicache.sh (dense LVG flow)
# -- single-GPU, same prepare + video2world_gr00t_lvg droid 2B inference, only
# swapping the DiCache flags for PISA and routing output to sampling_dataset/pisa/.
#
# Output: sampling_dataset/pisa/single_arm/output/singleview/dreamgen/{split}/iter_XXXXXX_static16fps/{view}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_singleview_dreamgen_pisa.sh
#
# Env: ITER_NAME, DIT_PATH, CKPT_BASE, MODEL_SIZE, FPS, GUIDANCE, ASPECT_RATIO,
#   VIEWS, SPLITS, NUM_CHUNKS, FORCE_REGENERATE, PROMPT_REFINE, PROMPT_PREFIX,
#   SOURCE_ROOT, INPUT_ROOT, OUTPUT_ROOT, PISA_{DENSITY,BLOCK_SIZE,START_LAYER_IDX}
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export VENV="${VENV:-${ROOT}/models/dreamgen/.venv}"
_PYVER="$(basename "$(ls -d ${VENV}/lib/python3.* 2>/dev/null | head -1)")"
export LD_LIBRARY_PATH="${VENV}/lib/${_PYVER}/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/${_PYVER}/site-packages/nvidia/cusparse/lib:${VENV}/lib/${_PYVER}/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

CKPT_BASE="${CKPT_BASE:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/singleview/single_arm_static_singleview_16fps_v2}"
CKPT_MODEL_DIR="${CKPT_BASE}/checkpoints_40_per/model"
if [[ -z "${ITER_NAME:-}" ]]; then
  latest_file="${CKPT_BASE}/checkpoints_40_per/latest_checkpoint.txt"
  if [[ -f "${latest_file}" ]]; then ITER_NAME="$(basename "$(cat "${latest_file}")" .pt)"; else ITER_NAME="iter_000090000"; fi
fi
DIT_PATH="${DIT_PATH:-${CKPT_MODEL_DIR}/${ITER_NAME}.pt}"
if [[ ! -f "${DIT_PATH}" ]]; then echo "ERROR: Checkpoint not found at ${DIT_PATH}" >&2; ls "${CKPT_MODEL_DIR}/"*.pt 2>/dev/null >&2 || true; exit 1; fi

NUM_GPUS=1
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
ASPECT_RATIO="${ASPECT_RATIO:-4:3}"
NUM_CONDITIONAL_FRAMES="${NUM_CONDITIONAL_FRAMES:-1}"
VIEWS="${VIEWS:-global}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT}/datasets/single_arm/singleview}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/radial/single_arm/input/singleview/dreamgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/radial/single_arm/output/singleview/dreamgen}"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-2 3}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
PROMPT_REFINE="${PROMPT_REFINE:-0}"
PROMPT_PREFIX="${PROMPT_PREFIX:-}"
SOURCE_FPS=15
OUT_ITER="${ITER_NAME}_static16fps"

RADIAL_DECAY_FACTOR="${RADIAL_DECAY_FACTOR:-1.0}"
RADIAL_BLOCK_SIZE="${RADIAL_BLOCK_SIZE:-64}"
RADIAL_START_LAYER_IDX="${RADIAL_START_LAYER_IDX:-4}"
RADIAL_MODEL_TYPE="${RADIAL_MODEL_TYPE:-hunyuan}"

echo "============================================"
echo "DreamGen Single-arm Singleview — Radial (LVG, single-GPU)"
echo "  Checkpoint: ${DIT_PATH} | Iter tag: ${OUT_ITER}"
echo "  Model: ${MODEL_SIZE} | FPS: ${FPS} | Aspect: ${ASPECT_RATIO} | Views: ${VIEWS}"
echo "  Radial: decay=${RADIAL_DECAY_FACTOR} block=${RADIAL_BLOCK_SIZE} start=${RADIAL_START_LAYER_IDX} type=${RADIAL_MODEL_TYPE}"
echo "  Splits: ${SPLITS} | Chunks: ${NUM_CHUNKS} | Output: ${OUTPUT_ROOT}"
echo "============================================"

prepare_batch_inputs() {
  local split="$1" view="$2"
  local view_cam="observation.images.${view}"
  local split_dir="${SOURCE_ROOT}/${split}"
  local input_dir="${INPUT_ROOT}/${view}/${split}"
  local frames_dir="${input_dir}/first_frames"
  [[ -d "${split_dir}" ]] || { echo "  [SKIP] ${split}/${view}: source dir not found"; return; }
  mkdir -p "${frames_dir}"
  python3 - "${split_dir}" "${input_dir}" "${frames_dir}" "${view_cam}" "${OUTPUT_ROOT}/${split}/${OUT_ITER}/${view}" "${SOURCE_FPS}" <<'PY'
import json, math, re, subprocess, sys
from collections import defaultdict
from pathlib import Path
split_dir=Path(sys.argv[1]); input_dir=Path(sys.argv[2]); frames_dir=Path(sys.argv[3])
view_cam=sys.argv[4]; output_dir=sys.argv[5]; source_fps=float(sys.argv[6])
FRAMES_PER_CHUNK=93; FRAMES_PER_ADDITIONAL=92; MODEL_FPS=16
def load_tasks(sd):
    ep_task={}; ef=sd/"meta"/"episodes.jsonl"; tf=sd/"meta"/"tasks.jsonl"
    if not ef.exists(): return ep_task
    for line in ef.open():
        line=line.strip()
        if not line: continue
        rec=json.loads(line); ei=rec.get("episode_index",-1); ts=rec.get("tasks",[])
        if ei>=0 and ts and ts[0]: ep_task[ei]=ts[0]
    if ep_task: return ep_task
    tm={}
    if tf.exists():
        for line in tf.open():
            line=line.strip()
            if not line: continue
            rec=json.loads(line)
            if rec.get("task"): tm[rec["task_index"]]=rec["task"]
    if tm and ef.exists():
        for line in ef.open():
            line=line.strip()
            if not line: continue
            rec=json.loads(line); ei=rec.get("episode_index",-1); ti=rec.get("task_index",-1)
            if ei>=0 and ti in tm: ep_task[ei]=tm[ti]
    return ep_task
videos_root=split_dir/"videos"; episodes=[]
for mp4 in sorted(videos_root.glob(f"chunk-*/{view_cam}/*.mp4")):
    m=re.search(r"(\d+)$", mp4.stem)
    if m: episodes.append((int(m.group(1)), mp4))
if not episodes: print(f"  No episodes for {view_cam}"); sys.exit(0)
ep_tasks=load_tasks(split_dir)
PREF=["A robot arm","A single robotic arm","A tabletop robot arm","A robot manipulator"]
GEN="A robot arm performing a manipulation task."
def cap(raw,ei):
    if not raw or raw==GEN: return GEN
    return f"{PREF[ei%len(PREF)]} {raw}."
groups=defaultdict(list)
for ei,vp in episodes:
    fp=Path(frames_dir)/f"episode_{ei:06d}.png"
    if not fp.exists():
        r=subprocess.run(["ffmpeg","-y","-i",str(vp),"-vf","scale=640:480","-vframes","1","-q:v","2",str(fp)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if r.returncode!=0: print(f"  [WARN] frame extract failed {vp.name}"); continue
    r=subprocess.run(["ffprobe","-v","quiet","-show_entries","stream=nb_frames","-of","csv=p=0",str(vp)],capture_output=True,text=True)
    nf=None
    if r.returncode==0 and r.stdout.strip():
        try: nf=int(r.stdout.strip().split("\n")[0])
        except ValueError: pass
    if nf:
        tgt=nf*(MODEL_FPS/source_fps)
        nc=1 if tgt<=FRAMES_PER_CHUNK else math.ceil((tgt-FRAMES_PER_CHUNK)/FRAMES_PER_ADDITIONAL)+1
    else: nc=2
    groups[nc].append({"input_video":str(fp),"prompt":cap(ep_tasks.get(ei,GEN),ei),"output_video":f"{output_dir}/episode_{ei:06d}.mp4"})
tot=0
for nc,items in sorted(groups.items()):
    oj=Path(input_dir)/f"batch_input_{nc}chunks.json"; oj.write_text(json.dumps(items,indent=2)); tot+=len(items)
    print(f"  {oj.name}: {len(items)} episodes")
print(f"  Total: {tot} episodes across {len(groups)} chunk groups")
PY
}

echo ""; echo "--- Step 1: Prepare batch inputs ---"
for split in ${SPLITS}; do for view in ${VIEWS}; do prepare_batch_inputs "${split}" "${view}"; done; done

echo ""; echo "--- Step 2: Run inference ---"
run_inference() {
  local batch_json="$1" label="$2" num_chunks="$3"
  [[ -f "${batch_json}" ]] || { echo "  [SKIP] ${label}: batch not found"; return; }
  local item_count; item_count="$(python3 -c "import json;print(len(json.load(open('${batch_json}'))))")"
  local tmp_batch_json="" run_json="${batch_json}"
  if [[ "${FORCE_REGENERATE}" != "1" ]]; then
    tmp_batch_json="$(mktemp --suffix=_dreamgen_radial.json)"
    local pending; pending="$(python3 -c "
import json
from pathlib import Path
items=json.loads(Path('${batch_json}').read_text())
pend=[x for x in items if not Path(x['output_video']).is_file()]
Path('${tmp_batch_json}').write_text(json.dumps(pend,indent=2)); print(len(pend))")"
    if [[ "${pending}" -eq 0 ]]; then echo "  [SKIP] ${label}: all ${item_count} done"; rm -f "${tmp_batch_json}"; return; fi
    echo "  === Running: ${label} (${pending}/${item_count} pending, chunks=${num_chunks}) ==="; run_json="${tmp_batch_json}"
  else echo "  === Running: ${label} (${item_count} episodes, chunks=${num_chunks}) ==="; fi
  local pr_flag=""; [[ "${PROMPT_REFINE}" == "1" ]] && pr_flag="--enable_prompt_refiner_per_chunk"
  python -m models.dreamgen.examples.video2world_gr00t_lvg \
    --model_size "${MODEL_SIZE}" --gr00t_variant droid --dit_path "${DIT_PATH}" \
    --batch_input_json "${run_json}" --num_chunks "${num_chunks}" --guidance "${GUIDANCE}" \
    --num_gpus "${NUM_GPUS}" --aspect_ratio "${ASPECT_RATIO}" \
    --num_conditional_frames "${NUM_CONDITIONAL_FRAMES}" --prompt_prefix "${PROMPT_PREFIX}" \
    --disable_guardrail --no_skip_existing ${pr_flag} \
    --use_radial --radial_decay_factor "${RADIAL_DECAY_FACTOR}" --radial_block_size "${RADIAL_BLOCK_SIZE}" \
    --radial_start_layer_idx "${RADIAL_START_LAYER_IDX}" --radial_model_type "${RADIAL_MODEL_TYPE}"
  [[ -n "${tmp_batch_json}" ]] && rm -f "${tmp_batch_json}"; echo "  === Done: ${label} ==="
}
for split in ${SPLITS}; do for view in ${VIEWS}; do
  input_dir="${INPUT_ROOT}/${view}/${split}"
  [[ -d "${input_dir}" ]] || { echo "  [SKIP] ${view}/${split}: input dir not found"; continue; }
  for nc in ${NUM_CHUNKS}; do run_inference "${input_dir}/batch_input_${nc}chunks.json" "single_arm_radial/${view}/${split}/${nc}chunks" "${nc}"; done
done; done

echo ""; echo "Done! Results: ${OUTPUT_ROOT}/{split}/${OUT_ITER}/{view}/"

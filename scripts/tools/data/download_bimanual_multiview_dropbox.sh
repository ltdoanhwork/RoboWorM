#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/datasets/bimanual/multiview"

usage() {
  cat <<'EOF'
Download and extract the GR00T bimanual multiview Dropbox tasks.

Usage:
  bash scripts/tools/data/download_bimanual_multiview_dropbox.sh [options] [task_name ...]

Options:
  --category makovian|non_makovian  Download only one split.
  --output-root PATH                Override output root.
  --force                           Re-download and overwrite extracted task dirs.
  --keep-zips                       Keep downloaded zip files under .downloads/.
  --list                            Print the embedded task manifest and exit.
  -h, --help                        Show this help.

Examples:
  bash scripts/tools/data/download_bimanual_multiview_dropbox.sh
  bash scripts/tools/data/download_bimanual_multiview_dropbox.sh --category non_makovian
  bash scripts/tools/data/download_bimanual_multiview_dropbox.sh close_toolbox insert_usb_cable

Notes:
  - Output layout matches datasets/bimanual/multiview/{makovian,non_makovian}/<task>/
  - Existing non-empty task directories are skipped unless --force is used.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

manifest() {
  cat <<'EOF'
non_makovian	brush_screws_into_dustpan_human_brush_left_hold	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ABQCFW5HLSyivpYQpzRsAgg/data/gr00t/brush_screws_into_dustpan_human_brush_left_hold?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
non_makovian	brush_screws_into_dustpan_left_brush_human_hold	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AHgy4x95OLbl04yHDw7IqqI/data/gr00t/brush_screws_into_dustpan_left_brush_human_hold?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	close_cardboard_box	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ACnI8RPDbWA9msMM0rX1_Xo/data/gr00t/close_cardboard_box?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	close_toolbox	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AONAPYNvhZNtiKMuMOHvHJ4/data/gr00t/close_toolbox?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	find_hole_and_insert_into_gear	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ABhzbzBluSd85DYlgHT5B58/data/gr00t/find_hole_and_insert_into_gear?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	find_insert_large_gear_shaft	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AINcWNtzLGhS_FzI3Lf5MWI/data/gr00t/find_insert_large_gear_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	find_insert_small_gear_shaft	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AKkjdJU9oXcPF3LEFiDhhDw/data/gr00t/find_insert_small_gear_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	fit_large_gear_shaft	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AJRxziYsrEd7EHOC1r9BdBk/data/gr00t/fit_large_gear_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	fit_small_gear_shaft	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ADyb3_zrmEjmJ3W30IGuwTc/data/gr00t/fit_small_gear_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
non_makovian	fold_bath_towel	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/APX3uyoq6RM9uVPVCcqrH08/data/gr00t/fold_bath_towel?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
non_makovian	fold_big_towel	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AIlA8EILMD-qArXYKTXU_IY/data/gr00t/fold_big_towel?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
non_makovian	fold_blue_towel	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AHGRwiMtscgUoFF5bwucFMM/data/gr00t/fold_blue_towel?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
non_makovian	fold_green_towel	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AFfYrj8EiiPJrjRYKxPDK78/data/gr00t/fold_green_towel?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
non_makovian	fold_light_blue_towel	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ANvXLKPkOPYUv1kiBQveORk/data/gr00t/fold_light_blue_towel?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
non_makovian	fold_orange_towel	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AP_wBpVW5skU6g0u946MBAM/data/gr00t/fold_orange_towel?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
non_makovian	fold_towel_assist	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ALJFsCCQEi9su0nxMIioJwE/data/gr00t/fold_towel_assist?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
non_makovian	fold_towel_in_random_places	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AA7cxdu5SwhzDeJ-FCBgxl4/data/gr00t/fold_towel_in_random_places?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
non_makovian	fold_yellow_towel	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AEZRF3x26iROaMlM0bkHl6c/data/gr00t/fold_yellow_towel?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
non_makovian	handover_and_wipe_sponge	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ALMTPQGnShstWMSAAKzIvio/data/gr00t/handover_and_wipe_sponge?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
non_makovian	handover_clear_zip_bag_upright	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ACGtaeyN40UK0-3LSxOkROo/data/gr00t/handover_clear_zip_bag_upright?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
non_makovian	handover_metallic_zip_bag_upright	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ACZOmKyuBDwR5T0JUTmRYZU/data/gr00t/handover_metallic_zip_bag_upright?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	hit_mark_with_hammer	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AM9LQZq43EHxl0W7k_HTihc/data/gr00t/hit_mark_with_hammer?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	hook_cable_8pin	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AG0WpnaOmp5j4z7X3tXWsm0/data/gr00t/hook_cable_8pin?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	hook_cable_narrow_8pin	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ADbEe5GDgrw6LnQ1mvic8r0/data/gr00t/hook_cable_narrow_8pin?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	hook_rubber_shaft	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AC9UiUcRRXtEdY7sH_U9FQA/data/gr00t/hook_rubber_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_gear_onto_shaft_sliding	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AD5JRM8YqEMVZWOlvMwWz4M/data/gr00t/insert_gear_onto_shaft_sliding?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_lan_cable_into_the_hub	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ADrhGfTGCDRY3Uzvm-A5jXQ/data/gr00t/insert_lan_cable_into_the_hub?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_large_gear_shaft	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AE46iX-ZsUjAQO5uXcmaekE/data/gr00t/insert_large_gear_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_random_shaft_random_medium_gear	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/APbrrl5RFt3VnVD2MXW3_Jw/data/gr00t/insert_random_shaft_random_medium_gear?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_random_shaft_random_small_gear	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ACEhPyU1hNfV4EY55W3uLLw/data/gr00t/insert_random_shaft_random_small_gear?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_rod_board	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AIlPo-ooDKg4ZuWMvpvkGOE/data/gr00t/insert_rod_board?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_rod_moved_board	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AM94k3jyBMZ5xp9yRDfkuHQ/data/gr00t/insert_rod_moved_board?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_shaft_into_gear	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AMLTcOVuZci-xNjnwp2AoeE/data/gr00t/insert_shaft_into_gear?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_small_gear_shaft	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AP6Ueermvw4bDt6KyBQusmU/data/gr00t/insert_small_gear_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_usb_cable	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ABiQzdMEaR-7ZvkBBk1SWlM/data/gr00t/insert_usb_cable?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_usb_cable_board	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ADZih17xms9qLmda3U6sA1c/data/gr00t/insert_usb_cable_board?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_usb_cable_fix_parts	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AIdL9EvMO2gApRkgYTd9RDk/data/gr00t/insert_usb_cable_fix_parts?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_usb_cable_fixed_by_robot	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AKwIgy3sl_JoZtNK0NhMHYE/data/gr00t/insert_usb_cable_fixed_by_robot?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_usb_pcbox	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AHyVCUFAc84GUtk7OAIyJ2U/data/gr00t/insert_usb_pcbox?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_usb_port_hub	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AH4mcBhwUDDoltT848Osl9s/data/gr00t/insert_usb_port_hub?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_washer_shaft	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AFL7RqZu2OMTJX50cagxxG8/data/gr00t/insert_washer_shaft?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	insert_washer_shaft_both_hands	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AKLXZKEpCH6kJuc8m3xXabs/data/gr00t/insert_washer_shaft_both_hands?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	wipe_the_desk_with_a_blue_cloth	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AMibO4nLGXVc9WnYvfPmgRA/data/gr00t/wipe_the_desk_with_a_blue_cloth?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	wipe_the_desk_with_a_sponge	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AOJ6Ym5zCIDehwGr1NonQ8I/data/gr00t/wipe_the_desk_with_a_sponge?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
non_makovian	take_out_reel_clear_zip_bag	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/APIqxuoWkm6cQNwU_9M8pvY/data/gr00t/take_out_reel_clear_zip_bag?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	tape_closed_cardboard	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ADnOB_uBb2pweSO-Uqg0P1o/data/gr00t/tape_closed_cardboard?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	turn_towel_over	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AAhtW5rPDugeOZ6eH27ek5A/data/gr00t/turn_towel_over?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	spoon_yellow_block_bowl	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/ALzSwJTUYhEFlCTlniaq6Dg/data/gr00t/spoon_yellow_block_bowl?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	spoon_red_block_bowl	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AOL52QGW3zMZ3XawKWV9SWU/data/gr00t/spoon_red_block_bowl?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	remove_box_from_case_right_arm	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AE1aieZSLWx-75Ca4qmjU64/data/gr00t/remove_box_from_case_right_arm?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	remove_box_from_case	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AI1Ln8ryvKGEyNiKE1Q4qaA/data/gr00t/remove_box_from_case?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
non_makovian	put_tape_white_case_alternating_arms	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AJQO0XN02iDKRzz-Qf_Sg4Y/data/gr00t/put_tape_white_case_alternating_arms?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	put_cup_moved_red_handle_holder	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AK4Fz8EnnTiplbXtuVassz0/data/gr00t/put_cup_moved_red_handle_holder?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
makovian	place_the_chinese_spoons_on_the_cloth	https://www.dropbox.com/scl/fo/zumqdjhk47uk2k8kcefmu/AFGVgKQpj3R8NP98F8qFoJ0/data/gr00t/place_the_chinese_spoons_on_the_cloth?rlkey=bhkxfigxkt8fmbxb88qnq1og7&dl=1
EOF
}

OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
CATEGORY_FILTER=""
FORCE=0
KEEP_ZIPS=0
LIST_ONLY=0
declare -a REQUESTED_TASKS=()

while (($# > 0)); do
  case "$1" in
    --category)
      CATEGORY_FILTER="${2:-}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --keep-zips)
      KEEP_ZIPS=1
      shift
      ;;
    --list)
      LIST_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      REQUESTED_TASKS+=("$@")
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      REQUESTED_TASKS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$CATEGORY_FILTER" && "$CATEGORY_FILTER" != "makovian" && "$CATEGORY_FILTER" != "non_makovian" ]]; then
  echo "Invalid --category: $CATEGORY_FILTER" >&2
  exit 1
fi

require_cmd wget
require_cmd unzip
require_cmd mktemp
require_cmd python3

mapfile -t MANIFEST_LINES < <(manifest)

if ((LIST_ONLY)); then
  printf '%s\n' "${MANIFEST_LINES[@]}"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/makovian" "${OUTPUT_ROOT}/non_makovian"
DOWNLOAD_DIR="${OUTPUT_ROOT}/.downloads"
mkdir -p "$DOWNLOAD_DIR"

declare -A REQUESTED_TASK_SET=()
if ((${#REQUESTED_TASKS[@]} > 0)); then
  for task in "${REQUESTED_TASKS[@]}"; do
    REQUESTED_TASK_SET["$task"]=1
  done
fi

copy_visible_contents() {
  local src_dir="$1"
  local dst_dir="$2"
  local entries=()

  shopt -s dotglob nullglob
  entries=("${src_dir}"/*)
  shopt -u dotglob nullglob

  if ((${#entries[@]} == 0)); then
    echo "No files extracted into ${src_dir}" >&2
    return 1
  fi

  mv "${entries[@]}" "$dst_dir"/
}

extract_zip_safely() {
  local zip_path="$1"
  local out_dir="$2"

  python3 - "$zip_path" "$out_dir" <<'PY'
import os
import shutil
import stat
import sys
import zipfile

zip_path, out_dir = sys.argv[1], sys.argv[2]
out_dir = os.path.abspath(out_dir)

with zipfile.ZipFile(zip_path) as zf:
    for info in zf.infolist():
        name = info.filename.replace("\\", "/").lstrip("/")
        if not name or name == ".":
            continue

        target = os.path.abspath(os.path.normpath(os.path.join(out_dir, name)))
        if os.path.commonpath([out_dir, target]) != out_dir:
            raise RuntimeError(f"Refusing to extract outside target dir: {info.filename}")

        mode = (info.external_attr >> 16) & 0o170000
        if stat.S_ISLNK(mode):
            continue

        if info.is_dir():
            os.makedirs(target, exist_ok=True)
            continue

        os.makedirs(os.path.dirname(target), exist_ok=True)
        with zf.open(info, "r") as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
PY
}

download_and_extract() {
  local category="$1"
  local task="$2"
  local url="$3"
  local task_dir="${OUTPUT_ROOT}/${category}/${task}"
  local stamp_file="${task_dir}/.download_complete"
  local zip_path="${DOWNLOAD_DIR}/${task}.zip"
  local tmp_extract
  local top_entries=()
  local payload_dir

  if [[ -f "$stamp_file" && $FORCE -eq 0 ]]; then
    echo "[skip] ${category}/${task} already complete"
    return
  fi

  if [[ -d "$task_dir" && $FORCE -eq 0 ]]; then
    if find "$task_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      echo "[skip] ${category}/${task} already has files; use --force to replace"
      return
    fi
  fi

  if [[ -d "$task_dir" && $FORCE -eq 1 ]]; then
    rm -rf "$task_dir"
  fi

  mkdir -p "$task_dir"

  echo "[download] ${category}/${task}"
  (
    cd "$DOWNLOAD_DIR"
    wget -c --content-disposition --trust-server-names "$url"
  )

  if [[ ! -f "$zip_path" ]]; then
    echo "Expected zip was not downloaded: $zip_path" >&2
    exit 1
  fi

  unzip -tq "$zip_path" >/dev/null

  tmp_extract=$(mktemp -d "${task_dir}.tmp.XXXXXX")
  extract_zip_safely "$zip_path" "$tmp_extract"

  shopt -s dotglob nullglob
  top_entries=("${tmp_extract}"/*)
  shopt -u dotglob nullglob

  payload_dir="$tmp_extract"
  if ((${#top_entries[@]} == 1)) && [[ -d "${top_entries[0]}" ]]; then
    payload_dir="${top_entries[0]}"
  fi

  copy_visible_contents "$payload_dir" "$task_dir"
  touch "$stamp_file"
  rm -rf "$tmp_extract"

  if ((KEEP_ZIPS == 0)); then
    rm -f "$zip_path"
  fi

  echo "[done] ${category}/${task} -> ${task_dir}"
}

selected_count=0
for line in "${MANIFEST_LINES[@]}"; do
  IFS=$'\t' read -r category task url <<< "$line"

  if [[ -n "$CATEGORY_FILTER" && "$category" != "$CATEGORY_FILTER" ]]; then
    continue
  fi

  if ((${#REQUESTED_TASKS[@]} > 0)) && [[ -z "${REQUESTED_TASK_SET[$task]+x}" ]]; then
    continue
  fi

  selected_count=$((selected_count + 1))
  download_and_extract "$category" "$task" "$url"
done

if ((selected_count == 0)); then
  echo "No tasks matched the requested filters." >&2
  exit 1
fi

echo "Processed ${selected_count} task(s) into ${OUTPUT_ROOT}"

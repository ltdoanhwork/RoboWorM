# #!/bin/bash

# set -e 

# ----------------------------------------------------------------------
# SINGLE ARM
# ----------------------------------------------------------------------

# # 1. Định nghĩa các đường dẫn gốc (Base paths)
# DATA_ROOT="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/preprocessing/single_arm"
# OUT_ROOT="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/output/output_p90_tol03_no-alpha/single_arm"
# TASK_TYPE=("makovian" "non_makovian")

# VIEWS_REAL="observation.images.exterior_1_left,observation.images.exterior_2_left"
# VIEWS_DENSE="view1,view2"

# # 2. Danh sách 3 thư mục
# FOLDERS=("real" "dreamgen" "ctrlworld")

# # 3. Vòng lặp chạy từng command
# for TASK_TYPE in "${TASK_TYPE[@]}"; do
#     for FOLDER in "${FOLDERS[@]}"; do
#         echo "======================================================================"
#         echo "🚀 Đang xử lý tập dữ liệu: ${FOLDER}"
#         echo "======================================================================"
        
#         # Gán biến CURRENT_VIEWS tùy thuộc vào tên folder đang chạy
#         case "$FOLDER" in
#             "real")
#                 CURRENT_VIEWS="${VIEWS_REAL}"
#                 ;;
#             "dreamgen")
#                 CURRENT_VIEWS="${VIEWS_DENSE}"
#                 ;;
#             "ctrlworld")
#                 CURRENT_VIEWS="${VIEWS_DENSE}"
#                 ;;
#             *)
#                 echo "❌ Lỗi: Không tìm thấy cấu hình VIEWS cho thư mục ${FOLDER}"
#                 exit 1
#                 ;;
#         esac

#         mkdir -p "${OUT_ROOT}/${FOLDER}"

#         # Chạy command với tập VIEWS tương ứng
#         uv run python -m multiview_consistency.run_video_metrics \
#             --episodes_root "${DATA_ROOT}/${FOLDER}/${TASK_TYPE}" \
#             --out_csv "${OUT_ROOT}/${FOLDER}/${TASK_TYPE}.csv" \
#             --views "${CURRENT_VIEWS}" \
#             --host 127.0.0.1 \
#             --port 6111
            
#         echo "✅ Xong: ${FOLDER}"
#         echo ""
#     done

# echo "🎉 Đã hoàn thành toàn bộ 3 thư mục!"




# # #!/bin/bash

# set -e 

# # ----------------------------------------------------------------------
# # BIMANUAL
# # ----------------------------------------------------------------------
# HOST="127.0.0.1"
# PORT="6111"

# CHECKPOINT="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/checkpoint/vggt_omega_1b_512.pt"

# DATA_ROOT_BASE="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/preprocessing/dense/bimanual"
# OUT_ROOT_BASE="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/output/output_indep/dense/bimanual"

# # FOLDERS=("real" "dreamgen")
# FOLDERS=("dreamgen" "ctrlworld")
# TASK_TYPES=("makovian" "non_makovian")

# # ----------------------------------------------------------------------
# # 2. Khởi chạy Model Service ở Background
# # ----------------------------------------------------------------------
# echo "======================================================================"
# echo "🤖 1. Đang khởi tạo Model Service (VGGT-Omega)..."
# echo "======================================================================"

# uv run python -m multiview_consistency.model_service \
#     --host "$HOST" \
#     --port "$PORT" \
#     --dtype float16 \
#     --omega_checkpoint "$CHECKPOINT" \
#     --omega_image_resolution 512 \
#     --omega_patch_size 16
#     # --no-compile &

# SERVICE_PID=$!

# cleanup() {
#     echo ""
#     echo "🛑 Đang dọn dẹp: Tắt Model Service (PID: $SERVICE_PID)..."
#     kill "$SERVICE_PID" 2>/dev/null || true
# }
# trap cleanup EXIT INT TERM

# echo "⏳ Đang chờ Model Service load checkpoint và mở port ${PORT}..."
# until python3 -c "
# from multiprocessing.connection import Client
# c = Client(('$HOST', int('$PORT')), authkey=b'vggt_omega')
# c.send({'cmd': 'ping'})
# print(c.recv())
# c.close()
# " 2>/dev/null; do
#     sleep 2
# done
# echo "✅ Model Service đã sẵn sàng!"
# echo ""

# # ----------------------------------------------------------------------
# # 3. Vòng lặp Sweep qua toàn bộ các thiết lập (FOLDERS x TASK_TYPES x Sub-tasks)
# # ----------------------------------------------------------------------
# for FOLDER in "${FOLDERS[@]}"; do
#     for TASK_TYPE in "${TASK_TYPES[@]}"; do

#         # Tự động chọn views tương ứng với nguồn dữ liệu (real vs gen)
#         case "$FOLDER" in
#             "real")
#                 CURRENT_VIEWS="observation.images.cam_high,observation.images.cam_low"
#                 ;;
#             "dreamgen"|"ctrlworld")
#                 CURRENT_VIEWS="view1,view2"
#                 ;;
#             *)
#                 echo "❌ Lỗi: Không tìm thấy cấu hình VIEWS cho folder ${FOLDER}"
#                 exit 1
#                 ;;
#         esac

#         TARGET_DATA_DIR="${DATA_ROOT_BASE}/${FOLDER}/${TASK_TYPE}"
#         TARGET_OUT_DIR="${OUT_ROOT_BASE}/${FOLDER}/${TASK_TYPE}"

#         # Kiểm tra nếu thư mục dữ liệu nguồn tồn tại thì mới tiến hành sweep
#         if [ ! -d "$TARGET_DATA_DIR" ]; then
#             echo "⚠️ Thư mục không tồn tại, bỏ qua: ${TARGET_DATA_DIR}"
#             continue
#         fi

#         mkdir -p "${TARGET_OUT_DIR}"

#         # Quét từng sub-task trong thư mục hiện tại
#         for task in "${TARGET_DATA_DIR}"/*/; do
#             [ -d "$task" ] || continue
#             name=$(basename "$task")

#             echo "======================================================================"
#             echo "🚀 [Folder: ${FOLDER}] [Type: ${TASK_TYPE}] [Task: ${name}]"
#             echo "======================================================================"

#             uv run python -m multiview_consistency.run_video_metrics \
#                 --episodes_root "$task" \
#                 --out_csv "${TARGET_OUT_DIR}/${name}.csv" \
#                 --views "${CURRENT_VIEWS}" \
#                 --host "$HOST" \
#                 --port "$PORT"

#             echo "✅ Hoàn thành: ${FOLDER}/${TASK_TYPE}/${name}"
#             echo ""
#         done
#     done
# done

# echo "🎉 Đã hoàn thành sweep toàn bộ thiết lập dữ liệu Bimanual!"



# #!/bin/bash

# set -e 

# # ----------------------------------------------------------------------
# # HUMANOID
# # ----------------------------------------------------------------------
# HOST="127.0.0.1"
# PORT="6111"

# CHECKPOINT="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/checkpoint/vggt_omega_1b_512.pt"

# DATA_ROOT_BASE="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/preprocessing/dense/humanoid"
# OUT_ROOT_BASE="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/output/output_indep/dense/humanoid"

# FOLDERS=("ctrlworld" "dreamgen")
# # FOLDERS=("real")
# # FOLDERS=("ctrlworld")
# # FOLDERS=("dreamgen")
# TASK_TYPES=("makovian" "non_makovian")

# # SINGLE_TASK="Airbot_MMK2_Airbot_MMK2_storage_mango_pomegranate"

# # ----------------------------------------------------------------------
# # 2. Khởi chạy Model Service ở Background
# # ----------------------------------------------------------------------
# echo "======================================================================"
# echo "🤖 1. Đang khởi tạo Model Service (VGGT-Omega)..."
# echo "======================================================================"

# uv run python -m multiview_consistency.model_service \
#     --host "$HOST" \
#     --port "$PORT" \
#     --dtype float16 \
#     --omega_checkpoint "$CHECKPOINT" \
#     --omega_image_resolution 512 \
#     --omega_patch_size 16 &
#     # --no-compile &

# SERVICE_PID=$!

# cleanup() {
#     echo ""
#     echo "🛑 Đang dọn dẹp: Tắt Model Service (PID: $SERVICE_PID)..."
#     kill "$SERVICE_PID" 2>/dev/null || true
# }
# trap cleanup EXIT INT TERM

# echo "⏳ Đang chờ Model Service load checkpoint và mở port ${PORT}..."
# until python3 -c "
# from multiprocessing.connection import Client
# c = Client(('$HOST', int('$PORT')), authkey=b'vggt_omega')
# c.send({'cmd': 'ping'})
# print(c.recv())
# c.close()
# " 2>/dev/null; do
#     sleep 2
# done
# echo "✅ Model Service đã sẵn sàng!"
# echo ""

# # ----------------------------------------------------------------------
# # 3. Vòng lặp Sweep qua toàn bộ các thiết lập (FOLDERS x TASK_TYPES x Sub-tasks)
# # ----------------------------------------------------------------------
# for FOLDER in "${FOLDERS[@]}"; do
#     for TASK_TYPE in "${TASK_TYPES[@]}"; do

#         TARGET_DATA_DIR="${DATA_ROOT_BASE}/${FOLDER}/${TASK_TYPE}"
#         TARGET_OUT_DIR="${OUT_ROOT_BASE}/${FOLDER}/${TASK_TYPE}"

#         if [ ! -d "$TARGET_DATA_DIR" ]; then
#             echo "⚠️ Thư mục không tồn tại, bỏ qua: ${TARGET_DATA_DIR}"
#             continue
#         fi

#         mkdir -p "${TARGET_OUT_DIR}"

#         # Quét từng sub-task trong thư mục hiện tại
#         for task in "${TARGET_DATA_DIR}"/*/; do
#             [ -d "$task" ] || continue
#             name=$(basename "$task")
            
#             # if [ -n "$SINGLE_TASK" ] && [ "$name" != "$SINGLE_TASK" ]; then
#             #     continue
#             # fi
#             # ----------------------------------------------------------
#             # Tự động chọn VIEWS dựa theo FOLDER và TASK NAME
#             # ----------------------------------------------------------
#             if [ "$FOLDER" = "real" ]; then
#                 case "$name" in
#                     "AIRBOT_MMK2_mobile_car" | "AIRBOT_MMK2_boxs_storage" | "Airbot_MMK2_storage_bottle_part")
#                         CURRENT_VIEWS="observation.images.cam_high_rgb,observation.images.cam_third_view"
#                         ;;
#                     "Airbot_MMK2_move_pan" | "Airbot_MMK2_Airbot_MMK2_storage_mango_pomegranate" | "Airbot_MMK2_doodled_line" )
#                         CURRENT_VIEWS="observation.images.cam_front_rgb,observation.images.cam_head_rgb"
#                         ;;
#                     *)
#                         echo "⚠️ Warning: Chưa cấu hình views cho real task: ${name}"
#                         continue
#                         ;;
#                 esac
#             else
#                 # For generated datasets ("dreamgen", "ctrlworld", etc.)
#                 CURRENT_VIEWS="view1,view2"
#             fi

#             echo "======================================================================"
#             echo "🚀 [Folder: ${FOLDER}] [Type: ${TASK_TYPE}] [Task: ${name}] [Views: ${CURRENT_VIEWS}]"
#             echo "======================================================================"

#             uv run python -m multiview_consistency.run_video_metrics \
#                 --episodes_root "$task" \
#                 --out_csv "${TARGET_OUT_DIR}/${name}.csv" \
#                 --views "${CURRENT_VIEWS}" \
#                 --host "$HOST" \
#                 --port "$PORT"

#             echo "✅ Hoàn thành: ${FOLDER}/${TASK_TYPE}/${name}"
#             echo ""
#         done
#     done
# done





# set -e 

# # ----------------------------------------------------------------------
# # SINGLE ARM
# # ----------------------------------------------------------------------
# HOST="127.0.0.1"
# PORT="6111"

# CHECKPOINT="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/checkpoint/vggt_omega_1b_512.pt"
# DATA_ROOT="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/preprocessing/dense/single_arm"
# OUT_ROOT="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/output/output_indep/dense/single_arm"

# TASK_TYPES=("makovian" "non_makovian")
# FOLDERS=("dreamgen" "ctrlworld")

# VIEWS_REAL="observation.images.exterior_1_left,observation.images.exterior_2_left"
# VIEWS_DENSE="view1,view2"

# echo "======================================================================"
# echo "🤖 1. Đang khởi tạo Model Service (VGGT-Omega)..."
# echo "======================================================================"

# uv run python -m multiview_consistency.model_service \
#     --host "$HOST" \
#     --port "$PORT" \
#     --dtype float16 \
#     --omega_checkpoint "$CHECKPOINT" \
#     --omega_image_resolution 512 \
#     --omega_patch_size 16 \
#     --no-compile &

# # Lấy PID của Model Service vừa chạy ngầm
# SERVICE_PID=$!

# # Đảm bảo tắt Model Service khi script dừng (thành công, lỗi, hoặc bấm Ctrl+C)
# cleanup() {
#     echo ""
#     echo "🛑 Đang dọn dẹp: Tắt Model Service (PID: $SERVICE_PID)..."
#     kill "$SERVICE_PID" 2>/dev/null || true
# }
# trap cleanup EXIT INT TERM

# # Chờ cho đến khi Model Service sẵn sàng lắng nghe ở PORT
# echo "⏳ Đang chờ Model Service load checkpoint và mở port ${PORT}..."
# until python3 -c "
# from multiprocessing.connection import Client
# c = Client(('127.0.0.1', 6111), authkey=b'vggt_omega')
# c.send({'cmd': 'ping'})
# print(c.recv())
# c.close()
# " 2>/dev/null; do
#     sleep 2
# done
# echo "✅ Model Service đã sẵn sàng!"
# echo ""

# # ----------------------------------------------------------------------
# # 3. Vòng lặp tính Metrics
# # ----------------------------------------------------------------------
# for TASK_TYPE in "${TASK_TYPES[@]}"; do
#     for FOLDER in "${FOLDERS[@]}"; do
#         echo "======================================================================"
#         echo "🚀 Đang xử lý tập dữ liệu: ${FOLDER} | Task: ${TASK_TYPE}"
#         echo "======================================================================"
        
#         case "$FOLDER" in
#             "real")
#                 CURRENT_VIEWS="${VIEWS_REAL}"
#                 ;;
#             "dreamgen"|"ctrlworld")
#                 CURRENT_VIEWS="${VIEWS_DENSE}"
#                 ;;
#             *)
#                 echo "❌ Lỗi: Không tìm thấy cấu hình VIEWS cho thư mục ${FOLDER}"
#                 exit 1
#                 ;;
#         esac

#         mkdir -p "${OUT_ROOT}/${FOLDER}"

#         uv run python -m multiview_consistency.run_video_metrics \
#             --episodes_root "${DATA_ROOT}/${FOLDER}/${TASK_TYPE}" \
#             --out_csv "${OUT_ROOT}/${FOLDER}/${TASK_TYPE}.csv" \
#             --views "${CURRENT_VIEWS}" \
#             --frame_batch_size 4 \
#             --host "$HOST" \
#             --port "$PORT"
            
#         echo "✅ Xong: ${FOLDER} (${TASK_TYPE})"
#         echo ""
#     done
# done

# echo "🎉 Đã hoàn thành toàn bộ công việc!"




set -euo pipefail
START_TIME=$SECONDS
HOST="127.0.0.1"
PORT="6111"
FRAME_BATCH_SIZE=4
VIEWS="view1,view2"
CHECKPOINT="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/checkpoint/vggt_omega_1b_512.pt"
DATA_ROOT="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/preprocessing"
OUT_ROOT="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/output/output_indep"
VIDEO_TYPES=(${VIDEO_TYPES:-fastercache})
ROBOT_TYPES=(${ROBOT_TYPES:-bimanual single_arm humanoid})
MODEL_TYPES=(${MODEL_TYPES:-ctrlworld dreamgen})
TASK_TYPES=(${TASK_TYPES:-makovian non_makovian})
# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
is_episode_dir() {
    local dir="$1"
    [ -d "${dir}/view1" ] && [ -d "${dir}/view2" ]
}
has_episode_children() {
    local dir="$1"
    local child
    for child in "$dir"/*/; do
        [ -d "$child" ] || continue
        is_episode_dir "$child" && return 0
    done
    return 1
}
# Yield directories whose immediate children are episode folders (view1+view2).
find_episode_roots() {
    local dir="$1"
    [ -d "$dir" ] || return 0
    if has_episode_children "$dir"; then
        echo "$dir"
        return 0
    fi
    local child
    for child in "$dir"/*/; do
        [ -d "$child" ] || continue
        find_episode_roots "$child"
    done
}
rel_path() {
    echo "${1#"${DATA_ROOT}/"}"
}
run_metrics() {
    local episodes_root="$1"
    local out_csv="$2"
    local rel
    rel="$(rel_path "$episodes_root")"
    mkdir -p "$(dirname "$out_csv")"
    echo "======================================================================"
    echo "🚀 episodes_root: ${rel}"
    echo "   out_csv:       ${out_csv#${OUT_ROOT}/}"
    echo "======================================================================"
    if ! uv run python -m multiview_consistency.run_video_metrics \
        --episodes_root "$episodes_root" \
        --out_csv "$out_csv" \
        --views "$VIEWS" \
        --frame_batch_size "$FRAME_BATCH_SIZE" \
        --host "$HOST" \
        --port "$PORT"; then
        echo "⚠️  Failed: ${rel}" >&2
    else
        echo "✅ Done: ${rel}"
    fi
    echo ""
}
sweep_model_task() {
    local robot_root="$1"   # .../itm/bimanual
    local model_type="$2"
    local task_type_dir root rel out_csv found=0
    for task_type in "${TASK_TYPES[@]}"; do
        task_type_dir="${robot_root}/${model_type}/${task_type}"
        [ -d "$task_type_dir" ] || continue
        while IFS= read -r root; do
            [ -n "$root" ] || continue
            found=1
            rel="$(rel_path "$root")"
            rel="${rel%/}"
            out_csv="${OUT_ROOT}/${rel}.csv"
            run_metrics "$root" "$out_csv"
        done < <(find_episode_roots "$task_type_dir")
        if [ "$found" -eq 0 ]; then
            echo "⚠️  No episodes with view1+view2 under: ${task_type_dir#${DATA_ROOT}/}"
            echo "    (run frame_extractor first?)"
        fi
    done
}
# ----------------------------------------------------------------------
# 1. Model service
# ----------------------------------------------------------------------
echo "======================================================================"
echo "🤖 Starting Model Service (VGGT-Omega)..."
echo "======================================================================"
uv run python -m multiview_consistency.model_service \
    --host "$HOST" \
    --port "$PORT" \
    --dtype float16 \
    --omega_checkpoint "$CHECKPOINT" \
    --omega_image_resolution 512 \
    --omega_patch_size 16 \
    --no-compile &
SERVICE_PID=$!
cleanup() {
    echo ""
    echo "🛑 Stopping Model Service (PID: $SERVICE_PID)..."
    kill "$SERVICE_PID" 2>/dev/null || true

    local duration=$(( SECONDS - START_TIME ))
    local hours=$(( duration / 3600 ))
    local mins=$(( (duration % 3600) / 60 ))
    local secs=$(( duration % 60 ))
    
    echo "----------------------------------------------------------------------"
    printf "⏱️ Total Execution Time: %02dh:%02dm:%02ds (%d seconds)\n" "$hours" "$mins" "$secs" "$duration"
    echo "----------------------------------------------------------------------"
}
trap cleanup EXIT INT TERM
echo "⏳ Waiting for Model Service on port ${PORT}..."
until python3 -c "
from multiprocessing.connection import Client
c = Client(('$HOST', int('$PORT')), authkey=b'vggt_omega')
c.send({'cmd': 'ping'})
print(c.recv())
c.close()
" 2>/dev/null; do
    sleep 2
done
echo "✅ Model Service ready!"
echo ""
# ----------------------------------------------------------------------
# 2. Sweep: itm/{robot}/{model}/{task_type}/[{task_name}/]{episode_id}/
# ----------------------------------------------------------------------
total_roots=0
for video_type in "${VIDEO_TYPES[@]}"; do
    for robot_type in "${ROBOT_TYPES[@]}"; do
        robot_root="${DATA_ROOT}/${video_type}/${robot_type}"
        [ -d "$robot_root" ] || continue
        for model_type in "${MODEL_TYPES[@]}"; do
            [ -d "${robot_root}/${model_type}" ] || continue
            echo "----------------------------------------------------------------------"
            echo "📂 ${video_type} / ${robot_type} / ${model_type}"
            echo "----------------------------------------------------------------------"
            sweep_model_task "$robot_root" "$model_type"
        done
    done
done
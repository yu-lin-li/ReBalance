set -euo pipefail

PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}

LAYER_ID=18  # v0:27; v1:33; v2:26
THRESHOLD=0.75  # v0:0.75; v1:0.74; v2:0.74
DATASET_DIR=./outputs

python hidden_analysis_mixed_auto_pangu.py \
  --layer_id $LAYER_ID \
  --jsonl_path "$DATASET_DIR/openPangu-Embedded-7B-V1.1/Math_Math/origin_temp0.7_maxlen32000.jsonl" \
  --hidden_dir "$DATASET_DIR/openPangu-Embedded-7B-V1.1/Math_Math/" \
  --save_path  "$DATASET_DIR/openPangu-Embedded-7B-V1.1/Math_Math/steer_vector_layer${LAYER_ID}_conf_mixed.pt" \
  --threshold $THRESHOLD \
  --max_files 500 \
  --expected_offset 1 \
  --device npu \
  --report_path  "$DATASET_DIR/data1/mamingrui/dyn_yt_32000/beta/openPangu-Embedded-7B-V1.1/Math_Math/analysis_layer${LAYER_ID}.json" \
  --verbose

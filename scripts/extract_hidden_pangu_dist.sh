PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}

model=/data1/mamingrui/openPangu-Embedded-7B-V1.1
max_generated_tokens=32000
torchrun \
	--nproc_per_node=8 \
	--master_port=29501 \
	transformer_inference_dp_pangu_dist.py \
	--model_name_or_path "$model" \
	--dataset_dir "./Data" \
	--dataset Math_Math \
	--output_path "./outputs" \
  	--max_generated_tokens $max_generated_tokens \
	--trust_remote_code

python \
	merged_base.py \
	--model_name_or_path "/data1/mamingrui/openPangu-Embedded-7B-V1.1" \
	--dataset Math_Math \
	--output_path "./outputs" \
    --max_generated_tokens $max_generated_tokens \




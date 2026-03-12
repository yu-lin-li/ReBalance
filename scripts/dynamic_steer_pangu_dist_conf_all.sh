PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}
# Dataset switches (1=run, 0=skip)
run_aime2025=${run_aime2025:-1}
run_gsm8k=${run_gsm8k:-1}


# Build datasets array based on switches
datasets=()
[[ $run_aime2025 -eq 1 ]] && datasets+=(Math_AIME2025)
[[ $run_gsm8k -eq 1 ]] && datasets+=(Math_GSM8K)

echo "=== Datasets to run: ${datasets[@]} ==="
steer_vector_path=./outputs/openPangu-Embedded-7B-V1.1/Math_Math/steer_vector_layer18_conf_mixed.pt
steer_layer=18
run_id=v0
seed=42
q25=0.75
q75=0.92
low_val=-3.3
tau=0.01
max_generated_tokens=32000
model=/data1/mamingrui/openPangu-Embedded-7B-V1.1
budget_num=3
# Multi-node inference
for ds in "${datasets[@]}"; do
    echo "=== Running dataset: ${ds} ==="
    torchrun \
        --nproc_per_node=8 \
        --master_port=29501 \
        transformer_inference_steer_dp_pangu_dist_conf_baseline.py \
        --model_name_or_path "$model" \
        --dataset_dir "./Data" \
        --dataset "$ds" \
        --output_path "./outputs/outputs_steer_dynamic_conf" \
        --steer_vector_path "$steer_vector_path" \
        --steer_layer $steer_layer \
        --steer_coef -1 \
        --run_id $run_id \
        --max_generated_tokens $max_generated_tokens \
        --seed $seed \
        --q25 $q25 \
        --q75 $q75 \
        --low_val $low_val \
        --tau $tau \
        --budget_num $budget_num  \
    
    python \
        merge_steering_baseline.py \
        --model_name_or_path "$model" \
        --dataset "$ds" \
        --dataset_dir "./Data" \
        --output_path "./outputs/outputs_steer_dynamic_conf" \
        --steer_layer $steer_layer \
        --run_id $run_id \
        --max_generated_tokens $max_generated_tokens \
        --seed $seed \
        --q25 $q25 \
        --q75 $q75 \
        --low_val $low_val \
        --tau $tau \
        --num_samples $budget_num \

    torchrun \
        --nproc_per_node=8 \
        --master_port=29501 \
        transformer_inference_steer_dp_pangu_dist_conf_all.py \
        --model_name_or_path "$model" \
        --dataset_dir "./Data" \
        --dataset "$ds" \
        --output_path "./outputs/outputs_steer_dynamic_conf" \
        --steer_vector_path "$steer_vector_path" \
        --steer_layer $steer_layer \
        --steer_coef -1 \
        --run_id $run_id \
        --max_generated_tokens $max_generated_tokens \
        --seed $seed \
        --q25 $q25 \
        --q75 $q75 \
        --low_val $low_val \
        --tau $tau \
        --budget_num $budget_num  \

    echo "=== merge_all_shards: ${ds} ==="
    python \
        merge_steering.py \
        --model_name_or_path "$model" \
        --dataset "$ds" \
        --dataset_dir "./Data" \
        --output_path "./outputs/outputs_steer_dynamic_conf" \
        --steer_layer $steer_layer \
        --run_id $run_id \
        --max_generated_tokens $max_generated_tokens \
        --seed $seed \
        --q25 $q25 \
        --q75 $q75 \
        --low_val $low_val \
        --tau $tau
    echo "=== Finished ${ds} ==="
done

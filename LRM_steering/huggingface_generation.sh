#!/bin/bash

export HF_HOME="/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"
# The new script uses 'auto' device mapping, so CUDA_VISIBLE_DEVICES might be less critical
# but we'll set it to be consistent with the original environment.
export CUDA_VISIBLE_DEVICES=6,7

# This should be updated to the path or HF name of your custom model
model_name="/home/kdh0901/Desktop/cache_dir/kdh0901/custom_r1_distill_pc" 

# Define the datasets to run on
declare -A datasets
datasets=(
    ["Maxwell-Jia/AIME_2024"]=4
    ["yentinglin/aime_2025"]=4
    # ["gpqa"]=2
)

# Get the absolute path of the script's directory
SCRIPT_DIR=$(cd $(dirname "$0") && pwd)
# Navigate up to the project root from utils/LRM_analysis/scripts
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PYTHON_SCRIPT_PATH="/home/kdh0901/Desktop/Underthinking/LRM_steering/huggingface_generation.py"

echo "Project root detected as: $PROJECT_ROOT"
echo "Python script path: $PYTHON_SCRIPT_PATH"

if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT_PATH"
    exit 1
fi

for name in "${!datasets[@]}"; do
    echo "--------------------------------------------------"
    echo "Running generation for dataset: $name"
    echo "--------------------------------------------------"
    # Note: The new python script does not have a --rollout_num argument.
    # It generates one response per problem instance.
    # If multiple rollouts are needed, the python script would need modification.
    n_samples=${datasets[$name]}
    python3 "$PYTHON_SCRIPT_PATH" \
        --model_name "$model_name" \
        --dataset_name "$name" \
        --rollout_num "$n_samples"
done

echo "All generation tasks are complete."

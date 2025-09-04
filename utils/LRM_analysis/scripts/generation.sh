#!/bin/bash

# --- 1. Configuration ---
# Set environment variables for cache and visible GPUs
export HF_HOME="/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Model and Server configuration
MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=131072 # Set a reasonable max length
HOST="localhost"
PORT=8000

# Define the datasets and their corresponding rollout numbers
declare -A datasets
datasets=(
    ["agentica-org/DeepScaleR-Preview-Dataset"]=1
    # ["Maxwell-Jia/AIME_2024"]=4
    # ["yentinglin/aime_2025"]=4
    # ["gpqa"]=2
)

# --- 2. Server Management ---
echo "🚀 Starting VLLM server for model: $MODEL_NAME"

# ** CORRECTED COMMAND **
# Start the server in the background using the official `vllm serve` command
vllm serve "$MODEL_NAME" \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --reasoning-parser qwen3 \
    --host "$HOST" \
    --port $PORT &

# Capture the Process ID (PID) of the server
SERVER_PID=$!

# Define a cleanup function to be called on script exit
function cleanup {
    echo "🧹 Shutting down VLLM server (PID: $SERVER_PID)..."
    kill $SERVER_PID
    # Wait for the process to terminate
    wait $SERVER_PID 2>/dev/null
    echo "✅ Server shut down."
}

# Set a trap to run the cleanup function when the script exits
trap cleanup EXIT

echo "⏳ Waiting for the server to become ready (PID: $SERVER_PID)..."

# Wait until the server's health check endpoint is available, with a 2-minute timeout
timeout 120s bash -c "until curl --silent --fail http://$HOST:$PORT/health > /dev/null; do sleep 2; done"

if [ $? -ne 0 ]; then
    echo "❌ Server failed to start within the timeout period. Exiting."
    exit 1
fi

echo "✅ Server is up and running!"

# --- 3. Client Execution Loop ---
for name in "${!datasets[@]}"; do
    n_samples=${datasets[$name]}
    echo "--------------------------------------------------------"
    echo "🧪 Running client for dataset: '$name' with $n_samples rollouts"
    echo "--------------------------------------------------------"

    # The client script remains the same and does not need any changes
    python generation.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "$name" \
        --rollout_num $n_samples \
        --host "$HOST" \
        --port $PORT \
        --batch_size 1024

done

echo "🎉 All datasets processed successfully."

# The 'trap' will automatically call the cleanup function to stop the server.
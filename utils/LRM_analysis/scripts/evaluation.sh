export HF_HOME="/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=4,5

model_name=Qwen/Qwen3-4B-Thinking-2507
datasets=("agentica-org/DeepScaleR-Preview-Dataset")

for name in "${datasets[@]}"; do
    python3 -m evaluation --model_name $model_name --dataset_name $name
done
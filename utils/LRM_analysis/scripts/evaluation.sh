export HF_HOME="/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=4,5

model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
datasets=("Maxwell-Jia/AIME_2024" "yentinglin/aime_2025" "gpqa")

for name in "${datasets[@]}"; do
    python3 -m evaluation.py --model_name $model_name --dataset_name $name
done
export HF_HOME="/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=4,5

model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

declare -A datasets
datasets=( 
    ["Maxwell-Jia/AIME_2024"]=4
    ["yentinglin/aime_2025"]=4
    ["gpqa"]=2
)

for name in "${!datasets[@]}"; do
    n_samples=${datasets[$name]}
    python3 -m generation --model_name $model_name --dataset_name $name --rollout_num $n_samples
done
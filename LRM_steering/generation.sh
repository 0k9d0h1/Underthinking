export CUDA_VISIBLE_DEVICES=0,1
export TARGET_LAYER=10
export TARGET_HEAD=3
export FIRE_THRESHOLD=0.50
export SUB_ALPHA=0.15

declare -A datasets
datasets=( 
    ["Maxwell-Jia/AIME_2024"]=4
    ["yentinglin/aime_2025"]=4
    ["gpqa"]=2
)

for name in "${!datasets[@]}"; do
    n_samples=${datasets[$name]}
    python /home/kdh0901/Underthinking/LRM_steering/generate_with_ctranslate2.py \
    --model_path /home/kdh0901/custom_r1_distill/custom_r1_distill_ct2 \
    --tokenizer_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name $name \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens 32768 \
    --rollout_num $n_samples
done
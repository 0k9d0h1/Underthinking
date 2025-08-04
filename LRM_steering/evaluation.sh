model_name=custom_r1_distill_qwen_7b
datasets=("Maxwell-Jia/AIME_2024" "yentinglin/aime_2025" "gpqa")

for name in "${datasets[@]}"; do
    python3 -m evaluation --model_name $model_name --dataset_name $name
done
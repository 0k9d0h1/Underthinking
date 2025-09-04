model_name=custom_r1_distill_pc
datasets=("Maxwell-Jia/AIME_2024" "yentinglin/aime_2025" "gpqa")

for name in "${datasets[@]}"; do
    python3 -m LRM_steering.evaluation --model_name $model_name --dataset_name $name
done
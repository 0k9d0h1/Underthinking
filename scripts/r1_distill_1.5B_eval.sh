set -x

# List of datasets to evaluate
datasets=("aime_2024" "aime_2025" "math500" "minerva_math" "olympiad")

for name in "${datasets[@]}"; do
    data_path="/home/kdh0901/Desktop/cache_dir/kdh0901/eval_data/${name}/r1_distill_1.5B_rm_120_${name}_generation_test.parquet"

    CUDA_VISIBLE_DEVICES=4,5 python3 -m verl.trainer.main_eval \
        data.path=$data_path \
        data.response_key=responses \
        data.data_source_key=data_source \
        data.reward_model_key=reward_model \
        ray_init.num_cpus=8
done
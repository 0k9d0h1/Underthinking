set -x

data_path=/home/kdh0901/Desktop/cache_dir/kdh0901/eval_data/aime_2024/r1_distill_1.5B_base_aime_2024_generation_test.parquet

CUDA_VISIBLE_DEVICES=4,5 python3 -m verl.trainer.main_eval \
    data.path=$data_path \
    data.response_key=responses \
    data.data_source_key=data_source \
    data.reward_model_key=reward_model \
    ray_init.num_cpus=8 \

export HF_HOME="/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"

set -x

model_path=/home/kdh0901/Desktop/cache_dir/kdh0901/ppl_rm_merged_120

# Define dataset names and n_samples as pairs
declare -A datasets
datasets=( 
    ["aime_2024"]=16
    ["aime_2025"]=16
    ["math500"]=2
    ["minerva_math"]=5
    ["olympiad"]=2
)

for name in "${!datasets[@]}"; do
    data_path="/home/kdh0901/Desktop/cache_dir/kdh0901/verl_dataset/${name}/test.parquet"
    save_path="/home/kdh0901/Desktop/cache_dir/kdh0901/eval_data/${name}/r1_distill_1.5B_rm_120_${name}_generation_test.parquet"
    n_samples=${datasets[$name]}

    CUDA_VISIBLE_DEVICES=4,5 python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=2 \
        data.path=$data_path \
        data.prompt_key=prompt \
        data.n_samples=$n_samples \
        data.output_path=$save_path \
        model.path=$model_path \
        +model.trust_remote_code=True \
        rollout.temperature=0.6 \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.prompt_length=2048 \
        rollout.response_length=16384 \
        ++rollout.max_num_batched_tokens=18432 \
        rollout.tensor_model_parallel_size=1 \
        rollout.gpu_memory_utilization=0.8
done
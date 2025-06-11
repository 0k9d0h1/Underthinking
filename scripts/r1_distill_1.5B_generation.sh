export HF_HOME="/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"

set -x

data_path=/home/kdh0901/Desktop/cache_dir/kdh0901/verl_dataset/aime_2024/test.parquet
save_path=/home/kdh0901/Desktop/cache_dir/kdh0901/eval_data/aime_2024/r1_distill_1.5B_base_aime_2024_generation_test.parquet
model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

CUDA_VISIBLE_DEVICES=4,5 python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=8 \
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
    rollout.gpu_memory_utilization=0.8 \

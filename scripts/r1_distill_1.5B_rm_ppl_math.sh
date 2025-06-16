export HF_HOME="/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"
export NCCL_ASYNC_ERROR_HANDLING=1       # fail fast on rank divergence

set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

math_train_path=/home/kdh0901/Desktop/cache_dir/kdh0901/verl_dataset/math/train.parquet
math_test_path=/home/kdh0901/Desktop/cache_dir/kdh0901/verl_dataset/math/test.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path']"

# Algorithm
temperature=0.6
top_p=0.95
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.95

CUDA_VISIBLE_DEVICES=4,5 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ppl \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    custom_reward_function.path=/home/kdh0901/Desktop/Underthinking/verl/verl/utils/reward_score/perplexity_reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=18432 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=True \
    reward_model.model.path=/home/kdh0901/Desktop/cache_dir/kdh0901/prm_checkpoints/checkpoint-85000\
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=64 \
    reward_model.use_dynamic_bsz=False \
    reward_model.forward_max_token_len_per_gpu=98304 \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager='perplexity' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rm_ppl_reward' \
    trainer.experiment_name='r1_distill_1.5B_rm_ppl_math' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir=/home/kdh0901/Desktop/cache_dir/kdh0901/checkpoints/rm_ppl_reward \
    trainer.validation_data_dir=/home/kdh0901/Desktop/cache_dir/kdh0901/val_data/rm_ppl_reward \
    trainer.save_freq=2 \
    trainer.test_freq=20 \
    trainer.val_before_train=False \
    trainer.total_epochs=5 $@

export HF_HOME="/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=4,5

model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

python3 -m activation_analysis --model_name $model_name
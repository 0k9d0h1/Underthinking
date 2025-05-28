import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import torch.nn as nn
from tqdm import tqdm
import deepspeed

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# Create DeepSpeed config first
# ds_config = {
#     "tensor_parallel": {"tp_size": 2},
#     "dtype": "bfloat16",
#     "replace_method": "auto",
#     "replace_with_kernel_inject": True,
#     "enable_cuda_graph": False,
# }

# Load directly with device_map for initial distribution
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with DeepSpeed integration
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    cache_dir="/home/kdh0901/Desktop/cache_dir/kdh0901",
    device_map="auto",  # This helps with initial allocation
    low_cpu_mem_usage=True  # This is important for large models
)

# Initialize DeepSpeed after model is distributed
# model = deepspeed.init_inference(
#     model,
#     replace_method="auto",
#     replace_with_kernel_inject=True,
#     config=ds_config
# ).module

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load the dataset
dataset_name = "Idavidrein/gpqa"
dataset = load_dataset(dataset_name, name="gpqa_diamond", split="train")

# Open a JSONL file to save the results
output_file = f"./{dataset_name.split("/")[1].replace("-", "_")}_{model_name.split("/")[1].replace("-", "_")}_results.jsonl"
with open(output_file, "a") as f:
    for i, data in enumerate(tqdm(dataset)):  # Assuming evaluation is on the 'test' split
        problem = data["Question"]
        correct_answer = data["Correct Answer"]

        # Prepare the prompt
        messages = [
            {"role": "user", "content": problem}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize and generate response
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=65536
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Save the result
        result = {
            "problem": problem,
            "correct_answer": correct_answer,
            "model_response": response
        }
        f.write(json.dumps(result) + "\n")
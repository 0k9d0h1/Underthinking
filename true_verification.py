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
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load model with DeepSpeed integration
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    cache_dir="/home/kdh0901/Desktop/cache_dir/kdh0901",
    device_map="auto",  # This helps with initial allocation
    low_cpu_mem_usage=True  # This is important for large models
)

input_file = "./DeepSeek_R1_Distill_Qwen_14B_deceiving.jsonl"
output_file = "./DeepSeek_R1_Distill_Qwen_14B_verification.jsonl"

with open(output_file, "a") as f, open(input_file, "r") as infile:
    for i, line in enumerate(tqdm(infile)):
        data = json.loads(line)
        id = data["id"]
        problem = data["problem"]
        correct_answer = data["correct_answer"]
        modified_response = data["modified_response"]

        messages = [
                    {"role": "user", "content": problem}
                ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) + modified_response

        # Tokenize and generate response
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=65536,
            use_cache=True,
        )

        generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
        response = modified_response + tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        result = {
            "id": id,
            "problem": problem,
            "correct_answer": correct_answer,
            "model_response": response
        }
        f.write(json.dumps(result) + "\n")
        f.flush()

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
    cache_dir="../cache_dir/kdh0901",
    device_map="auto",  # This helps with initial allocation
    low_cpu_mem_usage=True  # This is important for large models
)

messages = [
            {"role": "user", "content": ""}
        ]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
) + "Hello! My name is"

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
print(response)
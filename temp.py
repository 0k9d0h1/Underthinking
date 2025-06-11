import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import torch.nn as nn
from tqdm import tqdm


model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(tokenizer.encode("Alternatively"))
print(tokenizer.encode("alternatively"))
print(tokenizer.encode("Wait"))
print(tokenizer.encode("But wait"))
print(tokenizer.encode("Hmm"))
print(tokenizer.encode("hmm"))
print(tokenizer.decode([57021], clean_up_tokenization_spaces=False))
print(tokenizer.decode([82], clean_up_tokenization_spaces=False))
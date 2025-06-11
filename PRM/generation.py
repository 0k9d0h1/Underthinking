# vllm_generation.py
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
os.environ["HF_HOME"] = "/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"
from datasets import load_dataset
from vllm import LLM, SamplingParams
from pathlib import Path
from tqdm.auto import tqdm

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUT_DIR  = Path("/home/kdh0901/Desktop/Underthinking/PRM/gen_vllm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1-a. load MATH-lighteval (train split)
ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")  # :contentReference[oaicite:0]{index=0}

# 1-b. fire up vLLM (one line!)
llm = LLM(model=MODEL_ID, dtype="bfloat16")                               # :contentReference[oaicite:1]{index=1}
sam = SamplingParams(
    max_tokens   = 16384,
    temperature  = 0.6,
    top_p        = 0.95,
)

prompts = [
    f"{ex['problem']} Let's think step by step and output the final answer within \\boxed{{}}."
    for ex in ds
]

batch_size = 32        # vLLM pads nothing → large batches are cheap
outs = []
for i in tqdm(range(0, len(prompts), batch_size)):
    chunk = prompts[i : i + batch_size]
    out   = llm.generate(chunk, sam)
    outs.extend(o.outputs[0].text for o in out)

ds = ds.add_column("model_output", outs)
ds.save_to_disk(OUT_DIR)
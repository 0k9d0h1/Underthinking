import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import json
from tqdm import tqdm

# your hyperparameters
alpha = 3
beta = 600
special_token_ids = [92014, 80022, 14190]

# 1) define a stateful logits processor
class MitigateProcessor:
    def __init__(self, alpha: float, beta: int, special_token_ids: list):
        self.alpha = alpha
        self.beta = beta
        self.special_token_ids = special_token_ids
        self.thought_lengths = None  # will become a Tensor of shape (batch_size,)

    def __call__(self, token_ids: torch.LongTensor, logits: torch.FloatTensor):
        # token_ids: (batch_size, seq_len), logits: (batch_size, vocab_size)
        batch_size, device = token_ids.shape[0], logits.device

        # init/reset thought_lengths if needed
        if self.thought_lengths is None or self.thought_lengths.shape[0] != batch_size:
            self.thought_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

        last_tokens = token_ids[:, -1]
        # whenever special_token_id was just generated, reset that sequence’s counter
        reset_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for sid in self.special_token_ids:
            reset_mask |= (last_tokens == sid)
        self.thought_lengths[reset_mask] = 0
        # otherwise increment
        self.thought_lengths[~reset_mask] += 1

        # apply –alpha to the special token’s logit for all sequences still below beta
        active_mask = self.thought_lengths < self.beta
        if active_mask.any():
            for sid in self.special_token_ids:
                logits[active_mask, sid] -= self.alpha

        return logits

# 2) instantiate vLLM engine
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
llm = LLM(
    model=model_name,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
)

# 3) prepare SamplingParams with our custom processor
processors = [MitigateProcessor(alpha, beta, special_token_ids)]
sampling_params = SamplingParams(
    temperature=1.0,   # pure softmax sampling
    top_p=1.0,         # no nucleus filtering
    top_k=0,           # no top-k filtering
    logits_processors=processors,  # inject our mitigation logic
) 

# 4) load tokenizer & dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")

output_file = (
    f"./Underthinking_Reproduction_AIME_{model_name.split('/')[-1].replace('-', '_')}.jsonl"
)

# 5) loop and generate with vLLM
with open(output_file, "a") as fout:
    for i, data in enumerate(tqdm(dataset)):
        if i < 886 or data["Year"] < 2020:
            continue

        # build the chat prompt exactly as before
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": data["Question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # call vLLM.generate (auto–tokenizes/detokenizes for us)
        outputs = llm.generate([text], sampling_params)

        # each output.outputs is a list of choices; grab the first
        generated = outputs[0].outputs[0].text

        fout.write(
            json.dumps({
                "id": data["ID"],
                "problem": data["Question"],
                "correct_answer": data["Answer"],
                "model_response": generated,
            })
            + "\n"
        )

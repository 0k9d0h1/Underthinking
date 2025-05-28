import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["VLLM_USE_V1"] = "0"

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import json
from tqdm import tqdm

# your hyperparameters
alpha = 3
beta = 600
special_token_ids = [92014, 80022, 14190, 11489]

# 1) define a stateful logits processor
class MitigateProcessor:
    def __init__(self, alpha: float, beta: int, special_token_ids: list):
        self.alpha = alpha
        self.beta = beta
        self.special_token_ids = special_token_ids
        self.thought_lengths = 0  # will become a Tensor of shape (batch_size,)

    def __call__(self, token_ids: torch.LongTensor, logits: torch.FloatTensor):
        # token_ids: (batch_size, seq_len), logits: (batch_size, vocab_size)

        last_tokens = token_ids[-1] if len(token_ids) > 0 else -1
        # whenever special_token_id was just generated, reset that sequence’s counter
        reset = False
        for sid in self.special_token_ids:
            reset |= (last_tokens == sid)
        if reset:
            self.thought_lengths = 0
        else:
            self.thought_lengths += 1

        # apply –alpha to the special token’s logit for all sequences still below beta
        active_mask = self.thought_lengths < self.beta
        if active_mask:
            for sid in self.special_token_ids:
                logits[sid] -= self.alpha

        return logits

# 2) instantiate vLLM engine
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
llm = LLM(
    model=model_name,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    download_dir="/home/kdh0901/Desktop/cache_dir/kdh0901",
    max_model_len=65536
)

# 3) prepare SamplingParams with our custom processor
processors = [MitigateProcessor(alpha, beta, special_token_ids)]
sampling_params = SamplingParams(
    temperature=0.6,   # pure softmax sampling
    top_p=0.95,         # no nucleus filtering
    logits_processors=processors,  # inject our mitigation logic
    max_tokens=65536
) 

# 4) load tokenizer & dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset_name = "Maxwell-Jia/AIME_2024"
dataset = load_dataset(dataset_name, split="train")

output_file = f"./outputs/Underthinking_Reproduction_{dataset_name.split('/')[-1].replace('-', '_')}_{model_name.split('/')[-1].replace('-', '_')}.jsonl"


# 5) loop and generate with vLLM
with open(output_file, "a") as fout:
    for i, data in enumerate(tqdm(dataset)):
        problem = data["Problem"]
        correct_answer = data["Answer"]

        # Prepare the prompt
        messages = [
            {"role": "user", "content": problem}
        ]
        # build the chat prompt exactly as before
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # call vLLM.generate (auto–tokenizes/detokenizes for us)
        outputs = llm.generate([text], sampling_params)

        # each output.outputs is a list of choices; grab the first
        generated = outputs[0].outputs[0].text

        fout.write(
            json.dumps({
                "id": i,
                "problem": data["Problem"],
                "correct_answer": data["Answer"],
                "model_response": generated,
            })
            + "\n"
        )
        fout.flush()

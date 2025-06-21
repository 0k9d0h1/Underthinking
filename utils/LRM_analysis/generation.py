import os
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import json
from tqdm import tqdm
import argparse
import pandas
import random

dataset_to_problem_answer_id = {
    "Maxwell-Jia/AIME_2024": {
        "problem": "Problem",
        "answer": "Answer",
    },
    "yentinglin/aime_2025": {
        "problem": "problem",
        "answer": "answer",
    },
}

def main(args):
    model_name = args.model_name
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=2,
        gpu_memory_utilization=args.gpu_memory_utilization,
        download_dir="/home/kdh0901/Desktop/cache_dir/kdh0901",
        max_model_len=args.max_tokens
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,   # pure softmax sampling
        top_p=args.top_p,         # no nucleus filtering
        max_tokens=args.max_tokens
    )

    # 4) load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset_name = args.dataset_name
    if "aime" in dataset_name.lower():
        dataset = load_dataset(dataset_name, split="train")
    elif "gpqa" in dataset_name.lower():
        df = pandas.read_csv(
            "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(0)
        dataset = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]

    os.makedirs(f"./{model_name.split('/')[-1].replace('-', '_')}", exist_ok=True)
    output_file = f"./{model_name.split('/')[-1].replace('-', '_')}/{dataset_name.split('/')[-1].replace('-', '_')}_generation.jsonl"

    # 5) loop and generate with vLLM
    with open(output_file, "a") as fout:
        for i, data in enumerate(tqdm(dataset)):
            if "aime" in dataset_name.lower():
                problem = data[dataset_to_problem_answer_id[dataset_name]["problem"]]
                correct_answer = data[dataset_to_problem_answer_id[dataset_name]["answer"]]
                instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

                # Prepare the prompt
                messages = [
                    {"role": "user", "content": problem + " " + instruction_following},
                ]
            elif "gpqa" in dataset_name.lower():    
                choices = [
                    data["Correct Answer"],
                    data["Incorrect Answer 1"],
                    data["Incorrect Answer 2"],
                    data["Incorrect Answer 3"],
                ]
                choices = [choices[i] for i in data["permutation"]]
                correct_index = choices.index(data["Correct Answer"])
                correct_answer = "ABCD"[correct_index]
                choices_dict = dict(
                    A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=data["Question"]
                )
                problem = f"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{choices_dict['Question']}

A) {choices_dict['A']}
B) {choices_dict['B']}
C) {choices_dict['C']}
D) {choices_dict['D']}
""".strip()
                messages = [
                    {"role": "user", "content": problem},
                ]

            # build the chat prompt exactly as before
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # call vLLM.generate (auto–tokenizes/detokenizes for us)
            for _ in range(args.rollout_num):
                # Generate the output
                outputs = llm.generate([text], sampling_params)

                # each output.outputs is a list of choices; grab the first
                generated = outputs[0].outputs[0].text

                fout.write(
                    json.dumps({
                        "id": i,
                        "problem": problem,
                        "correct_answer": correct_answer,
                        "model_response": generated,
                    })
                    + "\n"
                )
                fout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", help="Name of the model to use")
    parser.add_argument("--dataset_name", type=str, default="Maxwell-Jia/AIME_2024", help="Name of the dataset to use")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Maximum number of tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for nucleus sampling")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--rollout_num", type=int, default=1, help="Number of rollouts to perform")
    args = parser.parse_args()
    main(args)
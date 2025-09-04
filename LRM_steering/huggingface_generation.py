import argparse
import json
import os
import random
from tqdm import tqdm

import pandas
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_qwen2_custom import Qwen2ForCausalLMCustom


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(SimpleClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# Set environment variables if needed, e.g., for cache directories
# os.environ["HF_HOME"] = "/path/to/your/cache"

dataset_to_problem_answer_id = {
    "Maxwell-Jia/AIME_2024": {"problem": "Problem", "answer": "Answer"},
    "yentinglin/aime_2025": {"problem": "problem", "answer": "answer"},
}


def get_prompt_and_answer(data, dataset_name):
    """Constructs the prompt and gets the correct answer based on the dataset."""
    if "aime" in dataset_name.lower():
        problem = data[dataset_to_problem_answer_id[dataset_name]["problem"]]
        correct_answer = data[dataset_to_problem_answer_id[dataset_name]["answer"]]
        instruction = (
            "Let's think step by step and output the final answer within \boxed{}."
        )
        prompt = f"{problem} {instruction}"
        messages = [{"role": "user", "content": prompt}]
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
        choices_dict = {
            "A": choices[0],
            "B": choices[1],
            "C": choices[2],
            "D": choices[3],
        }
        problem = f"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{data['Question']}

A) {choices_dict['A']}
B) {choices_dict['B']}
C) {choices_dict['C']}
D) {choices_dict['D']}
""".strip()
        messages = [{"role": "user", "content": problem}]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return problem, messages, correct_answer


def main(args):
    # 1) Model and Tokenizer Initialization
    print(f"Loading model: {args.model_name}")
    model = Qwen2ForCausalLMCustom.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    classifier = SimpleClassifier(input_dim=3584)
    classifier.load_state_dict(torch.load(args.classifier_dir, map_location="cpu"))

    # Move classifier to the same device as the target layer
    target_device = model.model.layers[27].self_attn.q_proj.weight.device
    classifier.to(target_device)

    classifier.eval()
    model.model.layers[27].self_attn.classifier = classifier
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("Model and tokenizer loaded.")

    # 2) Dataset Loading
    print(f"Loading dataset: {args.dataset_name}")
    if "aime" in args.dataset_name.lower():
        dataset = load_dataset(args.dataset_name, split="train")
    elif "gpqa" in args.dataset_name.lower():
        df = pandas.read_csv(
            "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(0)
        dataset = [
            example | {"permutation": rng.sample(range(4), 4)} for example in examples
        ]
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    print(f"Dataset loaded with {len(dataset)} examples.")

    # 3) Output File Setup
    output_dir = os.path.join(
        args.output_dir, args.model_name.split("/")[-1].replace("-", "_")
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f'{args.dataset_name.split("/")[-1].replace("-", "_")}_generation.jsonl',
    )
    print(f"Outputs will be saved to {output_file}")

    # 4) Generation Loop
    with open(output_file, "a") as fout:
        for i, data in enumerate(tqdm(dataset, desc="Generating responses")):
            problem, messages, correct_answer = get_prompt_and_answer(
                data, args.dataset_name
            )

            # Prepare the prompt for the model
            text_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)
            prompt_len = inputs["input_ids"].shape[1]

            for _ in range(args.rollout_num):
                # Reset the fire recorder before each generation
                model.fire_recorder.reset()
                # Reset the state of custom attention layers
                for layer in model.model.layers:
                    if hasattr(layer.self_attn, "reset"):
                        layer.self_attn.reset()
                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )

                generated_ids = outputs[0, prompt_len:]
                generated_text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )

                # Analyze fire events
                fire_counts = model.fire_recorder.fire_counts
                print(fire_counts)
                fired_seq_lengths = model.fire_recorder.fired_seq_lengths
                generation_fired_tokens = []
                if fire_counts:
                    # We assume batch size is 1, so sample_idx is 0
                    seq_lengths = fired_seq_lengths.get(0, [])
                    for seq_len in seq_lengths:
                        token_idx = seq_len - 1
                        if token_idx >= prompt_len:
                            gen_token_idx = token_idx - prompt_len
                            if gen_token_idx < len(generated_ids):
                                gen_token_id = generated_ids[gen_token_idx]
                                generation_fired_tokens.append(int(gen_token_id))

                decoded_fired_tokens = tokenizer.convert_ids_to_tokens(
                    generation_fired_tokens
                )

                # Save results
                result = {
                    "id": i,
                    "problem": problem,
                    "correct_answer": correct_answer,
                    "model_response": generated_text,
                    "response_tokens": generated_ids.tolist(),
                    "fire_info": {
                        "total_fires_in_generation": len(generation_fired_tokens),
                        "fired_token_ids": generation_fired_tokens,
                        "fired_tokens": decoded_fired_tokens,
                    },
                }
                fout.write(json.dumps(result) + "\n")
                fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate responses for benchmarks using Hugging Face models."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Path or HF repo name of the model.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Maxwell-Jia/AIME_2024",
        help="Name of the dataset to use (e.g., 'Maxwell-Jia/AIME_2024', 'gpqa').",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16384,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Temperature for sampling."
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Top-p for nucleus sampling."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_outputs",
        help="Directory to save the generation results.",
    )
    parser.add_argument(
        "--rollout_num", type=int, default=1, help="Number of rollouts to perform"
    )
    parser.add_argument(
        "--classifier_dir",
        type=str,
        default="/home/kdh0901/Desktop/Underthinking/model_outputs/best_classifier_layer_27.pt",
        help="Path to the classifier model.",
    )
    args = parser.parse_args()
    main(args)

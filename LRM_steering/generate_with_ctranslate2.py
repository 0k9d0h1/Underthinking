import os
import argparse
import json
import random
import pandas
import ctranslate2
import transformers
from datasets import load_dataset
from tqdm import tqdm

# Mapping of dataset names to their respective column names for problem and answer
dataset_to_problem_answer_id = {
    "Maxwell-Jia/AIME_2024": {"problem": "Problem", "answer": "Answer"},
    "yentinglin/aime_2025": {"problem": "problem", "answer": "answer"},
}

def main(args):
    # 1) Initialize CTranslate2 Generator and Tokenizer
    # CTranslate2 automatically uses available GPUs. We can specify which ones.
    device_indices = [int(i) for i in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(',')]
    generator = ctranslate2.Generator(args.model_path, device="cuda", device_index=device_indices)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 2) Load the dataset
    dataset_name = args.dataset_name
    if "aime" in dataset_name.lower():
        dataset = load_dataset(dataset_name, split="train")
    elif "gpqa" in dataset_name.lower():
        # GPQA dataset loading logic from the original script
        df = pandas.read_csv("https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv")
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(0)
        dataset = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 3) Prepare output file
    model_name_formatted = args.tokenizer_path.split('/')[-1].replace('-', '_')
    dataset_name_formatted = dataset_name.split('/')[-1].replace('-', '_')
    output_dir = f"./{model_name_formatted}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{dataset_name_formatted}_generation_ct2.jsonl"

    # 4) Main generation loop
    with open(output_file, "a") as fout:
        for i, data in enumerate(tqdm(dataset, desc=f"Generating on {dataset_name}")):
            if "aime" in dataset_name.lower():
                problem = data[dataset_to_problem_answer_id[dataset_name]["problem"]]
                correct_answer = data[dataset_to_problem_answer_id[dataset_name]["answer"]]
                instruction = "Let's think step by step and output the final answer within \\boxed{}."
                messages = [{"role": "user", "content": f"{problem} {instruction}"}]
            
            elif "gpqa" in dataset_name.lower():
                choices = [
                    data["Correct Answer"], data["Incorrect Answer 1"],
                    data["Incorrect Answer 2"], data["Incorrect Answer 3"],
                ]
                choices = [choices[i] for i in data["permutation"]]
                correct_index = choices.index(data["Correct Answer"])
                correct_answer = "ABCD"[correct_index]
                
                problem = f"""Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{data['Question']}

A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}
""".strip()
                messages = [{"role": "user", "content": problem}]

            # Apply chat template to format the prompt
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Tokenize the prompt for CTranslate2
            prompt_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt_text))

            # Generate multiple rollouts
            for _ in range(args.rollout_num):
                # CTranslate2 generation call
                results = generator.generate_batch(
                    [prompt_tokens],
                    max_length=args.max_tokens,
                    sampling_temperature=args.temperature,
                    sampling_topp=args.top_p,
                    include_prompt_in_result=False,
                )
                
                # Decode the first result
                generated_ids = results[0].sequences_ids[0]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Write output to JSONL file
                fout.write(
                    json.dumps({
                        "id": i,
                        "problem": problem,
                        "correct_answer": str(correct_answer),
                        "model_response": generated_text,
                    }) + "\n"
                )
                fout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using CTranslate2.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the converted CTranslate2 model directory.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the original Hugging Face model for the tokenizer.")
    parser.add_argument("--dataset_name", type=str, default="Maxwell-Jia/AIME_2024", help="Name of the dataset to use (e.g., 'Maxwell-Jia/AIME_2024', 'yentinglin/aime_2025', 'gpqa').")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens for generation.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for nucleus sampling.")
    parser.add_argument("--rollout_num", type=int, default=1, help="Number of responses to generate per problem.")
    args = parser.parse_args()
    main(args)

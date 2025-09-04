# vllm_client.py
import os
import asyncio
import openai
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import re
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import argparse
import pandas
import random
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

# --- Constants and Helper Functions (Unchanged) ---
dataset_to_problem_answer_id = {
    "Maxwell-Jia/AIME_2024": {"problem": "Problem", "answer": "Answer"},
    "yentinglin/aime_2025": {"problem": "problem", "answer": "answer"},
    "agentica-org/DeepScaleR-Preview-Dataset": {
        "problem": "problem",
        "answer": "answer",
    },
}
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"


def acc_reward(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score
    return ret_score


# --- Async Request Function (Unchanged) ---
async def get_completion(client, model_name, prompt, sampling_params, semaphore):
    async with semaphore:
        try:
            response = await client.completions.create(
                model=model_name,
                prompt=prompt,
                **sampling_params,
            )
            return response.choices[0].text
        except Exception as e:
            print(f"An error occurred while processing a request: {e}")
            return None


async def main(args):
    # 1) Setup OpenAI client
    client = openai.AsyncOpenAI(
        api_key="vllm",
        base_url=f"http://{args.host}:{args.port}/v1",
    )

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "n": args.rollout_num,
    }

    # 2) Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset_name = args.dataset_name

    if "gpqa" not in dataset_name.lower():
        dataset = load_dataset(dataset_name, split="train")
    else:
        df = pandas.read_csv(
            "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(0)
        dataset = [
            example | {"permutation": rng.sample(range(4), 4)} for example in examples
        ]

    # --- 3. MODIFIED BATCH PROCESSING LOGIC ---
    model_folder_name = args.model_name.split("/")[-1].replace("-", "_")
    os.makedirs(f"./{model_folder_name}", exist_ok=True)
    output_file = f"./{model_folder_name}/{dataset_name.split('/')[-1].replace('-', '_')}_generation.jsonl"

    total_score = 0
    num_total_prompts = len(dataset)
    semaphore = asyncio.Semaphore(args.concurrency)

    # Open file once in write mode
    with open(output_file, "w") as fout:
        # Create a new outer loop to process the dataset in batches
        for i in tqdm(
            range(0, num_total_prompts, args.batch_size), desc="Processing Batches"
        ):
            start_index = i
            end_index = min(i + args.batch_size, num_total_prompts)

            current_batch = dataset[start_index:end_index]
            if "gpqa" not in dataset_name.lower():
                current_batch_list = [
                    {"problem": problem, "answer": answer}
                    for problem, answer in zip(
                        current_batch[
                            dataset_to_problem_answer_id[dataset_name]["problem"]
                        ],
                        current_batch[
                            dataset_to_problem_answer_id[dataset_name]["answer"]
                        ],
                    )
                ]
            else:
                current_batch_list = [
                    {
                        "question": data["Question"],
                        "choices": [
                            data["Correct Answer"],
                            data["Incorrect Answer 1"],
                            data["Incorrect Answer 2"],
                            data["Incorrect Answer 3"],
                        ],
                    }
                    for data in current_batch
                ]

            batch_prompts_data = []
            # Prepare prompts for the current batch
            for data in current_batch_list:
                if "gpqa" not in dataset_name.lower():
                    problem = data["problem"]
                    correct_answer = data["answer"]
                    instruction_following = "\nLet's think step by step and output the final answer within \\boxed{}."
                    content = problem + " " + instruction_following
                else:
                    choices = [
                        data["Correct Answer"],
                        data["Incorrect Answer 1"],
                        data["Incorrect Answer 2"],
                        data["Incorrect Answer 3"],
                    ]
                    choices = [choices[i] for i in data["permutation"]]
                    correct_index = choices.index(data["Correct Answer"])
                    correct_answer = "ABCD"[correct_index]
                    content = f"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{data['Question']}

A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}
""".strip()

                messages = [{"role": "user", "content": content}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                batch_prompts_data.append(
                    {
                        "prompt": text,
                        "problem": content,
                        "correct_answer": correct_answer,
                    }
                )

            # Create and run async tasks for the current batch
            tasks = [
                get_completion(
                    client, args.model_name, data["prompt"], sampling_params, semaphore
                )
                for data in batch_prompts_data
            ]

            # Use tqdm_asyncio for progress within the batch gather
            results = await tqdm_asyncio.gather(
                *tasks, desc=f"Batch {i//args.batch_size + 1}"
            )

            # Process and save results for the current batch
            for j, generated_text in enumerate(results):
                if generated_text is None:
                    continue

                original_index = start_index + j
                data = batch_prompts_data[j]

                if "gpqa" not in dataset_name.lower():
                    score = acc_reward(generated_text, str(data["correct_answer"]))
                else:
                    match = re.search(ANSWER_PATTERN_MULTICHOICE, generated_text)
                    extracted_answer = match.group(1) if match else None
                    score = 1.0 if extracted_answer == data["correct_answer"] else 0.0

                total_score += score
                fout.write(
                    json.dumps(
                        {
                            "id": original_index,
                            "problem": data["problem"],
                            "correct_answer": data["correct_answer"],
                            "model_response": generated_text,
                            "score": score,
                        }
                    )
                    + "\n"
                )

            # Flush the file buffer to ensure the batch is written to disk
            fout.flush()

    print(f"Final Accuracy: {total_score / num_total_prompts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLLM Client for Batched Inference")
    # Server and Model args
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-9b-it",
        help="Name of the model being served",
    )

    # Dataset and Output args
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Maxwell-Jia/AIME_2024",
        help="Name of the dataset to use",
    )

    # Sampling and Request args
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=81920,
        help="Maximum number of tokens for generation",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Top-p for nucleus sampling"
    )
    parser.add_argument(
        "--rollout_num", type=int, default=1, help="Number of generation choices (n)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Number of concurrent requests to send",
    )

    # --- NEW ARGUMENT ---
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of prompts to process and save in a single batch",
    )

    args = parser.parse_args()
    asyncio.run(main(args))

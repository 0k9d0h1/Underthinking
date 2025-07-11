# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os
import random

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_ import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    math500_source = "HuggingFaceH4/MATH-500"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    math500_dataset = datasets.load_dataset(math500_source, trust_remote_code=True)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    math500_test = math500_dataset["test"]

    # Step 1: Make a set of all "problems" in MATH-500 test set
    math500_problems = set([ex["problem"].strip() for ex in math500_test])

    # Step 2: Filter out examples from MATH-lighteval test set that overlap
    filtered_test = test_dataset.filter(
        lambda ex: ex["problem"].strip() not in math500_problems
    )

    print(f"Original MATH-lighteval test size: {len(test_dataset)}")
    print(f"Filtered MATH-lighteval test size (no overlap): {len(filtered_test)}")
    assert len(filtered_test) + len(math500_test) == len(
        test_dataset
    ), "Filtered test set size mismatch!"

    # Step 3: Randomly select 500 problems (or all, if less than 500 left)
    num_samples = min(500, len(filtered_test))
    selected_indices = random.sample(range(len(filtered_test)), num_samples)
    filtered_test_500 = filtered_test.select(selected_indices)

    print(f"Final test set size: {len(filtered_test_500)}")

    instruction_following = (
        "Let's think step by step and output the final answer within \\boxed{}."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            question = question + " " + instruction_following

            answer = example.pop("solution")
            solution = extract_solution(answer)
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = filtered_test_500.map(
        function=make_map_fn("test"), with_indices=True
    )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

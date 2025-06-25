import os
import json
from itertools import groupby
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# === NEW IMPORTS FOR EMBEDDINGS & METRICS ===
import torch
from itertools import combinations
import numpy as np
import math
import random
from collections import defaultdict
import pickle
from statistics import mean
from tabulate import tabulate
import argparse
import re


def main(args):
    model_name = args.model_name
    thought_split_file = (
        f"./{model_name.split('/')[-1].replace('-', '_')}/thought_split.jsonl"
    )
    thought_cluster_file = (
        f"./{model_name.split('/')[-1].replace('-', '_')}/thought_clustering.jsonl"
    )

    # accumulators
    total = 0
    thoughts_correct = thoughts_incorrect = 0
    cluster_correct = cluster_incorrect = 0

    cluster_thoughts_ratio_correct = []
    cluster_thoughts_ratio_incorrect = []
    # === NEW: collect cluster sequences for revisit-rate ===
    cluster_sequences_correct = []
    cluster_sequences_incorrect = []

    def revisit_rate(seqs):
        total = 0.0
        for seq in seqs:
            if not seq:
                continue
            segments = [k for k, _ in groupby(seq)]
            seen, revisits, changes = {segments[0]}, 0, 0
            for prev, curr in zip(segments, segments[1:]):
                if curr != prev:
                    changes += 1
                    if curr in seen:
                        revisits += 1
                seen.add(curr)
            total += (revisits / changes) if changes else 0.0
        return total / len(seqs) if seqs else 0.0

    with open(thought_split_file, "r") as f1, open(thought_cluster_file, "r") as f2:

        thoughts_split = [json.loads(line) for line in f1]
        thoughts_cluster = [json.loads(line) for line in f2]

        for i, (split, cluster) in tqdm(
            enumerate(zip(thoughts_split, thoughts_cluster))
        ):
            phrases = split["phrases"]
            filtered = [p for p in phrases if len(p) > 100]
            n_phr = len(filtered)
            if n_phr == 0 or n_phr > 1000:
                continue

            correct = cluster["correctness"]
            gpt4o = cluster["gpt4o_answer"]
            gpt4o = re.sub(r"\([^)]*\)", "", gpt4o)

            # build mapping: thought index -> cluster label
            thought_to_cluster = {}
            for part in gpt4o.split("Cluster"):
                if ":" not in part or "Thoughts:" not in part:
                    continue
                cluster_num, rest = part.split(":", 1)
                ids = rest.split("Thoughts:")[1]
                for tok in ids.split(","):
                    tok = tok.strip()
                    if tok.isdigit():
                        idx = int(tok)
                        if idx >= len(phrases):
                            continue
                        if len(phrases[idx]) <= 100:
                            continue  # skip short phrases
                        thought_to_cluster[idx] = cluster_num.strip()

            # sort by thought idx -> get sequence of cluster labels
            sorted_items = sorted(thought_to_cluster.items(), key=lambda x: x[0])
            cluster_seq = [label for _, label in sorted_items]
            total += 1

            # --- APPEND to the correct list for revisit-rate ---
            if correct:
                cluster_sequences_correct.append(cluster_seq)
            else:
                cluster_sequences_incorrect.append(cluster_seq)

            # update counts & ratios
            if correct:
                thoughts_correct += n_phr
                cluster_correct += len(set(cluster_seq))
                cluster_thoughts_ratio_correct.append(len(set(cluster_seq)) / n_phr)
            else:
                thoughts_incorrect += n_phr
                cluster_incorrect += len(set(cluster_seq))
                cluster_thoughts_ratio_incorrect.append(len(set(cluster_seq)) / n_phr)

    # === FINAL REPORT ===
    print(f"Total problems: {total}")
    print(f"Thoughts Correct:   {thoughts_correct / total}")
    print(f"Thoughts Incorrect: {thoughts_incorrect / total}")
    print(f"Clusters Correct:   {cluster_correct / total}")
    print(f"Clusters Incorrect: {cluster_incorrect / total}")

    print(f"Revisit Rate (✔): {revisit_rate(cluster_sequences_correct):.4f}")
    print(f"Revisit Rate (✘): {revisit_rate(cluster_sequences_incorrect):.4f}")

    print(
        f"Avg #clusters/phrase (✔): {sum(cluster_thoughts_ratio_correct)/len(cluster_thoughts_ratio_correct):.4f}"
    )
    print(
        f"Avg #clusters/phrase (✘): {sum(cluster_thoughts_ratio_incorrect)/len(cluster_thoughts_ratio_incorrect):.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze thought clustering and splitting results."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Name of the model to use for analysis.",
    )
    args = parser.parse_args()

    main(args)

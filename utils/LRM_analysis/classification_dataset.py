import os, json, math, random, pickle, gc
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2Model,
    Qwen2ForCausalLM,
)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from math import sqrt
import types
import re
import argparse

from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns  # purely for prettier heat-maps; comment out if unwelcome
from scipy.ndimage import maximum_filter
import pyvene as pv
import pandas
import datasets

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


def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)

    return wrapped


class Collector:
    collect_state = True
    collect_action = False

    def __init__(self, multiplier, head):
        self.head = head
        self.states = []
        self.actions = []

    def reset(self):
        self.states = []
        self.actions = []

    def __call__(self, b, s):
        # b's shape is (batch_size, seq_len, hidden_size)
        if self.head == -1:
            self.states.append(
                b[0].detach().clone()
            )  # store for the whole sequence, remove batch dim
        else:
            b_reshaped = b.permute(1, 0, 2).squeeze(1)
            b_head_wise = b_reshaped.view(
                b_reshaped.shape[0], self.num_heads, -1
            ).permute(1, 0, 2)
            self.states.append(b_head_wise[self.head].detach().clone())
        return b


def get_activations_pyvene(collected_model, collectors, prompt, device, model_config):
    with torch.no_grad():
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        prompt = prompt.to(device)
        _, output = collected_model({"input_ids": prompt, "output_hidden_states": True})

    # Layer-wise hidden states (output of each transformer layer)
    hidden_states = output.hidden_states
    hidden_states = torch.stack(
        hidden_states, dim=0
    )  # (n_layers+1, batch_size, seq_len, hidden_size)
    hidden_states = hidden_states.squeeze(1)
    hidden_states = hidden_states.detach().cpu().to(torch.float32).numpy()

    # Head-wise hidden states
    head_wise_hidden_states_collected = []
    for collector in collectors:
        if collector.collect_state:
            # collector.states[0] shape: (seq_len, hidden_size)
            head_wise_hidden_states_collected.append(collector.states[0].to("cuda:0"))
        else:
            head_wise_hidden_states_collected.append(None)
        collector.reset()

    if any(h is not None for h in head_wise_hidden_states_collected):
        hwhs_tensor = torch.stack(
            head_wise_hidden_states_collected, dim=0
        ).cpu()  # (num_layers, seq_len, hidden_size)

        num_layers, seq_len, hidden_size = hwhs_tensor.shape
        num_heads = model_config.num_attention_heads
        head_dim = hidden_size // num_heads

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) not divisible by num_heads ({num_heads})"
            )

        head_wise_activations = hwhs_tensor.reshape(
            num_layers, seq_len, num_heads, head_dim
        )
        head_wise_activations = head_wise_activations.to(torch.float32).numpy()
    else:
        head_wise_activations = np.array([])

    mlp_wise_hidden_states = []  # Not implemented

    return hidden_states, head_wise_activations, mlp_wise_hidden_states


def load_data(split_path, cluster_path):
    with open(split_path) as f1, open(cluster_path) as f2:
        splits = [json.loads(l) for l in f1]
        clusters = [json.loads(l) for l in f2]
    assert len(splits) == len(clusters)
    return splits, clusters


def incremental_run(
    model,
    tokenizer,
    phrase_texts,
    batch_size_limit,
    want_attn=True,
    want_hidden=True,
    q_chunk=1024,
):
    """
    Returns
    -------
    attn_rows : list[list[torch.Tensor]]
        attn_rows[l][k] = Tensor(H, qk, prev+qk)  on CPU, one per micro-step
        or None if `want_attn=False`.
    hs_rows   : list[list[torch.Tensor]]
        hs_rows[l][k] = Tensor(qk, D)  on CPU, one per micro-step
        or None if `want_hidden=False`.
    phrase_token_ranges : list[(start,end)]
        token boundaries so you can build `ranges` later.
    """
    L = model.config.num_hidden_layers
    H = model.config.num_attention_heads
    D = model.config.hidden_size

    # tokenise every phrase --------------------------------------------
    ids_per_phrase = [
        tokenizer.encode(t, add_special_tokens=False) for t in phrase_texts
    ]
    input_ids = torch.cat([torch.Tensor(ids) for ids in ids_per_phrase], dim=0)
    input_ids = input_ids.to(torch.long)

    attn_rows = [[] for _ in range(L)] if want_attn else None
    hs_rows = [[] for _ in range(L + 1)] if want_hidden else None
    past_kv = None
    cursor = 0
    tok_ranges = []

    for i, p_ids in enumerate(ids_per_phrase):
        phrase_start = cursor
        # micro-batch the phrase so GPU sees ≤ q_chunk tokens
        pos = 0
        if (i > 0 and phrase_end + len(p_ids) > batch_size_limit) or (
            i == 0 and len(p_ids) > batch_size_limit
        ):
            print(f"Phrases too long ({phrase_end} tokens) – stop running.")
            break
        while pos < len(p_ids):
            piece = torch.tensor([p_ids[pos : pos + q_chunk]], device=model.device)
            pos += q_chunk
            q_len = piece.size(1)

            with torch.no_grad():
                out = model(
                    input_ids=piece,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_attentions=want_attn,
                    output_hidden_states=want_hidden,
                    attn_mode="torch",
                )

            past_kv = out.past_key_values  # extend cache

            if want_attn:
                for l, a in enumerate(out.attentions):  # (1,H,q,prev+q)
                    attn_rows[l].append(a[0].cpu())  # to CPU, drop batch
            if want_hidden:
                for l, h in enumerate(out.hidden_states):  # (1,q,D)
                    hs_rows[l].append(h[0].cpu())

            cursor += q_len

        phrase_end = cursor
        tok_ranges.append((phrase_start, phrase_end))
    return input_ids, attn_rows, hs_rows, tok_ranges


def get_prompts(model_name, dataset_names):
    prompt_list = []
    for dataset_name in dataset_names:
        generation_jsonl_path = f"./{model_name.split('/')[-1].replace('-', '_')}/{dataset_name}_generation.jsonl"
        with open(generation_jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line)
                prompt_list.append(data["problem"])

    return prompt_list


def main(args):
    model_name = args.model_name
    rng_seed = args.rng_seed
    seq_len_limit = args.seq_len_limit
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    split_path = f"./{model_name.split('/')[-1].replace('-', '_')}/thought_split.jsonl"
    cluster_path = (
        f"./{model_name.split('/')[-1].replace('-', '_')}/thought_clustering.jsonl"
    )
    torch.manual_seed(rng_seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    dataset_names = ["AIME_2024", "gpqa", "aime_2025"]
    prompts = get_prompts(model_name, dataset_names)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    # patch_qwen2_cpu(model)  # patch to return attentions on CPU
    model.eval()

    # ------------------- data ------------------------------------------------------
    splits, clusters = load_data(split_path, cluster_path)
    # ------------------- iterate ---------------------------------------------------
    from tqdm import tqdm

    label_list = []
    layer_wise_activations_list = []
    head_wise_activations_list = []

    for idx, (sp, cl, prompt) in enumerate(zip(tqdm(splits), clusters, prompts), 1):
        phrases = sp["phrases"]
        if not phrases:
            continue
        phrases.insert(0, prompt)  # insert the prompt as the first phrase
        debug_jsonl = "./debug.jsonl"
        with open(debug_jsonl, "a") as f:
            f.write(
                json.dumps(
                    {
                        "idx": idx,
                        "phrases": phrases,
                        "gpt4o_answer": cl["gpt4o_answer"],
                    }
                )
                + "\n"
            )

        # derive ground-truth cluster sequence
        phr_to_cluster = {}
        gpt4o = re.sub(r"\([^)]*\)", "", cl["gpt4o_answer"])
        for part in gpt4o.split("Cluster"):
            if ":" not in part or "Thoughts:" not in part:
                continue
            cnum, rest = part.split(":", 1)
            for tok in rest.split("Thoughts:")[1].split(","):
                tok = tok.strip()
                if tok.isdigit():
                    phi = int(tok)
                    if phi < len(phrases):
                        phr_to_cluster[phi + 1] = cnum.strip()
        cluster_seq = [phr_to_cluster.get(i, "UNK") for i in range(len(phrases))]
        cluster_transition_seq = []
        for i in range(2, len(cluster_seq)):
            if cluster_seq[i] != cluster_seq[i - 1]:
                cluster_transition_seq.append(i)

        collectors = []
        pv_config = []
        for layer in range(model.config.num_hidden_layers):
            collector = Collector(
                multiplier=0, head=-1
            )  # head=-1 to collect all head activations, multiplier doens't matter
            collectors.append(collector)
            pv_config.append(
                {
                    "component": f"model.layers[{layer}].self_attn.o_proj.input",
                    "intervention": wrapper(collector),
                }
            )
        collected_model = pv.IntervenableModel(pv_config, model)

        start_idx = []
        ids_per_phrase = [
            tokenizer.encode(t, add_special_tokens=False) for t in phrases
        ]
        cursor = 0
        for ids in ids_per_phrase:
            start_idx.append(cursor)
            cursor += len(ids)

        input_ids = torch.cat([torch.Tensor(ids) for ids in ids_per_phrase], dim=0)
        input_ids = input_ids.to(torch.long)
        layer_wise_activations, head_wise_activations, _ = get_activations_pyvene(
            collected_model, collectors, input_ids[:seq_len_limit], "cuda", model.config
        )
        true_label_indices = []
        staying_start_indices = []
        for i in range(2, len(cluster_seq)):
            if start_idx[i] > seq_len_limit:
                break
            if i in cluster_transition_seq:
                true_label_indices.append(start_idx[i])
            else:
                staying_start_indices.append(start_idx[i])

        overall_seq_len = layer_wise_activations.shape[1]
        prompt_len = start_idx[1]
        # Exclude prompt tokens and true label tokens from the pool of false candidates
        all_indices = set(range(prompt_len, min(overall_seq_len, seq_len_limit)))
        true_label_set = set(true_label_indices)
        staying_start_set = set(staying_start_indices)
        false_candidate_indices = list(all_indices - true_label_set - staying_start_set)

        # Sample from false candidates to balance the dataset (1:1 ratio)
        if len(staying_start_indices) <= 2 * len(true_label_indices):
            false_label_indices = random.sample(
                false_candidate_indices,
                min(
                    2 * len(true_label_indices) - len(staying_start_indices),
                    len(false_candidate_indices),
                ),
            )
            false_label_indices.extend(staying_start_indices)
        else:
            # If not enough false candidates, use all of them
            false_label_indices = staying_start_indices
        # print(true_label_indices, false_label_indices)

        # Combine and create labels
        selected_indices = true_label_indices + false_label_indices
        labels = [1] * len(true_label_indices) + [0] * len(false_label_indices)
        label_list.extend(labels)

        # Extract activations for the selected indices
        # Layer-wise activations shape: (n_layers+1, seq_len, hidden_size)
        layer_wise_selected = layer_wise_activations[:, selected_indices, :]
        # Head-wise activations shape: (num_layers, seq_len, num_heads, head_dim)
        head_wise_selected = head_wise_activations[:, selected_indices, :, :]
        for i in range(len(selected_indices)):
            layer_wise_activations_list.append(layer_wise_selected[:, i, :])
            head_wise_activations_list.append(head_wise_selected[:, i, :, :])

    print("Saving labels")
    np.save(
        f"/home/kdh0901/Desktop/cache_dir/kdh0901/classification_data/labels.npy",
        label_list,
    )

    print("Saving layer wise activations")
    np.save(
        f"/home/kdh0901/Desktop/cache_dir/kdh0901/classification_data/layer_wise.npy",
        layer_wise_activations_list,
    )

    print("Saving head wise activations")
    np.save(
        f"/home/kdh0901/Desktop/cache_dir/kdh0901/classification_data/head_wise.npy",
        head_wise_activations_list,
    )


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LRM activation analysis.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Name of the model to use for analysis.",
    )
    parser.add_argument(
        "--rng_seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--seq_len_limit",
        type=int,
        default=30000,
        help="Maximum number of tokens per forward pass.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/kdh0901/Desktop/cache_dir/kdh0901/classification_data",
        help="Directory to save output files.",
    )
    args = parser.parse_args()
    main(args)

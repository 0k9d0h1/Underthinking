import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
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

PREFIX = "Represent the reasoning strategy of this step:"
grid = [8, 10, 16, 32, 100]
agg = {k: defaultdict(list) for k in grid}       # {k: {"intra_mean":[..], ...}}
MIN_LEN      = 100         # length filter
SEED         = 42
rng          = random.Random(SEED)

def batch_encode(texts, model, batch_size=4):
    instruction = "Represent the reasoning strategy of this step."
    inputs = [[instruction, text] for text in texts]

    all_emb = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        emb_batch = model.encode(
            batch,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=False
        )
        all_emb.append(emb_batch)
    return torch.cat(all_emb, dim=0)

def remove_leading_words(text):
    starters = ["Wait", "Hmm", "Alternatively"]
    text = text.lstrip()  # remove leading whitespace
    for starter in starters:
        if text.lower().startswith(starter.lower()):
            # Remove the starter word and any immediate following punctuation (comma, period, space)
            text = text[len(starter):].lstrip(",. ")
            break
    return text

thought_split_file   = "../model_outputs/Underthinking_Reproduction_Thought_split_Deepseek_R1_Distill_Qwen_14B.jsonl"
thought_cluster_file = "../model_outputs/Underthinking_Reproduction_Thought_cluster_4.1_Deepseek_R1_Distill_Qwen_14B.jsonl"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load directly with device_map for initial distribution
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with DeepSpeed integration
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    cache_dir="/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface/hub",
    device_map="auto",  # This helps with initial allocation
    low_cpu_mem_usage=True  # This is important for large models
)

# accumulators
total = 0
thoughts_correct = thoughts_incorrect = 0
cluster_correct  = cluster_incorrect  = 0

cluster_thoughts_ratio_correct   = []
cluster_thoughts_ratio_incorrect = []

silhouette   = []

intra = []
inter = []

# === NEW: collect cluster sequences for revisit-rate ===
cluster_sequences_correct   = []
cluster_sequences_incorrect = []

def revisit_rate(seqs):
    total = 0.0
    for seq in seqs:
        if not seq:
            continue
        segments = [k for k,_ in groupby(seq)]
        seen, revisits, changes = {segments[0]}, 0, 0
        for prev, curr in zip(segments, segments[1:]):
            if curr != prev:
                changes += 1
                if curr in seen:
                    revisits += 1
            seen.add(curr)
        total += (revisits / changes) if changes else 0.0
    return total / len(seqs) if seqs else 0.0


def _pair_key(i, j):
    """canonical (smaller, larger) tuple for dict key"""
    return (i, j) if i < j else (j, i)


def compute_all_pairwise_ppl(phrases, tokenizer, model):
    """
    Returns:
        full_ppl  : dict[(i,j)] -> ppl
    """
    device = model.device
    full_ppl = {}
    for (i, text_i), (j, text_j) in tqdm(
            combinations(enumerate(phrases), 2),
            total=len(phrases)*(len(phrases)-1)//2,
            desc="exhaustive PPL"):

        text1 = text_i + "\n\nThat is, "
        text2 = remove_leading_words(text_j)
        enc   = tokenizer([text1 + text2], return_tensors="pt").to(device)
        t1_ids = tokenizer(text1, return_tensors="pt").input_ids.to(device)
        labels = enc.input_ids.clone(); labels[:, :t1_ids.size(1)] = -100

        with torch.no_grad(), torch.inference_mode(), \
             torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = model(enc.input_ids, labels=labels).loss
        full_ppl[_pair_key(i, j)] = math.exp(loss.item())

    return full_ppl

def summarize(sampled, discarded):
    def mean(lst): return sum(lst)/len(lst) if lst else float("nan")

    table = {}
    for bucket in ("intra", "inter"):
        full   = sampled[bucket] + discarded[bucket]
        table[bucket] = {
            "full_mean"      : mean(full),
            "sampled_mean"   : mean(sampled[bucket]),
            "discarded_mean" : mean(discarded[bucket]),
            "%kept"          : (len(sampled[bucket]) / len(full) * 100) if full else 0.0,
            "contrib_ratio"  : (mean(sampled[bucket]) / mean(full)) if full else float("nan")
        }
    return table

def study_sampling_curve(
    phrases: list[str],
    label_map,
    tokenizer,
    model,
    sample_grid=(1, 2, 4, 8, 16, 32),
    min_len  = 100,
    seed     = 42,
):
    """
    Run analyse_sampling_effect() for a range of `max_samples`
    values and print / return a summary table.

    sample_grid : iterable of ints  – how many previous thoughts to sample
    """
    rng = random.Random(seed)
    device = model.device

    # -------- pre-compute exhaustive PPL once (expensive) -------------------
    full_ppl = compute_all_pairwise_ppl(phrases, tokenizer, model)

    results = {}  # {max_samples: {bucket: stats}}
    for k in sample_grid:
        sampled, discarded = simulate_sampling(
            full_ppl, phrases, label_map,
            max_samples=k,
            min_len=min_len,
            rng=rng
        )
        results[k] = summarize(sampled, discarded)

    # -------- pretty print --------------------------------------------------
    header = (
        "max_k  |  intra_mean  contrib%  kept%  |  inter_mean  contrib%  kept%"
    )
    print(header)
    print("-"*len(header))
    for k in sample_grid:
        i = results[k]["intra"]
        o = results[k]["inter"]
        print(f"{k:5d} | "
              f"{i['sampled_mean']:11.4f}  {i['contrib_ratio']*100:7.2f}  {i['%kept']:5.1f} | "
              f"{o['sampled_mean']:11.4f}  {o['contrib_ratio']*100:7.2f}  {o['%kept']:5.1f}")

    return results

def simulate_sampling(full_ppl, phrases, label_map,
                      max_samples=10, min_len=100, rng=None):

    if rng is None:
        rng = random.Random()

    sampled, discarded = defaultdict(list), defaultdict(list)
    eligible = [len(p) >= min_len for p in phrases]
    N = len(phrases)

    def same_cluster(a, b):
        # unlabeled phrases get a unique negative id
        return label_map.get(a, -a-1) == label_map.get(b, -b-1)

    for i in range(1, N):
        if not eligible[i]:
            continue
        js = [j for j in range(i) if eligible[j]]
        if not js:
            continue

        take = js if len(js) <= max_samples else rng.sample(js, max_samples)

        for j in js:
            bucket = "intra" if same_cluster(i, j) else "inter"
            pair   = _pair_key(i, j)
            (sampled if j in take else discarded)[bucket].append(full_ppl[pair])

    return sampled, discarded

with open(thought_split_file,   'r') as f1, \
     open(thought_cluster_file, 'r') as f2:

    thoughts_split   = [json.loads(line) for line in f1]
    thoughts_cluster = [json.loads(line) for line in f2]

    for i, (split, cluster) in tqdm(enumerate(zip(thoughts_split, thoughts_cluster))):
        intra_each = []
        inter_each = []
        phrases = split["phrases"]
        filtered = [p for p in phrases if len(p) > 100]
        n_phr = len(filtered)
        if n_phr == 0 or n_phr > 1000:
            continue

        correct = cluster["correctness"]
        gpt4o  = cluster["gpt4o_answer"]

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
        cluster_seq   = [label for _, label in sorted_items]
        total += 1

        # --- APPEND to the correct list for revisit-rate ---
        if correct:
            cluster_sequences_correct.append(cluster_seq)
        else:
            cluster_sequences_incorrect.append(cluster_seq)

        # update counts & ratios
        if correct:
            thoughts_correct += n_phr
            cluster_correct  += len(set(cluster_seq))
            cluster_thoughts_ratio_correct.append(len(set(cluster_seq)) / n_phr)
        else:
            thoughts_incorrect += n_phr
            cluster_incorrect  += len(set(cluster_seq))
            cluster_thoughts_ratio_incorrect.append(len(set(cluster_seq)) / n_phr)

        # --- EMBEDDING ALIGNMENT ---
        if len(cluster_seq) > 1 and len(set(cluster_seq)) > 1:
            texts  = [phrases[idx] for idx, _ in sorted_items]
            label_map = {idx: int(cluster_num.strip()) for idx, cluster_num in thought_to_cluster.items()}
            pairwise_samples = []
            # for (idx1, text1), (idx2, text2) in combinations(enumerate(texts), 2):
            #     label1 = labels[idx1]
            #     label2 = labels[idx2]
            #     text1 = text1 + "\n\nThat is, "
            #     text2 = remove_leading_words(text2)
            #     model_inputs = tokenizer([text1 + text2], return_tensors="pt").to(model.device).input_ids
            #     text1_ids = tokenizer(text1, return_tensors="pt").input_ids
            #     labels_for_model = model_inputs.clone()

            #     labels_for_model[:, :len(text1_ids[0])]  = -100
            #     with torch.no_grad():
            #         loss = model(model_inputs, labels=labels_for_model).loss
            #         ppl = math.exp(loss.item())
            #     (intra_each if label1==label2 else inter_each).append(ppl)
            #     del model_inputs, labels_for_model, loss
            #     torch.cuda.empty_cache()
            # intra.append(intra_each)
            # inter.append(inter_each)

            results = study_sampling_curve(
                phrases=phrases,
                label_map=label_map,
                tokenizer=tokenizer,
                model=model,
                sample_grid=grid,
                min_len=100,
            )
            
            for k in grid:
                intra  = results[k]["intra"]
                inter  = results[k]["inter"]
                agg[k]["intra_mean"  ].append(intra["sampled_mean"])
                agg[k]["intra_ctr"   ].append(intra["contrib_ratio"]*100)
                agg[k]["intra_kept"  ].append(intra["%kept"])
                agg[k]["inter_mean"  ].append(inter["sampled_mean"])
                agg[k]["inter_ctr"   ].append(inter["contrib_ratio"]*100)
                agg[k]["inter_kept"  ].append(inter["%kept"])
                
rows = []
for k in grid:
    row = [k]
    for key in ("intra_mean", "intra_ctr", "intra_kept",
                "inter_mean", "inter_ctr", "inter_kept"):
        row.append(f"{np.nanmean(agg[k][key]):7.3f}")
    rows.append(row)

print("\n=== Overall averages across all problems ===")
print(tabulate(
    rows,
    headers=["k",
             "intra μ", "intra ctr%", "kept%",
             "inter μ", "inter ctr%", "kept%"],
    tablefmt="github"))

# # === FINAL REPORT ===
# print(f"Total problems: {total}")
# print(f"Thoughts Correct:   {thoughts_correct / total}")
# print(f"Thoughts Incorrect: {thoughts_incorrect / total}")
# print(f"Clusters Correct:   {cluster_correct / total}")
# print(f"Clusters Incorrect: {cluster_incorrect / total}")

# print(f"Revisit Rate (✔): {revisit_rate(cluster_sequences_correct):.4f}")
# print(f"Revisit Rate (✘): {revisit_rate(cluster_sequences_incorrect):.4f}")

# print(f"Avg #clusters/phrase (✔): {sum(cluster_thoughts_ratio_correct)/len(cluster_thoughts_ratio_correct):.4f}")
# print(f"Avg #clusters/phrase (✘): {sum(cluster_thoughts_ratio_incorrect)/len(cluster_thoughts_ratio_incorrect):.4f}")

# intra_flat = [item for sublist in intra for item in sublist]
# inter_flat = [item for sublist in inter for item in sublist]
# print(f"Mean intra-cluster perplexity : {sum(intra_flat)/len(intra_flat):.4f}")
# print(f"Mean inter-cluster perplexity : {sum(inter_flat)/len(inter_flat):.4f}")
# print(f"Std intra-cluster perplexity : {np.std(intra_flat):.4f}")
# print(f"Std inter-cluster perplexity : {np.std(inter_flat):.4f}")

# with open("intra.pkl", "wb") as f:
#     pickle.dump(intra, f)
# with open("inter.pkl", "wb") as f:
#     pickle.dump(inter, f)
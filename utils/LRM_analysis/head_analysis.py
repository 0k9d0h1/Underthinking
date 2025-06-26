import json
import re
import torch
from collections import defaultdict
from transformers import AutoTokenizer
from itertools import combinations


def load_data(split_path, cluster_path):
    with open(split_path) as f1, open(cluster_path) as f2:
        splits = [json.loads(l) for l in f1]
        clusters = [json.loads(l) for l in f2]
    assert len(splits) == len(clusters)
    return splits, clusters


def tokenise_phrases(phrases, tokenizer):
    """
    Tokenises every phrase *without* special tokens so we know exact boundaries,
    then concatenates them with a single EOS token in-between.  Returns

        input_ids             – torch.tensor([seq_len])
        phrase_token_ranges   – list[(start, end))  (end exclusive)
    """
    eos = tokenizer.eos_token_id
    all_ids, ranges = [], []
    cursor = 0

    for ph in phrases:
        ids = tokenizer.encode(ph, add_special_tokens=False)
        all_ids.extend(ids)
        ranges.append((cursor, cursor + len(ids)))
        cursor += len(ids)

        # separator
        all_ids.append(eos)
        cursor += 1

    input_ids = torch.tensor([all_ids], dtype=torch.long)
    return input_ids, ranges


def compute_ranges(
    tokenizer,
    phrase_texts,
):
    EOS = tokenizer.eos_token_id

    # tokenise every phrase --------------------------------------------
    ids_per_phrase = [
        tokenizer.encode(t, add_special_tokens=False) for t in phrase_texts
    ]
    for ids in ids_per_phrase[:-1]:  # EOS between phrases
        ids.append(EOS)

    cursor = 0
    tok_ranges = []

    for i, p_ids in enumerate(ids_per_phrase):
        phrase_start = cursor
        cursor += len(p_ids)
        phrase_end = cursor
        tok_ranges.append((phrase_start, phrase_end))
    return tok_ranges


model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
split_path = f"./{model_name.split('/')[-1].replace('-', '_')}/thought_split.jsonl"
cluster_path = (
    f"./{model_name.split('/')[-1].replace('-', '_')}/thought_clustering.jsonl"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
head_path = f"./{model_name.split('/')[-1].replace('-', '_')}/LRM_analysis_output/burst_heads.jsonl"


# ------------------- data ------------------------------------------------------
splits, clusters = load_data(split_path, cluster_path)
# ------------------- iterate ---------------------------------------------------
from tqdm import tqdm

total_transition_count = 0
total_intra_count = 0
for idx, (sp, cl) in enumerate(zip(tqdm(splits), clusters), 1):
    phrases = sp["phrases"]
    if not phrases:
        continue

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
                    phr_to_cluster[phi] = cnum.strip()
    cluster_seq = [phr_to_cluster.get(i, "UNK") for i in range(len(phrases))]

    # ------------- forward pass (may need micro-batching if sequence huge) ----
    input_ids, ranges = tokenise_phrases(phrases, tokenizer)
    ids_len = input_ids.size(1)

    ranges = compute_ranges(
        tokenizer,
        phrases,
    )
    if len(ranges) == 0:
        print(f"Skipping example #{idx} with no valid ranges.")
        continue
    cluster_seq = cluster_seq[: len(ranges)]  # truncate to match ranges
    cluster_transition_indices = []
    for i in range(1, len(cluster_seq)):
        if cluster_seq[i] != cluster_seq[i - 1]:
            cluster_transition_indices.append(i)
    total_transition_count += len(cluster_transition_indices)
    for i, j in combinations(cluster_seq, 2):
        if i == j:
            total_intra_count += 1
    # print(ranges)
    # print(cluster_seq)
    # print(cluster_transition_indices)

json_idx_by_prob_idx = []
head_to_transition = defaultdict(list)
head_to_transition_rate = defaultdict(float)
head_to_precision = defaultdict(float)
head_to_intra = defaultdict(list)
head_to_intra_rate = defaultdict(float)
head_to_intra_precision = defaultdict(float)
head_to_transition_F1 = defaultdict(float)
head_to_intra_f1 = defaultdict(float)
head_to_weight_score = defaultdict(list)
idx = 0
with open(head_path, "r") as f:
    burst_heads = [json.loads(line) for line in f]
    for i in range(len(burst_heads)):
        if burst_heads[i]["layer"] < burst_heads[i - 1]["layer"]:
            idx += 1
        json_idx_by_prob_idx.append(idx)
        head_to_transition[(burst_heads[i]["layer"], burst_heads[i]["head"])].append(
            burst_heads[i]["transition"]
        )
        head_to_intra[(burst_heads[i]["layer"], burst_heads[i]["head"])].append(
            burst_heads[i]["relation"] == "intra"
            and burst_heads[i]["q_tok_phrase_num"] != burst_heads[i]["k_tok_phrase_num"]
        )
        head_to_weight_score[(burst_heads[i]["layer"], burst_heads[i]["head"])].append(
            (
                burst_heads[i]["weight"],
                burst_heads[i]["relation"] == "intra"
                and burst_heads[i]["q_tok_phrase_num"]
                != burst_heads[i]["k_tok_phrase_num"],
                burst_heads[i]["transition"],
            )
        )
for layer_head, transitions in head_to_transition.items():
    head_to_transition_rate[layer_head] = (
        (sum(transitions) / len(transitions)) if transitions else 0.0
    )
    head_to_precision[layer_head] = (
        sum(transitions) / total_transition_count if total_transition_count > 0 else 0.0
    )
    head_to_intra_precision[layer_head] = (
        sum(head_to_intra[layer_head]) / total_intra_count
        if total_intra_count > 0
        else 0.0
    )
    head_to_intra_rate[layer_head] = (
        (sum(head_to_intra[layer_head]) / len(head_to_intra[layer_head]))
        if head_to_intra[layer_head]
        else 0.0
    )
    head_to_transition_F1[layer_head] = (
        2
        * head_to_transition_rate[layer_head]
        * head_to_precision[layer_head]
        / (head_to_transition_rate[layer_head] + head_to_precision[layer_head])
        if (head_to_transition_rate[layer_head] + head_to_precision[layer_head]) > 0
        else 0.0
    )
    head_to_intra_f1[layer_head] = (
        2
        * head_to_intra_rate[layer_head]
        * head_to_intra_precision[layer_head]
        / (head_to_intra_rate[layer_head] + head_to_intra_precision[layer_head])
        if (head_to_intra_rate[layer_head] + head_to_intra_precision[layer_head]) > 0
        else 0.0
    )
sorted_head_to_transition_F1 = sorted(
    head_to_transition_F1.items(), key=lambda x: x[1], reverse=True
)
sorted_head_to_intra_f1 = sorted(
    head_to_intra_f1.items(), key=lambda x: x[1], reverse=True
)
for layer_head, f1_score in sorted_head_to_transition_F1[:10]:
    transition_true_weight_avg = (
        sum([w[0] for w in head_to_weight_score[layer_head] if w[2] == True])
        / len(head_to_weight_score[layer_head])
        if head_to_weight_score[layer_head]
        else 0.0
    )
    transition_false_weight_avg = (
        sum([w[0] for w in head_to_weight_score[layer_head] if w[2] == False])
        / len(head_to_weight_score[layer_head])
        if head_to_weight_score[layer_head]
        else 0.0
    )
    print(
        f"Layer {layer_head}, Transition F1 Score: {f1_score}, True Weight Avg: {transition_true_weight_avg}, False Weight Avg: {transition_false_weight_avg}"
    )
for layer_head, f1_score in sorted_head_to_intra_f1[:10]:
    intra_true_weight_avg = (
        sum([w[0] for w in head_to_weight_score[layer_head] if w[1] == True])
        / len(head_to_weight_score[layer_head])
        if head_to_weight_score[layer_head]
        else 0.0
    )
    intra_false_weight_avg = (
        sum([w[0] for w in head_to_weight_score[layer_head] if w[1] == False])
        / len(head_to_weight_score[layer_head])
        if head_to_weight_score[layer_head]
        else 0.0
    )
    print(
        f"Layer {layer_head}, Intra F1 Score: {f1_score}, True Weight Avg: {intra_true_weight_avg}, False Weight Avg: {intra_false_weight_avg}"
    )

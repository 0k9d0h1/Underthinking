try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")
import re
import torch
import math
import numpy as np
import random
from verl import DataProto
from itertools import pairwise
from tensordict import TensorDict

MIN_LEN = 100
special_token_ids = [[92014], [80022], [14190], [151649], [3983, 3783]]
split_str = ["</think>", "Alternatively", "Wait", "Hmm", "But wait"]

def remove_leading_words(text):
    starters = ["Wait", "Hmm", "Alternatively", "But wait"]
    text = text.lstrip()  # remove leading whitespace
    for starter in starters:
        if text.lower().startswith(starter.lower()):
            # Remove the starter word and any immediate following punctuation (comma, period, space)
            text = text[len(starter):].lstrip(",. ")
            break
    return text

def acc_reward(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score


def split_string_and_ids(ids, delims, tok):
    delim_seqs = []
    for d in delims:
        for variant in (d, " " + d):
            seq = tok(variant, add_special_tokens=False)["input_ids"]
            if seq:                     # ignore empty encodings
                delim_seqs.append((seq, d))

    # longest first → avoids "But wait" being split as "But" + "wait"
    delim_seqs.sort(key=lambda x: len(x[0]), reverse=True)

    id_parts = []
    cur = []

    i = 0
    N = len(ids)
    while i < N:
        matched = False
        for seq, _ in delim_seqs:
            L = len(seq)
            if L and ids[i : i + L] == seq:       # exact token-sequence match
                if cur:                           # flush preceding fragment
                    id_parts.append(cur)
                cur = ids[i : i + L]              # start new slice with delim
                i += L
                matched = True
                break
        if not matched:                           # normal token
            cur.append(ids[i])
            i += 1

    if cur:
        id_parts.append(cur)

    text_parts = [
        tok.decode(part,
                   skip_special_tokens=True)
        for part in id_parts
    ]

    # Optional sanity check
    assert sum(id_parts, []) == ids, "Concatenated slices do not reproduce ids!"

    return text_parts, id_parts

def perplexity_reward(output_ids, ref_rollout_wg, tokenizer, n_gpus_per_node):
    previous_text_strategy = "random"
    max_samples = 8
    rng = random.Random(42)
    splitted_phrases, splitted_ids = split_string_and_ids(output_ids.tolist(), split_str, tokenizer)
    assert len(splitted_phrases) == len(splitted_ids), "Splitted phrases and ids must have the same length."

    phrases_token_lengths = [len(ids) for ids in splitted_ids]

    end_of_think_index = len(splitted_phrases)
    for i, phrase in enumerate(splitted_phrases):
        if "</think>" in phrase:
            end_of_think_index = i

    # 1) Collect all (i, text1, text2) pairs
    pairs = []
    for i in range(1, end_of_think_index):
        if len(splitted_phrases[i]) < MIN_LEN:
            continue

        # ----- possible j’s that also satisfy the length constraint --------
        eligible_js = [j for j in range(i) if len(splitted_phrases[j]) >= MIN_LEN]

        if not eligible_js:
            # No valid previous phrase; we’ll leave the reward for slice i = 0
            continue

        if previous_text_strategy == "random":
            if len(eligible_js) <= max_samples:
                selected_js = eligible_js
            else:
                selected_js = rng.sample(eligible_js, max_samples)
            
        elif previous_text_strategy == "all":
            selected_js = eligible_js

        for j in selected_js:
            text1 = splitted_phrases[j] + "\n\nThat is, "
            text2 = remove_leading_words(splitted_phrases[i])
            pairs.append((i, j, text1, text2))

    prefix_ids = tokenizer("\n\nThat is, ", add_special_tokens=False)["input_ids"]
    PREFIX_LEN = len(prefix_ids)
    
    texts = [p[2] + p[3] for p in pairs]  
    
    if len(texts) == 0:
        return np.array([0.0] * len(phrases_token_lengths)), phrases_token_lengths
    # tokenise + pad to the longest in this batch
    enc = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = enc.input_ids

    pad_id = tokenizer.pad_token_id
    labels = input_ids.clone()
    text1_lens = [
        PREFIX_LEN + phrases_token_lengths[p[1]]
        for p in pairs
    ]

    for idx, t1_len in enumerate(text1_lens):
        row        = input_ids[idx]
        # first index that is *not* PAD
        first_non_pad = (row != pad_id).nonzero(as_tuple=True)[0][0].item()
        t2_start      = first_non_pad + t1_len          # where t2 begins
        labels[idx, :t2_start] = -100                  # mask PAD + t1

    padding_size = 0
    if len(pairs) % n_gpus_per_node != 0:
        # pad to the next multiple of n_gpus_per_node
        padding_size = n_gpus_per_node - (len(pairs) % n_gpus_per_node)
        input_ids = torch.cat([input_ids, torch.full((padding_size, input_ids.shape[1]), pad_id)], dim=0)
        labels = torch.cat([labels, torch.full((padding_size, labels.shape[1]), -100)], dim=0)

    final_batch_size = len(pairs) + padding_size
    # pack into your DataProto
    batch_td = TensorDict(
        {
            "model_inputs": input_ids,
            "labels_for_model": labels,
        },
        batch_size=[final_batch_size],
    )
    batch_dp = DataProto(
        batch=batch_td,
        non_tensor_batch={
            "n_gpus_per_node": np.array([n_gpus_per_node] * final_batch_size),
            "pad_id": np.array([pad_id] * final_batch_size),
        }
    )

    # forward in one go
    loss = ref_rollout_wg.calculate_loss(batch_dp).batch["per_seq_loss"]
    # assume `loss` is a tensor of shape (batch_size,)
    loss = loss[loss >= 0].tolist()

    # 3) exponentiate to get per‐pair ppl
    all_ppls = [math.exp(l) for l in loss]

    # 4) regroup into per‐phrase lists and average
    ppls_per_phrase = {k: [] for k in range(1, end_of_think_index)}
    for (i, _, _, _), ppl in zip(pairs, all_ppls):
        ppls_per_phrase[i].append(ppl)

    ppl_rewards = []
    for i in range(1, end_of_think_index):
        group = ppls_per_phrase[i]
        ppl_rewards.append(sum(group) / len(group) if group else 0.0)

    # 5) normalize and pad as before
    arr = np.array(ppl_rewards)
    valid_mask = np.array([
        (len(splitted_phrases[i]) >= MIN_LEN) and (len(ppls_per_phrase[i]) > 0)
        for i in range(1, end_of_think_index)          # phrase indices 1..K
    ])
    if valid_mask.any():
        center = arr[valid_mask].mean()
        arr[valid_mask] = arr[valid_mask] - center

        scale = np.abs(arr[valid_mask]).max()
        if scale != 0.0:
            arr[valid_mask] = arr[valid_mask] / scale
    arr[~valid_mask] = 0.0
    # prepend 0.0 for the pre-amble slice
    arr = np.insert(arr, 0, 0.0)

    # pad with zeros after </think>
    arr = np.append(arr, [0.0] * (len(splitted_phrases) - end_of_think_index))

    return arr, phrases_token_lengths

def compute_score(data_source, solution_str, solution_ids, ground_truth, extra_info=None):
    ref_rollout_wg = extra_info["ref_rollout_wg"]
    tokenizer = extra_info["tokenizer"]
    n_gpus_per_node = extra_info["n_gpus_per_node"]

    ppl_rewards_for_phrases, phrase_token_lengths = perplexity_reward(solution_ids, ref_rollout_wg, tokenizer, n_gpus_per_node)
    return acc_reward(solution_str, ground_truth) + ppl_rewards_for_phrases, phrase_token_lengths

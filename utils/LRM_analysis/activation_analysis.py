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

sns.set_context("paper")


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


def cluster_metrics(true_seq, reps_layer, seed):
    """
    true_seq: list[str] cluster id *per phrase*.
    reps_layer: torch (P, H)
    Returns ARI & purity for this layer (or None if metrics ill-defined).
    """
    uniq = sorted({c for c in true_seq if c != "UNK"})
    if not uniq:
        return None
    kmeans = KMeans(len(uniq), n_init="auto", random_state=seed).fit(
        reps_layer.to(torch.float32).cpu().numpy()
    )
    pred = kmeans.labels_
    label_to_int = {lab: i for i, lab in enumerate(uniq)}
    true_int = [label_to_int.get(c, -1) for c in true_seq]
    mask = [t != -1 for t in true_int]
    if not any(mask):
        return None
    true_f = [t for t, m in zip(true_int, mask) if m]
    pred_f = [p for p, m in zip(pred, mask) if m]
    cm = confusion_matrix(true_f, pred_f)
    purity = cm.max(axis=0).sum() / cm.sum()
    return adjusted_rand_score(true_f, pred_f), purity


def reps_per_layer(hidden_states, ranges):
    """
    Returns list[length = n_layers+1] where item l is tensor (P, H)
    containing the mean hidden vector of every phrase at layer l.
    """
    reps_layers = []
    for layer_act in hidden_states:  # (1, S, H)
        layer_act = layer_act[0].to("cpu")  # drop batch
        reps = [layer_act[s:e].mean(0) for s, e in ranges]
        reps_layers.append(torch.stack(reps, 0))
    return reps_layers  # list[(P, H)]


def attention_within_cluster(attn, ranges, clusters):
    same = [
        (i, j)
        for i in range(len(ranges))
        for j in range(len(ranges))
        if clusters[i] == clusters[j] != "UNK" and i != j
    ]
    return _attention_mass(attn, ranges, same)


def attention_inter_cluster(attn, ranges, clusters):
    diff = [
        (i, j)
        for i in range(len(ranges))
        for j in range(len(ranges))
        if clusters[i] != clusters[j] and "UNK" not in (clusters[i], clusters[j])
    ]
    return _attention_mass(attn, ranges, diff)


def _attention_mass(attn, ranges, pairs):
    if not pairs:
        return 0.0
    masses = []
    for i, j in pairs:
        s_i, e_i = ranges[i]
        s_j, e_j = ranges[j]
        masses.append(attn[..., s_i:e_i, s_j:e_j].mean().item())
    return float(np.mean(masses))


def detect_burst_heads_2d(
    attn,  # (L, H, S, S)
    ranges,
    cluster_seq,
    cluster_transition_indices,
    tokenizer,
    layer,
    alpha=5,
    spike_th=10.0,  # keep heads with score > spike_th
    burst_z=3.0,  # 'bursts' above mean+burst_z*std
    max_per_head=30,
    max_examples=1_000,
    cache=None,
    nms_size=3,  # window size for local maxima
):
    attn = attn.to("cuda:1")  # move to GPU for speed
    L, H, S, _ = attn.shape
    ids = cache["ids"]
    examples = []

    for h in range(H):
        A = attn[0, h].to(torch.float32)  # (S, S)
        # ---------- 1) Compute a 2D spikiness score ----------
        score = _spike_score_2d(A)
        if score < spike_th:
            continue

        # ---------- 2) Find local 2D bursts ----------
        mu, sig = A.mean().item(), A.std(unbiased=False).item()
        thresh = mu + burst_z * sig

        # Boolean mask of potential bursts
        mask = A > thresh
        mask = mask

        # Find local maxima (NMS in 2D)
        if mask.sum() == 0:
            continue

        # local maximum filter (window size nms_size)
        def torch_maximum_filter_2d(A, size=3):
            # A: (S, S), 2D torch tensor
            # Add batch and channel dims for pooling
            A_ = A.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
            pool = F.max_pool2d(A_, kernel_size=size, stride=1, padding=size // 2)
            return pool[0, 0]  # (S, S)

        local_max = A == torch_maximum_filter_2d(A, size=nms_size)
        peaks = mask & local_max
        peak_indices = torch.nonzero(peaks, as_tuple=False)
        if peak_indices.numel() == 0:
            continue

        # strongest peaks first
        peak_vals = A[peak_indices[:, 0], peak_indices[:, 1]]
        top = torch.argsort(peak_vals, descending=True)[:max_per_head]
        peak_indices = peak_indices[top]

        for idx in peak_indices:
            if len(examples) >= max_examples:
                return examples
            q_tok, k_tok = int(idx[0]), int(idx[1])
            if abs(k_tok - q_tok) <= 2 * alpha or k_tok == 0:
                continue
            weight = float(A[q_tok, k_tok])

            q_tok_phrase_num = 0
            k_tok_phrase_num = 0
            for i, (s, e) in enumerate(ranges):
                if s <= q_tok < e:
                    q_tok_phrase_num = i
                if s <= k_tok < e:
                    k_tok_phrase_num = i
            q_tok_cluster = cluster_seq[q_tok_phrase_num]
            k_tok_cluster = cluster_seq[k_tok_phrase_num]
            transition = False
            if q_tok_cluster != "UNK":
                transition = (
                    int(q_tok_cluster) in cluster_transition_indices
                    and q_tok <= ranges[int(q_tok_cluster)][0] + 3
                ) or (
                    int(q_tok_cluster) + 1 in cluster_transition_indices
                    and ranges[int(q_tok_cluster)][1] - 3 <= q_tok
                )

            q0, q1 = max(0, q_tok - alpha), min(S, q_tok + alpha + 1)
            k0, k1 = max(0, k_tok - alpha), min(S, k_tok + alpha + 1)

            examples.append(
                {
                    "layer": layer,
                    "head": h,
                    "spike_scr": round(score, 2),
                    "q_tok": q_tok,
                    "k_tok": k_tok,
                    "weight": weight,
                    "q_ctx": tokenizer.decode(ids[q0:q1]),
                    "k_ctx": tokenizer.decode(ids[k0:k1]),
                    "relation": "intra" if q_tok_cluster == k_tok_cluster else "inter",
                    "transition": transition,
                    "q_tok_cluster": q_tok_cluster,
                    "k_tok_cluster": k_tok_cluster,
                    "q_tok_phrase_num": q_tok_phrase_num,
                    "k_tok_phrase_num": k_tok_phrase_num,
                }
            )

        del A, mask, local_max, peaks, peak_vals, top, idx
    torch.cuda.empty_cache()
    gc.collect()
    return examples


def _spike_score_2d(A):
    """Composite spikiness score for a 2D matrix."""
    mu = A.mean()
    sig = A.std(unbiased=False)
    if sig == 0 or mu == 0:
        return 0.0
    peak_z = (A.max() - mu) / sig
    cv = sig / mu
    # 2D kurtosis
    kurtosis = ((A - mu) ** 4).mean() / (sig**4 + 1e-9)
    return float(peak_z * cv * sqrt(kurtosis))


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
    EOS = tokenizer.eos_token_id

    # tokenise every phrase --------------------------------------------
    ids_per_phrase = [
        tokenizer.encode(t, add_special_tokens=False) for t in phrase_texts
    ]
    for ids in ids_per_phrase[:-1]:  # EOS between phrases
        ids.append(EOS)

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
    return attn_rows, hs_rows, tok_ranges


def pad_and_cat(slices, T_final):
    """
    slices : list of tensors  (H, q_len, cur_width)
    Returns (H, T_final, T_final) lower-tri matrix on CPU.
    """
    H = slices[0].shape[0]
    rows = []
    for blk in slices:
        pad_cols = T_final - blk.shape[2]
        if pad_cols:
            blk = torch.nn.functional.pad(blk, (0, pad_cols))  # right-pad
        rows.append(blk)
    return torch.cat(rows, dim=1)  # cat over the row axis


def save_heatmap(mat, title, fp):
    plt.figure(figsize=(6, 5))
    sns.heatmap(mat, cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fp)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis loop
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    model_name = args.model_name
    alpha = args.alpha
    spike_th = args.spike_th
    burst_z = args.burst_z
    nms_size = args.nms_size
    rng_seed = args.rng_seed
    batch_size_limit = args.batch_size_limit
    output_dir = f"./{model_name.split('/')[-1].replace('-', '_')}/{args.output_dir}"
    os.makedirs(output_dir, exist_ok=True)
    split_path = f"./{model_name.split('/')[-1].replace('-', '_')}/thought_split.jsonl"
    cluster_path = (
        f"./{model_name.split('/')[-1].replace('-', '_')}/thought_clustering.jsonl"
    )
    torch.manual_seed(rng_seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)

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

    # ------------------- accumulators ---------------------------------------------
    per_layer_ari = defaultdict(list)
    per_layer_pur = defaultdict(list)

    within_scores, inter_scores = [], []
    burst_examples = []

    # ------------------- iterate ---------------------------------------------------
    from tqdm import tqdm

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

        input_ids = input_ids.to(next(model.parameters()).device)
        attn_rows, hs_rows, ranges = incremental_run(
            model,
            tokenizer,
            phrases,
            batch_size_limit,
            want_attn=True,
            want_hidden=True,
            q_chunk=256,
        )
        if len(ranges) == 0:
            print(f"Skipping example #{idx} with no valid ranges.")
            continue
        cluster_seq = cluster_seq[: len(ranges)]  # truncate to match ranges
        cluster_transition_indices = []
        for i in range(1, len(cluster_seq)):
            if cluster_seq[i] != cluster_seq[i - 1]:
                cluster_transition_indices.append(i)

        total_len = ranges[-1][1]  # final token count
        print(f"Processed example #{idx} with {total_len} tokens.")
        # ------------- representations & clustering -------------------------------
        reps_layers = []
        for layer_act in hs_rows:  # len = L+1
            layer_cat = torch.cat(layer_act, 0)  # (T, D)
            reps = [layer_cat[s:e].mean(0) for s, e in ranges]
            reps_layers.append(torch.stack(reps, 0))
        for l, reps in enumerate((reps_layers)):
            res = cluster_metrics(cluster_seq, reps, rng_seed)
            if res:
                ari, pur = res
                per_layer_ari[l].append(ari)
                per_layer_pur[l].append(pur)
        print("Processed representations & clustering for example #", idx)
        # ------------- attention analysis -----------------------------------------
        per_layer_within_mass = []
        per_layer_inter_mass = []
        for l, layer_attn_chunks in enumerate(attn_rows):
            layer_attn_full = pad_and_cat(
                layer_attn_chunks, total_len
            )  # (H, T_final, T_final)
            attn_mean_heads_layer = layer_attn_full.mean(0)  # (T_final, T_final)
            w_mass = attention_within_cluster(
                attn_mean_heads_layer, ranges, cluster_seq
            )
            i_mass = attention_inter_cluster(attn_mean_heads_layer, ranges, cluster_seq)
            per_layer_within_mass.append(w_mass)
            per_layer_inter_mass.append(i_mass)

            layer_attn_full_unsqueezed = layer_attn_full.unsqueeze(
                0
            )  # (1, H, T_final, T_final)
            bursts_in_layer = detect_burst_heads_2d(
                layer_attn_full_unsqueezed,  # (1, H, T_final, T_final)
                ranges,
                cluster_seq,
                cluster_transition_indices,
                tokenizer,
                layer=l,
                alpha=alpha,
                spike_th=spike_th,  # adjust if you get too many / few heads
                burst_z=burst_z,
                cache={"ids": input_ids[0].cpu().tolist()},
                nms_size=nms_size,
            )
            burst_examples.extend(bursts_in_layer)
            print(f"Processed attention analysis for layer {l} in example #{idx}")
            del (
                layer_attn_full,
                attn_mean_heads_layer,
                bursts_in_layer,
                layer_attn_full_unsqueezed,
            )
        within_scores.append(per_layer_within_mass)
        inter_scores.append(per_layer_inter_mass)
        # ------------- tidy up GPU -------------------------------------------------
        del attn_rows, hs_rows, reps, reps_layers
        torch.cuda.empty_cache()
        gc.collect()

    # ────────────────────────────────────────────────────────────────────────
    # Summary & visualisation
    # ────────────────────────────────────────────────────────────────────────
    # ---- cluster metrics
    layers = sorted(per_layer_ari.keys())
    mean_ari = [np.mean(per_layer_ari[l]) for l in layers]
    std_ari = [np.std(per_layer_ari[l]) for l in layers]
    mean_pur = [np.mean(per_layer_pur[l]) for l in layers]
    std_pur = [np.std(per_layer_pur[l]) for l in layers]

    plt.figure(figsize=(7, 4))
    plt.errorbar(layers, mean_ari, yerr=std_ari, label="ARI", marker="o")
    plt.errorbar(layers, mean_pur, yerr=std_pur, label="Purity", marker="s")
    plt.xlabel("Layer index (0 = embeddings)")
    plt.ylabel("Score")
    plt.title("Per-layer clustering quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir + "/per_layer_clustering.png")
    plt.close()

    # ---- within vs inter cluster attention
    within_scores = list(map(list, zip(*within_scores)))
    inter_scores = list(map(list, zip(*inter_scores)))
    for i, (within_score, inter_score) in enumerate(zip(within_scores, inter_scores)):
        data = {"Within": within_score, "Inter": inter_score}
        plt.figure(figsize=(6, 4))
        sns.violinplot(data=data)
        plt.ylabel("Mean attention mass")
        plt.title("Within- vs. Inter-cluster attention")
        plt.tight_layout()
        plt.savefig(output_dir + f"/within_vs_inter_attention_{i}.png")
        plt.close()

    for i, (within_score, inter_score) in enumerate(zip(within_scores, inter_scores)):
        ratio = [w / i if i > 0 else 0 for w, i in zip(within_score, inter_score)]
        plt.figure(figsize=(6, 4))
        plt.hist(ratio, bins=30)
        plt.xlabel("Within / Inter attention ratio")
        plt.ylabel("#Problems")
        plt.tight_layout()
        plt.savefig(output_dir + f"/attention_ratio_hist_{i}.png")
        plt.close()

    # ---- save burst-head examples
    burst_fp = output_dir + "/burst_heads.jsonl"
    with open(burst_fp, "w") as f:
        for ex in burst_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(burst_examples)} burst-head snippets ➜ {burst_fp}")

    print(f"All figures & data stored in: {output_dir}")


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
        "--alpha",
        type=int,
        default=25,
        help="Number of tokens to show left/right in burst-head snippets.",
    )
    parser.add_argument(
        "--rng_seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--batch_size_limit",
        type=int,
        default=10000,
        help="Maximum number of tokens per forward pass.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="LRM_analysis_output",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--spike_th", type=float, default=100.0, help="Threshold for spike detection."
    )
    parser.add_argument(
        "--burst_z",
        type=float,
        default=5.0,
        help="Z-score threshold for burst detection.",
    )
    parser.add_argument(
        "--nms_size",
        type=int,
        default=5,
        help="Size of the non-maximum suppression window.",
    )
    args = parser.parse_args()
    main(args)

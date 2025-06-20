#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis of reasoning traces:

1.  Forward pass with output_attentions & output_hidden_states.
2.  Average hidden states for every phrase → activation vector.
3.  Cluster these vectors and compare with gold `cluster_seq`.
4.  Compute mean attention mass that tokens of phrase_i put on tokens
    of phrase_j when the two phrases share the same cluster label.
5.  Detect “burst” attention heads (usually quiet, rarely very high),
    extract contextual snippets around the query/key tokens.
6.  Save figures + JSONL with the burst-head cases.
"""
# ──────────────────────────────────────────────────────────────────────────────
# Imports & basic configuration
# ──────────────────────────────────────────────────────────────────────────────
import os, json, math, random, pickle, gc
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Model, Qwen2ForCausalLM
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from math import sqrt
import types

from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns   # purely for prettier heat-maps; comment out if unwelcome
from scipy.ndimage import maximum_filter
sns.set_context("paper")

# ------------------------------------------------------------------------------
# Paths & hyper-parameters
# ------------------------------------------------------------------------------
THOUGHT_SPLIT_FP   = "../model_outputs/Underthinking_Reproduction_Thought_split_Deepseek_R1_Distill_Qwen_14B.jsonl"
THOUGHT_CLUSTER_FP = "../model_outputs/Underthinking_Reproduction_Thought_cluster_4.1_Deepseek_R1_Distill_Qwen_14B.jsonl"
MODEL_NAME         = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

OUT_DIR            = Path("./analysis_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# analysis granularity
MAX_EXAMPLES       = None          # None → run on all; or e.g. 50 for a quick pass
LAYER_SELECTION    = "last"        # "last", "mean", or an int for a specific layer
ALPHA              = 25            # number of tokens left/right to display in burst-head snippets
BURST_Q            = 0.995         # global quantile; above this an attn weight counts as a burst
RNG_SEED           = 42

torch.manual_seed(RNG_SEED)
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# VRAM helpers
DTYPE              = torch.bfloat16
DEVICE_MAP         = "auto"        # let transformers split across GPUs
BATCH_SIZE_LIMIT   = 24000         # max #tokens per forward (rough heuristic)


class Qwen2AttentionCPU(Qwen2Attention):
    """Return attention maps on CPU to save GPU VRAM."""
    def forward(self, *args, **kwargs):
        out, attn = super().forward(*args, **kwargs)
        if attn is not None:
            attn = attn.to("cpu")
        return out, attn
# ----------------------------------------------------------------
# 2)  Apply the patch to a loaded model
# ----------------------------------------------------------------
def patch_qwen2_cpu(model: torch.nn.Module):
    """
    Patch a **loaded** Qwen2ForCausalLM or Qwen2Model so that:
      • every self-attention returns CPU maps
      • returned hidden_states & attentions are on CPU
    """
    # 2-A  Replace every self-attention block
    core = model.model if isinstance(model, Qwen2ForCausalLM) else model
    if not isinstance(core, Qwen2Model):
        raise TypeError("Expect Qwen2ForCausalLM or Qwen2Model.")

    for i, layer in enumerate(core.layers):
        old_attn = layer.self_attn
        new_attn = Qwen2AttentionCPU(old_attn.config, layer_idx=i).to(old_attn.o_proj.weight.device,
                                                                      dtype=old_attn.o_proj.weight.dtype)
        new_attn.load_state_dict(old_attn.state_dict(), strict=True)
        layer.self_attn = new_attn
        print(f"[patch]  layer {i:02d} self_attn ➜ CPU-returning")

    # 2-B  Wrap the forward pass of the *top* module so it off-loads
    #      hidden_states / attentions AFTER logits are done.
    target = model if isinstance(model, Qwen2ForCausalLM) else core
    orig_fwd = target.forward

    def forward_cpu(self, *args, **kwargs):
        outputs = orig_fwd(*args, **kwargs)
        # attn & hidden states can be very large – move them
        if getattr(outputs, "attentions", None) is not None:
            outputs.attentions = tuple(a.to("cpu") if a is not None else None
                                       for a in outputs.attentions)
        if getattr(outputs, "hidden_states", None) is not None:
            outputs.hidden_states = tuple(h.to("cpu") if h is not None else None
                                          for h in outputs.hidden_states)
        if hasattr(outputs, "last_hidden_state"):
            outputs.last_hidden_state = outputs.last_hidden_state.to("cpu")
        return outputs

    # bind the new method
    target.forward = types.MethodType(forward_cpu, target)
    print("[patch]  forward() wrapped to off-load hidden_states/attentions")


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def load_data():
    with open(THOUGHT_SPLIT_FP)   as f1, \
         open(THOUGHT_CLUSTER_FP) as f2:
        splits   = [json.loads(l) for l in f1]
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


def cluster_metrics(true_seq, reps_layer):
    """
    true_seq: list[str] cluster id *per phrase*.
    reps_layer: torch (P, H)
    Returns ARI & purity for this layer (or None if metrics ill-defined).
    """
    uniq = sorted({c for c in true_seq if c != "UNK"})
    if not uniq:
        return None
    kmeans = KMeans(len(uniq), n_init="auto", random_state=RNG_SEED)\
                .fit(reps_layer.to(torch.float32).cpu().numpy())
    pred = kmeans.labels_
    label_to_int = {lab: i for i, lab in enumerate(uniq)}
    true_int = [label_to_int.get(c, -1) for c in true_seq]
    mask = [t != -1 for t in true_int]
    if not any(mask):
        return None
    true_f = [t for t, m in zip(true_int, mask) if m]
    pred_f = [p for p, m in zip(pred,     mask) if m]
    cm = confusion_matrix(true_f, pred_f)
    purity = cm.max(axis=0).sum() / cm.sum()
    return adjusted_rand_score(true_f, pred_f), purity


def reps_per_layer(hidden_states, ranges):
    """
    Returns list[length = n_layers+1] where item l is tensor (P, H)
    containing the mean hidden vector of every phrase at layer l.
    """
    reps_layers = []
    for layer_act in hidden_states:            # (1, S, H)
        layer_act = layer_act[0].to("cpu")               # drop batch
        reps = [layer_act[s:e].mean(0) for s, e in ranges]
        reps_layers.append(torch.stack(reps, 0))
    return reps_layers                         # list[(P, H)]


def attention_within_cluster(attn, ranges, clusters):
    same = [(i, j) for i in range(len(ranges))
                    for j in range(len(ranges))
                    if clusters[i] == clusters[j] != "UNK" and i != j]
    return _attention_mass(attn, ranges, same)


def attention_inter_cluster(attn, ranges, clusters):
    diff = [(i, j) for i in range(len(ranges))
                    for j in range(len(ranges))
                    if clusters[i] != clusters[j] and
                       "UNK" not in (clusters[i], clusters[j])]
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
    attn,                          # (L, H, S, S)
    ranges,
    tokenizer,
    layer,
    alpha          = 5,
    spike_th       = 10.0,         # keep heads with score > spike_th
    burst_z        = 3.0,          # 'bursts' above mean+burst_z*std
    max_per_head   = 30,
    max_examples   = 1_000,
    cache          = None,
    nms_size       = 3,            # window size for local maxima
):
    L, H, S, _ = attn.shape
    ids = cache["ids"]
    examples = []

    for h in range(H):
        A = attn[0, h].to(torch.float32)          # (S, S)
        # ---------- 1) Compute a 2D spikiness score ----------
        score = _spike_score_2d(A)
        if score < spike_th:
            continue

        # ---------- 2) Find local 2D bursts ----------
        mu, sig = A.mean().item(), A.std(unbiased=False).item()
        thresh  = mu + burst_z * sig

        # Boolean mask of potential bursts
        mask = (A > thresh)

        # Find local maxima (NMS in 2D)
        if mask.sum() == 0:
            continue
        # local maximum filter (window size nms_size)
        local_max = (A == torch.from_numpy(maximum_filter(A.cpu().numpy(), size=nms_size))).to(A.device)
        peaks = mask & local_max
        peak_indices = torch.nonzero(peaks, as_tuple=False)
        if peak_indices.numel() == 0:
            continue

        # strongest peaks first
        peak_vals = A[peak_indices[:,0], peak_indices[:,1]]
        top = torch.argsort(peak_vals, descending=True)[:max_per_head]
        peak_indices = peak_indices[top]

        for idx in peak_indices:
            if len(examples) >= max_examples:
                return examples
            q_tok, k_tok = int(idx[0]), int(idx[1])
            if abs(k_tok - q_tok) <= 2 * alpha:
                continue
            weight = float(A[q_tok, k_tok])

            q0, q1 = max(0, q_tok-alpha), min(S, q_tok+alpha+1)
            k0, k1 = max(0, k_tok-alpha), min(S, k_tok+alpha+1)

            examples.append({
                "layer":     layer,
                "head":      h,
                "spike_scr": round(score, 2),
                "q_tok":     q_tok,
                "k_tok":     k_tok,
                "weight":    weight,
                "q_ctx":     tokenizer.decode(ids[q0:q1]),
                "k_ctx":     tokenizer.decode(ids[k0:k1]),
            })
    return examples

def _spike_score_2d(A):
    """Composite spikiness score for a 2D matrix."""
    mu  = A.mean()
    sig = A.std(unbiased=False)
    if sig == 0 or mu == 0:
        return 0.0
    peak_z   = (A.max() - mu) / sig
    cv       = sig / mu
    # 2D kurtosis
    kurtosis = ((A - mu)**4).mean() / (sig**4 + 1e-9)
    return float(peak_z * cv * sqrt(kurtosis))

def incremental_run(model, tokenizer, phrase_texts,
                    want_attn=True, want_hidden=True,
                    q_chunk=1024):
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
    L  = model.config.num_hidden_layers
    H  = model.config.num_attention_heads
    D  = model.config.hidden_size
    EOS = tokenizer.eos_token_id

    # tokenise every phrase --------------------------------------------
    ids_per_phrase = [tokenizer.encode(t, add_special_tokens=False)
                      for t in phrase_texts]
    for ids in ids_per_phrase[:-1]:   # EOS between phrases
        ids.append(EOS)

    attn_rows = [ [] for _ in range(L) ] if want_attn   else None
    hs_rows   = [ [] for _ in range(L+1) ] if want_hidden else None
    past_kv   = None
    cursor    = 0
    tok_ranges = []

    for p_ids in ids_per_phrase:
        phrase_start = cursor
        # micro-batch the phrase so GPU sees ≤ q_chunk tokens
        pos = 0
        while pos < len(p_ids):
            piece = torch.tensor([p_ids[pos:pos+q_chunk]],
                                 device=model.device)
            pos  += q_chunk
            q_len = piece.size(1)

            with torch.no_grad():
                out = model(input_ids       = piece,
                            past_key_values = past_kv,
                            use_cache       = True,
                            output_attentions   = want_attn,
                            output_hidden_states= want_hidden,
                            attn_mode="torch")

            past_kv = out.past_key_values   # extend cache

            if want_attn:
                for l, a in enumerate(out.attentions):      # (1,H,q,prev+q)
                    attn_rows[l].append(a[0].cpu())         # to CPU, drop batch
            if want_hidden:
                for l, h in enumerate(out.hidden_states):   # (1,q,D)
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
    return torch.cat(rows, dim=1)          # cat over the row axis


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
def main():
    # ------------------- load model ------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        device_map=DEVICE_MAP,
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
    )
    # patch_qwen2_cpu(model)  # patch to return attentions on CPU
    model.eval()

    # ------------------- data ------------------------------------------------------
    splits, clusters = load_data()
    if MAX_EXAMPLES:
        splits   = splits[:MAX_EXAMPLES]
        clusters = clusters[:MAX_EXAMPLES]

    # ------------------- accumulators ---------------------------------------------
    per_layer_ari   = defaultdict(list)
    per_layer_pur   = defaultdict(list)

    within_scores, inter_scores = [], []
    burst_examples  = []

    # ------------------- iterate ---------------------------------------------------
    from tqdm import tqdm
    for idx, (sp, cl) in enumerate(zip(tqdm(splits), clusters), 1):
        phrases      = sp["phrases"]
        if not phrases:
            continue

        # derive ground-truth cluster sequence
        phr_to_cluster = {}
        for part in cl["gpt4o_answer"].split("Cluster"):
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
        if ids_len > BATCH_SIZE_LIMIT:
            print(f"[warn] Sample #{idx} too long ({ids_len} tok) – skipping.")
            continue

        input_ids = input_ids.to(next(model.parameters()).device)
        attn_rows, hs_rows, ranges = incremental_run(
            model, tokenizer, phrases,
            want_attn=True, want_hidden=True, q_chunk=256
        )

        total_len = ranges[-1][1]                  # final token count
        print(f"Processed example #{idx} with {total_len} tokens.")
        # ------------- representations & clustering -------------------------------
        reps_layers = []
        for layer_act in hs_rows:                     # len = L+1
            layer_cat = torch.cat(layer_act, 0)       # (T, D)
            reps = [ layer_cat[s:e].mean(0) for s,e in ranges ]
            reps_layers.append(torch.stack(reps, 0))
        for l, reps in enumerate((reps_layers)):
            res = cluster_metrics(cluster_seq, reps)
            if res:
                ari, pur = res
                per_layer_ari[l].append(ari)
                per_layer_pur[l].append(pur)
        print("Processed representations & clustering for example #", idx)
        # ------------- attention analysis -----------------------------------------
        per_layer_within_mass = []
        per_layer_inter_mass  = []
        for l, layer_attn_chunks in enumerate(attn_rows):
            layer_attn_full = pad_and_cat(layer_attn_chunks, total_len)  # (H, T_final, T_final)
            attn_mean_heads_layer = layer_attn_full.mean(0)  # (T_final, T_final)
            w_mass = attention_within_cluster(attn_mean_heads_layer, ranges, cluster_seq)
            i_mass = attention_inter_cluster (attn_mean_heads_layer, ranges, cluster_seq)
            per_layer_within_mass.append(w_mass)
            per_layer_inter_mass.append(i_mass)

            layer_attn_full_unsqueezed = layer_attn_full.unsqueeze(0)  # (1, H, T_final, T_final)
            bursts_in_layer = detect_burst_heads_2d(
                layer_attn_full_unsqueezed,  # (1, H, T_final, T_final)
                ranges,
                tokenizer,
                layer=l,
                alpha=ALPHA,
                spike_th=100.0,               # adjust if you get too many / few heads
                burst_z=5.0,
                cache={"ids": input_ids[0].cpu().tolist()},
                nms_size=5,
            )
            burst_examples.extend(bursts_in_layer)
            print(f"Processed attention analysis for layer {l} in example #{idx}")
            del layer_attn_full, attn_mean_heads_layer, bursts_in_layer, layer_attn_full_unsqueezed
            gc.collect()
        within_scores.append(per_layer_within_mass)
        inter_scores.append(per_layer_inter_mass)
        # ------------- tidy up GPU -------------------------------------------------
        del reps, reps_layers
        torch.cuda.empty_cache()
        gc.collect()

    # ────────────────────────────────────────────────────────────────────────
    # Summary & visualisation
    # ────────────────────────────────────────────────────────────────────────
    # ---- cluster metrics
    layers = sorted(per_layer_ari.keys())
    mean_ari = [np.mean(per_layer_ari[l]) for l in layers]
    std_ari  = [np.std (per_layer_ari[l]) for l in layers]
    mean_pur = [np.mean(per_layer_pur[l]) for l in layers]
    std_pur  = [np.std (per_layer_pur[l]) for l in layers]

    plt.figure(figsize=(7,4))
    plt.errorbar(layers, mean_ari, yerr=std_ari, label="ARI", marker="o")
    plt.errorbar(layers, mean_pur, yerr=std_pur, label="Purity", marker="s")
    plt.xlabel("Layer index (0 = embeddings)")
    plt.ylabel("Score")
    plt.title("Per-layer clustering quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_layer_clustering.png"); plt.close()

    # ---- within vs inter cluster attention
    within_scores = list(map(list, zip(*within_scores)))
    inter_scores  = list(map(list, zip(*inter_scores)))
    for i, (within_score, inter_score) in enumerate(zip(within_scores, inter_scores)):
        data = {"Within": within_score, "Inter": inter_score}
        plt.figure(figsize=(6,4))
        sns.violinplot(data=data)
        plt.ylabel("Mean attention mass")
        plt.title("Within- vs. Inter-cluster attention")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"within_vs_inter_attention_{i}.png"); plt.close()

    for i, (within_score, inter_score) in enumerate(zip(within_scores, inter_scores)):
        ratio = [w/i if i>0 else 0 for w,i in zip(within_score, inter_score)]
        plt.figure(figsize=(6,4))
        plt.hist(ratio, bins=30)
        plt.xlabel("Within / Inter attention ratio")
        plt.ylabel("#Problems")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"attention_ratio_hist_{i}.png"); plt.close()

    # ---- save burst-head examples
    burst_fp = OUT_DIR / "burst_heads.jsonl"
    with open(burst_fp, "w") as f:
        for ex in burst_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(burst_examples)} burst-head snippets ➜ {burst_fp}")

    print(f"All figures & data stored in: {OUT_DIR.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
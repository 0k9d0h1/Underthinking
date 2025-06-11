import os, math, random, argparse
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
os.environ["HF_HOME"] = "/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"
from pathlib import Path
from tqdm.auto import tqdm

import torch, torch.nn.functional as F
from datasets import Dataset, Features, Value, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------
# constants / helpers
# -----------------------------------------------------------------
_MIN_LEN       = 100
_PREFIX        = "\n\nThat is, "
_SPLIT_STRINGS = ["</think>", "Alternatively", "Wait", "Hmm", "But wait"]
RNG            = random.Random(42)

def remove_leading_words(text: str) -> str:
    for w in ["Wait", "Hmm", "Alternatively", "But wait"]:
        if text.lstrip().lower().startswith(w.lower()):
            return text.lstrip()[len(w):].lstrip(" ,.")
    return text.lstrip()

def split_string_and_ids(ids, delims, tok):
    """Token-aligned split that keeps multi-token delimiters intact."""
    delim_seqs = []
    for d in delims:
        for v in (d, " " + d):
            seq = tok(v, add_special_tokens=False)["input_ids"]
            if seq: delim_seqs.append(seq)
    delim_seqs.sort(key=len, reverse=True)

    parts, cur, i, N = [], [], 0, len(ids)
    while i < N:
        matched = False
        for seq in delim_seqs:
            L = len(seq)
            if ids[i : i + L] == seq:
                if cur: parts.append(cur)
                cur, i, matched = ids[i : i + L], i + L, True
                break
        if not matched:
            cur.append(ids[i]); i += 1
    if cur: parts.append(cur)
    texts = [tok.decode(p, skip_special_tokens=True) for p in parts]
    return texts, parts

def batched_nll(
    model,
    tokenizer,
    batch_pairs,
    max_tokens: int = 4192,      # ← global token budget per forward
    max_seq_len: int | None = None # clip to model’s pos-emb size
):
    """
    Compute per-sequence NLL for a *list* of (t1, t2) pairs.
    Splits the batch into smaller chunks so that (#tokens ≤ max_tokens)
    which prevents CUDA OOM and lets us use very large nominal batch_size.
    """
    if not batch_pairs:
        return []

    # -----------------------------------------------------------------
    # 0. helper: process one packed chunk → per-seq loss list
    # -----------------------------------------------------------------
    def _process_chunk(chunk):
        """
        chunk : list of dicts with keys
                {"t1", "t2", "t2_len", "concat_text"}
        """
        texts = [d["concat_text"] for d in chunk]
        enc   = tokenizer(texts, return_tensors="pt", padding=True)
        ids, msk = enc.input_ids, enc.attention_mask
        labels   = ids.clone()

        # mask tokens *before* t2
        for row_idx, d in enumerate(chunk):
            valid_len = msk[row_idx].sum().item()
            boundary  = valid_len - d["t2_len"]   # first t2 token
            labels[row_idx, :boundary] = -100

        # to GPU-0 (DataParallel master)
        ids, msk, labels = ids.to("cuda:0"), msk.to("cuda:0"), labels.to("cuda:0")

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=msk).logits  # (b, L, V)

        ce = F.cross_entropy(
            logits[:, :-1, :].transpose(1, 2),  # (b, V, L-1)
            labels[:, 1:],                      # (b, L-1)
            reduction="none",
            ignore_index=-100,
        )                                       # (b, L-1)

        tok_mask = (labels[:, 1:] != -100)
        denom    = tok_mask.sum(1).clamp(min=1)     # avoid ÷0
        seq_loss = (ce * tok_mask).sum(1) / denom   # (b,)

        # tidy
        del logits, ce
        torch.cuda.empty_cache()
        return seq_loss.cpu().tolist()

    # -----------------------------------------------------------------
    # 1. greedy packing based on *true* token length (no padding cost)
    # -----------------------------------------------------------------
    packed_losses = []
    cur_chunk, cur_tokens = [], 0

    for t1, t2 in batch_pairs:
        # quick length probe (no tensors kept)
        enc_len  = tokenizer(t1 + t2, add_special_tokens=False,
                             return_length=True)["length"][0]
        t2_len   = tokenizer(t2, add_special_tokens=False,
                             return_length=True)["length"][0]

        if cur_tokens + enc_len > max_tokens and cur_chunk:
            # flush current chunk to GPU(s)
            packed_losses.extend(_process_chunk(cur_chunk))
            cur_chunk, cur_tokens = [], 0

        cur_chunk.append(
            {"t1": t1, "t2": t2, "t2_len": t2_len, "concat_text": t1 + t2}
        )
        cur_tokens += enc_len

    if cur_chunk:
        packed_losses.extend(_process_chunk(cur_chunk))

    return packed_losses


# -----------------------------------------------------------------
# end-to-end pipeline
# -----------------------------------------------------------------
def main(gen_dir: Path, reg_dir: Path,
         batch_size: int = 64, max_samples: int | None = None):

    # 1 . dataset & model --------------------------------------------------
    gen_ds = load_from_disk(gen_dir)
    if max_samples:
        gen_ds = gen_ds.select(range(max_samples))

    tok = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")                    # puts *first* replica on GPU-0

    model = base_model
    # # wrap for 2-GPU data parallel
    # if torch.cuda.device_count() >= 2:
    #     model = torch.nn.DataParallel(base_model, device_ids=[0, 1])
    # else:
    #     raise RuntimeError("Need at least 2 GPUs for this script.")
    model.eval()

    # 2 . iterate records, accumulate pairs, compute batched NLL ----------
    rows, pair_buf = [], []
    pbar_outer = tqdm(gen_ds, desc="Scanning CoT answers")

    flush = lambda: rows.extend(
        {"text": t1 + t2, "label": l}
        for (t1, t2), l in zip(pair_buf, batched_nll(model, tok, pair_buf))
    ) or pair_buf.clear()

    for rec in pbar_outer:
        out_ids = tok(rec["model_output"], add_special_tokens=False)["input_ids"]
        phrases, _ = split_string_and_ids(out_ids, _SPLIT_STRINGS, tok)
        end_idx = next((i for i, p in enumerate(phrases) if "</think>" in p),
                       len(phrases))

        sampled_first_indices = RNG.sample(
            [i for i in range(1, end_idx) if len(phrases[i]) >= _MIN_LEN],
            min(len([i for i in range(1, end_idx) if len(phrases[i]) >= _MIN_LEN]), 8)
        )

        for i in sampled_first_indices:
            if len(phrases[i]) < _MIN_LEN:          # too short → skip
                continue
            js = [j for j in range(i) if len(phrases[j]) >= _MIN_LEN]
            js = RNG.sample(js, min(len(js), 8))    # at most 8 refs
            for j in js:
                t1 = phrases[j] + _PREFIX
                t2 = remove_leading_words(phrases[i])
                pair_buf.append((t1, t2))
                if len(pair_buf) >= batch_size:
                    flush()

    if pair_buf: flush()       # leftovers

    # 3 . save regression dataset -----------------------------------------
    print(f"Built {len(rows):,} (concat, loss) samples → {reg_dir}")
    Dataset.from_dict(
        {"text":  [r["text"]  for r in rows],
         "label": [r["label"] for r in rows]},
        features=Features({"text": Value("string"), "label": Value("float32")})
    ).save_to_disk(reg_dir)


# -----------------------------------------------------------------
# CLI glue
# -----------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", type=Path,
                    default=Path("/home/kdh0901/Desktop/Underthinking/PRM/gen_vllm"),
                    help="folder with vLLM generations")
    ap.add_argument("--reg_dir", type=Path,
                    default=Path("/home/kdh0901/Desktop/Underthinking/PRM/reg_dp"),
                    help="output folder for regression dataset")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--max_samples", type=int,
                    help="debug: truncate the generation set")
    args = ap.parse_args()

    args.reg_dir.mkdir(parents=True, exist_ok=True)
    main(args.gen_dir, args.reg_dir,
         batch_size=args.batch_size,
         max_samples=args.max_samples)
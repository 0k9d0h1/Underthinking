import argparse, json, os, torch
os.environ["HF_HOME"] = "/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_qwen2_custom import Qwen2ForCausalLMCustom

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="HF repo or local path with the original weights")
    ap.add_argument("--dst", required=True,
                    help="Where to save the converted model (same dir can be the "
                         "one that already contains your *.py files)")
    ap.add_argument("--target_layer", type=int, default=10)
    ap.add_argument("--target_head", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.50)
    ap.add_argument("--alpha", type=float, default=0.15)
    args = ap.parse_args()

    # 1) Load original model (FP16 saves RAM, low_cpu_mem_usage avoids big spikes)
    base = AutoModelForCausalLM.from_pretrained(
        args.src,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,      # Deepseek publishes its own model class
    )
    tok = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)

    # 2) Copy its config and append our extra knobs
    cfg = base.config
    cfg.target_layer = args.target_layer
    cfg.target_head = args.target_head
    cfg.fire_threshold = args.tau
    cfg.sub_alpha = args.alpha
    cfg.architectures = ["Qwen2ForCausalLMCustom"]  # tells HF which class to pick
    cfg.auto_map = {
        "AutoModel": "modeling_qwen2_custom.Qwen2ForCausalLMCustom",
        "AutoModelForCausalLM": "modeling_qwen2_custom.Qwen2ForCausalLMCustom",
    }

    # 3) Build *our* model with that config
    custom = Qwen2ForCausalLMCustom(cfg)
    custom._tied_weights_keys = []
    base._tied_weights_keys = []  # avoid warnings about tied weights
    print(base.state_dict().keys())
    print(custom.state_dict().keys())
    # 4) Transfer weights (names are identical, so this is trivial)
    # We use strict=False because we know lm_head.weight might be missing from the base state_dict.
    missing, unexpected = custom.load_state_dict(base.state_dict(), strict=False)
    if unexpected or (missing and missing != ['lm_head.weight']):
        print("WARNING: Unexpected key matching issues.")
        print("  Missing keys:", missing)
        print("  Unexpected keys:", unexpected)

    # Manually copy the language model head weights from the base model.
    print("Manually copying lm_head.weight from base model...")
    custom.lm_head.weight.data.copy_(base.lm_head.weight.data)
    print("Copy complete.")
    print(custom._tied_weights_keys)

    # 5) Save – “safetensors” keeps one shard per 2 GB by default
    custom.save_pretrained(args.dst, safe_serialization=True)
    tok.save_pretrained(args.dst)

    # 6) Save updated config as JSON (optional – save_pretrained already did this)
    cfg_json = os.path.join(args.dst, "config.json")
    with open(cfg_json, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print("✔ Finished.  New model in:", args.dst)

if __name__ == "__main__":
    main()
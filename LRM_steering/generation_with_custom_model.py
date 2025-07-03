import argparse, json, os
os.environ["HF_HOME"] = "/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
from vllm import LLM, SamplingParams
import argparse
from transformers import AutoModelForCausalLM

def main(args):
    # vLLM automatically respects the new class once we allow remote code
    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16")

    out = llm.generate(
        [args.prompt],
        SamplingParams(max_tokens=128, temperature=0.7)
    )

    print(out[0].outputs[0].text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path or HF repo name; must contain the custom code + weights")
    parser.add_argument("--prompt", default="Hello, explain quantum tunnelling in one paragraph.")
    args = parser.parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True,
    )
    print(model.config.tie_word_embeddings)
    print(model.get_input_embeddings().weight.data_ptr()
      == model.lm_head.weight.data_ptr())
    exit()
    main(args)
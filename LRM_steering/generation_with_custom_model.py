import argparse, json, os
os.environ["HF_HOME"] = "/home/kdh0901/Desktop/cache_dir/kdh0901/.cache/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from vllm import LLM, SamplingParams
import argparse
from transformers import AutoModelForCausalLM
from modeling_qwen2_custom import reset_fire_recorder, FIRE_RECORDER

# Monkey-patch for ModelRegistry.is_text_generation_model(architects)
def new_is_text_generation_model(architectures):
    from vllm import ModelRegistry
    if "Qwen2ForCausalLMCustom" in architectures:
        # If our custom model is present, we know it's a text generation model
        return True
    return ModelRegistry.inspect_model_cls(architectures).is_text_generation_model


def register():
    from vllm import ModelRegistry
    from modeling_qwen2_custom import Qwen2ForCausalLMCustom
    ModelRegistry.is_text_generation_model = new_is_text_generation_model
    ModelRegistry.register_model(
        "Qwen2ForCausalLMCustom",
        Qwen2ForCausalLMCustom,
    )

def main(args):
    # vLLM automatically respects the new class once we allow remote code
    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16")

    # Ensure the recorder is clean before starting generation
    reset_fire_recorder()
    
    prompts = [args.prompt]
    tokenizer = llm.get_tokenizer()
    prompt_token_ids = tokenizer.encode(prompts[0])
    prompt_len = len(prompt_token_ids)

    out = llm.generate(
        prompts,
        SamplingParams(max_tokens=16384, temperature=0.6, top_p=0.95)
    )

    print("--- Generation Output ---")
    print(out[0].outputs[0].text)
    print("-------------------------\n")

    fire_counts = FIRE_RECORDER.fire_counts
    fired_seq_lengths = FIRE_RECORDER.fired_seq_lengths

    if not fire_counts:
        print("No fire events were recorded during generation.")
    else:
        print("--- Fire Events Report ---")
        for sample_idx in sorted(fire_counts.keys()):
            total_fire_count = fire_counts[sample_idx]
            seq_lengths = fired_seq_lengths.get(sample_idx, [])
            output_info = out[sample_idx].outputs[0]

            generation_fired_tokens = []
            for seq_len in seq_lengths:
                # The token being processed is at index (seq_len - 1).
                # We only care about fires during the generation phase.
                token_idx = seq_len - 1
                if token_idx >= prompt_len:
                    # Calculate index into the list of *generated* tokens
                    gen_token_idx = token_idx - prompt_len
                    if gen_token_idx < len(output_info.token_ids):
                        gen_token_id = output_info.token_ids[gen_token_idx]
                        generation_fired_tokens.append(gen_token_id)

            decoded_tokens = tokenizer.convert_ids_to_tokens(generation_fired_tokens)
            generation_fire_count = len(decoded_tokens)
            
            print(f"\nSample {sample_idx} (Prompt: '{prompts[sample_idx][:50]}...'):")
            print(f"  Total fires (prefill + generation): {total_fire_count}")
            print(f"  Fires during generation: {generation_fire_count}")
            if decoded_tokens:
                # Using repr() to make whitespace characters like ' ' visible
                print(f"  Fired on tokens (in generation): {[repr(t) for t in decoded_tokens]}")
        print("\n------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path or HF repo name; must contain the custom code + weights")
    parser.add_argument("--prompt", default="Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations: \n\\[\\log_2\\left({x \\over yz}\\right) = {1 \\over 2}\\]\n\\[\\log_2\\left({y \\over xz}\\right) = {1 \\over 3}\\]\n\\[\\log_2\\left({z \\over xy}\\right) = {1 \\over 4}\\]\nThen the value of $\\left|\\log_2(x^4y^3z^2)\\right|$ is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.")
    args = parser.parse_args()
    register()  # register the custom model class
    
    main(args)


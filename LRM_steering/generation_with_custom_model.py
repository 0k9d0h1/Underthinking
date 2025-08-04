import argparse, os
import ctranslate2
import transformers

def main(args):
    # CTranslate2 can run on multiple GPUs. We list the device indexes.
    device_indices = [0, 1]
    
    # Load the CTranslate2 model and the tokenizer
    generator = ctranslate2.Generator(args.model, device="cuda", device_index=device_indices)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Prepare the prompt and generate text
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    results = generator.generate_batch(
        [tokens],
        max_length=32768,
        sampling_temperature=0.7,
        include_prompt_in_result=False,
    )

    # Decode and print the output
    output_text = tokenizer.decode(results[0].sequences_ids[0])
    print(output_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to the converted CTranslate2 model directory")
    parser.add_argument("--tokenizer_path", required=True,
                        help="Path to the original Hugging Face model to load the tokenizer")
    parser.add_argument("--prompt", default="What is 123 * 123?")
    args = parser.parse_args()
    main(args)

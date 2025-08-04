import json
import argparse

def fix_config(model_path: str):
    """
    Loads the config.json from the specified model path, adds the
    necessary auto_map configuration for the custom Qwen2 model,
    and saves the config back to the same location.
    """
    config_path = f"{model_path}/config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Successfully loaded config.json.")

        # Ensure the custom model is the first architecture
        architectures = config.get("architectures", [])
        if "Qwen2ForCausalLMCustom" not in architectures:
            architectures.insert(0, "Qwen2ForCausalLMCustom")
        config["architectures"] = architectures

        # Add the auto_map for vLLM to find the custom class
        config["auto_map"] = {
            "AutoModelForCausalLM": "modeling_qwen2_custom.Qwen2ForCausalLMCustom"
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Successfully updated and saved {config_path}")
        print("You can now retry running the generation script.")

    except FileNotFoundError:
        print(f"Error: Could not find config.json at {config_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix the config.json for use with custom models in vLLM."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the model directory containing the config.json file.",
    )
    args = parser.parse_args()
    fix_config(args.model)

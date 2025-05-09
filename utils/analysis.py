import json
from transformers import AutoProcessor, AutoTokenizer
from difflib import SequenceMatcher
import re
from fuzzywuzzy import fuzz

def split_model_output(model_output, reference_text, threshold=75, chunk_size=50):
    """
    Split model output based on the first fuzzy match with a reference text.
    Returns text before and after, with the split occurring AFTER the reference text.
    
    Args:
        model_output (str): The complete text output from the model
        reference_text (str): The reference text to find in the model output
        threshold (int): Minimum similarity score (0-100) to consider a match
        chunk_size (int): Size of the sliding window for fuzzy matching
        
    Returns:
        tuple: (text_including_reference, text_after_reference)
    """
    # Use the reference text for matching
    # Limit the reference text length to avoid excessive computation
    reference_to_match = reference_text[:min(200, len(reference_text))]
    reference_len = len(reference_to_match)
    
    # Find the best match using a sliding window approach
    best_score = 0
    best_position = -1
    
    # Slide through the model output with overlapping windows
    for i in range(0, len(model_output) - chunk_size + 1):
        # Extract a chunk of the model output
        chunk = model_output[i:i + chunk_size]
        
        # Compare with reference start
        similarity = fuzz.ratio(chunk, reference_to_match[:chunk_size])
        
        if similarity > best_score:
            best_score = similarity
            best_position = i
    
    if best_score < threshold or best_position == -1:
        print(f"No good match found. Best score: {best_score}")
        return model_output, ""
    
    # Fine-tune the match position using token ratio
    start_position = best_position
    window_size = min(reference_len * 2, 200)  
    
    # Try positions around the best match to find the optimal starting point
    search_start = max(0, best_position - 20)
    search_end = min(len(model_output) - window_size, best_position + 20)
    
    for pos in range(search_start, search_end):
        window = model_output[pos:pos + window_size]
        score = fuzz.token_sort_ratio(window, reference_to_match)
        
        if score > best_score:
            best_score = score
            start_position = pos
    
    # First, find a good chunk of text that contains the full reference
    search_window = model_output[start_position:start_position + min(300, len(reference_text) * 2)]
    
    # Now find where the reference text ends by checking for the presence of the FULL reference text
    # Gradually increase the window until we capture the entire reference text
    end_position = start_position
    best_full_match_score = 0
    
    for window_length in range(len(reference_text), min(300, len(reference_text) * 2)):
        if start_position + window_length >= len(model_output):
            break
            
        potential_match = model_output[start_position:start_position + window_length]
        score = fuzz.token_set_ratio(potential_match, reference_text)
        
        # If we got a good match and it's better than previous matches
        if score > best_full_match_score:
            best_full_match_score = score
            end_position = start_position + window_length
            
            # If we have a very good match, we can stop
            if score > 95:
                break
    
    # Extract the text before (including the reference) and the text after the reference
    text_including_reference = model_output[:end_position]
    text_after_reference = model_output[end_position:]
    
    # print(f"Match found at position {start_position}-{end_position} with score {best_score}")
    return text_including_reference, text_after_reference


def display_split_result(before, after, context=50):
    """
    Display the split result with context around the split point.
    
    Args:
        before (str): Text before and including the reference
        after (str): Text after the reference
        context (int): Number of characters to show around the split point
    """
    if not after and before:
        print("No match found or match is at the end. Full text:")
        print(before)
        return
        
    print("\n" + "="*80)
    print("SPLIT POINT VISUALIZATION")
    print("="*80)
    
    # Show end of before text (which includes the reference)
    if len(before) > context:
        print(f"...{before[-context:]} ")
    else:
        print(f"{before} ")
    
    print("| <- SPLIT POINT")
    
    # Show beginning of after text
    if len(after) > context:
        print(f" {after[:context]}...")
    else:
        print(f" {after}")
    
    print("="*80)

task_names = ["gpqa", "AIME_2024"]
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
output_file = f"./{model_name.split('/')[1].replace('-', '_')}_to_modify.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_name)

all_token_lens = []
removing_after_token_lens = []

changed_correctness = [0, 0, 0, 0]

with open(output_file, "w") as f:
    for task_name in task_names:
        base_name = f"{task_name}_DeepSeek_R1_Distill_Qwen_14B_results"
        with open(f"./{base_name}.jsonl", "r") as f1, open(f"./{base_name}_first_answer.jsonl", "r") as f2, open(f"./{base_name}_evaluation.jsonl", "r") as f3:
            for i, (row1, row2, row3) in enumerate(zip(f1, f2, f3)):
                row1 = json.loads(row1)
                row2 = json.loads(row2)
                row3 = json.loads(row3)
                if row2["gpt4o_answer"] == "Rate limit exceeded":
                    continue

                if task_name == "AIME_2024":
                    correctness = row3["evaluation"]["is_correct"]
                elif task_name == "gpqa":
                    correctness = row3["gpt4o_answer"].split("The correctness of the answer is: ")[-1].lower() == "correct"
                
                changed = "is changed" in row2["gpt4o_answer"].split("\n")[-1]
                if correctness and changed:
                    changed_correctness[0] += 1
                elif correctness and not changed:
                    changed_correctness[1] += 1
                elif not correctness and changed:
                    changed_correctness[2] += 1
                else:
                    changed_correctness[3] += 1

                if correctness and not changed:
                    model_output = row1["model_response"].split("</think>")[0]
                    if len(row2["gpt4o_answer"].split("\"")) > 1:
                        reference_text = row2["gpt4o_answer"].split("\"")[1].split("\"")[0]
                    else:
                        reference_text = row2["gpt4o_answer"].split("'")[1].split("'")[0]
                    # Split the text
                    before, after = split_model_output(model_output, reference_text, threshold=60)

                    all_tokens = len(tokenizer(model_output, return_tensors="pt")["input_ids"][0])
                    after_tokens = len(tokenizer(after, return_tensors="pt")["input_ids"][0])
            
                    print(before[-500:])
                    print("="*80)

                    output = {
                        "id": i,
                        "to_modify": before[-500:],
                        "before_to_modify": before[:-500],
                    }
                    f.write(json.dumps(output) + "\n")
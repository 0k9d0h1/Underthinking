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


special_token_ids = [92014, 80022, 14190]

def remove_leading_words(text):
    starters = ["Wait", "Hmm", "Alternatively"]
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

def split_text_by_phrases(text):
    """
    Split text into phrases that start with "Alternatively", "Wait", or "Hmm",
    and also return the first phrase before any of these keywords appear.
    
    Args:
        text (str): The input text to split
        
    Returns:
        list: A list of phrases including the first phrase and those starting with the specified words
    """
    # Define the keywords we're looking for
    keywords = ["Alternatively", "Wait", "Hmm"]
    
    # Create a list to store all phrases
    all_phrases = []
    
    # First, check if there's text before the first keyword
    first_keyword_index = -1
    first_keyword = None
    
    # Find the first occurrence of any keyword
    for keyword in keywords:
        index = text.find(keyword)
        if index != -1 and (first_keyword_index == -1 or index < first_keyword_index):
            first_keyword_index = index
            first_keyword = keyword
    
    # If there's text before the first keyword, add it as the first phrase
    if first_keyword_index > 0:
        first_phrase = text[:first_keyword_index].strip()
        if first_phrase:
            all_phrases.append(first_phrase)
    
    # Pattern to match phrases starting with our target words
    pattern = r'(Alternatively|Wait|Hmm)(?:[^A-Z]|$).*?(?=(Alternatively|Wait|Hmm)(?:[^A-Z]|$)|$)'
    
    # Find all matches
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Extract the complete phrases from the matches
    for match in matches:
        # match[0] contains the starting word (Alternatively, Wait, or Hmm)
        # We need to reconstruct the full phrase
        starting_index = text.find(match[0])
        if starting_index != -1:
            # Find where the next phrase would start or the end of text
            next_phrase_index = -1
            for next_word in keywords:
                next_index = text.find(next_word, starting_index + len(match[0]))
                if next_index != -1 and (next_phrase_index == -1 or next_index < next_phrase_index):
                    next_phrase_index = next_index
            
            # Extract the phrase
            if next_phrase_index == -1:
                phrase = text[starting_index:]
            else:
                phrase = text[starting_index:next_phrase_index]
            
            all_phrases.append(phrase.strip())
            
            # Update the text to avoid re-finding the same phrase
            text = text[:starting_index] + text[starting_index + len(phrase):]
    
    return all_phrases

def split_by_special_tokens(token_ids, special_token_ids):
    result = []
    current = []
    for tid in token_ids:
        if tid in special_token_ids:
            # If current segment is not empty, save it before starting a new one
            if current:
                result.append(current)
            current = [tid]  # Start new segment with the special token
        else:
            current.append(tid)
    if current:
        result.append(current)
    return result

def perplexity_reward(model_output: str, output_ids, actor_module, tokenizer):
    splited_phrases = split_text_by_phrases(model_output)
    splited_ids = split_by_special_tokens(output_ids, special_token_ids)

    ppl_rewards_for_phrases = [0.0]
    phrases_token_lengths = [len(ids) for ids in splited_ids]

    for i in range(1, len(splited_phrases)):
        ppls = []
        for j in range(i):
            text1 = splited_phrases[j] + "\n\nThat is, "
            text2 = remove_leading_words(splited_phrases[i])
            model_inputs = tokenizer([text1 + text2], return_tensors="pt").to(actor_module.device).input_ids
            text1_ids = tokenizer(text1, return_tensors="pt").input_ids
            labels_for_model = model_inputs.clone()
            labels_for_model[:, :len(text1_ids[0])] = -100

            with torch.no_grad():
                loss = actor_module(model_inputs, labels=labels_for_model).loss
                ppl = math.exp(loss.item())
            ppls.append(ppl)
            del model_inputs, labels_for_model, loss
            torch.cuda.empty_cache()
        ppl_mean = sum(ppls) / len(ppls) if ppls else 0.0
        ppl_rewards_for_phrases.append(ppl_mean)
    ppl_rewards_for_phrases = np.array(ppl_rewards_for_phrases)
    ppl_rewards_for_phrases = ppl_rewards_for_phrases - ppl_rewards_for_phrases[1:].mean()
    ppl_rewards_for_phrases[0] = 0.0
    abs_max = np.abs(ppl_rewards_for_phrases).max()
    ppl_rewards_for_phrases /= abs_max

    return ppl_rewards_for_phrases, phrases_token_lengths

def compute_score(data_source, solution_str, solution_ids, ground_truth, extra_info=None):
    actor_module = extra_info["actor_module"]
    tokenizer = extra_info["tokenizer"]

    ppl_rewards_for_phrases, phrase_token_lengths = perplexity_reward(solution_str, solution_ids, actor_module, tokenizer)
    return acc_reward(solution_str, ground_truth) + ppl_rewards_for_phrases, phrase_token_lengths
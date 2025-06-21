import json, os, textwrap, itertools, collections, math, re

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
    keywords = ["Alternatively", "Wait", "Hmm", "But wait"]
    
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
    pattern = r'(Alternatively|But wait|Wait|Hmm)(?:[^A-Z]|$).*?(?=(Alternatively|But wait|Wait|Hmm)(?:[^A-Z]|$)|$)'
    
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

file20 = '/home/kdh0901/Desktop/cache_dir/kdh0901/val_data/rm_ppl_reward/20.jsonl'
file120 = '/home/kdh0901/Desktop/cache_dir/kdh0901/val_data/rm_ppl_reward/120.jsonl'
data20=[]
data120=[]
for line in open(file20,'r'):
    data20.append(json.loads(line))
for line in open(file120,'r'):
    data120.append(json.loads(line))
len20=len(data20); len120=len(data120)

map120 = {d['input']: d for d in data120}
diff_good=[]
for d in data20:
    if d['score']==1.0 and d['input'] in map120 and map120[d['input']]['score']!=1.0:
        diff_good.append((d, map120[d['input']]))

len_good = []
len_bad = []
thoughts_good = []
thoughts_bad = []
for diff in diff_good:
    # print(diff[0]['output'])
    for phrase in split_text_by_phrases(diff[0]['output']):
        len_good.append(len(phrase))
    thoughts_good.append(len(split_text_by_phrases(diff[0]['output'])) / len(diff[0]['output']))
    for phrase in split_text_by_phrases(diff[1]['output']):
        len_bad.append(len(phrase))
    thoughts_bad.append(len(split_text_by_phrases(diff[1]['output'])) / len(diff[1]['output']))
print(sum(len_good) / len(len_good))
print(sum(len_bad) / len(len_bad))
print(sum(thoughts_good) / len(thoughts_good))
print(sum(thoughts_bad) / len(thoughts_bad))
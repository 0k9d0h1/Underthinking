from glob import glob
import json
import re
import openai
import os
import time
from dotenv import load_dotenv
import argparse
import glob


load_dotenv()  # loads variables from .env into os.environ

api_key = os.getenv("OPENAI_API_KEY")
# Set your OpenAI API key
client = openai.OpenAI(api_key=api_key)


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
    pattern = r"(Alternatively|But wait|Wait|Hmm)(?:[^A-Z]|$).*?(?=(Alternatively|But wait|Wait|Hmm)(?:[^A-Z]|$)|$)"

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
                if next_index != -1 and (
                    next_phrase_index == -1 or next_index < next_phrase_index
                ):
                    next_phrase_index = next_index

            # Extract the phrase
            if next_phrase_index == -1:
                phrase = text[starting_index:]
            else:
                phrase = text[starting_index:next_phrase_index]

            all_phrases.append(phrase.strip())

            # Update the text to avoid re-finding the same phrase
            text = text[:starting_index] + text[starting_index + len(phrase) :]

    return all_phrases


def get_gpt4o_answer(question):
    """Function to get GPT-4o's answer with automatic retry on rate limit errors."""
    for i in range(10):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI answering visual questions.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                        ],
                    },
                ],
                max_completion_tokens=32768,
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError as e:
            retry_time = min(60, float(e.response.headers.get("Retry-After", 1)))
            print(f"Rate limit reached. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            if i == 9:
                print("Max retries reached. Exiting.")
                return "Rate limit exceeded"
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None


def main(args):
    model_name = args.model_name
    input_files = glob.glob(
        f"./{model_name.split('/')[-1].replace('-', '_')}/*_generation.jsonl"
    )
    evaluation_files = glob.glob(
        f"./{model_name.split('/')[-1].replace('-', '_')}/*_evaluation.jsonl"
    )
    output_file = (
        f"./{model_name.split('/')[-1].replace('-', '_')}/thought_clustering.jsonl"
    )
    split_output_file = (
        f"./{model_name.split('/')[-1].replace('-', '_')}/thought_split.jsonl"
    )

    for i, (input_file, evaluation_file) in enumerate(
        zip(input_files, evaluation_files)
    ):
        with open(input_file, "r") as f, open(evaluation_file, "r") as ef, open(
            output_file, "a"
        ) as of, open(split_output_file, "a") as sf:
            for j, (line, eval_line) in enumerate(zip(f, ef)):
                data = json.loads(line)
                eval_data = json.loads(eval_line)
                response = data["model_response"].split("</think>")[0]
                correctness = eval_data.get("is_correct", False)

                prompt = """You will be given a list of "thoughts," where each thought is a reasoning step generated by an LLM solving a problem.

Your task is to:
- Group the thoughts into clusters based on the similarity of the problem-solving approach or method used.
- Perform **fine-grained clustering**: distinguish even subtle differences in reasoning strategies, assumptions, or step structure.
- Even if two thoughts use the same general technique, separate them if their implementation or reasoning flow differs.
- Only group thoughts together if their underlying logic is very closely aligned.

---

**Rules**:
- Be conservative when clustering — prefer making more clusters over fewer.
- Do not cluster based on surface wording or phrasing similarity. Focus on method and logic.
- Do not include any explanation, commentary, or justification outside the required format.

---

**Output Format**:  
- Use the exact structure below for each cluster:
Cluster [cluster_number]: [short label for reasoning method]
Thoughts: [comma-separated list of thought numbers]

- Example:
Cluster 1: Apply Pythagorean theorem directly
Thoughts: 2, 5, 9

Cluster 2: Use triangle similarity with proportion setup
Thoughts: 1, 3, 7

---

Now classify the following thoughts using this format."""
                phrases = split_text_by_phrases(response)
                if len(phrases) > 1000 or len(phrases) == 0:
                    continue

                sf.write(json.dumps({"phrases": phrases}) + "\n")

                for i, phrase in enumerate(phrases):
                    prompt += f"\n\nThought {i}: [{phrase}]"

                response = get_gpt4o_answer(prompt)

                of.write(
                    json.dumps({"gpt4o_answer": response, "correctness": correctness})
                    + "\n"
                )
                of.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thought Clustering Script")
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Name of the model to use",
    )
    args = parser.parse_args()
    main(args)

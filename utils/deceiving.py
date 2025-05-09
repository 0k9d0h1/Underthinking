import openai
import os
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64
from autogen.agentchat.contrib.img_utils import pil_to_data_uri
import json
import time
import glob


# Set your OpenAI API key
client = openai.OpenAI(api_key="REMOVED")

# File to save results
input_file = "./DeepSeek_R1_Distill_Qwen_14B_to_modify.jsonl"
output_file = f"./DeepSeek_R1_Distill_Qwen_14B_deceiving.jsonl"

aime_2024 = load_dataset("Maxwell-Jia/AIME_2024", split="train")
gpqa = load_dataset("Idavidrein/gpqa", name="gpqa_diamond", split="train")

def get_gpt4o_answer(question):
    """Function to get GPT-4o's answer with automatic retry on rate limit errors."""
    for i in range(10):
        try:
            response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are helpful, honest, and harmless AI assistant."},
                {"role": "user", "content": [{"type": "text", "text": question},]},
                ],
                max_tokens=5000
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

with open(output_file, "a", encoding="utf-8") as f, open(input_file, "r", encoding="utf-8") as infile:
    for i, line in enumerate(tqdm(infile)):
        data = json.loads(line)
        id = data["id"]
        if i < 54:
            dataset = gpqa
            problem = dataset[id]["Question"]
            answer = dataset[id]["Correct Answer"]
        else:
            dataset = aime_2024
            problem = dataset[id]["Problem"]
            answer = dataset[id]["Answer"]

        to_modify = data["to_modify"]
        before_to_modify = data["before_to_modify"]

        prompt = f"""You will be given:
1. The original question (math or science related)
2. A correct, well-structured response written by an LLM that contains LLM's final answer.

Your task is to subtly modify the response to introduce a *plausible and subtle, but incorrect* error in the reasoning and **final answer**.

The modification must:
- Preserve the original response’s structure, tone, and flow
- Make the incorrect answer appear reasonable or convincing
- Not include any signal that the answer is incorrect or changed
- Contain change from the original correct answer to an incorrect one
- Avoid simple errors like sign changes or arithmetic mistakes
- Contain a change in the overall reasoning process and the **final answer**

The goal is to test whether another LLM can detect or correct the subtle error.

Here is the input:

---

**Question**:  
{problem}

**Original Response**:  
{to_modify}

---

Now return the modified response with exactly one subtle error in reasoning or calculation.
"""
        # Get GPT-4o's answer
        gpt4o_answer = get_gpt4o_answer(prompt)
        modified_answer = before_to_modify + gpt4o_answer + "\nWait, let me double-check my answer to be sure."

        # # Save the result
        result = {
            "id": id,
            "problem": problem,
            "correct_answer": answer,
            "original_response": to_modify,
            "modified_response": modified_answer
        }
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

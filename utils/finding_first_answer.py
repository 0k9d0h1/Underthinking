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
input_file = "./gpqa_DeepSeek_R1_Distill_Qwen_14B_results.jsonl"
output_file = f"./{input_file.split('/')[1].split('.')[0]}_first_answer.jsonl"

# Function to get image URL or create a data URI
def get_image_url(image):
    if isinstance(image, str) and image.startswith('http'):
        return image  # It's already a URL
    else:
        # Convert to data URI
        return pil_to_data_uri(image)

def get_gpt4o_answer(question):
    """Function to get GPT-4o's answer with automatic retry on rate limit errors."""
    for i in range(10):
        try:
            response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are an AI answering visual questions."},
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
        if i < 29:
            continue
        data = json.loads(line)
        problem = data["problem"]
        answer = data["model_response"].split("<think>")[-1].split("</think>")[0].strip()

        prompt = f"""I will give you a question and the response from a language model that attempts to answer it. Your job is to:
1. Identify the first point in the response where the model computes the actual final answer to the question (not intermediate steps, first computed {problem}). Quote that sentence or phrase.
2. Find the last final answer stated in the response (usually after a phrase like "Final Answer").
3. Determine whether the model changed its answer between the first computation and the final stated answer.

Please respond in the following format:
'The exact point where the model first computes the final answer is: <quote>
The model's last final answer is <changed/unchanged> from the model's first final answer.'


Question:
{problem}

Model Response:
{answer}"""
        # Get GPT-4o's answer
        gpt4o_answer = get_gpt4o_answer(prompt)

        # # Save the result
        result = {
            "gpt4o_answer": gpt4o_answer
        }
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

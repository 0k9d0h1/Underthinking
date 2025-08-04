import json
import re
import argparse
from pathlib import Path
import logging
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"


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


def evaluate_response(correct_answer, model_response):
    """
    Evaluates if the model response contains the correct answer.

    Args:
        correct_answer: The known correct answer (can be int, float, str, etc.)
        model_response: The text response from the model

    Returns:
        dict: Evaluation result with is_correct flag and details
    """
    # Convert correct_answer to string for text matching
    correct_answer_str = str(correct_answer)

    # Check if the exact answer appears in the response
    if correct_answer_str in model_response:
        return {
            "is_correct": True,
            "method": "exact_match",
            "details": f"Found exact answer '{correct_answer_str}' in response",
        }

    # Look for answer patterns like "answer is 33" or "m+n = 33"
    answer_patterns = [
        rf"answer\s+is\s+{re.escape(correct_answer_str)}",
        rf"=\s*{re.escape(correct_answer_str)}",
        rf"equals\s+{re.escape(correct_answer_str)}",
        rf"m\s*\+\s*n\s*=\s*{re.escape(correct_answer_str)}",
        rf"value\s+of\s+m\s*\+\s*n\s+is\s+{re.escape(correct_answer_str)}",
    ]

    for pattern in answer_patterns:
        if re.search(pattern, model_response, re.IGNORECASE):
            return {
                "is_correct": True,
                "method": "pattern_match",
                "pattern": pattern,
                "details": f"Found answer pattern with '{correct_answer_str}'",
            }

    # If we get here, no match was found
    return {
        "is_correct": False,
        "method": "no_match",
        "details": f"Could not find '{correct_answer_str}' in response",
    }


def process_jsonl_file(args):
    """
    Process each problem in the input JSONL file and write evaluation results to output JSONL file.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
    """
    results = []
    correct_count = 0
    total_count = 0
    model_name = args.model_name
    dataset_name = args.dataset_name
    input_file = f"./model_outputs/{model_name.split('/')[-1].replace('-', '_')}/{dataset_name.split('/')[-1].replace('-', '_')}_generation.jsonl"
    output_file = f"./model_outputs/{model_name.split('/')[-1].replace('-', '_')}/{dataset_name.split('/')[-1].replace('-', '_')}_evaluation.jsonl"

    logger.info(f"Processing input file: {input_file}")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse the JSON line
                    data = json.loads(line.strip())
                    total_count += 1

                    # Extract fields
                    problem = data.get("problem", "")
                    correct_answer = data.get("correct_answer")
                    model_response = data.get("model_response", "")

                    # Skip if missing required fields
                    if not all([problem, correct_answer is not None, model_response]):
                        logger.warning(
                            f"Line {line_num}: Missing required fields, skipping"
                        )
                        continue

                    # Evaluate the response
                    if "aime" in dataset_name.lower():
                        score = acc_reward(model_response, str(correct_answer))
                    elif "gpqa" in dataset_name.lower():
                        match = re.search(ANSWER_PATTERN_MULTICHOICE, model_response)
                        extracted_answer = match.group(1) if match else None
                        score = 1.0 if extracted_answer == correct_answer else 0.0

                    correct_count += score

                    # Prepare result object
                    result = {
                        "is_correct": score == 1.0,
                        "problem": problem,
                        "correct_answer": correct_answer,
                        "model_response": model_response,
                    }
                    results.append(result)

                except json.JSONDecodeError:
                    logger.error(f"Line {line_num}: Invalid JSON format, skipping")
                except Exception as e:
                    logger.error(f"Line {line_num}: Error processing - {str(e)}")

    except Exception as e:
        logger.error(f"Error reading input file: {str(e)}")
        return

    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    logger.info(
        f"Evaluation complete. Accuracy: {correct_count}/{total_count} ({accuracy:.2%})"
    )

    # Write results to output file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error writing output file: {str(e)}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate model responses in JSONL files"
    )
    parser.add_argument(
        "--model_name",
        help="Name of the model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument(
        "--dataset_name", help="Name of the dataset", default="AIME_2024"
    )
    args = parser.parse_args()

    # Process the file
    process_jsonl_file(args)


if __name__ == "__main__":
    main()

import json
import gradio as gr
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from transformers import AutoTokenizer

# Load the DeepSeek tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    tokenizer_loaded = True
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer_loaded = False

# Function to load JSONL data
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Function to properly format LaTeX for MathJax rendering in Gradio
def format_latex(text):
    if not isinstance(text, str):
        return str(text)  # Convert non-string values to string
    
    # First preserve any already properly formatted LaTeX
    # For inline math that's already properly formatted
    text = text.replace('\\(', 'PRESERVEINLINE')
    text = text.replace('\\)', 'PRESERVEINLINEEND')
    
    # For display math that's already properly formatted
    text = text.replace('\\[', 'PRESERVEDISPLAY')
    text = text.replace('\\]', 'PRESERVEDISPLAYEND')
    
    # Handle standard LaTeX delimiters
    # Convert single $ to inline math (but not $$)
    text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', r'\\(\1\\)', text)
    
    # Convert $$ to display math
    text = re.sub(r'\$\$(.*?)\$\$', r'\\[\1\\]', text)
    
    # Restore preserved content
    text = text.replace('PRESERVEINLINE', '\\(')
    text = text.replace('PRESERVEINLINEEND', '\\)')
    text = text.replace('PRESERVEDISPLAY', '\\[')
    text = text.replace('PRESERVEDISPLAYEND', '\\]')
    
    return text

# Function to calculate token lengths using DeepSeek tokenizer
def analyze_tokens(text, split_point=None):
    if not tokenizer_loaded:
        return {
            "total_tokens": "Tokenizer not loaded",
            "after_split_tokens": "Tokenizer not loaded",
            "percentage": "Tokenizer not loaded"
        }
    
    if not isinstance(text, str):
        text = str(text)
    
    # Tokenize the entire text
    total_tokens = len(tokenizer.encode(text))
    
    # If split point is provided, calculate tokens after that point
    after_split_tokens = 0
    percentage = 0
    
    if split_point and split_point in text:
        split_index = text.find(split_point) + len(split_point)
        after_text = text[split_index:]
        after_split_tokens = len(tokenizer.encode(after_text))
        percentage = (after_split_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    
    return {
        "total_tokens": total_tokens,
        "after_split_tokens": after_split_tokens,
        "percentage": percentage
    }

# Gradio interface
def create_interface():
    # State variables to persist across interactions
    state = {
        "data": [],
        "current_index": 0,
        "correct_count": 0,
        "total_evaluated": 0,
        "evaluations": {},  # Store evaluations by problem index
        "split_point": "Therefore, the answer is"  # Default split point
    }
    
    def upload_file(file):
        try:
            # Load JSONL data
            state["data"] = load_jsonl(file.name)
            state["current_index"] = 0
            state["correct_count"] = 0
            state["total_evaluated"] = 0
            state["evaluations"] = {}
            
            # Reset evaluation stats
            stats_html = "Upload successful! Total problems: " + str(len(state["data"]))
            return display_problem(0), stats_html, create_accuracy_chart(), create_token_analysis_chart(), state["split_point"]
        except Exception as e:
            return f"Error loading file: {str(e)}", "Error", None, None, state["split_point"]

    def display_problem(index):
        if not state["data"]:
            return "Please upload a JSONL file first."
        
        if index < 0 or index >= len(state["data"]):
            return f"Index out of range. Valid range: 0-{len(state['data'])-1}"
        
        # Update current index in state
        state["current_index"] = index
        
        problem_data = state["data"][index]
        
        # Format the problem and answers with proper LaTeX, handling different data types
        problem_text = str(problem_data.get("problem", "No problem text available"))
        correct_answer = problem_data.get("correct_answer", "No correct answer available")
        model_response = str(problem_data.get("model_response", "No model response available"))
        
        # Format LaTeX for MathJax rendering
        problem_text = format_latex(problem_text)
        correct_answer = format_latex(str(correct_answer))
        model_response = format_latex(model_response)
        
        # Format display
        display_text = f"""
## Problem {index + 1}/{len(state["data"])}

### Problem Statement
{problem_text}

### Correct Answer
{correct_answer}

### Model Response
{model_response}

"""
        
        # Add evaluation status if available
        if index in state["evaluations"]:
            status = "Correct ✅" if state["evaluations"][index] else "Incorrect ❌"
            display_text += f"\n### Evaluation: {status}"
            
        # Add token analysis
        token_analysis = analyze_tokens(model_response, state["split_point"])
        if tokenizer_loaded:
            display_text += f"""
### Token Analysis
- Total tokens: {token_analysis["total_tokens"]}
- Tokens after "{state["split_point"]}": {token_analysis["after_split_tokens"]}
- Percentage of tokens after split point: {token_analysis["percentage"]:.2f}%
"""
        
        return display_text

    def navigate(direction):
        new_index = state["current_index"] + direction
        if 0 <= new_index < len(state["data"]):
            return display_problem(new_index), get_stats_html(), create_accuracy_chart(), create_token_analysis_chart()
        return display_problem(state["current_index"]), get_stats_html(), create_accuracy_chart(), create_token_analysis_chart()

    def mark_correct():
        if not state["data"]:
            return "Please upload a JSONL file first.", get_stats_html(), create_accuracy_chart(), create_token_analysis_chart()
        
        index = state["current_index"]
        # Update if not already evaluated
        if index not in state["evaluations"]:
            state["evaluations"][index] = True
            state["correct_count"] += 1
            state["total_evaluated"] += 1
        # Change evaluation from incorrect to correct
        elif not state["evaluations"][index]:
            state["evaluations"][index] = True
            state["correct_count"] += 1
            
        return display_problem(index), get_stats_html(), create_accuracy_chart(), create_token_analysis_chart()

    def mark_incorrect():
        if not state["data"]:
            return "Please upload a JSONL file first.", get_stats_html(), create_accuracy_chart(), create_token_analysis_chart()
        
        index = state["current_index"]
        # Update if not already evaluated
        if index not in state["evaluations"]:
            state["evaluations"][index] = False
            state["total_evaluated"] += 1
        # Change evaluation from correct to incorrect
        elif state["evaluations"][index]:
            state["evaluations"][index] = False
            state["correct_count"] -= 1
            
        return display_problem(index), get_stats_html(), create_accuracy_chart(), create_token_analysis_chart()

    def get_stats_html():
        if state["total_evaluated"] == 0:
            return "No problems evaluated yet"
        
        accuracy = (state["correct_count"] / state["total_evaluated"]) * 100
        return f"""
### Evaluation Statistics
- Problems evaluated: {state["total_evaluated"]}/{len(state["data"])}
- Correct answers: {state["correct_count"]}
- Current accuracy: {accuracy:.2f}%
"""

    def create_accuracy_chart():
        if state["total_evaluated"] == 0:
            return None
            
        fig = Figure(figsize=(6, 4))
        ax = fig.subplots()
        
        # Data for pie chart
        labels = ['Correct', 'Incorrect']
        sizes = [state["correct_count"], state["total_evaluated"] - state["correct_count"]]
        colors = ['#4CAF50', '#F44336']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_title('Model Accuracy')
        
        return fig

    def create_token_analysis_chart():
        if not state["data"] or not tokenizer_loaded:
            return None
        
        # Analyze tokens for all evaluated problems
        token_data = {
            "total": [],
            "after_split": [],
            "percentages": []
        }
        
        problem_indices = []
        
        for i, item in enumerate(state["data"]):
            if i in state["evaluations"]:
                model_response = str(item.get("model_response", ""))
                analysis = analyze_tokens(model_response, state["split_point"])
                
                token_data["total"].append(analysis["total_tokens"])
                token_data["after_split"].append(analysis["after_split_tokens"])
                token_data["percentages"].append(analysis["percentage"])
                problem_indices.append(i + 1)  # 1-indexed for display
        
        if not problem_indices:
            return None
            
        fig = Figure(figsize=(10, 6))
        ax = fig.subplots()
        
        # Bar chart for tokens
        x = np.arange(len(problem_indices))
        width = 0.35
        
        ax.bar(x - width/2, token_data["total"], width, label='Total Tokens')
        ax.bar(x + width/2, token_data["after_split"], width, label=f'Tokens After Split')
        
        ax.set_xlabel('Problem Number')
        ax.set_ylabel('Token Count')
        ax.set_title('Token Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(problem_indices)
        ax.legend()
        
        # Add percentage as text above bars
        for i, (total, after, percent) in enumerate(zip(token_data["total"], token_data["after_split"], token_data["percentages"])):
            ax.text(i + width/2, after + 5, f"{percent:.1f}%", ha='center')
        
        fig.tight_layout()
        return fig

    def go_to_index(index_str):
        try:
            index = int(index_str) - 1  # Convert to 0-indexed
            if 0 <= index < len(state["data"]):
                return display_problem(index), get_stats_html(), create_accuracy_chart(), create_token_analysis_chart()
            else:
                return f"Invalid index. Please enter a number between 1 and {len(state['data'])}", get_stats_html(), create_accuracy_chart(), create_token_analysis_chart()
        except ValueError:
            return "Please enter a valid number", get_stats_html(), create_accuracy_chart(), create_token_analysis_chart()

    # Function to automatically check answers
    def auto_evaluate():
        if not state["data"]:
            return "Please upload a JSONL file first.", get_stats_html(), create_accuracy_chart(), create_token_analysis_chart()
        
        # Reset evaluation
        state["correct_count"] = 0
        state["total_evaluated"] = 0
        state["evaluations"] = {}
        
        for idx, item in enumerate(state["data"]):
            correct_answer = str(item.get("correct_answer", "")).strip()
            model_response = str(item.get("model_response", "")).strip()
            
            # Extract the final number from the model response
            model_numbers = re.findall(r'\b\d+\b', model_response)
            
            # Check if the correct answer appears in the model response
            if model_numbers and correct_answer in model_numbers:
                state["evaluations"][idx] = True
                state["correct_count"] += 1
            else:
                state["evaluations"][idx] = False
            
            state["total_evaluated"] += 1
        
        return display_problem(state["current_index"]), get_stats_html(), create_accuracy_chart(), create_token_analysis_chart()

    # Function to export evaluations with token analysis
    def export_evaluations():
        if not state["data"] or not state["evaluations"]:
            return "No evaluations to export."
        
        try:
            # Create export data
            export_data = []
            for idx, item in enumerate(state["data"]):
                if idx in state["evaluations"]:
                    model_response = str(item.get("model_response", ""))
                    token_analysis = analyze_tokens(model_response, state["split_point"])
                    
                    export_item = {
                        "problem_index": idx,
                        "problem": item.get("problem", ""),
                        "correct_answer": item.get("correct_answer", ""),
                        "model_response": model_response,
                        "evaluation": "correct" if state["evaluations"][idx] else "incorrect",
                        "token_analysis": {
                            "total_tokens": token_analysis["total_tokens"],
                            "after_split_tokens": token_analysis["after_split_tokens"],
                            "percentage": token_analysis["percentage"]
                        }
                    }
                    export_data.append(export_item)
            
            # Save to file
            filename = "model_evaluations_with_tokens.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            return f"Successfully exported {len(export_data)} evaluations to {filename}"
        except Exception as e:
            return f"Error exporting evaluations: {str(e)}"

    # Function to update split point
    def update_split_point(new_split_point):
        state["split_point"] = new_split_point
        return display_problem(state["current_index"]), create_token_analysis_chart()

    # Create the interface
    with gr.Blocks(css="""
        .math.inline {
            font-size: 16px !important;
        }
        .math.display {
            font-size: 18px !important;
            margin: 15px 0 !important;
        }
    """) as interface:
        gr.Markdown("# AIME Model Response Viewer and Evaluator")
        gr.Markdown("""
        This interface supports LaTeX rendering and token analysis. 
        
        **Tip for better LaTeX rendering**: After loading data, give the app a moment to properly render the mathematical expressions.
        """)
        
        with gr.Row():
            file_input = gr.File(label="Upload JSONL File")
            upload_button = gr.Button("Load Data")
        
        with gr.Row():
            with gr.Column(scale=3):
                problem_display = gr.Markdown("Please upload a JSONL file to begin.")
            
            with gr.Column(scale=1):
                stats_display = gr.Markdown("No data loaded")
                accuracy_chart = gr.Plot(label="Accuracy Chart")
        
        with gr.Row():
            prev_button = gr.Button("Previous Problem")
            index_input = gr.Textbox(label="Go to problem #", placeholder="Enter problem number")
            go_button = gr.Button("Go")
            next_button = gr.Button("Next Problem")
        
        with gr.Row():
            correct_button = gr.Button("Mark Correct ✅", variant="primary")
            incorrect_button = gr.Button("Mark Incorrect ❌", variant="secondary")
            auto_evaluate_button = gr.Button("Auto-Evaluate All", variant="primary")
        
        with gr.Row():
            export_button = gr.Button("Export Evaluations")
            export_result = gr.Markdown("")
        
        gr.Markdown("## Token Analysis Settings")
        with gr.Row():
            split_point_input = gr.Textbox(
                label="Split Point for Token Analysis", 
                placeholder="Enter text to analyze tokens after",
                value="Therefore, the answer is"
            )
            update_split_button = gr.Button("Update Split Point")
        
        token_analysis_chart = gr.Plot(label="Token Analysis Chart")
        
        # Wire up the events
        upload_button.click(
            upload_file, 
            inputs=[file_input], 
            outputs=[problem_display, stats_display, accuracy_chart, token_analysis_chart, split_point_input]
        )
        
        prev_button.click(
            lambda: navigate(-1), 
            outputs=[problem_display, stats_display, accuracy_chart, token_analysis_chart]
        )
        
        next_button.click(
            lambda: navigate(1), 
            outputs=[problem_display, stats_display, accuracy_chart, token_analysis_chart]
        )
        
        go_button.click(
            go_to_index,
            inputs=[index_input],
            outputs=[problem_display, stats_display, accuracy_chart, token_analysis_chart]
        )
        
        correct_button.click(
            mark_correct,
            outputs=[problem_display, stats_display, accuracy_chart, token_analysis_chart]
        )
        
        incorrect_button.click(
            mark_incorrect,
            outputs=[problem_display, stats_display, accuracy_chart, token_analysis_chart]
        )
        
        auto_evaluate_button.click(
            auto_evaluate,
            outputs=[problem_display, stats_display, accuracy_chart, token_analysis_chart]
        )
        
        export_button.click(
            export_evaluations,
            outputs=[export_result]
        )
        
        update_split_button.click(
            update_split_point,
            inputs=[split_point_input],
            outputs=[problem_display, token_analysis_chart]
        )
        
    return interface

# Launch the interface
if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)
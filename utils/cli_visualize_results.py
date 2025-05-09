import json
import re
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich import box
from rich.table import Table
from transformers import AutoTokenizer

console = Console(highlight=False)  # Disable syntax highlighting to prevent conflicts

# Load the DeepSeek tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    tokenizer_loaded = True
except Exception as e:
    console.print(f"Error loading tokenizer: {e}", style="bold red")
    tokenizer_loaded = False

class MathEvaluator:
    def __init__(self):
        self.data = []
        self.current_index = 0
        self.correct_count = 0
        self.total_evaluated = 0
        self.evaluations = {}  # Store evaluations by problem index
        self.split_point = "Therefore, the answer is"  # Default split point
    
    def load_jsonl(self, file_path):
        """Load JSONL data from a file."""
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            self.data = data
            self.current_index = 0
            self.correct_count = 0
            self.total_evaluated = 0
            self.evaluations = {}
            console.print(f"Successfully loaded {len(data)} problems from {file_path}", style="green")
            return True
        except Exception as e:
            console.print(f"Error loading file: {str(e)}", style="bold red")
            return False
    
    def format_latex(self, text):
        """Format LaTeX for terminal display without using markup tags."""
        if not isinstance(text, str):
            return str(text)  # Convert non-string values to string
        
        # For CLI, we'll keep the LaTeX notation as is, but clean it up a bit
        # Convert $$ to display math markers for clarity (avoiding Rich markup conflicts)
        text = re.sub(r'\$\$(.*?)\$\$', r'DISPLAY_MATH{ \1 }', text)
        
        # Convert single $ to inline math for clarity (avoiding Rich markup conflicts)
        text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', r'MATH{ \1 }', text)
        
        return text
    
    def analyze_tokens(self, text):
        """Calculate token lengths using DeepSeek tokenizer."""
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
        
        if self.split_point and self.split_point in text:
            split_index = text.find(self.split_point) + len(self.split_point)
            after_text = text[split_index:]
            after_split_tokens = len(tokenizer.encode(after_text))
            percentage = (after_split_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        
        return {
            "total_tokens": total_tokens,
            "after_split_tokens": after_split_tokens,
            "percentage": percentage
        }
    
    def display_problem(self):
        """Display the current problem in the console."""
        if not self.data:
            console.print("Please load a JSONL file first using the 'load' command.", style="yellow")
            return False
        
        if self.current_index < 0 or self.current_index >= len(self.data):
            console.print(f"Index out of range. Valid range: 0-{len(self.data)-1}", style="yellow")
            return False
        
        problem_data = self.data[self.current_index]
        
        # Format the problem and answers, handling different data types
        problem_text = str(problem_data.get("problem", "No problem text available"))
        correct_answer = problem_data.get("correct_answer", "No correct answer available")
        model_response = str(problem_data.get("model_response", "No model response available"))
        
        # Format for console display
        problem_text = self.format_latex(problem_text)
        correct_answer = self.format_latex(str(correct_answer))
        model_response = self.format_latex(model_response)
        
        # Create display panels for each section
        problem_panel = Panel(
            problem_text,
            title="Problem Statement",
            border_style="blue"
        )
        
        answer_panel = Panel(
            str(correct_answer),
            title="Correct Answer",
            border_style="green"
        )
        
        response_panel = Panel(
            model_response,
            title="Model Response",
            border_style="cyan"
        )
        
        # Main panel title
        console.print(f"\nProblem {self.current_index + 1}/{len(self.data)}", style="bold cyan")
        
        # Display panels
        console.print(problem_panel)
        console.print(answer_panel)
        console.print(response_panel)
        
        # Add evaluation status if available
        if self.current_index in self.evaluations:
            status = "Correct ✓" if self.evaluations[self.current_index] else "Incorrect ✗"
            status_style = "green" if self.evaluations[self.current_index] else "red"
            console.print(f"Evaluation: {status}", style=f"bold {status_style}")
        
        # Add token analysis
        if tokenizer_loaded:
            token_analysis = self.analyze_tokens(model_response)
            token_panel = Panel(
                f"Total tokens: {token_analysis['total_tokens']}\n"
                f"Tokens after \"{self.split_point}\": {token_analysis['after_split_tokens']}\n"
                f"Percentage of tokens after split point: {token_analysis['percentage']:.2f}%",
                title="Token Analysis",
                border_style="green"
            )
            console.print(token_panel)
        
        return True
    
    def navigate(self, direction):
        """Navigate to the next or previous problem."""
        new_index = self.current_index + direction
        if 0 <= new_index < len(self.data):
            self.current_index = new_index
            return self.display_problem()
        else:
            console.print("Reached the end of the problem set.", style="yellow")
            return False
    
    def go_to_index(self, index_str):
        """Go to a specific problem by index."""
        try:
            index = int(index_str) - 1  # Convert to 0-indexed
            if 0 <= index < len(self.data):
                self.current_index = index
                return self.display_problem()
            else:
                console.print(f"Invalid index. Please enter a number between 1 and {len(self.data)}", style="yellow")
                return False
        except ValueError:
            console.print("Please enter a valid number", style="yellow")
            return False
    
    def mark_correct(self):
        """Mark the current problem as correct."""
        if not self.data:
            console.print("Please load a JSONL file first.", style="yellow")
            return False
        
        index = self.current_index
        # Update if not already evaluated
        if index not in self.evaluations:
            self.evaluations[index] = True
            self.correct_count += 1
            self.total_evaluated += 1
        # Change evaluation from incorrect to correct
        elif not self.evaluations[index]:
            self.evaluations[index] = True
            self.correct_count += 1
        
        console.print(f"Problem {index + 1} marked as correct.", style="green")
        return True
    
    def mark_incorrect(self):
        """Mark the current problem as incorrect."""
        if not self.data:
            console.print("Please load a JSONL file first.", style="yellow")
            return False
        
        index = self.current_index
        # Update if not already evaluated
        if index not in self.evaluations:
            self.evaluations[index] = False
            self.total_evaluated += 1
        # Change evaluation from correct to incorrect
        elif self.evaluations[index]:
            self.evaluations[index] = False
            self.correct_count -= 1
        
        console.print(f"Problem {index + 1} marked as incorrect.", style="red")
        return True
    
    def show_stats(self):
        """Display evaluation statistics."""
        if not self.data:
            console.print("Please load a JSONL file first.", style="yellow")
            return False
        
        if self.total_evaluated == 0:
            console.print("No problems evaluated yet", style="yellow")
            return False
        
        accuracy = (self.correct_count / self.total_evaluated) * 100
        
        # Create a table for statistics
        table = Table(title="Evaluation Statistics", box=box.DOUBLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Problems evaluated", f"{self.total_evaluated}/{len(self.data)}")
        table.add_row("Correct answers", str(self.correct_count))
        table.add_row("Current accuracy", f"{accuracy:.2f}%")
        
        console.print(table)
        return True
    
    def auto_evaluate(self):
        """Automatically evaluate all problems based on the correct answer appearing in the response."""
        if not self.data:
            console.print("Please load a JSONL file first.", style="yellow")
            return False
        
        # Reset evaluation
        self.correct_count = 0
        self.total_evaluated = 0
        self.evaluations = {}
        
        with Progress() as progress:
            task = progress.add_task("Evaluating problems...", total=len(self.data))
            
            for idx, item in enumerate(self.data):
                correct_answer = str(item.get("correct_answer", "")).strip()
                model_response = str(item.get("model_response", "")).strip()
                
                # Extract the final number from the model response
                model_numbers = re.findall(r'\b\d+\b', model_response)
                
                # Check if the correct answer appears in the model response numbers
                if model_numbers and correct_answer in model_numbers:
                    self.evaluations[idx] = True
                    self.correct_count += 1
                else:
                    self.evaluations[idx] = False
                
                self.total_evaluated += 1
                progress.update(task, advance=1)
        
        console.print(f"Auto-evaluation complete. Evaluated {self.total_evaluated} problems.", style="green")
        self.show_stats()
        return True
    
    def export_evaluations(self, filename="model_evaluations_with_tokens.json"):
        """Export evaluations with token analysis to a JSON file."""
        if not self.data or not self.evaluations:
            console.print("No evaluations to export.", style="yellow")
            return False
        
        try:
            # Create export data
            export_data = []
            
            with Progress() as progress:
                task = progress.add_task("Exporting evaluations...", total=len(self.evaluations))
                
                for idx, item in enumerate(self.data):
                    if idx in self.evaluations:
                        model_response = str(item.get("model_response", ""))
                        token_analysis = self.analyze_tokens(model_response)
                        
                        export_item = {
                            "problem_index": idx,
                            "problem": item.get("problem", ""),
                            "correct_answer": item.get("correct_answer", ""),
                            "model_response": model_response,
                            "evaluation": "correct" if self.evaluations[idx] else "incorrect",
                            "token_analysis": {
                                "total_tokens": token_analysis["total_tokens"],
                                "after_split_tokens": token_analysis["after_split_tokens"],
                                "percentage": token_analysis["percentage"]
                            }
                        }
                        export_data.append(export_item)
                        progress.update(task, advance=1)
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            console.print(f"Successfully exported {len(export_data)} evaluations to {filename}", style="green")
            return True
        except Exception as e:
            console.print(f"Error exporting evaluations: {str(e)}", style="bold red")
            return False
    
    def update_split_point(self, new_split_point):
        """Update the split point for token analysis."""
        self.split_point = new_split_point
        console.print(f"Split point updated to: \"{new_split_point}\"", style="green")
        return True
    
    def show_token_chart(self):
        """Display a token analysis chart for evaluated problems."""
        if not self.data or not tokenizer_loaded:
            console.print("No data loaded or tokenizer not available.", style="yellow")
            return False
        
        if not self.evaluations:
            console.print("No problems evaluated yet.", style="yellow")
            return False
            
        # Analyze tokens for all evaluated problems
        token_data = {
            "total": [],
            "after_split": [],
            "percentages": []
        }
        
        problem_indices = []
        
        for i, item in enumerate(self.data):
            if i in self.evaluations:
                model_response = str(item.get("model_response", ""))
                analysis = self.analyze_tokens(model_response)
                
                token_data["total"].append(analysis["total_tokens"])
                token_data["after_split"].append(analysis["after_split_tokens"])
                token_data["percentages"].append(analysis["percentage"])
                problem_indices.append(i + 1)  # 1-indexed for display
        
        # Create the chart
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(problem_indices))
        width = 0.35
        
        plt.bar(x - width/2, token_data["total"], width, label='Total Tokens')
        plt.bar(x + width/2, token_data["after_split"], width, label=f'Tokens After Split')
        
        plt.xlabel('Problem Number')
        plt.ylabel('Token Count')
        plt.title('Token Analysis')
        plt.xticks(x, problem_indices)
        plt.legend()
        
        # Add percentage as text above bars
        for i, (total, after, percent) in enumerate(zip(token_data["total"], token_data["after_split"], token_data["percentages"])):
            plt.text(i + width/2, after + 5, f"{percent:.1f}%", ha='center')
        
        plt.tight_layout()
        
        # Save the chart to a file
        chart_file = "token_analysis_chart.png"
        plt.savefig(chart_file)
        plt.close()
        
        console.print(f"Token analysis chart saved to {chart_file}", style="green")
        
        # Depending on the OS, try to open the image
        try:
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                os.system(f"open {chart_file}")
            elif system == "Windows":
                os.system(f"start {chart_file}")
            elif system == "Linux":
                os.system(f"xdg-open {chart_file}")
                
            console.print("Chart opened in default image viewer.", style="green")
        except Exception:
            console.print("Chart saved but could not be opened automatically.", style="yellow")
            
        return True
        
    def show_accuracy_chart(self):
        """Display an accuracy pie chart."""
        if not self.data or not self.evaluations:
            console.print("No data loaded or no problems evaluated yet.", style="yellow")
            return False
            
        # Create the chart
        plt.figure(figsize=(8, 8))
        
        # Data for pie chart
        labels = ['Correct', 'Incorrect']
        sizes = [self.correct_count, self.total_evaluated - self.correct_count]
        colors = ['#4CAF50', '#F44336']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Model Accuracy')
        
        # Save the chart to a file
        chart_file = "accuracy_chart.png"
        plt.savefig(chart_file)
        plt.close()
        
        console.print(f"Accuracy chart saved to {chart_file}", style="green")
        
        # Depending on the OS, try to open the image
        try:
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                os.system(f"open {chart_file}")
            elif system == "Windows":
                os.system(f"start {chart_file}")
            elif system == "Linux":
                os.system(f"xdg-open {chart_file}")
                
            console.print("Chart opened in default image viewer.", style="green")
        except Exception:
            console.print("Chart saved but could not be opened automatically.", style="yellow")
            
        return True

def print_help():
    """Print help information about available commands."""
    help_text = """
    AIME Model Response Evaluator CLI

    Available commands:
      load <file>           - Load a JSONL file with problem data
      show                  - Display the current problem
      next                  - Go to the next problem
      prev                  - Go to the previous problem
      goto <number>         - Go to a specific problem number
      correct               - Mark the current problem as correct
      incorrect             - Mark the current problem as incorrect
      stats                 - Show evaluation statistics
      auto                  - Automatically evaluate all problems
      export [filename]     - Export evaluations with token analysis
      split <text>          - Update the split point for token analysis
      chart                 - Generate and show token analysis chart
      accuracy              - Generate and show accuracy pie chart
      help                  - Show this help message
      exit                  - Exit the program
      
    Token analysis measures how much of the model's response comes after the split point,
    which is by default "Therefore, the answer is".
    """
    console.print(Panel(help_text, title="Help", border_style="green"))

def main():
    """Main CLI application loop."""
    console.print(Panel(
        "AIME Model Response Evaluator CLI\n\n"
        "A command-line tool for evaluating mathematical problem responses.",
        title="Welcome",
        border_style="cyan"
    ))
    
    evaluator = MathEvaluator()
    
    # Check if arguments were provided
    parser = argparse.ArgumentParser(description="AIME Model Response Evaluator CLI")
    parser.add_argument("file", nargs="?", help="JSONL file to load")
    args = parser.parse_args()
    
    # If a file was provided as an argument, load it
    if args.file:
        evaluator.load_jsonl(args.file)
        evaluator.display_problem()
    else:
        console.print("\nType 'help' to see available commands.\n")
    
    # Main command loop
    while True:
        try:
            command = console.input(">>> ")
            command = command.strip()
            
            if command == "exit" or command == "quit":
                console.print("Goodbye!", style="cyan")
                break
                
            elif command == "help":
                print_help()
                
            elif command.startswith("load "):
                file_path = command[5:].strip()
                evaluator.load_jsonl(file_path)
                if evaluator.data:
                    evaluator.display_problem()
                    
            elif command == "show":
                evaluator.display_problem()
                
            elif command == "next":
                evaluator.navigate(1)
                
            elif command == "prev":
                evaluator.navigate(-1)
                
            elif command.startswith("goto "):
                index = command[5:].strip()
                evaluator.go_to_index(index)
                
            elif command == "correct":
                evaluator.mark_correct()
                evaluator.display_problem()
                
            elif command == "incorrect":
                evaluator.mark_incorrect()
                evaluator.display_problem()
                
            elif command == "stats":
                evaluator.show_stats()
                
            elif command == "auto":
                evaluator.auto_evaluate()
                evaluator.display_problem()
                
            elif command.startswith("export"):
                parts = command.split(" ", 1)
                if len(parts) > 1:
                    evaluator.export_evaluations(parts[1].strip())
                else:
                    evaluator.export_evaluations()
                    
            elif command.startswith("split "):
                new_split = command[6:].strip()
                evaluator.update_split_point(new_split)
                
            elif command == "chart":
                evaluator.show_token_chart()
                
            elif command == "accuracy":
                evaluator.show_accuracy_chart()
                
            else:
                console.print("Unknown command. Type 'help' to see available commands.", style="yellow")
        except Exception as e:
            console.print(f"Error: {str(e)}", style="bold red")

if __name__ == "__main__":
    main()
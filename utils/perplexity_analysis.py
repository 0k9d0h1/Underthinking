import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import os

with open("intra.pkl", "rb") as f:
    intra_perplexities_list = pickle.load(f)
with open("inter.pkl", "rb") as f:
    inter_perplexities_list = pickle.load(f)
# Example variables: replace with your actual nested lists
# intra_perplexities_list = [...]
# inter_perplexities_list = [...]

# Compute per-problem statistics
problem_ids = np.arange(len(intra_perplexities_list))
mean_intra = [np.mean(x) if len(x) > 0 else np.nan for x in intra_perplexities_list]
std_intra  = [np.std(x, ddof=1) if len(x) > 1 else np.nan for x in intra_perplexities_list]
mean_inter = [np.mean(x) if len(x) > 0 else np.nan for x in inter_perplexities_list]
std_inter  = [np.std(x, ddof=1) if len(x) > 1 else np.nan for x in inter_perplexities_list]
reward     = [mi - mni for mi, mni in zip(mean_inter, mean_intra)]

# Build a DataFrame
df = pd.DataFrame({
    'problem_id': problem_ids,
    'mean_intra': mean_intra,
    'std_intra': std_intra,
    'mean_inter': mean_inter,
    'std_inter': std_inter,
    'reward': reward
})


# Plot 1: Histogram of rewards
plt.figure()
plt.hist(df['reward'].dropna(), bins=30)
plt.title('Distribution of Reward (mean_inter - mean_intra)')
plt.xlabel('Reward')
plt.ylabel('Count')
plt.savefig('./plots/reward_distribution.png')

# Plot 2: Scatter plot mean_intra vs mean_inter
plt.figure()
plt.scatter(df['mean_intra'], df['mean_inter'])
plt.title('Mean Intra vs Mean Inter Perplexity')
plt.xlabel('Mean Intra Perplexity')
plt.ylabel('Mean Inter Perplexity')
plt.savefig('./plots/mean_intra_vs_mean_inter.png')

# Plot 3: Reward over problem index
plt.figure()
plt.plot(df['problem_id'], df['reward'])
plt.title('Reward by Problem ID')
plt.xlabel('Problem ID')
plt.ylabel('Reward (mean_inter - mean_intra)')
plt.savefig('./plots/reward_by_problem_id.png')

# Configuration
random.seed(42)
num_problems = len(intra_perplexities_list)
sample_size = min(20, num_problems)
sample_ids = random.sample(range(num_problems), k=sample_size)

# Directory to save plots
output_dir = './plots'
os.makedirs(output_dir, exist_ok=True)

saved_files = []
for pid in sample_ids:
    intra = intra_perplexities_list[pid]
    inter = inter_perplexities_list[pid]
    
    intra = np.array(intra) if intra else None
    inter = np.array(inter) if inter else None
    intra = intra - intra.mean() if intra is not None else None
    inter = inter - inter.mean() if inter is not None else None
    
    abs_max_intra = np.abs(intra).max() if intra is not None else None
    abs_max_inter = np.abs(inter).max() if inter is not None else None
    
    if abs_max_inter is None:
        a = abs_max_intra
    elif abs_max_intra is None:
        a = abs_max_inter
    else:
        a = max(abs_max_inter, abs_max_intra)
    plt.figure(figsize=(6, 4))
    if intra is not None:
        plt.scatter(range(len(intra)), intra/a, color='blue', label='intra-cluster')
    if inter is not None:
        plt.scatter(range(len(inter)), inter/a, color='red', label='inter-cluster')
    plt.title(f'Problem {pid} Perplexity Distribution')
    plt.xlabel('Sample Index')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'problem_{pid}_perplexity.png')
    plt.savefig(filepath)
    plt.close()
    saved_files.append(filepath)

# Print out where plots were saved
for fp in saved_files:
    print(fp)
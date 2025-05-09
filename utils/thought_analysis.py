import json
from itertools import groupby

def revisit_rate(seqs):
    sum = 0
    # 1) collapse consecutive duplicates into a list of segments
    for seq in seqs:
        if len(seq) == 0:
            continue
        segments = [k for k,_ in groupby(seq)]
        seen      = set([segments[0]])
        revisits  = 0
        changes   = 0

        # walk through each transition in segments
        for prev, curr in zip(segments, segments[1:]):
            if curr != prev:
                changes += 1
                if curr in seen:
                    revisits += 1
            seen.add(curr)
        score = revisits / changes if changes else 0.0
        sum += score
    # 2) average the scores
    if len(seqs) > 0:
        return sum / len(seqs)
    
thought_split_file = "../outputs/Thought_split_Deepseek_R1_Distill_Qwen_14B.jsonl"
thought_cluster_file = "../outputs/Thought_cluster_4.1_Deepseek_R1_Distill_Qwen_14B.jsonl"

thoughts_correct = 0
thoughts_incorrect = 0
cluster_correct = 0
cluster_incorrect = 0
cluster_thoughts_ratio_correct = []
cluster_thoughts_ratio_incorrect = []

cluster_sequences_correct = []
cluster_sequences_incorrect = []

with open(thought_split_file, 'r') as f, open(thought_cluster_file, 'r') as f2:
    thoughts_split = [json.loads(line) for line in f]
    thoughts_cluster = [json.loads(line) for line in f2]
    
    for thought_split, thought_cluster in zip(thoughts_split, thoughts_cluster):
        thought_filtered = [thought for thought in thought_split["phrases"] if len(thought) > 100]
        num_thoughts = len(thought_filtered)
        
        if num_thoughts > 1000 or num_thoughts == 0:
            continue
        correctness = thought_cluster["correctness"]
        gpt4o_answer = thought_cluster["gpt4o_answer"]

        thought_to_cluster = {}
        cluster_seq = []
        clusters = [cluster for cluster in gpt4o_answer.split("Cluster") if len(cluster) > 3]
        for cluster in clusters:
            cluster_num = cluster.split(":")[0].strip()
            if len(cluster.split("Thoughts:")) < 2:
                continue
            cluster = cluster.split("Thoughts:")[1].strip()
            thoughts = [int(thought.strip()) for thought in cluster.split(",") if thought.strip().isdigit()]

            for thought in thoughts:
                if thought < len(thought_split["phrases"]) and len(thought_split["phrases"][thought]) <= 100:
                    continue
                thought_to_cluster[thought] = cluster_num
        # Sort cluster numbers by their corresponding thought numbers
        sorted_clusters = sorted(thought_to_cluster.items(), key=lambda x: x[0])
        cluster_seq = [cluster_num for _, cluster_num in sorted_clusters]

        if correctness == True:
            thoughts_correct += num_thoughts
            cluster_correct += len(clusters)
            cluster_thoughts_ratio_correct.append(len(clusters) / num_thoughts)
            cluster_sequences_correct.append(cluster_seq)
        else:
            thoughts_incorrect += num_thoughts
            cluster_incorrect += len(clusters)
            cluster_thoughts_ratio_incorrect.append(len(clusters) / num_thoughts)
            cluster_sequences_incorrect.append(cluster_seq)

print(f"Thoughts Correct: {thoughts_correct}")
print(f"Thoughts Incorrect: {thoughts_incorrect}")
print(f"Cluster Correct: {cluster_correct}")
print(f"Cluster Incorrect: {cluster_incorrect}")
print(f"Revisit Rate Correct: {revisit_rate(cluster_sequences_correct)}")
print(f"Revisit Rate Incorrect: {revisit_rate(cluster_sequences_incorrect)}")
print(f"Cluster Thoughts Ratio Correct: {sum(cluster_thoughts_ratio_correct) / len(cluster_thoughts_ratio_correct)}")
print(f"Cluster Thoughts Ratio Incorrect: {sum(cluster_thoughts_ratio_incorrect) / len(cluster_thoughts_ratio_incorrect)}")

# import matplotlib.pyplot as plt

# # Plot a histogram of thought lengths
# plt.hist(thought_len, bins=range(min(thought_len), max(thought_len) + 10, 10), color='blue', alpha=0.7)
# plt.title('Histogram of Thought Lengths')
# plt.xlabel('Length of Thoughts')
# plt.ylabel('Frequency')
# plt.xlim(0, 200)
# plt.grid(axis='y', alpha=0.75)

# # Save the histogram as a PNG file
# plt.savefig('thought_length_histogram.png')
# plt.close()


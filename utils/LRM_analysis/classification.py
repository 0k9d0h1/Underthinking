import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os


# Define the PyTorch-based classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(SimpleClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# Load the datasets
layer_wise_activations = np.load(
    "/home/kdh0901/Desktop/cache_dir/kdh0901/classification_data/layer_wise.npy"
)
labels = np.load(
    "/home/kdh0901/Desktop/cache_dir/kdh0901/classification_data/labels.npy"
)

print("labels shape:", labels.shape)
print("layer_wise_activations shape:", layer_wise_activations.shape)

# Split indices for consistent train/dev sets across evaluations
indices = np.arange(len(labels))
train_indices, dev_indices = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=labels
)

y_train = labels[train_indices]
y_dev = labels[dev_indices]

# Evaluate layer-wise activations
num_layers_layerwise = layer_wise_activations.shape[1]

print("\n--- Layer-wise activation classification ---")
layer_wise_results = []
best_f1 = 0.0
best_model_state = None
best_layer = -1

for layer in range(num_layers_layerwise):
    # Extract data for the specific layer
    X = layer_wise_activations[:, layer, :]
    X_train_np = X[train_indices]
    X_dev_np = X[dev_indices]

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_dev = torch.tensor(X_dev_np, dtype=torch.float32)
    y_dev_tensor = torch.tensor(y_dev, dtype=torch.float32).view(-1, 1)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize model, loss, and optimizer
    input_dim = X_train.shape[1]
    model = SimpleClassifier(input_dim).to("cpu")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_logits = model(X_dev)
        y_pred = torch.sigmoid(y_pred_logits).round().numpy().flatten()

    accuracy = accuracy_score(y_dev, y_pred)
    f1 = f1_score(y_dev, y_pred, average="macro")
    layer_wise_results.append({"layer": layer, "accuracy": accuracy, "f1": f1})

    if f1 > best_f1:
        best_f1 = f1
        best_model_state = model.state_dict()
        best_layer = layer

# Save the best model
if best_model_state is not None:
    output_dir = "/home/kdh0901/Desktop/Underthinking/model_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"best_classifier_layer_{best_layer}.pt")
    torch.save(best_model_state, save_path)
    print(
        f"\nBest model saved for layer {best_layer} with accuracy {best_f1:.4f} at {save_path}"
    )


y_all_true = np.ones(len(y_dev))  # Dummy true labels for all heads
y_all_false = np.zeros(len(y_dev))  # Dummy false labels for all heads
y_random = np.random.randint(0, 2, size=len(y_dev))  # Random labels for all heads
print(
    f"All true labels: Accuracy = {accuracy_score(y_dev, y_all_true):.4f}, F1 Score = {f1_score(y_dev, y_all_true, average='macro'):.4f}"
)
print(
    f"All false labels: Accuracy = {accuracy_score(y_dev, y_all_false):.4f}, F1 Score = {f1_score(y_dev, y_all_false, average='macro'):.4f}"
)
print(
    f"Random labels: Accuracy = {accuracy_score(y_dev, y_random):.4f}, F1 Score = {f1_score(y_dev, y_random, average='macro'):.4f}"
)

# Sort results by F1 score
sorted_layer_wise_results = sorted(
    layer_wise_results, key=lambda x: x["f1"], reverse=True
)

# Print sorted results
for result in sorted_layer_wise_results[:10]:
    print(
        f"Layer {result['layer']}: Accuracy = {result['accuracy']:.4f}, F1 Score = {result['f1']:.4f}"
    )

print("\n--- Evaluation complete ---")

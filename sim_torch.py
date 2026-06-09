"""
SIM_TORCH.PY
============
PyTorch implementation of the 8-bit Logic Gate Neural Simulator.

This script demonstrates the "Pythonic" way of building neural networks using 
PyTorch's OOP-based architecture. It includes:
1. A custom nn.Module class for the simulator.
2. Use of DataLoaders and TensorDatasets for efficient batching.
3. Explicit training loop with backpropagation using torch.optim.
4. Integration with CPU/GPU (defaults to CPU for simplicity).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from data_utils import generate_data, parse_input, calculate_raw_result

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'logic_gate_model.pth')
DATASET_PATH = os.path.join(SCRIPT_DIR, 'test_dataset.npz')

# =============================================================================
# 1. MODEL ARCHITECTURE
# =============================================================================

class LogicGateSimulator(nn.Module):
    """
    Standard PyTorch Module defining a 3-layer MLP.
    Inherits from nn.Module and implements the forward method.
    """
    def __init__(self):
        super(LogicGateSimulator, self).__init__()
        # Using nn.Sequential for a clean linear flow
        self.network = nn.Sequential(
            nn.Linear(19, 128),     # Input to Hidden 1
            nn.ReLU(),
            nn.Linear(128, 128),    # Hidden 1 to Hidden 2
            nn.ReLU(),
            nn.Linear(128, 8),      # Hidden 2 to Output (8 bits)
            nn.Sigmoid()            # Squash output to [0, 1] range
        )

    def forward(self, x):
        """Defines the computation performed at every call."""
        return self.network(x)

# =============================================================================
# 2. TRAINING LOOP
# =============================================================================

def train():
    """
    Orchestrates the training process including data preparation, 
    optimization, and persistence.
    """
    print("Generating training data in memory...")
    X_raw, y_raw = generate_data(100000)
    
    # Convert NumPy arrays to PyTorch Tensors
    X_train = torch.tensor(X_raw, dtype=torch.float32)
    y_train = torch.tensor(y_raw, dtype=torch.float32)
    
    # Create Dataset and DataLoader for easy mini-batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    model = LogicGateSimulator()
    # Binary Cross Entropy Loss is standard for bitwise prediction
    criterion = nn.BCELoss()
    # Adam optimizer usually converges faster than standard SGD
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting PyTorch training (25 epochs)...")
    for epoch in range(25):
        model.train()  # Set to training mode
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            # 1. Zero the parameter gradients
            optimizer.zero_grad()
            
            # 2. Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 3. Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation Check (every epoch)
        model.eval()
        with torch.no_grad():
            # Check accuracy on a subset of training data
            sample_preds = (model(X_train[:1000]) > 0.5).float()
            # All-bits match accuracy
            acc = (sample_preds == y_train[:1000]).all(dim=1).float().mean()
            print(f"Epoch {epoch+1}/25 | Avg Loss: {total_loss/len(train_loader):.4f} | Accuracy: {acc:.2%}")

    # Save only the state dictionary (weights)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")
    return model

# =============================================================================
# 3. INTERACTIVE SIMULATION
# =============================================================================

def run_interactive(model):
    """
    Provides an interactive CLI for testing specific logic gate operations.
    """
    op_names = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR", "NOT A", "NOT B"]
    
    while True:
        print("\n" + "="*45)
        print("   8-BIT LOGIC GATE NEURAL SIMULATOR (Torch)   ")
        print("="*45)
        for i, name in enumerate(op_names): 
            print(f"  [{i}] {name}")
        print("  [Q] Quit")
        print("="*45)
        
        choice = input("\nSelect Operation (0-7): ").strip().upper()
        if choice == 'Q': break
        if not (choice.isdigit() and 0 <= int(choice) <= 7): continue
        
        op_idx = int(choice)
        a_bits, b_bits = None, None
        
        # Get and parse user input
        while a_bits is None: a_bits = parse_input(input("Input A (Bin/Hex/Dec): "))
        while b_bits is None: b_bits = parse_input(input("Input B (Bin/Hex/Dec): "))
        
        # Construct the input tensor
        op_bits = [int(c) for c in format(op_idx, '03b')]
        input_tensor = torch.tensor([a_bits + b_bits + op_bits], dtype=torch.float32)
        
        # Inference mode
        model.eval()
        with torch.no_grad():
            pred_out = model(input_tensor)
            # Threshold output to get discrete bits
            pred_bits = (pred_out > 0.5).int().tolist()[0]
            pred_str = "".join(map(str, pred_bits))
            
        # Ground Truth comparison
        a_val = int("".join(map(str, a_bits)), 2)
        b_val = int("".join(map(str, b_bits)), 2)
        true_res = calculate_raw_result(a_val, b_val, op_idx)
        true_str = format(true_res, '08b')
        status = "PASSED" if pred_str == true_str else "FAILED"
            
        print("-" * 40)
        print(f" OPERATION: {op_names[op_idx]}")
        print(f" RESULT:    {pred_str} (Dec: {int(pred_str, 2)})")
        print(f" TRUE:      {true_str}")
        print(f" CHECK:     {status}")
        print("-" * 40)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Handle Test Dataset
    if os.path.exists(DATASET_PATH):
        print(f"Loading shared test dataset from {DATASET_PATH}...")
        data = np.load(DATASET_PATH)
        X_test = torch.tensor(data['X_test'], dtype=torch.float32)
        y_test = torch.tensor(data['y_test'], dtype=torch.float32)
    else:
        print("Test dataset not found. Generating...")
        X_raw, y_raw = generate_data(10000)
        np.savez_compressed(DATASET_PATH, X_test=X_raw, y_test=y_raw)
        X_test, y_test = torch.tensor(X_raw, dtype=torch.float32)
        y_test = torch.tensor(y_raw, dtype=torch.float32)

    # Initialize Model
    model = LogicGateSimulator()
    
    # Load weights or train
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model = train()

    # Final Batch Evaluation
    model.eval()
    with torch.no_grad():
        test_preds = (model(X_test) > 0.5).float()
        # Accuracy: Percentage of samples where every bit matches
        acc = (test_preds == y_test).all(dim=1).float().mean()
        print(f"\nFinal Test Accuracy (Exact Match): {acc:.2%}")

    # Launch UI
    run_interactive(model)

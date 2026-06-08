import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from data_utils import generate_data, parse_input, calculate_raw_result

# Define paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'logic_gate_model.pth')
DATASET_PATH = os.path.join(SCRIPT_DIR, 'test_dataset.npz')

# 1. Model Architecture
class LogicGateSimulator(nn.Module):
    def __init__(self):
        super(LogicGateSimulator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(19, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 2. Training Loop
def train():
    print("Generating training data in memory...")
    X_raw, y_raw = generate_data(100000)
    X_train, y_train = torch.tensor(X_raw), torch.tensor(y_raw)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    model = LogicGateSimulator()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training (25 epochs)...")
    for epoch in range(25):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            preds = (model(X_train[:1000]) > 0.5).float()
            acc = (preds == y_train[:1000]).all(dim=1).float().mean()
            print(f"Epoch {epoch+1}/25, Sample Accuracy: {acc:.2%}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

# 3. Interactive Logic
def run_interactive(model):
    op_names = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR", "NOT A", "NOT B"]
    while True:
        print("\n" + "="*45)
        print("   8-BIT LOGIC GATE NEURAL SIMULATOR (Torch)   ")
        print("="*45)
        for i, name in enumerate(op_names): print(f"  [{i}] {name}")
        print("  [Q] Quit")
        print("="*45)
        
        choice = input("\nSelect Operation (0-7): ").strip().upper()
        if choice == 'Q': break
        if not (choice.isdigit() and 0 <= int(choice) <= 7): continue
        
        op_idx = int(choice)
        a_bits, b_bits = None, None
        while a_bits is None: a_bits = parse_input(input("Input A (Bin/Hex/Dec): "))
        while b_bits is None: b_bits = parse_input(input("Input B (Bin/Hex/Dec): "))
        
        op_bits = [int(c) for c in format(op_idx, '03b')]
        input_tensor = torch.tensor([a_bits + b_bits + op_bits], dtype=torch.float32)
        
        model.eval()
        with torch.no_grad():
            pred_out = model(input_tensor)
            pred_bits = (pred_out > 0.5).int().tolist()[0]
            pred_str = "".join(map(str, pred_bits))
            
        # Accuracy check
        a_val = int("".join(map(str, a_bits)), 2)
        b_val = int("".join(map(str, b_bits)), 2)
        true_res = calculate_raw_result(a_val, b_val, op_idx)
        true_str = format(true_res, '08b')
        status = "PASSED" if pred_str == true_str else "FAILED"
            
        print("-" * 40)
        print(f" RESULT ({op_names[op_idx]}): {pred_str}")
        print(f" TRUE RESULT:   {true_str}")
        print(f" CHECK:         {status}")
        print("-" * 40)

if __name__ == "__main__":
    if os.path.exists(DATASET_PATH):
        print(f"Loading shared dataset from {DATASET_PATH}...")
        data = np.load(DATASET_PATH)
        X_test, y_test = torch.tensor(data['X_test']), torch.tensor(data['y_test'])
    else:
        print("Dataset not found. Generating...")
        X_raw, y_raw = generate_data(10000)
        np.savez_compressed(DATASET_PATH, X_test=X_raw, y_test=y_raw)
        X_test, y_test = torch.tensor(X_raw), torch.tensor(y_raw)

    model = LogicGateSimulator()
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model = train()

    model.eval()
    with torch.no_grad():
        test_preds = (model(X_test) > 0.5).float()
        acc = (test_preds == y_test).all(dim=1).float().mean()
        print(f"\nFinal Test Accuracy: {acc:.2%}")

    run_interactive(model)

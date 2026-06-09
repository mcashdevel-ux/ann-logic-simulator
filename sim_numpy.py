"""
SIM_NUMPY.PY
============
NumPy-based Multi-Layer Perceptron (MLP) implementation for 8-bit Logic Gates.

This script demonstrates a "from-scratch" implementation of a neural network
using only NumPy for matrix operations. It includes:
1. A manual forward and backward pass implementation.
2. He and Xavier weight initialization.
3. Training loop with SGD and batching.
4. Interactive CLI for real-time testing.
"""

import numpy as np
import os
from data_utils import generate_data, parse_input, calculate_raw_result

# --- CONFIGURATION & PATHS ---
# Determine paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, 'numpy_weights.npz')
SHARED_DATASET_PATH = os.path.join(SCRIPT_DIR, 'test_dataset.npz')

class NumpyMLP:
    """
    A 3-layer MLP implemented purely in NumPy.
    Architecture: Input(19) -> Hidden(128) -> Hidden(128) -> Output(8)
    Activations: ReLU for hidden layers, Sigmoid for output.
    """
    def __init__(self, input_size=19, hidden_size=128, output_size=8):
        # LAYER 1: He initialization (standard for ReLU)
        # Weights: (input_size x hidden_size), Biases: (1 x hidden_size)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # LAYER 2: He initialization
        # Weights: (hidden_size x hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        
        # LAYER 3: Xavier initialization (standard for Sigmoid)
        # Weights: (hidden_size x output_size), Biases: (1 x output_size)
        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.b3 = np.zeros((1, output_size))

    def sigmoid(self, x):
        """Numerically stable sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def relu(self, x):
        """Rectified Linear Unit activation."""
        return np.maximum(0, x)

    def forward(self, X):
        """
        Performs the forward pass through the network.
        Stores intermediate activations for the backward pass.
        """
        # Input to Hidden 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Hidden 1 to Hidden 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        # Hidden 2 to Output
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        
        return self.a3

    def backward(self, X, y, output, lr):
        """
        Manual implementation of backpropagation.
        Calculates gradients and updates weights using Stochastic Gradient Descent.
        """
        m = y.shape[0]  # Number of samples in the batch
        
        # 1. Output Layer Gradient (Binary Cross Entropy + Sigmoid)
        dz3 = output - y
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # 2. Hidden Layer 2 Gradient
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * (self.z2 > 0)  # Derivative of ReLU
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 3. Hidden Layer 1 Gradient
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # Derivative of ReLU
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 4. Update Weights and Biases
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def save(self, path):
        """Saves current weights and biases to a compressed NumPy file."""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    def load(self, path):
        """Loads weights and biases from a file."""
        data = np.load(path)
        self.W1, self.b1 = data['W1'], data['b1']
        self.W2, self.b2 = data['W2'], data['b2']
        self.W3, self.b3 = data['W3'], data['b3']

def train_model():
    """Logic for initializing and training the NumPy model."""
    print("Generating training data in memory...")
    X, y = generate_data(100000)
    mlp = NumpyMLP()
    
    epochs, batch_size, lr = 100, 128, 0.1
    print(f"Training NumPy MLP for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Shuffle the dataset every epoch
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]
        
        # Mini-batch loop
        for i in range(0, X.shape[0], batch_size):
            X_b, y_b = X[i:i+batch_size], y[i:i+batch_size]
            mlp.backward(X_b, y_b, mlp.forward(X_b), lr)
            
        # Logging progress
        if (epoch + 1) % 10 == 0:
            preds = (mlp.forward(X[:1000]) > 0.5).astype(int)
            acc = np.mean(np.all(preds == y[:1000], axis=1))
            print(f"Epoch {epoch+1}/{epochs}, Sample Accuracy: {acc:.2%}")
            
    mlp.save(WEIGHTS_PATH)
    return mlp

def evaluate(mlp, X_test, y_test):
    """Evaluates the model on a separate test set."""
    print("\nEvaluating NumPy model on shared test dataset...")
    preds = (mlp.forward(X_test) > 0.5).astype(int)
    # Exact Match: Every bit must be correct for the sample to count
    acc = np.mean(np.all(preds == y_test, axis=1))
    print(f"Test Exact Match Accuracy (NumPy): {acc:.4%}")

def interactive_mode(mlp):
    """Provides a CLI interface for users to test specific logic operations."""
    op_names = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR", "NOT A", "NOT B"]
    while True:
        print("\n" + "="*45)
        print("   8-BIT LOGIC GATE NUMPY SIMULATOR   ")
        print("="*45)
        for i, name in enumerate(op_names): 
            print(f"  [{i}] {name}")
        print("  [Q] Quit")
        
        choice = input("\nSelect Operation: ").strip().upper()
        if choice == 'Q': break
        if not (choice.isdigit() and 0 <= int(choice) <= 7): continue
        
        op_idx = int(choice)
        a_bits, b_bits = None, None
        
        # Input Loop for Operand A
        while a_bits is None: 
            a_bits = parse_input(input("Input A (Bin/Hex/Dec): "))
            
        # Input Loop for Operand B
        while b_bits is None: 
            b_bits = parse_input(input("Input B (Bin/Hex/Dec): "))
        
        # Prepare Input Vector: [A bits] + [B bits] + [3-bit Op Code]
        X_input = np.array([a_bits + b_bits + [int(c) for c in format(op_idx, '03b')]])
        
        # Run Inference
        pred_bits = (mlp.forward(X_input) > 0.5).astype(int)[0]
        pred_str = "".join(map(str, pred_bits))
        
        # Ground Truth Verification
        a_val = int("".join(map(str, a_bits)), 2)
        b_val = int("".join(map(str, b_bits)), 2)
        true_res = calculate_raw_result(a_val, b_val, op_idx)
        true_str = format(true_res, '08b')
        status = "PASSED" if pred_str == true_str else "FAILED"
        
        # Print results
        print("-" * 40)
        print(f" OPERATION: {op_names[op_idx]}")
        print(f" RESULT:    {pred_str} (Dec: {int(pred_str, 2)})")
        print(f" TRUE:      {true_str}")
        print(f" CHECK:     {status}")
        print("-" * 40)

if __name__ == "__main__":
    # Load or generate the shared test dataset
    if os.path.exists(SHARED_DATASET_PATH):
        data = np.load(SHARED_DATASET_PATH)
        X_test, y_test = data['X_test'], data['y_test']
    else:
        X_test, y_test = generate_data(10000)
        np.savez_compressed(SHARED_DATASET_PATH, X_test=X_test, y_test=y_test)

    # Initialize model and load weights if available
    mlp = NumpyMLP()
    if os.path.exists(WEIGHTS_PATH):
        mlp.load(WEIGHTS_PATH)
    else:
        mlp = train_model()
        
    evaluate(mlp, X_test, y_test)
    interactive_mode(mlp)

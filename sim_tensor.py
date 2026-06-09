"""
SIM_TENSOR.PY
=============
TensorFlow/Keras implementation of the 8-bit Logic Gate Neural Simulator.

This script leverages high-level deep learning APIs to:
1. Define a Sequential model with Dense layers.
2. Train using the Adam optimizer and Binary Crossentropy loss.
3. Save/Load models in the modern .keras format.
4. Provide a high-performance interactive simulation interface.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from data_utils import generate_data, parse_input, calculate_raw_result

# --- PATH & ENVIRONMENT CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'logic_gate_model.keras')
DATASET_PATH = os.path.join(SCRIPT_DIR, 'test_dataset.npz')

# Suppress TensorFlow logging for a cleaner CLI experience
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =============================================================================
# 1. MODEL ARCHITECTURE
# =============================================================================

def create_model():
    """
    Constructs a Sequential neural network using Keras.
    
    Architecture:
    - Input: 19 features (8-bit A + 8-bit B + 3-bit Op)
    - Hidden 1: 128 units, ReLU activation
    - Hidden 2: 128 units, ReLU activation
    - Output: 8 units (Result bits), Sigmoid activation
    """
    model = models.Sequential([
        layers.Input(shape=(19,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(8, activation='sigmoid')
    ])
    
    # Using Adam optimizer and Binary Crossentropy (ideal for multi-label classification)
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model

# =============================================================================
# 2. TRAINING LOGIC
# =============================================================================

def train():
    """
    Handles data generation, model creation, and training.
    """
    print("Generating training data in memory...")
    X_train, y_train = generate_data(100000)
    
    model = create_model()
    
    print("Starting TensorFlow training (20 epochs)...")
    # verbose=1 shows the progress bar for each epoch
    model.fit(X_train, y_train, epochs=20, batch_size=128, verbose=1)
    
    # Save the trained weights and architecture
    model.save(MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")
    return model

# =============================================================================
# 3. INTERACTIVE SIMULATION
# =============================================================================

def run_interactive(model):
    """
    Main loop for user-driven logic simulations.
    """
    op_names = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR", "NOT A", "NOT B"]
    
    while True:
        print("\n" + "="*45)
        print("   8-BIT LOGIC GATE NEURAL SIMULATOR (Tensor)  ")
        print("="*45)
        for i, name in enumerate(op_names): 
            print(f"  [{i}] {name}")
        print("  [Q] Quit")
        print("="*45)
        
        choice = input("\nSelect Operation (0-7): ").strip().upper()
        if choice == 'Q': break
        if not (choice.isdigit() and 0 <= int(choice) <= 7): 
            print("Invalid selection.")
            continue
        
        op_idx = int(choice)
        a_bits, b_bits = None, None
        
        # Prompt for inputs using shared parser
        while a_bits is None: a_bits = parse_input(input("Input A: "))
        while b_bits is None: b_bits = parse_input(input("Input B: "))
        
        # Prepare data for prediction (TensorFlow expects a batch dimension)
        input_data = np.array([a_bits + b_bits + [int(c) for c in format(op_idx, '03b')]], dtype=np.float32)
        
        # Inference
        pred = model.predict(input_data, verbose=0)
        # Threshold at 0.5 to get binary bits
        pred_bits = (pred > 0.5).astype(int)[0]
        pred_str = "".join(map(str, pred_bits))
        
        # Calculate true result for comparison
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
    # Load or generate evaluation data
    if os.path.exists(DATASET_PATH):
        data = np.load(DATASET_PATH)
        X_test, y_test = data['X_test'], data['y_test']
    else:
        X_test, y_test = generate_data(10000)
        np.savez_compressed(DATASET_PATH, X_test=X_test, y_test=y_test)

    # Load existing model or train a new one
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing TensorFlow model from {MODEL_PATH}...")
        model = models.load_model(MODEL_PATH)
    else:
        model = train()

    # Perform a batch evaluation on the test set
    print("\nEvaluating on shared test set...")
    preds = (model.predict(X_test, verbose=0) > 0.5).astype(int)
    # Average of samples where ALL 8 bits match exactly
    exact_acc = np.mean(np.all(preds == y_test, axis=1))
    print(f"Final Test Exact Match Accuracy: {exact_acc:.2%}")

    # Launch the interactive CLI
    run_interactive(model)

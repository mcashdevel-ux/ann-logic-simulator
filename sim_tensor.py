import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from data_utils import generate_data, parse_input, calculate_raw_result

# Define paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'logic_gate_model.keras')
DATASET_PATH = os.path.join(SCRIPT_DIR, 'test_dataset.npz')

# 1. Model Architecture
def create_model():
    model = models.Sequential([
        layers.Input(shape=(19,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(8, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 2. Training Logic
def train():
    print("Generating training data in memory...")
    X_train, y_train = generate_data(100000)
    model = create_model()
    print("Starting training (20 epochs)...")
    model.fit(X_train, y_train, epochs=20, batch_size=128, verbose=1)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

# 3. Interactive Logic
def run_interactive(model):
    op_names = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR", "NOT A", "NOT B"]
    while True:
        print("\n" + "="*45)
        print("   8-BIT LOGIC GATE NEURAL SIMULATOR (Tensor)  ")
        print("="*45)
        for i, name in enumerate(op_names): print(f"  [{i}] {name}")
        print("  [Q] Quit")
        print("="*45)
        
        choice = input("\nSelect Operation (0-7): ").strip().upper()
        if choice == 'Q': break
        if not (choice.isdigit() and 0 <= int(choice) <= 7): continue
        
        op_idx = int(choice)
        a_bits, b_bits = None, None
        while a_bits is None: a_bits = parse_input(input("Input A: "))
        while b_bits is None: b_bits = parse_input(input("Input B: "))
        
        input_data = np.array([a_bits + b_bits + [int(c) for c in format(op_idx, '03b')]], dtype=np.float32)
        pred = model.predict(input_data, verbose=0)
        pred_bits = (pred > 0.5).astype(int)[0]
        pred_str = "".join(map(str, pred_bits))
        
        a_val = int("".join(map(str, a_bits)), 2)
        b_val = int("".join(map(str, b_bits)), 2)
        true_res = calculate_raw_result(a_val, b_val, op_idx)
        true_str = format(true_res, '08b')
        status = "PASSED" if pred_str == true_str else "FAILED"
            
        print("-" * 40)
        print(f" RESULT:  {pred_str} (Dec: {int(pred_str, 2)})")
        print(f" TRUE:    {true_str}")
        print(f" CHECK:   {status}")
        print("-" * 40)

if __name__ == "__main__":
    if os.path.exists(DATASET_PATH):
        data = np.load(DATASET_PATH)
        X_test, y_test = data['X_test'], data['y_test']
    else:
        X_test, y_test = generate_data(10000)
        np.savez_compressed(DATASET_PATH, X_test=X_test, y_test=y_test)

    if os.path.exists(MODEL_PATH):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        model = models.load_model(MODEL_PATH)
    else:
        model = train()

    print("\nEvaluating on shared test set...")
    preds = (model.predict(X_test, verbose=0) > 0.5).astype(int)
    exact_acc = np.mean(np.all(preds == y_test, axis=1))
    print(f"Final Test Exact Match: {exact_acc:.2%}")

    run_interactive(model)

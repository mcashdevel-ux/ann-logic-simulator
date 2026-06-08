import math
import json
import os
import random
import time
import sys

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_JSON = os.path.join(SCRIPT_DIR, 'py_weights.json')
NPZ_PATH = os.path.join(SCRIPT_DIR, 'numpy_weights.npz')

# 1. Pure Python Matrix & Vector Utilities
def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def matrix_vector_mul(matrix, vector):
    return [dot_product(row, vector) for row in matrix]

def vector_add(v1, v2):
    return [x + y for x, y in zip(v1, v2)]

def vector_sub(v1, v2):
    return [x - y for x, y in zip(v1, v2)]

def matrix_transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def relu(vector):
    return [max(0, x) for x in vector]

def sigmoid(vector):
    return [1 / (1 + math.exp(-max(min(x, 50), -50))) for x in vector]

# 2. Dataset Generation
def generate_data_py(num_samples):
    X, y = [], []
    for _ in range(num_samples):
        a_b = [random.randint(0, 1) for _ in range(8)]
        b_b = [random.randint(0, 1) for _ in range(8)]
        o_i = random.randint(0, 7)
        o_b = [int(c) for c in format(o_i, '03b')]
        a, b = int("".join(map(str, a_b)), 2), int("".join(map(str, b_b)), 2)
        res = 0
        if o_i == 0: res = a & b
        elif o_i == 1: res = a | b
        elif o_i == 2: res = a ^ b
        elif o_i == 3: res = ~(a & b) & 0xFF
        elif o_i == 4: res = ~(a | b) & 0xFF
        elif o_i == 5: res = ~(a ^ b) & 0xFF
        elif o_i == 6: res = (~a) & 0xFF
        elif o_i == 7: res = (~b) & 0xFF
        X.append(a_b + b_b + o_b)
        y.append([int(c) for c in format(res, '08b')])
    return X, y

# 3. Neural Network Class
class PurePythonMLP:
    def __init__(self, input_size=19, hidden_size=128, output_size=8):
        self.W1 = [[random.gauss(0, math.sqrt(2./input_size)) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.W2 = [[random.gauss(0, math.sqrt(2./hidden_size)) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.b2 = [0.0] * hidden_size
        self.W3 = [[random.gauss(0, math.sqrt(1./hidden_size)) for _ in range(hidden_size)] for _ in range(output_size)]
        self.b3 = [0.0] * output_size

    def forward(self, x):
        self.z1 = vector_add(matrix_vector_mul(self.W1, x), self.b1)
        self.a1 = relu(self.z1)
        self.z2 = vector_add(matrix_vector_mul(self.W2, self.a1), self.b2)
        self.a2 = relu(self.z2)
        self.z3 = vector_add(matrix_vector_mul(self.W3, self.a2), self.b3)
        self.a3 = sigmoid(self.z3)
        return self.a3

    def train_step(self, x, y, lr):
        output = self.forward(x)
        dz3 = vector_sub(output, y)
        dW3 = [[dz3[i] * self.a2[j] for j in range(len(self.a2))] for i in range(len(dz3))]
        da2 = matrix_vector_mul(matrix_transpose(self.W3), dz3)
        dz2 = [da2[i] * (1.0 if self.z2[i] > 0 else 0.0) for i in range(len(da2))]
        dW2 = [[dz2[i] * self.a1[j] for j in range(len(self.a1))] for i in range(len(dz2))]
        da1 = matrix_vector_mul(matrix_transpose(self.W2), dz2)
        dz1 = [da1[i] * (1.0 if self.z1[i] > 0 else 0.0) for i in range(len(da1))]
        dW1 = [[dz1[i] * x[j] for j in range(len(x))] for i in range(len(dz1))]
        
        for i in range(len(self.W3)):
            self.b3[i] -= lr * dz3[i]
            for j in range(len(self.W3[0])): self.W3[i][j] -= lr * dW3[i][j]
        for i in range(len(self.W2)):
            self.b2[i] -= lr * dz2[i]
            for j in range(len(self.W2[0])): self.W2[i][j] -= lr * dW2[i][j]
        for i in range(len(self.W1)):
            self.b1[i] -= lr * dz1[i]
            for j in range(len(self.W1[0])): self.W1[i][j] -= lr * dW1[i][j]

    def save_to_json(self, path):
        data = {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2, "W3": self.W3, "b3": self.b3}
        with open(path, 'w') as f: json.dump(data, f)

    def load_from_json(self, path):
        with open(path, 'r') as f: data = json.load(f)
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = data["W1"], data["b1"], data["W2"], data["b2"], data["W3"], data["b3"]

# 4. Progress Bar Utility
def print_progress(iteration, total, prefix='', suffix='', length=30, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: sys.stdout.write('\n')

# 5. UI and Training Loop
def run_training(mlp, mode_name, num_samples, epochs):
    print(f"\n--- {mode_name.upper()} MODE ---")
    X, y = generate_data_py(num_samples)
    lr = 0.05
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        for i in range(num_samples):
            mlp.train_step(X[i], y[i], lr)
            if i % 100 == 0 or i == num_samples - 1:
                print_progress(i + 1, num_samples, prefix=f'Epoch {epoch+1}/{epochs}', length=30)
        
        # Accuracy check
        correct = 0
        test_size = min(200, num_samples)
        for i in range(test_size):
            if [1 if v > 0.5 else 0 for v in mlp.forward(X[i])] == y[i]: correct += 1
        
        elapsed = time.time() - epoch_start
        print(f"  Result: Acc {correct/test_size:.1%} | Time: {elapsed:.1f}s")
        
    print(f"\nTotal Time: {time.time()-start_time:.1f}s")
    mlp.save_to_json(WEIGHTS_JSON)

def convert_from_numpy():
    try:
        import numpy as np
    except:
        print("Error: NumPy required.")
        return False
    if not os.path.exists(NPZ_PATH): return False
    data = np.load(NPZ_PATH)
    weights = {
        "W1": data['W1'].T.tolist(), "b1": data['b1'].flatten().tolist(),
        "W2": data['W2'].T.tolist(), "b2": data['b2'].flatten().tolist(),
        "W3": data['W3'].T.tolist(), "b3": data['b3'].flatten().tolist()
    }
    with open(WEIGHTS_JSON, 'w') as f: json.dump(weights, f)
    print(f"Weights converted to {WEIGHTS_JSON}")
    return True

def run_interactive(mlp):
    op_names = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR", "NOT A", "NOT B"]
    while True:
        print("\n" + "="*45 + "\n   8-BIT LOGIC GATE NEURAL SIMULATOR (Pure Python)\n" + "="*45)
        for i, name in enumerate(op_names): print(f"  [{i}] {name}")
        print("-" * 45)
        print("  [S] Slow Training (5k samples)")
        print("  [V] Very Slow Training (20k samples)")
        if os.path.exists(NPZ_PATH): print("  [C] Convert from NumPy Weights")
        print("  [Q] Quit")
        print("="*45)
        
        choice = input("\nSelect: ").strip().upper()
        if choice == 'Q': break
        if choice == 'S': run_training(mlp, "Slow", 5000, 10); continue
        if choice == 'V': run_training(mlp, "Very Slow", 20000, 20); continue
        if choice == 'C' and os.path.exists(NPZ_PATH):
            if convert_from_numpy(): mlp.load_from_json(WEIGHTS_JSON)
            continue
            
        if not (choice.isdigit() and 0 <= int(choice) <= 7): continue
        op_idx = int(choice)
        
        def parse(p):
            try:
                if p.startswith('0b'): v = int(p, 2)
                elif p.startswith('0x'): v = int(p, 16)
                else: v = int(p, 10)
                return [int(c) for c in format(v & 0xFF, '08b')]
            except: return None

        a_bits, b_bits = None, None
        while a_bits is None: a_bits = parse(input("Input A: "))
        while b_bits is None: b_bits = parse(input("Input B: "))
        
        input_data = a_bits + b_bits + [int(c) for c in format(op_idx, '03b')]
        pred_bits = [1 if x > 0.5 else 0 for x in mlp.forward(input_data)]
        pred_str = "".join(map(str, pred_bits))
        print("-" * 40 + f"\n RESULT:  {pred_str} (Dec: {int(pred_str, 2)})\n" + "-" * 40)

if __name__ == "__main__":
    mlp = PurePythonMLP()
    if os.path.exists(WEIGHTS_JSON):
        print(f"Loading weights from {WEIGHTS_JSON}...")
        mlp.load_from_json(WEIGHTS_JSON)
    run_interactive(mlp)

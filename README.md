# 🧠 8-Bit Logic Gate Neural Simulator

A comprehensive suite of Artificial Neural Networks (ANN) that simulate 8-bit digital logic gates across four different implementation levels: **PyTorch**, **TensorFlow**, **Pure NumPy**, and **Zero-Dependency Pure Python**.

This project demonstrates how a Multi-Layer Perceptron (MLP) can learn non-linear bitwise operations (like XOR, NAND, and XNOR) and generalize them across an entire 8-bit number space (0-255).

---

## 🚀 Key Features

- **Multi-Framework Support:** Identical architectures implemented in PyTorch, TensorFlow, NumPy, and Pure Python.
- **8-Bit Logic Support:** Simulates AND, OR, XOR, NAND, NOR, XNOR, NOT A, and NOT B.
- **Intelligent Persistence:** Each script automatically detects existing models and datasets to skip redundant training.
- **Cross-Framework Compatibility:** All models share the same 10,000-sample test dataset for benchmarking.
- **Pure Python Backpropagation:** A zero-dependency implementation including manual matrix math and gradient descent.
- **Framework Bridge:** Built-in utility to export optimized weights from NumPy to JSON for use in the Pure Python version.
- **Interactive CLI:** Smart input parsing for Binary (`0b`), Hex (`0x`), and Decimal values.

---

## 🏗️ Neural Network Architecture

All implementations utilize the same "brain" structure to ensure consistency:

- **Input Layer:** 19 Neurons
    - 8 bits for Input A
    - 8 bits for Input B
    - 3 bits for Operation Selector (0-7)
- **Hidden Layer 1:** 128 Neurons (ReLU activation)
- **Hidden Layer 2:** 128 Neurons (ReLU activation)
- **Output Layer:** 8 Neurons (Sigmoid activation)

### Logic Gate Mapping (3-bit Selector)
| Index | Binary | Gate | Logic |
| :--- | :--- | :--- | :--- |
| 0 | `000` | **AND** | `A & B` |
| 1 | `001` | **OR** | `A | B` |
| 2 | `010` | **XOR** | `A ^ B` |
| 3 | `011` | **NAND** | `~(A & B) & 0xFF` |
| 4 | `100` | **NOR** | `~(A | B) & 0xFF` |
| 5 | `101` | **XNOR** | `~(A ^ B) & 0xFF` |
| 6 | `110` | **NOT A** | `~A & 0xFF` |
| 7 | `111` | **NOT B** | `~B & 0xFF` |

---

## 📂 Project Structure

```text
ann_logic_simulator/
├── sim_torch.py       # PyTorch Implementation (Fastest GPU/CPU)
├── sim_tensor.py      # TensorFlow/Keras Implementation
├── sim_numpy.py       # Pure NumPy (Manual Backprop)
├── sim_py.py          # Zero-Dependency Python (Standard Lists only)
├── data_utils.py      # Shared logic for data gen and input parsing
├── test_dataset.npz   # The shared ground-truth dataset
├── logic_gate_model.pth    # Saved PyTorch Weights
├── logic_gate_model.keras  # Saved TensorFlow Model
├── numpy_weights.npz       # Saved NumPy Weights
└── py_weights.json         # Saved JSON Weights (Human Readable)
```

---

## 🛠️ Usage Instructions

### 1. PyTorch Implementation
Requires `torch` and `numpy`.
```bash
python3 ann_logic_simulator/sim_torch.py
```

### 2. TensorFlow Implementation
Requires `tensorflow` and `numpy`.
```bash
python3 ann_logic_simulator/sim_tensor.py
```

### 3. NumPy Implementation
Requires only `numpy`. This implementation features manual backpropagation.
```bash
python3 ann_logic_simulator/sim_numpy.py
```

### 4. Pure Python Implementation (Zero Dependencies)
Requires **no external libraries**.
```bash
python3 ann_logic_simulator/sim_py.py
```
**Training Modes in Pure Python:**
- **Slow Training:** 5,000 samples, 10 epochs. (~5 min)
- **Very Slow Training:** 20,000 samples, 20 epochs. (~30 min)
- **Convert from NumPy:** Instantly imports `numpy_weights.npz` into `py_weights.json`.

---

## 📊 Data Organization

The data is organized into high-performance blocks rather than alternating sets.

### Input Vector ($X$) - 19 Elements
`[A0, A1, A2... A7, B0, B1, B2... B7, OP0, OP1, OP2]`
Floating point values (0.0 or 1.0).

### Output Vector ($y$) - 8 Elements
`[R0, R1, R2, R3, R4, R5, R6, R7]`
The target bitwise result for the given inputs and operation.

---

## ⌨️ Interactive Mode

Each simulator includes a built-in interactive CLI. When prompted for inputs, you can use:
- **Binary:** `0b10101010`
- **Hexadecimal:** `0xFF`
- **Decimal:** `255`

Every interaction performs a **Real-Time Accuracy Check**, comparing the Neural Network's prediction against the mathematical ground truth.

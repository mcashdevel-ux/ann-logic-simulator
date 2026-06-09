"""
EXHAUSTIVE TEST SCRIPT FOR ALL SIMULATION BACKENDS
=================================================
Verifies every implementation against all possible inputs:
- A: 0..255 (8 bits)
- B: 0..255 (8 bits)
- Operation: 0..7 (3 bits)

Total test cases per backend: 256 * 256 * 8 = 524,288
"""

import importlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import time

import numpy as np
from data_utils import calculate_raw_result
from sim_numpy import NumpyMLP
from sim_py import PurePythonMLP

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    from tensorflow.keras import models as keras_models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

GATE_NAMES = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR", "NOT A", "NOT B"]


def bits_from_int(x, n_bits=8):
    """Convert integer to list of bits (MSB first)."""
    return [(x >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]


class PyModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        outputs = [self.model.forward(x.tolist()) for x in np.asarray(X, dtype=np.float32)]
        return np.array(outputs, dtype=np.float32)


class NumpyModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.forward(np.asarray(X, dtype=np.float32))


class TorchModelWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, X):
        with torch.no_grad():
            X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
            output = self.model(X_tensor)
        return output.cpu().numpy()


class TensorFlowModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(np.asarray(X, dtype=np.float32), verbose=0)


def load_numpy_backend():
    model = NumpyMLP()
    weights_path = os.path.join(os.path.dirname(__file__), 'numpy_weights.npz')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)
    model.load(weights_path)
    return NumpyModelWrapper(model)


def load_python_backend():
    model = PurePythonMLP()
    weights_path = os.path.join(os.path.dirname(__file__), 'py_weights.json')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)
    model.load_from_json(weights_path)
    return PyModelWrapper(model)


def load_torch_backend():
    if not TORCH_AVAILABLE:
        raise ImportError('PyTorch is not installed')
    sim_torch = importlib.import_module('sim_torch')
    model = sim_torch.LogicGateSimulator()
    weights_path = sim_torch.MODEL_PATH
    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)

    state_dict = torch.load(weights_path, map_location='cpu')
    if all(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', 'network.'): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return TorchModelWrapper(model)


def load_tensorflow_backend():
    if not TF_AVAILABLE:
        raise ImportError('TensorFlow is not installed')
    sim_tensor = importlib.import_module('sim_tensor')
    weights_path = sim_tensor.MODEL_PATH
    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)
    model = keras_models.load_model(weights_path)
    return TensorFlowModelWrapper(model)


def evaluate_backend(name, predictor, batch_size=16384):
    total_samples = 256 * 256 * 8
    total_bits = total_samples * 8
    exact_matches = 0
    total_bits_correct = 0
    gate_exact = [0] * 8

    bit_repr = {v: bits_from_int(v) for v in range(256)}
    batch_inputs = []
    batch_truth = []
    batch_ops = []
    start_time = time.time()

    def flush_batch():
        nonlocal exact_matches, total_bits_correct, gate_exact
        X_batch = np.asarray(batch_inputs, dtype=np.float32)
        y_batch_true = np.asarray(batch_truth, dtype=np.int8)
        ops = np.asarray(batch_ops, dtype=np.int8)

        pred_probs = predictor(X_batch)
        pred_bits = (pred_probs > 0.5).astype(np.int8)
        exact = np.all(pred_bits == y_batch_true, axis=1)

        exact_matches += np.sum(exact)
        total_bits_correct += np.sum(pred_bits == y_batch_true)

        for op in range(8):
            gate_exact[op] += int(np.sum(exact[ops == op]))

        batch_inputs.clear()
        batch_truth.clear()
        batch_ops.clear()

    for a in range(256):
        a_bits = bit_repr[a]
        for b in range(256):
            b_bits = bit_repr[b]
            for op in range(8):
                batch_inputs.append(a_bits + b_bits + bits_from_int(op, n_bits=3))
                batch_truth.append(bits_from_int(calculate_raw_result(a, b, op)))
                batch_ops.append(op)
                if len(batch_inputs) >= batch_size:
                    flush_batch()

    if batch_inputs:
        flush_batch()

    elapsed = time.time() - start_time
    return {
        'name': name,
        'exact_matches': exact_matches,
        'total_bits_correct': total_bits_correct,
        'total_bits': total_bits,
        'total_samples': total_samples,
        'elapsed': elapsed,
        'gate_exact': gate_exact,
    }


def print_results(result):
    exact_acc = result['exact_matches'] / result['total_samples']
    bit_acc = result['total_bits_correct'] / result['total_bits']

    print('\n' + '=' * 60)
    print(f"BACKEND: {result['name']}")
    print('=' * 60)
    print(f"Total test samples      : {result['total_samples']:,}")
    print(f"Exact match count       : {result['exact_matches']:,} / {result['total_samples']:,}")
    print(f"Exact match accuracy    : {exact_acc:.6%}")
    print(f"Bit-wise correct bits   : {result['total_bits_correct']:,} / {result['total_bits']:,}")
    print(f"Bit-wise accuracy       : {bit_acc:.6%}")
    print(f"Time taken              : {result['elapsed']:.2f} seconds")

    print('\nPer-gate exact match accuracy:')
    for op, name in enumerate(GATE_NAMES):
        print(f"  {name:8s} : {result['gate_exact'][op]:7d} / {256*256:7d}  ({result['gate_exact'][op]/(256*256):.4%})")


def main():
    backends = []

    try:
        backends.append(('NumPy', load_numpy_backend()))
    except Exception as exc:
        print(f"SKIPPED NumPy backend: {exc}")

    try:
        backends.append(('Pure Python', load_python_backend()))
    except Exception as exc:
        print(f"SKIPPED Pure Python backend: {exc}")

    try:
        backends.append(('PyTorch', load_torch_backend()))
    except Exception as exc:
        print(f"SKIPPED PyTorch backend: {exc}")

    try:
        backends.append(('TensorFlow', load_tensorflow_backend()))
    except Exception as exc:
        print(f"SKIPPED TensorFlow backend: {exc}")

    if not backends:
        print('No backends are available for exhaustive testing.')
        return

    for name, wrapper in backends:
        print(f"\n=== Testing {name} backend ===")
        result = evaluate_backend(name, wrapper.predict)
        print_results(result)


if __name__ == '__main__':
    main()

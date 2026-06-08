import numpy as np

def generate_data(num_samples):
    """
    Generates training/testing data for 8-bit logic gates.
    Inputs (X): 19 bits (8 for A, 8 for B, 3 for Op)
    Outputs (y): 8 bits
    """
    X = np.random.randint(0, 2, (num_samples, 19)).astype(np.float32)
    y = np.zeros((num_samples, 8), dtype=np.float32)

    for i in range(num_samples):
        # Extract bits
        a_bits = X[i, :8].astype(np.uint8)
        b_bits = X[i, 8:16].astype(np.uint8)
        op_bits = X[i, 16:].astype(np.uint8)
        
        # Convert to integers
        a = int("".join(map(str, a_bits)), 2)
        b = int("".join(map(str, b_bits)), 2)
        op = int("".join(map(str, op_bits)), 2)
        
        # Perform operation
        res = calculate_raw_result(a, b, op)
        
        # Convert result back to bits
        y[i] = [int(c) for c in format(res, '08b')]

    return X, y

def calculate_raw_result(a, b, op_idx):
    """
    Standard bitwise operations for 8-bit integers.
    """
    if op_idx == 0:   # AND
        return a & b
    elif op_idx == 1: # OR
        return a | b
    elif op_idx == 2: # XOR
        return a ^ b
    elif op_idx == 3: # NAND
        return ~(a & b) & 0xFF
    elif op_idx == 4: # NOR
        return ~(a | b) & 0xFF
    elif op_idx == 5: # XNOR
        return ~(a ^ b) & 0xFF
    elif op_idx == 6: # NOT A
        return (~a) & 0xFF
    elif op_idx == 7: # NOT B
        return (~b) & 0xFF
    return 0

def parse_input(val_str):
    """
    Parses binary, hex, or decimal strings into 8-bit lists.
    """
    val_str = val_str.strip().lower()
    try:
        if val_str.startswith('0b'): v = int(val_str, 2)
        elif val_str.startswith('0x'): v = int(val_str, 16)
        else: v = int(val_str, 10)
        
        if 0 <= v <= 255:
            return [int(c) for c in format(v, '08b')]
        return None
    except:
        return None

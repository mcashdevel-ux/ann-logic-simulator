"""
DATA_UTILS.PY
=============
Utility functions for generating training data and processing logic gate inputs.

This module provides the foundation for the ANN Logic Simulator by:
1. Generating large datasets of bitwise operations for neural network training.
2. Implementing raw bitwise logic to serve as a 'ground truth' for verification.
3. Parsing user inputs in various formats (Binary, Hex, Decimal) into bit arrays.

Operation Mapping (3-bit Op Code):
[0] AND   [1] OR    [2] XOR   [3] NAND
[4] NOR   [5] XNOR  [6] NOT A [7] NOT B
"""

import numpy as np

def generate_data(num_samples):
    """
    Generates a synthetic dataset for training and testing 8-bit logic gate models.
    
    The input vector X consists of 19 bits:
    - Bits 0-7:  First 8-bit operand (A)
    - Bits 8-15: Second 8-bit operand (B)
    - Bits 16-18: 3-bit Operation Code (Op)
    
    The output vector y consists of 8 bits representing the result of the operation.

    Args:
        num_samples (int): Number of training examples to generate.

    Returns:
        tuple: (X, y) where X is (num_samples, 19) and y is (num_samples, 8), both float32.
    """
    # Initialize random binary inputs (0 or 1)
    X = np.random.randint(0, 2, (num_samples, 19)).astype(np.float32)
    y = np.zeros((num_samples, 8), dtype=np.float32)

    for i in range(num_samples):
        # Extract individual bit components from the input vector
        a_bits = X[i, :8].astype(np.uint8)   # Operand A bits
        b_bits = X[i, 8:16].astype(np.uint8) # Operand B bits
        op_bits = X[i, 16:].astype(np.uint8) # 3-bit Op Code
        
        # Convert bit arrays to decimal integers for calculation
        # Joining the bits as strings and parsing base-2
        a = int("".join(map(str, a_bits)), 2)
        b = int("".join(map(str, b_bits)), 2)
        op = int("".join(map(str, op_bits)), 2)
        
        # Calculate the ground truth result using standard bitwise logic
        res = calculate_raw_result(a, b, op)
        
        # Convert the decimal result back to an 8-bit binary list for the label
        # format(res, '08b') ensures exactly 8 bits with leading zeros
        y[i] = [int(c) for c in format(res, '08b')]

    return X, y

def calculate_raw_result(a, b, op_idx):
    """
    Performs standard 8-bit bitwise operations to provide ground truth.
    
    Args:
        a (int): First 8-bit operand (0-255).
        b (int): Second 8-bit operand (0-255).
        op_idx (int): Operation index (0-7).

    Returns:
        int: The result of the operation, masked to 8 bits (0-255).
    """
    if op_idx == 0:   # AND: 1 if both bits are 1
        return a & b
    elif op_idx == 1: # OR: 1 if at least one bit is 1
        return a | b
    elif op_idx == 2: # XOR: 1 if bits are different
        return a ^ b
    elif op_idx == 3: # NAND: Inverse of AND
        return ~(a & b) & 0xFF
    elif op_idx == 4: # NOR: Inverse of OR
        return ~(a | b) & 0xFF
    elif op_idx == 5: # XNOR: Inverse of XOR
        return ~(a ^ b) & 0xFF
    elif op_idx == 6: # NOT A: Flip all bits in A
        return (~a) & 0xFF
    elif op_idx == 7: # NOT B: Flip all bits in B
        return (~b) & 0xFF
    return 0

def parse_input(val_str):
    """
    Flexible parser that converts user strings into 8-bit lists.
    Supports:
    - Binary (e.g., '0b1010')
    - Hexadecimal (e.g., '0xFF')
    - Decimal (e.g., '255')
    
    Args:
        val_str (str): Input string from the user.

    Returns:
        list[int] | None: 8-bit list if valid, else None.
    """
    val_str = val_str.strip().lower()
    try:
        # Detect format based on prefix
        if val_str.startswith('0b'): 
            v = int(val_str, 2)
        elif val_str.startswith('0x'): 
            v = int(val_str, 16)
        else: 
            v = int(val_str, 10)
        
        # Check range for 8-bit unsigned integer
        if 0 <= v <= 255:
            return [int(c) for c in format(v, '08b')]
        return None
    except (ValueError, TypeError):
        # Return None if parsing fails
        return None

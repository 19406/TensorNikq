"""
    HELPERS
"""

# Convert Scalars into Tensors
def ensure_tensor(x):
    from .tensor import Tensor
    
    if isinstance(x, Tensor): return x
    return Tensor(x, requires_grad=False)

# Create a zeros Tensor
def zeros_like(data):
    if isinstance(data, list):
        return [zeros_like(x) for x in data]
    return 0.0

# Create an ones Tensor
def ones_like(data):
    if isinstance(data, list):
        return [ones_like(x) for x in data]
    return 1.0

# Calculate numel from shape
def numel(shape):
    if shape == (): return 1
    n = 1
    for s in shape: n *= s
    return n

# Infer shape for reshape
def infer_shape_for_reshape(new_shape, total):
    if new_shape.count(-1) > 1:
        raise ValueError("Only one -1 allowed in reshape!")

    known = 1
    for s in new_shape:
        if s != -1: known *= s

    if total % known != 0: raise ValueError("Cannot infer shape!")

    inferred = total // known
    return tuple(inferred if s == -1 else s for s in new_shape)

# Data Formatting
def format_data(data, digits=4):
    if isinstance(data, list):
        return [format_data(d, digits) for d in data]
    return round(data, digits)
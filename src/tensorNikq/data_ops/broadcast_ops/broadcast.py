"""
    BROADCAST
"""

# Broadcast shape
def broadcast_shape(shape_a, shape_b):
    from itertools import zip_longest

    result = []
    for dim_a, dim_b in zip_longest(reversed(shape_a), reversed(shape_b), fillvalue=1):
        if dim_a == dim_b: result.append(dim_a)
        elif dim_a == 1: result.append(dim_b)
        elif dim_b == 1: result.append(dim_a)
        else: raise ValueError("Cannot broadcast!")
    return tuple(reversed(result))

def _broadcast_batch_matmul_shape(shape_a, shape_b):
    from itertools import zip_longest

    result = []
    for dim_a, dim_b in zip_longest(reversed(shape_a), reversed(shape_b), fillvalue=1):
        if dim_a == dim_b: result.append(dim_a)
        elif dim_a == 1: result.append(dim_b)
        elif dim_b == 1: result.append(dim_a)
        else: raise ValueError("Batch dimensions can not be broadcast!")
    return tuple(reversed(result))

def infer_batch_matmul_shape(shape_a, shape_b):
    m, k1 = shape_a[-2:]
    k2, n = shape_b[-2:]

    if k1 != k2: raise ValueError("Shape mismatch for matmul!")

    batch_a = shape_a[:-2]
    batch_b = shape_b[:-2]

    batch = _broadcast_batch_matmul_shape(batch_a, batch_b)

    return batch + (m, n)

# Expand (broadcast) data to new shape
def broadcast_to(data, current_shape, target_shape):
    if current_shape == target_shape: return data

    # Insert some dimension 1 if not adequate
    if len(current_shape) < len(target_shape):
        for _ in range(len(target_shape) - len(current_shape)): data = [data]
        current_shape = (1,) * (len(target_shape) - len(current_shape)) + current_shape

    if len(target_shape) == 0: return data

    if current_shape[0] == target_shape[0]:
        return [broadcast_to(d, current_shape[1:], target_shape[1:]) for d in data]

    if current_shape[0] == 1:
        return [
            broadcast_to(data[0], current_shape[1:], target_shape[1:]) 
            for _ in range(target_shape[0])
        ]

    raise ValueError("Cannot broadcast!")
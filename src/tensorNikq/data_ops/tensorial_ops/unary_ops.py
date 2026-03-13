import itertools

# ---------- Transposition ----------

def elementwise_transpose_2D(data):
    if (
        not isinstance(data, list) or
        not isinstance(data[0], list) or
        isinstance(data[0][0], list)
    ):
        raise ValueError("Transpose only supports 2D tensor!")
    return [list(row) for row in zip(*data)]

def transpose_last2(data):
    if not isinstance(data[0][0], list): return elementwise_transpose_2D(data)
    return [transpose_last2(sub) for sub in data]

def _get_shape(data):
    shape = []
    while isinstance(data, list):
        shape.append(len(data))
        data = data[0]
        
    return tuple(shape)

def _get_item(data, idx):
    for i in idx: data = data[i]
    return data

def _set_item(data, idx, value):
    for i in idx[:-1]: data = data[i]
    data[idx[-1]] = value
    
def _create_tensor(shape):
    if len(shape) == 1: return [None] * shape[0]
    return [_create_tensor(shape[1:]) for _ in range(shape[0])]

def _all_indices(shape):
    return itertools.product(*[range(s) for s in shape])

def transpose_nd(data, axes=None):
    shape = _get_shape(data)
    n = len(shape)
    
    if axes is None: axes = tuple(reversed(range(n)))
    
    new_shape = tuple(shape[i] for i in axes)
    result = _create_tensor(new_shape)
    
    for new_idx in _all_indices(new_shape):
        old_idx = [0]*n
        for i, ax in enumerate(axes):
            old_idx[ax] = new_idx[i]
            
        value = _get_item(data, old_idx)
        _set_item(result, new_idx, value)
        
    return result

# ---------- Reshape ----------

# Tensor Flattening
def flatten(data):    
    if isinstance(data, list):
        result = []
        for d in data: result.extend(flatten(d))
        return result
    return [data]

# Build new Tensor from 1D Tensor
def build_from_flat(flat, shape):
    if len(shape) == 0: return flat[0]

    size = shape[0]
    rest = shape[1:]
    step = len(flat) // size

    result = []
    for i in range(size):
        chunk = flat[i * step:(i + 1) * step]
        result.append(build_from_flat(chunk, rest))

    return result

# Reshape
def elementwise_reshape(data, new_shape):
    flat = flatten(data)
    return build_from_flat(flat, new_shape)
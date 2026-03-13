from ..math_ops.binary_ops import elementwise_add

def shape_after_sum(shape, dim=None, keepdim=False):
    if dim is None:
        if keepdim: return tuple(1 for _ in shape)
        return ()

    if isinstance(dim, int): dim = [dim]

    dim = [d if d >= 0 else d + len(shape) for d in dim]

    new_shape = []

    for i, s in enumerate(shape):
        if i in dim:
            if keepdim: new_shape.append(1)
        else: new_shape.append(s)

    return tuple(new_shape)

def _get_ndim(data):
    ndim = 0
    while isinstance(data, list):
        ndim += 1
        data = data[0]
    return ndim

# Sum along a specific dim
def sum_dim(data, dim=None, keepdim=False):
    # Sum all
    if dim is None:
        if not isinstance(data, list):
            result = data
            ndim = 0
        else:
            result = sum_dim(data[0])
            for d in data[1:]:
                result = elementwise_add(result, sum_dim(d))
            ndim = _get_ndim(data)

        if keepdim:
            for _ in range(ndim):
                result = [result]

        return result
    
    if dim == 0:
        result = data[0]
        for d in data[1:]: result = elementwise_add(result, d)

        if keepdim: return [result]
        return result

    return [sum_dim(sub, dim-1, keepdim) for sub in data]

# Reduce data to original shape
def reduce_to_shape(data, current_shape, target_shape):
    # Reduce current shape until it is equal to target shape by sum the outermost dim
    while len(current_shape) > len(target_shape):
        data = sum_dim(data, dim=0)
        current_shape = current_shape[1:]

    # The initial value of dim was 1, but it was later changed by broadcasting
    for i, (current, target) in enumerate(zip(current_shape, target_shape)):
        if target == 1 and current != 1:
            data = sum_dim(data, dim=i, keepdim=True)
            current_shape = (current_shape[:i] + (1,) + current_shape[i+1:])

    return data
from ..arith_logic_ops.rel_ops import elementwise_max

def _where(condition, a, b):
    if not isinstance(condition, list):
        return a if condition else b
    return [_where(c, ai, bi) for c, ai, bi in zip(condition, a, b)]

def masked_fill(x, mask, value):
    return _where(mask, value, x)

def _get_ndim(data):
    ndim = 0
    while isinstance(data, list):
        ndim += 1
        data = data[0]
    return ndim

def max_dim(data, dim=None, keepdim=False):
    if dim is None:
        if not isinstance(data, list):
            result = data
            ndim = 0
        else:
            result = max_dim(data[0])
            for d in data[1:]: result = elementwise_max(result, max_dim(d))
            ndim = _get_ndim(data)
        
        if keepdim:
            for _ in range(ndim): result = [result]
        return result

    if dim == 0:
        result = data[0]
        for d in data[1:]: result = elementwise_max(result, d)
        
        if keepdim: return [result]
        return result

    return [max_dim(sub, dim-1, keepdim) for sub in data]
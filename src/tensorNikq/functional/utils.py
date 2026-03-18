from ..tensor import Tensor
from ..data_ops import sum_dim, elementwise_exp
import random

def randint(low, high, shape):
    data = [
        [random.randint(low, high-1) for _ in range(shape[1])]
        for _ in range(shape[0])
    ]
    
    return Tensor(data)

def stack(tensors, dim=0):
    data = [t.data for t in tensors]
    return Tensor(data)

def cat(tensors, dim=0):    
    data = []
    for t in tensors: data.extend(t.data)
    return Tensor(data)

def randn(shape):
    data = [
        [random.gauss(0,1) for _ in range(shape[1])]
        for _ in range(shape[0])
    ]
    
    return Tensor(data)

def zeros(shape):    
    return Tensor([[0]*shape[1] for _ in range(shape[0])])

def ones(shape):
    return Tensor([[1]*shape[1] for _ in range(shape[0])])

def tril(n):    
    m = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(1 if j <= i else 0)
        m.append(row)
    return Tensor(m)

def _where(condition, a, b):
    if not isinstance(condition, list):
        return a if condition else b
    return [_where(c, ai, bi) for c, ai, bi in zip(condition, a, b)]

def _equal(a, b):
    if not isinstance(a, list): return a == b
    return [_equal(x, b) for x in a]

def masked_fill(x, mask, value):
    cond = _equal(mask, 0)
    return _where(cond, value, x)

def softmax(x, dim=-1):
    e = elementwise_exp(x)
    return e / sum_dim(e, dim, keepdim=True)
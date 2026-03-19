from ..tensor import Tensor
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

def rand_like(x): return Tensor(randn(*x.shape))

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
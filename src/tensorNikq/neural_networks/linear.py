from ..tensor import Tensor
from ..functional import randn, zeros
from .module import Module

"""
    Linear Neural Network Class
"""

class Linear(Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.weight = Tensor(randn((in_dim, out_dim)), requires_grad=True)

        if bias: self.bias = Tensor(zeros(out_dim), requires_grad=True)
        else: self.bias = None
        
    def __call__(self, x):
        y = x @ self.weight
        if self.bias is not None: y = y + self.bias
        return y
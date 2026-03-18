from ..functional import randn

"""
    Linear Neural Network Class
"""

class Linear:
    def __init__(self, in_dim, out_dim, bias=True):
        self.W = randn((in_dim, out_dim))
        self.b = randn((out_dim,)) if bias else None
        
    def __call__(self, x):
        y = x @ self.W
        if self.b is not None: y = y + self.b
        return y
        
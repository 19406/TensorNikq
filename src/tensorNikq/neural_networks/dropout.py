from .module import Module
from ..functional import rand_like

class Dropout(Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training: return x
        mask = (rand_like(x) > self.p)
        return x * mask / (1 - self.p)
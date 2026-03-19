from ..tensor import Tensor

class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "training", True)
        
    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)
            return
        object.__setattr__(self, name, value)
        
    def parameters(self):
        params = list(self._parameters.value())
        for m in self._modules.values(): params += m.parameters()
        return params
    
    def register_buffer(self, name, tensor):
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def train(self):
        self.training = True
        for m in self._modules.values(): m.train()
        
    def eval(self):
        self.training = False
        for m in self._modules.values(): m.eval()
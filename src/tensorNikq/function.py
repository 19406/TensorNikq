"""
    TENSOR LAYER
"""

# ---------- Context ----------

class Context:
    def __init__(self):
        self.saved = ()

    def save_for_backward(self, *info):
        self.saved = info

# ---------- Function Base ----------

class Function:
    differentiable = True
    
    @classmethod
    def apply(cls, *tensors):
        from .tensor import Tensor
        
        ctx = Context()
        
        out_data = cls.forward(ctx, *tensors)
        requires_grad = (
            cls.differentiable and
            any(
                t.requires_grad if isinstance(t, Tensor) else False
                for t in tensors
            )
        )

        out = Tensor(out_data, requires_grad)

        if requires_grad:
            out._ctx = ctx
            out._op = cls
            out._prev = tuple(t for t in tensors if isinstance(t, Tensor))

        return out
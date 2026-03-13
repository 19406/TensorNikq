from ..function import Function
from ..data_ops import *

# ---------- Negation ----------

class Neg(Function):
    @staticmethod
    def forward(ctx, a): return elementwise_neg(a.data)

    @staticmethod
    def backward(ctx, grad_out):
        # grad_a = - grad_out
        return elementwise_neg(grad_out)

# ---------- Sum ----------

class Sum(Function):
    @staticmethod
    def forward(ctx, a, dim=None, keepdim=False):
        original_shape = a.shape
        out_shape = shape_after_sum(a.shape, dim, keepdim)
        ctx.save_for_backward(original_shape, out_shape)
        return sum_dim(a.data, dim, keepdim)

    @staticmethod
    def backward(ctx, grad_out):
        original_shape, out_shape = ctx.saved
        return broadcast_to(grad_out, out_shape, original_shape)
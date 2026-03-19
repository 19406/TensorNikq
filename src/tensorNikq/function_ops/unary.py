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

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.a_data = a.data
        return elementwise_exp(a.data)
    
    @staticmethod
    def backward(ctx, grad_out):
        a_data = ctx.a_data
        
        # grad_a = grad_out * out
        return elementwise_mul(
            grad_out,
            elementwise_exp(a_data)
        )

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

# ---------- Max ----------
    
class Max(Function):
    differentiable = False
    
    @staticmethod
    def forward(ctx, a, dim=None, keepdim=False):
        return max_dim(a.data, dim, keepdim)
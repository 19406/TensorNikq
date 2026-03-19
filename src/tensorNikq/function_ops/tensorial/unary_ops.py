from ...function import Function
from ...data_ops import elementwise_transpose_2D, elementwise_reshape

# ---------- Transposition ----------

class Transpose(Function):
    @staticmethod
    def forward(ctx, a): return elementwise_transpose_2D(a.data)

    @staticmethod
    def backward(ctx, grad_out):
        # grad_a = grad_out.T
        return elementwise_transpose_2D(grad_out)

# ---------- Reshape ----------

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, new_shape):
        ctx.original_shape = a.shape
        
        return elementwise_reshape(a.data, new_shape)

    @staticmethod
    def backward(ctx, grad_out):
        original_shape = ctx.original_shape
        
        return elementwise_reshape(grad_out, original_shape)
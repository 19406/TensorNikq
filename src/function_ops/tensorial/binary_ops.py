from ...function import Function
from ...data_ops import *

# ---------- Matrix Multiplication ----------

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        shape_a, shape_b = a.shape, b.shape
        out_shape = infer_matmul_shape(shape_a, shape_b)
        
        a_data = broadcast_to(a.data, shape_a, out_shape[:-2] + shape_a[-2:])
        b_data = broadcast_to(b.data, shape_b, out_shape[:-2] + shape_b[-2:])
        ctx.save_for_backward(a_data, b_data, shape_a, shape_b, out_shape)
        
        return elementwise_matmul(a_data, b_data)

    @staticmethod
    def backward(ctx, grad_out):
        a_data, b_data, shape_a, shape_b, out_shape = ctx.saved        
        
        # vector . vector
        if len(shape_a) == 1 and len(shape_b) == 1:
            grad_a = [grad_out * x for x in b_data]
            grad_b = [grad_out * x for x in a_data]

        # matrix @ vector
        elif len(shape_a) == 2 and len(shape_b) == 1:
            grad_a = outer(grad_out, b_data)
            grad_b = elementwise_matmul(elementwise_transpose_2D(a_data), grad_out)

        # vector @ matrix
        elif len(shape_a) == 1 and len(shape_b) == 2:
            grad_a = elementwise_matmul(grad_out, elementwise_transpose_2D(b_data))
            grad_b = outer(a_data, grad_out)

        # matrix @ matrix
        else:
            grad_a = elementwise_matmul(grad_out, transpose_last2(b_data))
            grad_b = elementwise_matmul(transpose_last2(a_data), grad_out)
        
        return (
            # grad_a = grad_out @ b.T
            reduce_to_shape(grad_a, out_shape, shape_a),

            # grad_b = a.T @ grad_out
            reduce_to_shape(grad_b, out_shape, shape_b),
        )
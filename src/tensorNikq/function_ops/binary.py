from ..function import Function
from ..data_ops import *

# ---------- Addition ----------
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        shape_a, shape_b = a.shape, b.shape
        out_shape = broadcast_shape(shape_a, shape_b)
        ctx.save_for_backward(shape_a, shape_b, out_shape)

        a_data = broadcast_to(a.data, shape_a, out_shape)
        b_data = broadcast_to(b.data, shape_b, out_shape)
        return elementwise_add(a_data, b_data)
        
    @staticmethod
    def backward(ctx, grad_out):
        shape_a, shape_b, out_shape = ctx.saved
        return (
            # grad_a = grad_b = grad_out
            reduce_to_shape(grad_out, out_shape, shape_a),
            reduce_to_shape(grad_out, out_shape, shape_b)
        )

# ---------- Subtraction ----------
class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        shape_a, shape_b = a.shape, b.shape
        out_shape = broadcast_shape(shape_a, shape_b)
        ctx.save_for_backward(shape_a, shape_b, out_shape)

        a_data = broadcast_to(a.data, shape_a, out_shape)
        b_data = broadcast_to(b.data, shape_b, out_shape)
        return elementwise_sub(a_data, b_data)
        
    @staticmethod
    def backward(ctx, grad_out):
        shape_a, shape_b, out_shape = ctx.saved
        return (
            # grad_a = grad_out
            reduce_to_shape(grad_out, out_shape, shape_a),

            # grad_b = - grad_out
            reduce_to_shape(elementwise_neg(grad_out), out_shape, shape_b)
        )

# ---------- Multiplication ----------
class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        shape_a, shape_b = a.shape, b.shape
        out_shape = broadcast_shape(shape_a, shape_b)
        
        a_data = broadcast_to(a.data, shape_a, out_shape)
        b_data = broadcast_to(b.data, shape_b, out_shape)
        ctx.save_for_backward(a_data, b_data, shape_a, shape_b, out_shape)
        
        return elementwise_mul(a_data, b_data)

    @staticmethod
    def backward(ctx, grad_out):
        a_data, b_data, shape_a, shape_b, out_shape = ctx.saved
        return (
            # grad_a = grad_out * b
            reduce_to_shape(elementwise_mul(grad_out, b_data), out_shape, shape_a),

            # grad_b = grad_out * a
            reduce_to_shape(elementwise_mul(grad_out, a_data), out_shape, shape_b)
        )

# ---------- Division ----------
class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        shape_a, shape_b = a.shape, b.shape
        out_shape = broadcast_shape(shape_a, shape_b)
        
        a_data = broadcast_to(a.data, shape_a, out_shape)
        b_data = broadcast_to(b.data, shape_b, out_shape)
        ctx.save_for_backward(a_data, b_data, shape_a, shape_b, out_shape)

        return elementwise_div(a_data, b_data)

    @staticmethod
    def backward(ctx, grad_out):
        a_data, b_data, shape_a, shape_b, out_shape = ctx.saved
        return (
            # grad_a = grad_out / b
            reduce_to_shape(elementwise_div(grad_out, b_data), out_shape, shape_a),
            
            # grad_b = - grad_out * a / (b * b)
            reduce_to_shape(
                elementwise_neg(
                    elementwise_div(
                        elementwise_mul(grad_out, a_data),
                        elementwise_square(b_data)
                    )
                ), out_shape, shape_b
            )
        )
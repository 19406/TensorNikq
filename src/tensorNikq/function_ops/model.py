from ..function import Function
from ..data_ops import *

class MaskedFill(Function):
    @staticmethod
    def forward(ctx, a, mask, value):
        shape_a, shape_mask = a.shape, mask.shape
        out_shape = broadcast_shape(shape_a, shape_mask)
        
        a_data = broadcast_to(a.data, shape_a, out_shape)
        mask_data = broadcast_to(mask.data, shape_mask, out_shape)
        ctx.save_for_backward(shape_a, out_shape, mask_data)
        
        return masked_fill(a_data, mask_data, value)
    
    @staticmethod
    def backward(ctx, grad_out):
        shape_a, out_shape, mask_data = ctx.saved
        
        # grad_a = grad_out * (mask == 0)
        return reduce_to_shape(
            elementwise_mul(
                grad_out,
                elementwise_not(mask_data)
            )
            , out_shape, shape_a)
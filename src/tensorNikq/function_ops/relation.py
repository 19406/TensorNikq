from ..function import Function
from ..data_ops import *

class GreaterThan(Function):
    differentiable = False
    
    @staticmethod
    def forward(ctx, a, b):
        shape_a, shape_b = a.shape, b.shape
        out_shape = broadcast_shape(shape_a, shape_b)

        a_data = broadcast_to(a.data, shape_a, out_shape)
        b_data = broadcast_to(b.data, shape_b, out_shape)
        return elementwise_gt(a_data, b_data)
    
class GreaterOrEqual(Function):
    differentiable = False
    
    @staticmethod
    def forward(ctx, a, b):
        shape_a, shape_b = a.shape, b.shape
        out_shape = broadcast_shape(shape_a, shape_b)

        a_data = broadcast_to(a.data, shape_a, out_shape)
        b_data = broadcast_to(b.data, shape_b, out_shape)
        return elementwise_ge(a_data, b_data)
    
class LessThan(Function):
    differentiable = False
    
    @staticmethod
    def forward(ctx, a, b):
        shape_a, shape_b = a.shape, b.shape
        out_shape = broadcast_shape(shape_a, shape_b)

        a_data = broadcast_to(a.data, shape_a, out_shape)
        b_data = broadcast_to(b.data, shape_b, out_shape)
        return elementwise_lt(a_data, b_data)
    
class LessOrEqual(Function):
    differentiable = False
    
    @staticmethod
    def forward(ctx, a, b):
        shape_a, shape_b = a.shape, b.shape
        out_shape = broadcast_shape(shape_a, shape_b)

        a_data = broadcast_to(a.data, shape_a, out_shape)
        b_data = broadcast_to(b.data, shape_b, out_shape)
        return elementwise_le(a_data, b_data)
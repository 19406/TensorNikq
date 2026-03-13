from .data_ops import elementwise_add, broadcast_shape, broadcast_to
from .utils import ones_like

# Broadcast tensors
def broadcast(a, b):
    from .tensor import Tensor
    
    target_shape = broadcast_shape(a.shape, b.shape)

    a_data = broadcast_to(a.data, a.shape, target_shape)
    b_data = broadcast_to(b.data, b.shape, target_shape)

    return a_data, b_data

# Backward engine
def backward(tensor, grad=None):
    from .tensor import Tensor
    
    """
    Skip if no gradient required
    """
    if not tensor.requires_grad: return

    """
    Root gradient:
    If tensor is scalar -> grad = 1.0
    Else -> grad = ones_like raw data
    """
    if grad is None:
        grad_data = 1.0 if tensor.shape == () else ones_like(tensor.data)
    else:
        grad_data = grad.data if isinstance(grad, Tensor) else grad
    
    tensor.grad = Tensor(grad_data, requires_grad=False)

    """
       _ topo is list of tensors in forward order
       _ visited is list of tensors have been visited
    """
    topo = []
    visited = set()

    """
        At tensor T:
            If T haven't been visited before: add T to visited
            Recursive step:
                Call all tensors directly computing to T
            Append T to topo
    """
    def build_topo(t):
        if t not in visited:
            visited.add(t)
            for child in t._prev:
                build_topo(child)
            topo.append(t)

    build_topo(tensor)

    """
        Traverse in backward order (however, this means following the backward tree):
            _ Apply the backward function of each tensor
            _ The gradient of each tensor is accumulated through backward functions
            _ At tensor T: T.grad = ∂R/∂T where R is the tensor performing backward process
    """
    for t in reversed(topo):
        if t._op is None: continue

        grad_datas = t._op.backward(t._ctx, t.grad.data)
        if not isinstance(grad_datas, tuple): grad_datas = (grad_datas,)

        grads = tuple(Tensor(g, requires_grad=False) for g in grad_datas)

        for p, g in zip(t._prev, grads):
            if not p.requires_grad or g is None: continue

            # Accumulate raw gradient            
            if p.grad is None: p.grad = g
            else:
                p_grad_data, g_data = broadcast(p.grad, g)
                new_grad_data = elementwise_add(p_grad_data, g_data)
                p.grad = Tensor(new_grad_data, requires_grad=False)
from .engine import backward
from . import utils
from .function_ops import *

class Tensor:
    def __init__(self, data=None, requires_grad=False):
        """
        There are some cases:
            _ Scalar                  : int | float
            _ 1D Tensor               : list
            _ Multi-Dimensional Tensor: list of list
        """

        self.data = data        
        self.shape = self._infer_shape(data)

        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None
        self._op = None
        self._prev = ()
        

    # ---------- Inferencing tensor shape ----------

    # Inference shape
    def _infer_shape(self, data=None):
        if data is None: raise ValueError("Data is required!")
        
        if not isinstance(data, list):
            if isinstance(data, (int, float)): return ()
            else: raise TypeError(f"'{type(data).__name__}' is not supported! Only support 'int' and 'float'.")
        
        if isinstance(data, list):
            if len(data) == 0: return (0,)

            first_shape = self._infer_shape(data[0])
            for item in data[1:]:
                if self._infer_shape(item) != first_shape:
                    raise ValueError("Ragged tensor is not supported!")
            
            return (len(data),) + first_shape

    # Categorize tensors
    @property
    # The number of dimensions
    def ndim(self): return len(self.shape)

    @property
    # Scalar
    def is_scalar(self): return self.shape == ()

    @property
    # Vector (1D)
    def is_vector(self): return self.ndim == 1

    @property
    # Matrix (2D)
    def is_matrix(self): return self.ndim == 2

    # ---------- Representing ----------
    
    # Shape Formatting
    def _format_shape(self):
        if len(self.shape) == 1:
            return f"({self.shape[0]})"
        return str(self.shape)
    
    def __repr__(self):
        return f"Tensor(data={utils.format_data(self.data)}, shape={self._format_shape()})"
    
    # ---------- Equality ----------

    # (Hasn't been done yet!)
    
    # ---------- Indexing ----------
    def __getitem__(self, index):
        if self.shape == (): raise IndexError("Cannot index a 0-dim tensor!")
        t = Tensor(self.data[index], self.requires_grad)
        t._prev = {self}
        return t

    # ---------- Element-wise Operatations ----------
    
    # Negation
    def __neg__(self):
        return Neg.apply(self)
    
    # Addition
    def __add__(self, other):
        other = utils.ensure_tensor(other)
        return Add.apply(self, other)
        
    def __radd__(self, other): return self.__add__(other)

    # Subtraction
    def __sub__(self, other):
        other = utils.ensure_tensor(other)
        return Sub.apply(self, other)

    def __rsub__(self, other):
        other = utils.ensure_tensor(other)
        return other.__sub__(self)

    # Multiplication
    def __mul__(self, other):
        other = utils.ensure_tensor(other)
        return Mul.apply(self, other)

    def __rmul__(self, other): return self.__mul__(other)

    # Division
    def __truediv__(self, other):
        other = utils.ensure_tensor(other)
        return Div.apply(self, other)

    def __rtruediv__(self, other):
        other = utils.ensure_tensor(other)
        return other.__truediv__(self)

    # ---------- Matrix Operatations ----------

    # Matrix Transposition
    def transpose(self): return Transpose.apply(self) 

    @property
    def T(self): return self.transpose()
    
    # Matrix Multiplication
    def matmul(self, other):
        other = utils.ensure_tensor(other)
        if self.ndim == 0 and other.ndim == 0: raise ValueError("Both operands are scalar tensors!")
        elif self.ndim == 0: raise ValueError("The first operand is a scalar tensor!")
        elif other.ndim == 0: raise ValueError("The second operand is a scalar tensor!")
        return MatMul.apply(self, other)

    def __matmul__(self, other): return self.matmul(other)

    def __rmatmul__(self, other):
        other = utils.ensure_tensor(other)
        return other.matmul(self)

    def sum(self, dim=None, keepdim=False): return Sum.apply(self, dim, keepdim)

    # ---------- Shape Operations ----------

    # Tensor Reshape
    def reshape(self, *new_shape):
        total = utils.numel(self.shape)

        # Case -1
        if -1 in new_shape:
            new_shape = utils.infer_shape_for_reshape(new_shape, total)

        # Validate
        if utils.numel(new_shape) != total:
            raise ValueError(f"Cannot reshape tensor of size {total} into shape {new_shape}!")

        return Reshape.apply(self, new_shape)

    # ---------- Autograd ----------
    
    def backward(self, grad=None):
        backward(self, grad)
        
    # ---------- Hardware ----------
    
    def to(self, device):
        if device != "cpu": raise NotImplementedError("Only CPU supported!")
        return self
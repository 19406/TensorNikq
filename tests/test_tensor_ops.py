from src import Tensor
import pytest
import re

# ----- UNARY -----

def test_Neg_scalar():
    a = Tensor(7)
    b = -a
    assert str(b) == "Tensor(data=-7, shape=())"
    
def test_Neg_vector():
    a = Tensor([7, -56, 18])
    b = -a
    assert str(b) == "Tensor(data=[-7, 56, -18], shape=(3))"
    
def test_Neg_matrix():
    a = Tensor([[7, -56, 18], [-5, -13, 6]])
    b = -a
    assert str(b) == "Tensor(data=[[-7, 56, -18], [5, 13, -6]], shape=(2, 3))"
    
def test_Neg_multidim():
    a = Tensor([[[7, -56, 18], [-5, -13, 6]], [[-12, -61, 8], [73, -34, 49]], [[23, -1, -59], [38, -29, 10]]])
    b = -a
    assert str(b) == "Tensor(data=[[[-7, 56, -18], [5, 13, -6]], [[12, 61, -8], [-73, 34, -49]], [[-23, 1, 59], [-38, 29, -10]]], shape=(3, 2, 3))"
    
def test_Sum_no_keepdim():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a.sum()
    assert str(b) == "Tensor(data=21, shape=())"
    
def test_Sum_keepdim():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a.sum(keepdim=True)
    assert str(b) == "Tensor(data=[[21]], shape=(1, 1))"

def test_Sum_dim_0_no_keepdim():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a.sum(0)
    assert str(b) == "Tensor(data=[5, 7, 9], shape=(3))"

def test_Sum_dim_0_keepdim():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a.sum(0, keepdim=True)
    assert str(b) == "Tensor(data=[[5, 7, 9]], shape=(1, 3))"

# ----- BINARY -----

def test_Add_scalar():
    a = Tensor(1)
    b = Tensor(2)
    c = a + b
    assert str(c) == "Tensor(data=3, shape=())"

def test_Add_vector():
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    assert str(c) == "Tensor(data=[5, 7, 9], shape=(3))"
    
def test_Add_matrix():
    a = Tensor([[1, 2], [3, 4], [5, 6]])
    b = Tensor([[7, 8], [9, 10], [11, 12]])
    c = a + b
    assert str(c) == "Tensor(data=[[8, 10], [12, 14], [16, 18]], shape=(3, 2))"
    
def test_Add_multidim():
    a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    b = Tensor([[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]])
    c = a + b
    assert str(c) == "Tensor(data=[[[14, 16], [18, 20]], [[22, 24], [26, 28]], [[30, 32], [34, 36]]], shape=(3, 2, 2))"
    
def test_Sub_scalar():
    a = Tensor(1)
    b = Tensor(2)
    c = a - b
    assert str(c) == "Tensor(data=-1, shape=())"

def test_Sub_vector():
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a - b
    assert str(c) == "Tensor(data=[-3, -3, -3], shape=(3))"
    
def test_Sub_matrix():
    a = Tensor([[1, 2], [3, 4], [5, 6]])
    b = Tensor([[7, 8], [9, 10], [11, 12]])
    c = a - b
    assert str(c) == "Tensor(data=[[-6, -6], [-6, -6], [-6, -6]], shape=(3, 2))"
    
def test_Sub_multidim():
    a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    b = Tensor([[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]])
    c = a - b
    assert str(c) == "Tensor(data=[[[-12, -12], [-12, -12]], [[-12, -12], [-12, -12]], [[-12, -12], [-12, -12]]], shape=(3, 2, 2))"
    
def test_Mul_scalar():
    a = Tensor(1)
    b = Tensor(2)
    c = a * b
    assert str(c) == "Tensor(data=2, shape=())"

def test_Mul_vector():
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a * b
    assert str(c) == "Tensor(data=[4, 10, 18], shape=(3))"
    
def test_Mul_matrix():
    a = Tensor([[1, 2], [3, 4], [5, 6]])
    b = Tensor([[7, 8], [9, 10], [11, 12]])
    c = a * b
    assert str(c) == "Tensor(data=[[7, 16], [27, 40], [55, 72]], shape=(3, 2))"
    
def test_Mul_multidim():
    a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    b = Tensor([[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]])
    c = a * b
    assert str(c) == "Tensor(data=[[[13, 28], [45, 64]], [[85, 108], [133, 160]], [[189, 220], [253, 288]]], shape=(3, 2, 2))"
    
def test_Div_scalar():
    a = Tensor(1)
    b = Tensor(2)
    c = a / b
    assert str(c) == "Tensor(data=0.5, shape=())"

def test_Div_vector():
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a / b
    assert str(c) == "Tensor(data=[0.25, 0.4, 0.5], shape=(3))"
    
def test_Div_matrix():
    a = Tensor([[1, 2], [3, 4], [5, 6]])
    b = Tensor([[7, 8], [9, 10], [11, 12]])
    c = a / b
    assert str(c) == "Tensor(data=[[0.1429, 0.25], [0.3333, 0.4], [0.4545, 0.5]], shape=(3, 2))"
    
def test_Div_multidim():
    a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    b = Tensor([[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]])
    c = a / b
    assert str(c) == "Tensor(data=[[[0.0769, 0.1429], [0.2, 0.25]], [[0.2941, 0.3333], [0.3684, 0.4]], [[0.4286, 0.4545], [0.4783, 0.5]]], shape=(3, 2, 2))"

# ----- TENSORIAL -----

# --- Unary ---

def test_Transpose():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a.transpose()
    assert str(b) == "Tensor(data=[[1, 4], [2, 5], [3, 6]], shape=(3, 2))"
    
def test_T():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a.T
    assert str(b) == "Tensor(data=[[1, 4], [2, 5], [3, 6]], shape=(3, 2))"
    
def test_Reshape():
    a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    b = a.reshape(6, 2)
    assert str(b) == "Tensor(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], shape=(6, 2))"
    
def test_Reshape_infer_shape():
    a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    b = a.reshape(3, -1)
    assert str(b) == "Tensor(data=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], shape=(3, 4))"
    
# --- Binary ---

def test_MatMul_vector_vector():
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a @ b
    assert str(c) == "Tensor(data=32, shape=())"
    
def test_MatMul_matrix_vector():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = Tensor([4, 5, 6])
    c = a @ b
    assert str(c) == "Tensor(data=[32, 77], shape=(2))"
    
def test_MatMul_vector_matrix():
    a = Tensor([1, 2, 3])
    b = Tensor([[1, 2], [3, 4], [5, 6]])
    c = a @ b
    assert str(c) == "Tensor(data=[22, 28], shape=(2))"
    
def test_MatMul_matrix_matrix():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = Tensor([[1, 2], [3, 4], [5, 6]])
    c = a @ b
    assert str(c) == "Tensor(data=[[22, 28], [49, 64]], shape=(2, 2))"
    
def test_MatMul_batch_matmul():
    a = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9],[10, 11, 12]]])
    b = Tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
    c = a @ b
    assert str(c) == "Tensor(data=[[[38, 44, 50, 56], [83, 98, 113, 128]], [[416, 440, 464, 488], [569, 602, 635, 668]]], shape=(2, 2, 4))"
    
def test_Matmul_by_matmul_function():
    a = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9],[10, 11, 12]]])
    b = Tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
    c = a.matmul(b)
    assert str(c) == "Tensor(data=[[[38, 44, 50, 56], [83, 98, 113, 128]], [[416, 440, 464, 488], [569, 602, 635, 668]]], shape=(2, 2, 4))"

# ----- ERROR -----

def test_error_Transpose_scalar():
    with pytest.raises(ValueError, match="Transpose only supports 2D tensor!"):
        a = Tensor(1)
        b = a.transpose()    

def test_error_Transpose_vector():
    with pytest.raises(ValueError, match="Transpose only supports 2D tensor!"):
        a = Tensor([1, 2, 3])
        b = a.transpose()    
        
def test_error_Transpose_multidim_tensor():
    with pytest.raises(ValueError, match="Transpose only supports 2D tensor!"):
        a = Tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]])
        b = a.transpose()
        
def test_error_Reshape_wrong_shape():
    with pytest.raises(ValueError, match=re.escape("Cannot reshape tensor of size 12 into shape (5, 3)!")):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        b = a.reshape(5, 3)

def test_error_Reshape_multiple_minus_ones():
    with pytest.raises(ValueError, match="Only one -1 allowed in reshape!"):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        b = a.reshape(-1, -1)
        
def test_error_Reshape_auto_infer_shape():
    with pytest.raises(ValueError, match="Cannot infer shape!"):
        a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        b = a.reshape(5, -1)
        
def test_error_MatMul_both_are_scalars():
        with pytest.raises(ValueError, match="Both operands are scalar tensors!"):
            a = Tensor(1)
            b = Tensor(2)
            c = a @ b
            
def test_error_MatMul_first_is_scalar():
        with pytest.raises(ValueError, match="The first operand is a scalar tensor!"):
            a = Tensor(1)
            b = Tensor([1, 2, 3])
            c = a @ b
            
def test_error_MatMul_second_is_scalar():
        with pytest.raises(ValueError, match="The second operand is a scalar tensor!"):
            a = Tensor([1, 2, 3])
            b = Tensor(1)
            c = a @ b
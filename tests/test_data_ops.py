from src.tensorNikq.data_ops import *
import math
import pytest

# ----- UNARY -----

def test_neg():
    a = [1, 2, 3]
    b = elementwise_neg(a)
    assert b == [-1, -2, -3]
    
def test_square():
    a = [1, 2, 3]
    b = elementwise_square(a)
    assert b == [1, 4, 9]
    
def test_sqrt():
    a = [1, 9, 576]
    b = elementwise_sqrt(a)
    assert b == [1, 3, 24]

def test_exp():
    a = [1, 2, 3]
    b = elementwise_exp(a)
    assert b == [2.718281828459045, 7.38905609893065, 20.085536923187668]

# ----- BINARY -----

def test_add():
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = elementwise_add(a, b)
    assert c == [5, 7, 9]
    
def test_sub():
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = elementwise_sub(a, b)
    assert c == [-3, -3, -3]
    
def test_mul():
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = elementwise_mul(a, b)
    assert c == [4, 10, 18]

def test_div():
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = elementwise_div(a, b)
    assert c == [0.25, 0.4, 0.5]
    
def test_safe_div():
    a = [1, 2, 3]
    b = [4, 0, 6]
    c = elementwise_div(a, b)
    assert c == [0.25, math.inf, 0.5]
    
# ----- TENSORIAL -----

# --- Unary ---

def test_transpose_2D_matrix():
    a = [[1, 2, 3], [4, 5, 6]]
    b = elementwise_transpose_2D(a)
    assert b == [[1, 4], [2, 5], [3, 6]]

def test_transpose_2D_matrix_one_vector():
    a = [[1, 2, 3]]
    b = elementwise_transpose_2D(a)
    assert b == [[1], [2], [3]]
    
def test_transpose_last2_matrix():
    a = [[1, 2, 3], [4, 5, 6]]
    b = transpose_last2(a)
    assert b == [[1, 4], [2, 5], [3, 6]]
    
def test_transpose_last2_matrix_one_vector():
    a = [[1, 2, 3]]
    b = transpose_last2(a)
    assert b == [[1], [2], [3]]
    
def test_transpose_last2_multidim_tensor():
    a = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]
    b = transpose_last2(a)
    assert b == [[[1, 3, 5], [2, 4, 6]], [[7, 9, 11], [8, 10, 12]], [[13, 15, 17], [14, 16, 18]], [[19, 21, 23], [20, 22, 24]]]
    
def test_transpose_nd_matrix():
    a = [[1, 2, 3], [4, 5, 6]]
    b = transpose_nd(a)
    assert b == [[1, 4], [2, 5], [3, 6]]
    
def test_transpose_nd_matrix_one_vector():
    a = [[1, 2, 3]]
    b = transpose_nd(a)
    assert b == [[1], [2], [3]]
    
def test_transpose_nd_matrix_explicit():
    a = [[1, 2, 3], [4, 5, 6]]
    b = transpose_nd(a, (1, 0))
    assert b == [[1, 4], [2, 5], [3, 6]]

def test_transpose_nd_matrix_no_swap():
    a = [[1, 2, 3], [4, 5, 6]]
    b = transpose_nd(a, (0, 1))
    assert b == [[1, 2, 3], [4, 5, 6]]
    
def test_transpose_nd_multidim_tensor():
    a = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]
    b = transpose_nd(a)
    assert b == [[[1, 7, 13, 19], [3, 9, 15, 21], [5, 11, 17, 23]], [[2, 8, 14, 20], [4, 10, 16, 22], [6, 12, 18, 24]]]

# no axes = reverse dims
def test_transpose_nd_multidim_tensor_explicit():
    a = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]
    b = transpose_nd(a, (2, 1, 0))
    assert b == [[[1, 7, 13, 19], [3, 9, 15, 21], [5, 11, 17, 23]], [[2, 8, 14, 20], [4, 10, 16, 22], [6, 12, 18, 24]]]
    
def test_transpose_nd_multidim_tensor_last2():
    a = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]
    b = transpose_nd(a, (0, 2, 1))
    assert b == [[[1, 3, 5], [2, 4, 6]], [[7, 9, 11], [8, 10, 12]], [[13, 15, 17], [14, 16, 18]], [[19, 21, 23], [20, 22, 24]]]
    
def test_reshape():
    a = [[1, 2, 3], [4, 5, 6]]
    b = elementwise_reshape(a, (3, 2))
    assert b == [[1, 2], [3, 4], [5, 6]]
    
def test_reshape_no_change():
    a = [[1, 2, 3], [4, 5, 6]]
    b = elementwise_reshape(a, (2, 3))
    assert b == [[1, 2, 3], [4, 5, 6]]
    
# --- Binary ---

# dot product
def test_elementwise_matmul_vector_vector():
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = elementwise_matmul(a, b)
    assert c == 32

def test_elementwise_matmul_matrix_vector():
    a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    b = [1, 2, 3, 4]
    c = elementwise_matmul(a, b)
    assert c == [30, 70, 110]
    
def test_elementwise_matmul_vector_matrix():
    a = [1, 2, 3]
    b = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    c = elementwise_matmul(a, b)
    assert c == [38, 44, 50, 56]

def test_elementwise_matmul_matrix_matrix():
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    c = elementwise_matmul(a, b)
    assert c == [[38, 44, 50, 56], [83, 98, 113, 128]]
    
def test_elementwise_matmul_batch():
    a = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9],[10, 11, 12]]]
    b = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]
    c = elementwise_matmul(a, b)
    assert c == [[[38, 44, 50, 56], [83, 98, 113, 128]], [[416, 440, 464, 488], [569, 602, 635, 668]]]
    
def test_outer_vector_vector():
    a = [1, 2, 3]
    b = [4, 5, 6, 7, 8]
    c = outer(a, b)
    assert c == [[4, 5, 6, 7, 8], [8, 10, 12, 14, 16], [12, 15, 18, 21, 24]]
    
# Incorrect but not crashed due to Python
def test_outer_matrix_vector():
    a = [[1, 2, 3], [4, 5, 6]]
    b = [1, 2, 3, 4, 5, 6]
    c = outer(a, b)
    assert c == [
        [[1, 2, 3],
         [1, 2, 3, 1, 2, 3],
         [1, 2, 3, 1, 2, 3, 1, 2, 3],
         [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
         [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
         [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]],
        [[4, 5, 6],
         [4, 5, 6, 4, 5, 6],
         [4, 5, 6, 4, 5, 6, 4, 5, 6],
         [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6],
         [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6],
         [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6]]
    ]
    
# Incorrect but not crashed due to Python
def test_outer_vector_matrix():
    a = [1, 2, 3, 4, 5, 6]
    b = [[1, 2, 3], [4, 5, 6]]
    c = outer(a, b)
    assert c == [
        [[1, 2, 3], [4, 5, 6]],
        [[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]],
        [[1, 2, 3, 1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6, 4, 5, 6]],
        [[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6]],
        [[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6]],
        [[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6]]
    ]

# ----- BROADCAST -----

def test_broadcast_shape():
    a = broadcast_shape((3, 4), (4,))
    assert a == (3, 4)
    
def test_broadcast_batch_matmul_shape():
    a = infer_batch_matmul_shape((1, 3, 4), (3, 4, 5))
    assert a == (3, 3, 5)
    
def test_broadcast_to():
    a = [1, 2, 3, 4]
    b = broadcast_to(a, (4,), (3, 4))
    assert b == [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    
# ----- INFER SHAPE -----

def test_infer_matmul_shape_vector_vector():
    a = infer_matmul_shape((3,), (3,))
    assert a == ()
    
def test_infer_matmul_shape_matrix_vector():
    a = infer_matmul_shape((3, 4), (4,))
    assert a == (3,)
    
def test_infer_matmul_shape_vector_matrix():
    a = infer_matmul_shape((3,), (3, 4))
    assert a == (4,)
    
def test_infer_matmul_shape_matrix_matrix():
    a = infer_matmul_shape((5, 3), (3, 4))
    assert a == (5, 4)

def test_infer_shape_after_sum_no_keepdim():
    a = shape_after_sum((2, 3))
    assert a == ()

def test_infer_shape_after_sum_keepdim():
    a = shape_after_sum((2, 3), keepdim=True)
    assert a == (1, 1)
    
def test_infer_shape_after_sum_dim_0_no_keepdim():
    a = shape_after_sum((2, 3), 0)
    assert a == (3,)

def test_infer_shape_after_sum_dim_0_keepdim():
    a = shape_after_sum((2, 3), 0, keepdim=True)
    assert a == (1, 3)
    
# ----- REDUCE -----

def test_sum_dim_no_keepdim():
    a = [[1, 2, 3], [4, 5, 6]]
    b = sum_dim(a)
    assert b == 21
    
def test_sum_dim_keepdim():
    a = [[1, 2, 3], [4, 5, 6]]
    b = sum_dim(a, keepdim=True)
    assert b == [[21]]

def test_sum_dim_0_no_keepdim():
    a = [[1, 2, 3], [4, 5, 6]]
    b = sum_dim(a, 0)
    assert b == [5, 7, 9]

def test_sum_dim_0_keepdim():
    a = [[1, 2, 3], [4, 5, 6]]
    b = sum_dim(a, 0, keepdim=True)
    assert b == [[5, 7, 9]]
    
def test_reduce_to_shape():
    a = [[1, 2, 3], [3, 4, 5]]
    b = reduce_to_shape(a, (2, 3), (3,))
    assert b == [4, 6, 8]

# ----- ERROR -----

def test_error_elementwise_matmul_vector_vector_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch for dot product!"):
        a = [1, 2, 3]
        b = [4, 5, 6, 7]
        c = elementwise_matmul(a, b)
        
def test_error_elementwise_matmul_matrix_vector_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch for matmul!"):
        a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        b = [1, 2, 3, 4, 5]
        c = elementwise_matmul(a, b)
        
def test_error_elementwise_matmul_vector_matrix_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch for matmul!"):
        a = [1, 2, 3]
        b = [[1, 2, 3, 4], [5, 6, 7, 8]]
        c = elementwise_matmul(a, b)
        
def test_error_elementwise_matmul_matrix_matrix_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch for matmul!"):
        a = [[1, 2, 3], [4, 5, 6]]
        b = [[1, 2, 3, 4], [5, 6, 7, 8]]
        c = elementwise_matmul(a, b)
        
def test_error_elementwise_matmul_batch():
    with pytest.raises(ValueError, match="Shape mismatch for matmul!"):
        a = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        b = [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]
        c = elementwise_matmul(a, b)
        
def test_error_broadcast_shape():
    with pytest.raises(ValueError, match="Cannot broadcast!"):
        a = broadcast_shape((3,), (3, 4))        
        
def test_error_infer_batch_matmul_shape():
    with pytest.raises(ValueError, match="Shape mismatch for matmul!"):
        a = infer_batch_matmul_shape((3, 4), (2, 5))
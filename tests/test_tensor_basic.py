from src.tensorNikq import Tensor
import pytest

# ---------- Scalar Tensor ----------

def test_scalar_tensor_shape():
    t = Tensor(1)
    assert t.shape == ()

def test_scalar_tensor_data_int():
    t = Tensor(2)
    assert t.data == 2
    
def test_scalar_tensor_data_float():
    t = Tensor(2.0)
    assert t.data == 2.0
    
def test_scalar_tensor_datatype_int():
    t = Tensor(3)
    assert isinstance(t.data, int)

def test_scalar_tensor_datatype_float():
    t = Tensor(3.5)
    assert isinstance(t.data, float)

def test_scalar_tensor_repr_int():
    t = Tensor(4)
    assert str(t) == "Tensor(data=4, shape=())"
    
def test_scalar_tensor_repr_float():
    t = Tensor(4.78)
    assert str(t) == "Tensor(data=4.78, shape=())"

def test_scalar_tensor_is_scalar():
    t = Tensor(5)
    assert t.is_scalar

# ---------- Vector Tensor ----------

def test_vector_tensor_shape():
    t = Tensor([1, 2, 3])
    assert t.shape == (3,)
    
def test_vector_tensor_data_int():
    t = Tensor([2, 3, 4])
    assert t.data == [2, 3, 4]
    
def test_vector_tensor_data_float():
    t = Tensor([2.31, 3.14, 4.79])
    assert t.data == [2.31, 3.14, 4.79]

def test_vector_tensor_repr_int():
    t = Tensor([3, 4, 5])
    assert str(t) == "Tensor(data=[3, 4, 5], shape=(3))"
    
def test_vector_tensor_repr_float():
    t = Tensor([3.65, 4.15, 5.72])
    assert str(t) == "Tensor(data=[3.65, 4.15, 5.72], shape=(3))"

def test_vector_tensor_is_vector():
    t = Tensor([4, 5, 6])
    assert t.is_vector
    
# ---------- Matrix Tensor ----------

def test_matrix_tensor_shape():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)
    
def test_matrix_tensor_data_int():
    t = Tensor([[2, 3, 4], [5, 6, 7]])
    assert t.data == [[2, 3, 4], [5, 6, 7]]
    
def test_matrix_tensor_data_float():
    t = Tensor([[2.43, 3.25, 4.17], [5.95, 6.84, 7.29]])
    assert t.data == [[2.43, 3.25, 4.17], [5.95, 6.84, 7.29]]

def test_matrix_tensor_repr_int():
    t = Tensor([[3, 4], [5, 6]])
    assert str(t) == "Tensor(data=[[3, 4], [5, 6]], shape=(2, 2))"
    
def test_matrix_tensor_repr_float():
    t = Tensor([[3.78, 4.26], [5.89, 6.13]])
    assert str(t) == "Tensor(data=[[3.78, 4.26], [5.89, 6.13]], shape=(2, 2))"

def test_matrix_tensor_is_matrix():
    t = Tensor([[4, 5, 6], [7, 8, 9]])
    assert t.is_matrix
    
# ---------- Other ----------

def test_empty_list_tensor_shape():
    t = Tensor([])
    assert t.shape == (0,)

def test_empty_list_tensor_data():
    t = Tensor([])
    assert t.data == []
    
def test_multidim_tensor_shape():
    t = Tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]], [[17, 18, 19, 20], [21, 22, 23, 24]]])
    assert t.shape == (3, 2, 4)
    
def test_multidim_tensor_data_int():
    t = Tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]], [[17, 18, 19, 20], [21, 22, 23, 24]]])
    assert t.data == [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]], [[17, 18, 19, 20], [21, 22, 23, 24]]]

def test_multidim_tensor_data_float():
    t = Tensor([[[1.23, 2.34, 3.45, 4.56], [5.67, 6.78, 7.89, 8.91]], [[9.1011, 10.1112, 11.1213, 12.1314], [13.1415, 14.1516, 15.1617, 16.1718]], [[17.1819, 18.192, 19.2021, 20.2122], [21.2223, 22.2324, 23.2425, 24.2526]]])
    assert t.data == [[[1.23, 2.34, 3.45, 4.56], [5.67, 6.78, 7.89, 8.91]], [[9.1011, 10.1112, 11.1213, 12.1314], [13.1415, 14.1516, 15.1617, 16.1718]], [[17.1819, 18.192, 19.2021, 20.2122], [21.2223, 22.2324, 23.2425, 24.2526]]]

def test_multidim_tensor_repr_int():
    t = Tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]], [[17, 18, 19, 20], [21, 22, 23, 24]]])
    assert str(t) == "Tensor(data=[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]], [[17, 18, 19, 20], [21, 22, 23, 24]]], shape=(3, 2, 4))"
    
def test_multidim_tensor_repr_float():
    t = Tensor([[[1.23, 2.34, 3.45, 4.56], [5.67, 6.78, 7.89, 8.91]], [[9.1011, 10.1112, 11.1213, 12.1314], [13.1415, 14.1516, 15.1617, 16.1718]], [[17.1819, 18.192, 19.2021, 20.2122], [21.2223, 22.2324, 23.2425, 24.2526]]])
    assert str(t) == "Tensor(data=[[[1.23, 2.34, 3.45, 4.56], [5.67, 6.78, 7.89, 8.91]], [[9.1011, 10.1112, 11.1213, 12.1314], [13.1415, 14.1516, 15.1617, 16.1718]], [[17.1819, 18.192, 19.2021, 20.2122], [21.2223, 22.2324, 23.2425, 24.2526]]], shape=(3, 2, 4))"

def test_tensor_ndim():
    t1 = Tensor(5)
    t2 = Tensor([1, 2, 3])
    t3 = Tensor([[1, 2, 3], [4, 5, 6]])
    t4 = Tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]], [[17, 18, 19, 20], [21, 22, 23, 24]]])
    assert t1.ndim == 0
    assert t2.ndim == 1
    assert t3.ndim == 2
    assert t4.ndim == 3
    
# ----- GET ITEMS -----
    
def test_vector_tensor_get_item():
    t = Tensor([1, 2, 3])
    item = t[1]
    assert str(item) == "Tensor(data=2, shape=())"
    
def test_matrix_tensor_get_item():
    t = Tensor([[1, 2], [3, 4], [5, 6]])
    item = t[2]
    assert str(item) == "Tensor(data=[5, 6], shape=(2))"
    
def test_multidim_tensor_get_item():
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    item = t[1]
    assert str(item) == "Tensor(data=[[5, 6], [7, 8]], shape=(2, 2))"
    
# ----- TENSOR ERROR -----

def test_error_ragged_tensor_shape():
    with pytest.raises(ValueError, match="Ragged tensor is not supported!"):
        Tensor([[1, 2, 3], [4, 5]])    

def test_error_empty_tensor():
    with pytest.raises(ValueError, match="Data is required!"):
        Tensor() 
    
def test_error_tensor_data_unsupported_str():
    with pytest.raises(TypeError, match="'str' is not supported! Only support 'int' and 'float'."):
        Tensor("Hello, world!")
    
def test_error_tensor_data_unsupported_str_list():
    with pytest.raises(TypeError, match="'str' is not supported! Only support 'int' and 'float'."):
        Tensor(["Hello", ",", "world", "!"])

def test_error_tensor_data_unsupported_tuple():
    with pytest.raises(TypeError, match="'tuple' is not supported! Only support 'int' and 'float'."):
        Tensor((1, 2, 3))
        
def test_error_tensor_data_unsupported_tuple_list():
    with pytest.raises(TypeError, match="'tuple' is not supported! Only support 'int' and 'float'."):
        Tensor([(1, 2), (3, 4, 5), (6)])
        
def test_error_scalar_tensor_get_item():
    with pytest.raises(IndexError, match="Cannot index a 0-dim tensor!"):
        t = Tensor(1)
        item = t[0]
from src import Tensor

# ----- MATH -----

def test_Add_scalar():
    a = Tensor(1)
    b = 2
    c = a + b
    assert str(c) == "Tensor(data=3, shape=())"
    
def test_Reverse_Add_scalar():
    a = 2
    b = Tensor(3)
    c = a + b
    assert str(c) == "Tensor(data=5, shape=())"

def test_Add_vector():
    a = Tensor([1, 2, 3])
    b = [4, 5, 6]
    c = a + b
    assert str(c) == "Tensor(data=[5, 7, 9], shape=(3))"

def test_Reverse_Add_vector():
    a = [1, 2, 3]
    b = Tensor([4, 5, 6])
    c = a + b
    assert str(c) == "Tensor(data=[5, 7, 9], shape=(3))"
    
def test_Add_matrix():
    a = Tensor([[1, 2], [3, 4], [5, 6]])
    b = [[7, 8], [9, 10], [11, 12]]
    c = a + b
    assert str(c) == "Tensor(data=[[8, 10], [12, 14], [16, 18]], shape=(3, 2))"
    
def test_Reverse_Add_matrix():
    a = [[1, 2], [3, 4], [5, 6]]
    b = Tensor([[7, 8], [9, 10], [11, 12]])
    c = a + b
    assert str(c) == "Tensor(data=[[8, 10], [12, 14], [16, 18]], shape=(3, 2))"
    
def test_Add_multidim():
    a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    b = [[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]]
    c = a + b
    assert str(c) == "Tensor(data=[[[14, 16], [18, 20]], [[22, 24], [26, 28]], [[30, 32], [34, 36]]], shape=(3, 2, 2))"
    
def test_Reverse_Add_multidim():
    a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
    b = Tensor([[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]])
    c = a + b
    assert str(c) == "Tensor(data=[[[14, 16], [18, 20]], [[22, 24], [26, 28]], [[30, 32], [34, 36]]], shape=(3, 2, 2))"
    
def test_Sub_scalar():
    a = Tensor(2)
    b = 4
    c = a - b
    assert str(c) == "Tensor(data=-2, shape=())"
    
def test_Reverse_Sub_scalar():
    a = 4
    b = Tensor(2)
    c = a - b
    assert str(c) == "Tensor(data=2, shape=())"
    
def test_Sub_vector():
    a = Tensor([1, 2, 3])
    b = [4, 5, 6]
    c = a - b
    assert str(c) == "Tensor(data=[-3, -3, -3], shape=(3))"

def test_Reverse_Sub_vector():
    a = [4, 5, 6]
    b = Tensor([1, 2, 3])
    c = a - b
    assert str(c) == "Tensor(data=[3, 3, 3], shape=(3))"
    
def test_Sub_matrix():
    a = Tensor([[1, 2], [3, 4], [5, 6]])
    b = [[7, 8], [9, 10], [11, 12]]
    c = a - b
    assert str(c) == "Tensor(data=[[-6, -6], [-6, -6], [-6, -6]], shape=(3, 2))"
    
def test_Reverse_Sub_matrix():
    a = [[7, 8], [9, 10], [11, 12]]
    b = Tensor([[1, 2], [3, 4], [5, 6]])
    c = a - b
    assert str(c) == "Tensor(data=[[6, 6], [6, 6], [6, 6]], shape=(3, 2))"
    
def test_Sub_multidim():
    a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    b = [[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]]
    c = a - b
    assert str(c) == "Tensor(data=[[[-12, -12], [-12, -12]], [[-12, -12], [-12, -12]], [[-12, -12], [-12, -12]]], shape=(3, 2, 2))"
    
def test_Reverse_Sub_multidim():
    a = [[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]]
    b = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    c = a - b
    assert str(c) == "Tensor(data=[[[12, 12], [12, 12]], [[12, 12], [12, 12]], [[12, 12], [12, 12]]], shape=(3, 2, 2))"
    
def test_Mul_scalar():
    a = Tensor(1)
    b = 2
    c = a * b
    assert str(c) == "Tensor(data=2, shape=())"
    
def test_Reverse_Mul_scalar():
    a = 1
    b = Tensor(2)
    c = a * b
    assert str(c) == "Tensor(data=2, shape=())"

def test_Mul_vector():
    a = Tensor([1, 2, 3])
    b = [4, 5, 6]
    c = a * b
    assert str(c) == "Tensor(data=[4, 10, 18], shape=(3))"
    
def test_Reverse_Mul_vector():
    a = [1, 2, 3]
    b = Tensor([4, 5, 6])
    c = a * b
    assert str(c) == "Tensor(data=[4, 10, 18], shape=(3))"
    
def test_Mul_matrix():
    a = Tensor([[1, 2], [3, 4], [5, 6]])
    b = [[7, 8], [9, 10], [11, 12]]
    c = a * b
    assert str(c) == "Tensor(data=[[7, 16], [27, 40], [55, 72]], shape=(3, 2))"
    
def test_Reverse_Mul_matrix():
    a = [[1, 2], [3, 4], [5, 6]]
    b = Tensor([[7, 8], [9, 10], [11, 12]])
    c = a * b
    assert str(c) == "Tensor(data=[[7, 16], [27, 40], [55, 72]], shape=(3, 2))"
    
def test_Mul_multidim():
    a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    b = [[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]]
    c = a * b
    assert str(c) == "Tensor(data=[[[13, 28], [45, 64]], [[85, 108], [133, 160]], [[189, 220], [253, 288]]], shape=(3, 2, 2))"

def test_Reverse_Mul_multidim():
    a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
    b = Tensor([[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]])
    c = a * b
    assert str(c) == "Tensor(data=[[[13, 28], [45, 64]], [[85, 108], [133, 160]], [[189, 220], [253, 288]]], shape=(3, 2, 2))"
    
def test_Div_scalar():
    a = Tensor(1)
    b = 2
    c = a / b
    assert str(c) == "Tensor(data=0.5, shape=())"
    
def test_Reverse_Div_scalar():
    a = 2
    b = Tensor(1)
    c = a / b
    assert str(c) == "Tensor(data=2.0, shape=())"

def test_Div_vector():
    a = Tensor([1, 2, 3])
    b = [4, 5, 6]
    c = a / b
    assert str(c) == "Tensor(data=[0.25, 0.4, 0.5], shape=(3))"

def test_Reverse_Div_vector():
    a = [4, 5, 6]
    b = Tensor([1, 2, 3])
    c = a / b
    assert str(c) == "Tensor(data=[4.0, 2.5, 2.0], shape=(3))"
    
def test_Div_matrix():
    a = Tensor([[1, 2], [3, 4], [5, 6]])
    b = Tensor([[7, 8], [9, 10], [11, 12]])
    c = a / b
    assert str(c) == "Tensor(data=[[0.1429, 0.25], [0.3333, 0.4], [0.4545, 0.5]], shape=(3, 2))"

def test_Reverse_Div_matrix():
    a = [[7, 8], [9, 10], [11, 12]]
    b = Tensor([[1, 2], [3, 4], [5, 6]])
    c = a / b
    assert str(c) == "Tensor(data=[[7.0, 4.0], [3.0, 2.5], [2.2, 2.0]], shape=(3, 2))"
    
def test_Div_multidim():
    a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    b = [[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]]
    c = a / b
    assert str(c) == "Tensor(data=[[[0.0769, 0.1429], [0.2, 0.25]], [[0.2941, 0.3333], [0.3684, 0.4]], [[0.4286, 0.4545], [0.4783, 0.5]]], shape=(3, 2, 2))"
    
def test_Reverse_Div_multidim():
    a = [[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]]
    b = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    c = a / b
    assert str(c) == "Tensor(data=[[[13.0, 7.0], [5.0, 4.0]], [[3.4, 3.0], [2.7143, 2.5]], [[2.3333, 2.2], [2.0909, 2.0]]], shape=(3, 2, 2))"

# ----- TENSORIAL -----
    
def test_MatMul_vector_vector():
    a = Tensor([1, 2, 3])
    b = [4, 5, 6]
    c = a @ b
    assert str(c) == "Tensor(data=32, shape=())"
    
def test_Reverse_MatMul_vector_vector():
    a = [4, 5, 6]
    b = Tensor([1, 2, 3])
    c = a @ b
    assert str(c) == "Tensor(data=32, shape=())"
    
def test_MatMul_matrix_vector():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = [4, 5, 6]
    c = a @ b
    assert str(c) == "Tensor(data=[32, 77], shape=(2))"
    
def test_Reverse_MatMul_matrix_vector():
    a = [[1, 2], [3, 4], [5, 6]]
    b = Tensor([4, 5])
    c = a @ b
    assert str(c) == "Tensor(data=[14, 32, 50], shape=(3))"
    
def test_MatMul_vector_matrix():
    a = Tensor([1, 2, 3])
    b = [[1, 2], [3, 4], [5, 6]]
    c = a @ b
    assert str(c) == "Tensor(data=[22, 28], shape=(2))"
    
def test_Reverse_MatMul_vector_matrix():
    a = [1, 2]
    b = Tensor([[1, 2, 3], [4, 5, 6]])
    c = a @ b
    assert str(c) == "Tensor(data=[9, 12, 15], shape=(3))"
    
def test_MatMul_matrix_matrix():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = [[1, 2], [3, 4], [5, 6]]
    c = a @ b
    assert str(c) == "Tensor(data=[[22, 28], [49, 64]], shape=(2, 2))"
    
def test_Reverse_MatMul_matrix_matrix():
    a = [[1, 2], [3, 4], [5, 6]]
    b = Tensor([[1, 2, 3], [4, 5, 6]])
    c = a @ b
    assert str(c) == "Tensor(data=[[9, 12, 15], [19, 26, 33], [29, 40, 51]], shape=(3, 3))"
    
def test_MatMul_batch_matmul():
    a = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9],[10, 11, 12]]])
    b = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]
    c = a @ b
    assert str(c) == "Tensor(data=[[[38, 44, 50, 56], [83, 98, 113, 128]], [[416, 440, 464, 488], [569, 602, 635, 668]]], shape=(2, 2, 4))"
    
def test_Reverse_MatMul_batch_matmul():
    a = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
    b = Tensor([[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], [[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]]])
    c = a @ b
    assert str(c) == "Tensor(data=[[[15, 18, 21, 24, 27, 30], [31, 38, 45, 52, 59, 66], [47, 58, 69, 80, 91, 102]], [[243, 258, 273, 288, 303, 318], [307, 326, 345, 364, 383, 402], [371, 394, 417, 440, 463, 486]]], shape=(2, 3, 6))"
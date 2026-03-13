from src.tensorNikq import Tensor

# ----- MATH -----

def test_Add_1():
    a = Tensor(1)
    b = Tensor([1, 2, 3])
    c = a + b
    assert str(c) == "Tensor(data=[2, 3, 4], shape=(3))"
    
def test_Add_2():
    a = Tensor([1, 2, 3])
    b = Tensor([[1, 2, 3], [4, 5, 6]])
    c = a + b
    assert str(c) == "Tensor(data=[[2, 4, 6], [5, 7, 9]], shape=(2, 3))"
    
def test_Add_3():
    a = Tensor([[1, 2, 3]])
    b = Tensor([[1, 2, 3], [4, 5, 6]])
    c = a + b
    assert str(c) == "Tensor(data=[[2, 4, 6], [5, 7, 9]], shape=(2, 3))"
    
def test_Add_4():
    a = Tensor([1, 2])
    b = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    c = a + b
    assert str(c) == "Tensor(data=[[[2, 4], [4, 6]], [[6, 8], [8, 10]]], shape=(2, 2, 2))"
    
def test_Sub_1():
    a = Tensor(1)
    b = Tensor([1, 2, 3])
    c = a - b
    assert str(c) == "Tensor(data=[0, -1, -2], shape=(3))"
    
def test_Sub_2():
    a = Tensor([1, 2, 3])
    b = Tensor([[1, 2, 3], [4, 5, 6]])
    c = a - b
    assert str(c) == "Tensor(data=[[0, 0, 0], [-3, -3, -3]], shape=(2, 3))"
    
def test_Sub_3():
    a = Tensor([[1, 2, 3]])
    b = Tensor([[1, 2, 3], [4, 5, 6]])
    c = a - b
    assert str(c) == "Tensor(data=[[0, 0, 0], [-3, -3, -3]], shape=(2, 3))"
    
def test_Sub_4():
    a = Tensor([1, 2])
    b = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    c = a - b
    assert str(c) == "Tensor(data=[[[0, 0], [-2, -2]], [[-4, -4], [-6, -6]]], shape=(2, 2, 2))"
    
def test_Mul_1():
    a = Tensor(1)
    b = Tensor([1, 2, 3])
    c = a * b
    assert str(c) == "Tensor(data=[1, 2, 3], shape=(3))"
    
def test_Mul_2():
    a = Tensor([1, 2, 3])
    b = Tensor([[1, 2, 3], [4, 5, 6]])
    c = a * b
    assert str(c) == "Tensor(data=[[1, 4, 9], [4, 10, 18]], shape=(2, 3))"
    
def test_Mul_3():
    a = Tensor([[1, 2, 3]])
    b = Tensor([[1, 2, 3], [4, 5, 6]])
    c = a * b
    assert str(c) == "Tensor(data=[[1, 4, 9], [4, 10, 18]], shape=(2, 3))"
    
def test_Mul_4():
    a = Tensor([1, 2])
    b = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    c = a * b
    assert str(c) == "Tensor(data=[[[1, 4], [3, 8]], [[5, 12], [7, 16]]], shape=(2, 2, 2))"
    
def test_Div_1():
    a = Tensor(1)
    b = Tensor([1, 2, 3])
    c = a / b
    assert str(c) == "Tensor(data=[1.0, 0.5, 0.3333], shape=(3))"
    
def test_Div_2():
    a = Tensor([1, 2, 3])
    b = Tensor([[1, 2, 3], [4, 5, 6]])
    c = a / b
    assert str(c) == "Tensor(data=[[1.0, 1.0, 1.0], [0.25, 0.4, 0.5]], shape=(2, 3))"
    
def test_Div_3():
    a = Tensor([[1, 2, 3]])
    b = Tensor([[1, 2, 3], [4, 5, 6]])
    c = a / b
    assert str(c) == "Tensor(data=[[1.0, 1.0, 1.0], [0.25, 0.4, 0.5]], shape=(2, 3))"
    
def test_Div_4():
    a = Tensor([1, 2])
    b = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    c = a / b
    assert str(c) == "Tensor(data=[[[1.0, 1.0], [0.3333, 0.5]], [[0.2, 0.3333], [0.1429, 0.25]]], shape=(2, 2, 2))"
    
# ----- TENSORIAL -----

def test_MatMul_batch_matmul():
    a = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9],[10, 11, 12]]])
    b = Tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]], [[[25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]], [[37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48]]], [[[49, 50, 51, 52], [53, 54, 55, 56], [57, 58, 59, 60]], [[61, 62, 63, 64], [65, 66, 67, 68], [69, 70, 71, 72]]]])
    c = a @ b
    assert str(c) == "Tensor(data=[[[[38, 44, 50, 56], [83, 98, 113, 128]], [[416, 440, 464, 488], [569, 602, 635, 668]]], [[[182, 188, 194, 200], [443, 458, 473, 488]], [[992, 1016, 1040, 1064], [1361, 1394, 1427, 1460]]], [[[326, 332, 338, 344], [803, 818, 833, 848]], [[1568, 1592, 1616, 1640], [2153, 2186, 2219, 2252]]]], shape=(3, 2, 2, 4))"
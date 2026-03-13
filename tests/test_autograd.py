from src import Tensor

# ----- UNARY -----

def test_Neg_autograd_1():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = -a
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[-1.0, -1.0, -1.0], shape=(3))"

# Actually, the second operator is a subtraction operator     
def test_Neg_autograd_2():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = -a -a
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[-2.0, -2.0, -2.0], shape=(3))"
    
def test_Neg_autograd_1():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = -a
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[-1.0, -1.0, -1.0], shape=(3))"

# Actually, the second operator is a subtraction operator     
def test_Neg_autograd_2():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = -a -a
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[-2.0, -2.0, -2.0], shape=(3))"
    
def test_Sum_no_keepdim_autograd():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = a.sum()
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[1.0, 1.0, 1.0], shape=(3))"

def test_Sum_keepdim_autograd():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = a.sum(keepdim=True)
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[1.0, 1.0, 1.0], shape=(3))"
    
def test_Sum_dim_0_no_keepdim_autograd():
    a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = a.sum(0)
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], shape=(2, 3))"

def test_Sum_dim_0_keepdim_autograd():
    a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = a.sum(0, keepdim=True)
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], shape=(2, 3))"

# ----- BINARY -----

def test_Add_autograd_1():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = a + 2
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[1.0, 1.0, 1.0], shape=(3))"
    
def test_Add_autograd_2():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = a + a
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[2.0, 2.0, 2.0], shape=(3))"
    
def test_Add_autograd_3():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1, 2, 3], requires_grad=True)
    c = a + b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[1.0, 1.0, 1.0], shape=(3))"
    assert str(b.grad) == "Tensor(data=[1.0, 1.0, 1.0], shape=(3))"
    
def test_Add_autograd_4():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a + b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[2.0, 2.0, 2.0], shape=(3))"
    assert str(b.grad) == "Tensor(data=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], shape=(2, 3))"

def test_Sub_autograd_1():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = a - 2
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[1.0, 1.0, 1.0], shape=(3))"
    
def test_Sub_autograd_2():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = a - a
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[0.0, 0.0, 0.0], shape=(3))"
    
def test_Sub_autograd_3():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1, 2, 3], requires_grad=True)
    c = a - b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[1.0, 1.0, 1.0], shape=(3))"
    assert str(b.grad) == "Tensor(data=[-1.0, -1.0, -1.0], shape=(3))"
    
def test_Sub_autograd_4():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a - b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[2.0, 2.0, 2.0], shape=(3))"
    assert str(b.grad) == "Tensor(data=[[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]], shape=(2, 3))"
    
def test_Sub_autograd_1():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = a - 2
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[1.0, 1.0, 1.0], shape=(3))"
    
def test_Sub_autograd_2():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = a - a
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[0.0, 0.0, 0.0], shape=(3))"
    
def test_Sub_autograd_3():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1, 2, 3], requires_grad=True)
    c = a - b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[1.0, 1.0, 1.0], shape=(3))"
    assert str(b.grad) == "Tensor(data=[-1.0, -1.0, -1.0], shape=(3))"
    
def test_Sub_autograd_4():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a - b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[2.0, 2.0, 2.0], shape=(3))"
    assert str(b.grad) == "Tensor(data=[[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]], shape=(2, 3))"
    
def test_Mul_autograd_1():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = a * 2
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[2.0, 2.0, 2.0], shape=(3))"
    
def test_Mul_autograd_2():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = a * a
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[2.0, 4.0, 6.0], shape=(3))"
    
def test_Mul_autograd_3():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1, 2, 3], requires_grad=True)
    c = a * b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[1.0, 2.0, 3.0], shape=(3))"
    assert str(b.grad) == "Tensor(data=[1.0, 2.0, 3.0], shape=(3))"
    
def test_Mul_autograd_4():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a * b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[5.0, 7.0, 9.0], shape=(3))"
    assert str(b.grad) == "Tensor(data=[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], shape=(2, 3))"
    
def test_Div_autograd_1():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = a / 2
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[0.5, 0.5, 0.5], shape=(3))"
    
def test_Div_autograd_2():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = a / a
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[0.0, 0.0, 0.0], shape=(3))"
    
def test_Div_autograd_3():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1, 2, 3], requires_grad=True)
    c = a / b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[1.0, 0.5, 0.3333], shape=(3))"
    assert str(b.grad) == "Tensor(data=[-1.0, -0.5, -0.3333], shape=(3))"
    
def test_Div_autograd_4():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a / b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[1.25, 0.7, 0.5], shape=(3))"
    assert str(b.grad) == "Tensor(data=[[-1.0, -0.5, -0.3333], [-0.0625, -0.08, -0.0833]], shape=(2, 3))"
    
# ----- TENSORIAL -----

# --- Unary ---

def test_Transpose():
    a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = a.transpose()
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], shape=(2, 3))"

def test_Reshape():
    a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = a.reshape(3, 2)
    b.backward()
    
    assert str(a.grad) == "Tensor(data=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], shape=(2, 3))"
    
def test_MatMul_vector_vector():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)
    c = a @ b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[4.0, 5.0, 6.0], shape=(3))"
    assert str(b.grad) == "Tensor(data=[1.0, 2.0, 3.0], shape=(3))"
    
def test_MatMul_matrix_vector():
    a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)
    c = a @ b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]], shape=(2, 3))"
    assert str(b.grad) == "Tensor(data=[5.0, 7.0, 9.0], shape=(3))"
    
def test_MatMul_vector_matrix():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
    c = a @ b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[3.0, 7.0, 11.0], shape=(3))"
    assert str(b.grad) == "Tensor(data=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], shape=(3, 2))"
    
def test_MatMul_matrix_matrix():
    a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
    c = a @ b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[[3.0, 7.0, 11.0], [3.0, 7.0, 11.0]], shape=(2, 3))"
    assert str(b.grad) == "Tensor(data=[[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]], shape=(3, 2))"
    
def test_MatMul_batch_matmul_1():
    a = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9],[10, 11, 12]]], requires_grad=True)
    b = Tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]], requires_grad=True)
    c = a @ b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[[[10.0, 26.0, 42.0], [10.0, 26.0, 42.0]], [[58.0, 74.0, 90.0], [58.0, 74.0, 90.0]]], shape=(2, 2, 3))"
    assert str(b.grad) == "Tensor(data=[[[5.0, 5.0, 5.0, 5.0], [7.0, 7.0, 7.0, 7.0], [9.0, 9.0, 9.0, 9.0]], [[17.0, 17.0, 17.0, 17.0], [19.0, 19.0, 19.0, 19.0], [21.0, 21.0, 21.0, 21.0]]], shape=(2, 3, 4))"

def test_MatMul_batch_matmul_2():
    a = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9],[10, 11, 12]]], requires_grad=True)
    b = Tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]], [[[25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]], [[37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48]]], [[[49, 50, 51, 52], [53, 54, 55, 56], [57, 58, 59, 60]], [[61, 62, 63, 64], [65, 66, 67, 68], [69, 70, 71, 72]]]], requires_grad=True)
    c = a @ b
    c.backward()
    
    assert str(a.grad) == "Tensor(data=[[[318.0, 366.0, 414.0], [318.0, 366.0, 414.0]], [[462.0, 510.0, 558.0], [462.0, 510.0, 558.0]]], shape=(2, 2, 3))"
    assert str(b.grad) == "Tensor(data=[[[[5.0, 5.0, 5.0, 5.0], [7.0, 7.0, 7.0, 7.0], [9.0, 9.0, 9.0, 9.0]], [[17.0, 17.0, 17.0, 17.0], [19.0, 19.0, 19.0, 19.0], [21.0, 21.0, 21.0, 21.0]]], [[[5.0, 5.0, 5.0, 5.0], [7.0, 7.0, 7.0, 7.0], [9.0, 9.0, 9.0, 9.0]], [[17.0, 17.0, 17.0, 17.0], [19.0, 19.0, 19.0, 19.0], [21.0, 21.0, 21.0, 21.0]]], [[[5.0, 5.0, 5.0, 5.0], [7.0, 7.0, 7.0, 7.0], [9.0, 9.0, 9.0, 9.0]], [[17.0, 17.0, 17.0, 17.0], [19.0, 19.0, 19.0, 19.0], [21.0, 21.0, 21.0, 21.0]]]], shape=(3, 2, 3, 4))"
    
# ----- COMBINATION -----

def test_chain_basic():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)

    c = a * b
    d = c + a
    e = d.sum()
    e.backward()

    assert str(a.grad) == "Tensor(data=[5.0, 6.0, 7.0], shape=(3))"
    assert str(b.grad) == "Tensor(data=[1.0, 2.0, 3.0], shape=(3))"
    
def test_square_sum():
    a = Tensor([1, 2, 3], requires_grad=True)

    b = a * a
    c = b.sum()
    c.backward()

    assert str(a.grad) == "Tensor(data=[2.0, 4.0, 6.0], shape=(3))"

def test_dot_sum():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)

    c = a @ b
    d = c * 2
    d.backward()

    assert str(a.grad) == "Tensor(data=[8.0, 10.0, 12.0], shape=(3))"
    assert str(b.grad) == "Tensor(data=[2.0, 4.0, 6.0], shape=(3))"

def test_matvec_sum():
    a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor([1, 2, 3], requires_grad=True)

    c = a @ b
    d = c.sum()
    d.backward()

    assert str(a.grad) == "Tensor(data=[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], shape=(2, 3))"
    assert str(b.grad) == "Tensor(data=[5.0, 7.0, 9.0], shape=(3))"
    
def test_matmul_chain():
    a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)

    c = a @ b
    d = c.sum()
    d.backward()

    assert str(a.grad) == "Tensor(data=[[3.0, 7.0, 11.0], [3.0, 7.0, 11.0]], shape=(2, 3))"
    assert str(b.grad) == "Tensor(data=[[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]], shape=(3, 2))"
    
def test_complex_graph():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)

    c = a * b
    d = c + a
    e = d * b
    f = e.sum()
    f.backward()

    assert str(a.grad) == "Tensor(data=[20.0, 30.0, 42.0], shape=(3))"
    assert str(b.grad) == "Tensor(data=[9.0, 22.0, 39.0], shape=(3))"
    
def test_broadcast_add_sum():
    a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor([1, 1, 1], requires_grad=True)

    c = a + b
    d = c.sum()
    d.backward()

    assert str(a.grad) == "Tensor(data=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], shape=(2, 3))"
    assert str(b.grad) == "Tensor(data=[2.0, 2.0, 2.0], shape=(3))"
    
def test_linear_layer():
    x = Tensor([1, 2, 3], requires_grad=True)
    W = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)

    y = x @ W
    z = y * y
    loss = z.sum()
    loss.backward()

    assert y.data == [22, 28]
    assert loss.data == 1268
    assert str(y.grad) == "Tensor(data=[44.0, 56.0], shape=(2))"
    assert str(x.grad) == "Tensor(data=[156.0, 356.0, 556.0], shape=(3))"
    assert str(W.grad) == "Tensor(data=[[44.0, 56.0], [88.0, 112.0], [132.0, 168.0]], shape=(3, 2))"

# ----- OTHER -----

def test_no_requires_grad_autograd():
    a = Tensor([1, 2, 3])
    b = a + 2
    b.backward()
    
    assert a.grad == None
    
def test_output_autograd():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = a + 2
    b.backward()
    
    assert str(b.grad) == "Tensor(data=[1.0, 1.0, 1.0], shape=(3))"
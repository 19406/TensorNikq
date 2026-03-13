from ..broadcast_ops import broadcast_shape, infer_batch_matmul_shape

# ---------- Matrix Multiplication ----------

def outer(a, b):
    return [[x * y for y in b] for x in a]

def infer_matmul_shape(shape_a, shape_b):
    # vector . vector
    if len(shape_a) == 1 and len(shape_b) == 1:
        if shape_a[0] != shape_b[0]: raise ValueError("Shape mismatch for dot product!")
        return ()

    # matrix @ vector
    if len(shape_a) == 2 and len(shape_b) == 1:
        if shape_a[1] != shape_b[0]: raise ValueError("Shape mismatch for matmul!")
        return (shape_a[0],)

    # vector @ matrix
    if len(shape_a) == 1 and len(shape_b) == 2:
        if shape_a[0] != shape_b[0]: raise ValueError("Shape mismatch for matmul!")
        return (shape_b[1],)

    # general
    if shape_a[-1] != shape_b[-2]: raise ValueError("Shape mismatch for matmul!")
    return infer_batch_matmul_shape(shape_a, shape_b)

def elementwise_matmul(a, b):
    # ----- vector · vector -----
    if not isinstance(a[0], list) and not isinstance(b[0], list):
        if len(a) != len(b):
            raise ValueError("Shape mismatch for dot product!")
        return sum(a[i] * b[i] for i in range(len(a)))

    # ----- matrix @ vector -----
    if isinstance(a[0], list) and not isinstance(b[0], list):

        m = len(a)
        k = len(a[0])

        if k != len(b):
            raise ValueError("Shape mismatch for matmul!")

        return [
            sum(a[i][j] * b[j] for j in range(k))
            for i in range(m)
        ]

    # ----- vector @ matrix -----
    if not isinstance(a[0], list) and isinstance(b[0], list):

        n = len(a)
        p = len(b[0])

        if n != len(b):
            raise ValueError("Shape mismatch for matmul!")

        return [
            sum(a[i] * b[i][j] for i in range(n))
            for j in range(p)
        ]

    # ----- matrix @ matrix -----
    if isinstance(a[0][0], (int, float)) and isinstance(b[0][0], (int, float)):

        m = len(a)
        n = len(a[0])
        p = len(b[0])

        if n != len(b):
            raise ValueError("Shape mismatch for matmul!")

        result = []
        for i in range(m):
            row = []
            for j in range(p):
                s = 0
                for k in range(n):
                    s += a[i][k] * b[k][j]
                row.append(s)
            result.append(row)

        return result

    # ----- batch matmul -----

    if len(a) != len(b): raise ValueError("Batch dimensions mismatch!")

    return [
        elementwise_matmul(a[i], b[i])
        for i in range(len(a))
    ]
import math

def elementwise_unary(a, op):
    # If scalar
    if not isinstance(a, list): return op(a)

    # If list
    return [elementwise_unary(x, op) for x in a]

def elementwise_binary(a, b, op):
    # If both a and b are scalars
    if not isinstance(a, list) and not isinstance(b, list): return op(a, b)

    # If a - list; b - scalar
    if isinstance(a, list) and not isinstance(b, list):
        return [elementwise_binary(x, b, op) for x in a]

    # If a - scalar; b - list
    if not isinstance(a, list) and isinstance(b, list):
        return [elementwise_binary(a, y, op) for y in b]
    
    # If both a and b are lists
    return [elementwise_binary(x, y, op) for x, y in zip(a, b)]

# Safe Division
def safe_div(x, y):
    if y == 0:
        if x == 0: return math.nan
        return math.inf if x > 0 else -math.inf
    return x / y
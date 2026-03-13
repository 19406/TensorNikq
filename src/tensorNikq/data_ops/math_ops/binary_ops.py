from .generic_op import elementwise_binary, safe_div

# ---------- Element-wise ----------

# Addition
def elementwise_add(a, b): return elementwise_binary(a, b, lambda x, y: x + y)

# Subtraction
def elementwise_sub(a, b): return elementwise_binary(a, b, lambda x, y: x - y)

# Multiplication
def elementwise_mul(a, b): return elementwise_binary(a, b, lambda x, y: x * y)

# Division
def elementwise_div(a, b): return elementwise_binary(a, b, safe_div)

# ---------- Tensorial ----------
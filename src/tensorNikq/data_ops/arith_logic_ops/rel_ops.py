from .generic_op import elementwise_binary

# ---------- Element-wise ----------

# Greater than
def elementwise_gt(a, b): return elementwise_binary(a, b, lambda x, y: float(x > y))

# Greater than or equal
def elementwise_ge(a, b): return elementwise_binary(a, b, lambda x, y: float(x >= y))

# Less than
def elementwise_lt(a, b): return elementwise_binary(a, b, lambda x, y: float(x < y))

# Less than or equal
def elementwise_le(a, b): return elementwise_binary(a, b, lambda x, y: float(x <= y))

# Max
def elementwise_max(a, b): return elementwise_binary(a, b, max)

# Min
def elementwise_min(a, b): return elementwise_binary(a, b, min)
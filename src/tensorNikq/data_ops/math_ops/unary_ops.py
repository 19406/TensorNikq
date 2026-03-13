from .generic_op import elementwise_unary
import math

# ---------- Unary ----------

# Negation
def elementwise_neg(a): return elementwise_unary(a, lambda x: -x)

# Square
def elementwise_square(a): return elementwise_unary(a, lambda x: x * x)

# Square Root
def elementwise_sqrt(a): return elementwise_unary(a, math.sqrt)

# Exp
def elementwise_exp(a): return elementwise_unary(a, math.exp)

# ---------- Tensorial ----------
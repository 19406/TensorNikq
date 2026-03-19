from .generic_op import elementwise_unary

# Inversion
def elementwise_not(a): return elementwise_unary(a, lambda x: not x)
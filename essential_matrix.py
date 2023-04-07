from sympy import *


E = MatrixSymbol('E', 3, 3)
X1 = MatrixSymbol('X1', 3, 1)
X2 = MatrixSymbol('X2', 3, 1)


res = X2.T * E * X1
print(res.as_explicit())
print(res.diff(E).as_explicit())

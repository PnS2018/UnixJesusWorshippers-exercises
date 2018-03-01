# SESSION 1, EXERCISES
import numpy as np
from keras import backend as K

# EX.1

a=K.placeholder(shape=(5,))
b=K.placeholder(shape=(5,))
c=K.placeholder(shape=(5,))

funcOut= a**2 + b**2 + c**2 + 2*b*c

sumx= K.function(inputs=(a, b, c), outputs=(funcOut,))

# EX.2

# EX.3




# EX.4